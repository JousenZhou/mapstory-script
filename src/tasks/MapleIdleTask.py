import os  # 导入标准库 os，用于按核心数决定并发匹配线程数。
import threading  # 导入标准库 threading，用于并发匹配线程池懒加载的互斥锁。
import time  # 导入标准库 time，用于定时位移、转身与攻击节奏计时。
from concurrent.futures import ThreadPoolExecutor  # 导入线程池，用于把一帧内彼此独立的模板匹配铺到多个 CPU 核心上并发执行。

import cv2  # 导入 OpenCV，用于镜像模板匹配和实时画面标注绘制。
import numpy as np  # 导入 NumPy，用于镜像匹配的阈值筛选。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from ok.feature.Box import Box  # 导入 Box 类，用于镜像匹配结果的包装。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

from src.tasks.MyBaseTask import MyBaseTask  # 导入项目任务基类，导入时会同时生效标注文件 UTF-8 读取补丁。

# 测谎监控与自动解题已抽离为独立服务 src/liedetector/service.py（由 Globals 后台常驻），
# 任务不再直接解测谎，只需把持有的按键上报给服务（见 pop_held_keys）以便暂停时被释放。

MOVE_LEFT_KEY = "left"  # 左方向键：单击用于换方向，定时位移时按住用于移动。
MOVE_RIGHT_KEY = "right"  # 右方向键：单击用于换方向，定时位移时按住用于移动。

CAPTURE_FPS = 30  # 截图采集固定帧率：子类 MaplePatrolTask 的主循环节拍按该帧率复用。
CAPTURE_MIN_INTERVAL = 1.0 / CAPTURE_FPS  # 固定帧间隔秒数（约 0.0333）。
MATCH_WORKERS = max(2, min(12, os.cpu_count() or 4))  # 并发匹配线程数：OpenCV 的 matchTemplate 自身完全不用多核，线程数必须覆盖每帧提交的匹配任务数（角色原始+镜像、每个怪物分类原始+镜像、朝向左右、小地图，2 个怪物分类时共 9 个），任务数超过线程数就要排第二波反而更慢；实测 16 核机器上 8 线程 48.9ms、12 线程 32.8ms（4.77 倍加速），12 之后受内存带宽限制不再提升，并给截图采集与界面渲染留出核心。

_MATCH_POOL = None  # 并发匹配线程池，全进程共享一份并跨帧复用，避免每帧重建线程。
_MATCH_POOL_LOCK = threading.Lock()  # 保护线程池懒加载的互斥锁。


def get_match_pool():  # 取并发匹配线程池，首次调用时创建。
    global _MATCH_POOL  # 需要写模块级变量。
    if _MATCH_POOL is None:  # 双重检查锁定，避免多个任务同时启动时重复创建线程池。
        with _MATCH_POOL_LOCK:  # 加锁后再确认一次。
            if _MATCH_POOL is None:  # 确实还没有线程池。
                _MATCH_POOL = ThreadPoolExecutor(max_workers=MATCH_WORKERS, thread_name_prefix="match")  # 线程统一命名，便于日志与堆栈排查。
    return _MATCH_POOL  # 返回共享线程池。


class MatchBatch:  # 一帧画面全部模板匹配的并发批次：先一次性提交彼此独立的匹配，再按串行执行时的原顺序取结果，检测结果与串行完全一致。

    def __init__(self):  # 构造空批次。
        self._futures = {}  # 匹配键到 Future 的映射。

    def submit(self, key, fn):  # 提交一个匹配任务到线程池，键重复时后提交的覆盖前者。
        self._futures[key] = get_match_pool().submit(fn)  # 立即投递，任务在后台线程开始执行（matchTemplate 会释放 GIL）。

    def get(self, key, default=None):  # 阻塞取指定键的匹配结果，任务内异常原样抛出，与串行调用语义一致。
        future = self._futures.get(key)  # 取对应的 Future。
        if future is None:  # 该匹配本轮未提交（如朝向校准未到间隔）。
            return default  # 返回默认值。
        return future.result()  # 等待完成并透传原始异常。

    def cancel(self):  # 放弃本批次全部未取结果的匹配：排队中的直接取消，已在线程上运行的无法中断会自行跑完。
        for future in self._futures.values():  # 逐个处理。
            future.cancel()  # 取消尚未开始的任务，不再占用线程池。
        self._futures.clear()  # 清空引用，本帧画面随即可被回收。


class MapleIdleTask(MyBaseTask):  # 定义冒险岛挂机任务，继承项目基类。

    def __init__(self, *args, **kwargs):  # 构造函数，先初始化父类再设置任务元数据。
        super().__init__(*args, **kwargs)  # 必须先调用父类构造。
        self.name = "单点单方向挂机(原地攻击不停)"  # 任务显示名称。
        self.description = "Single-spot camping: stay in place, single-tap direction key only to turn, keep attacking the nearest monster within attack range until it dies; supports optional periodic reposition moves and turn-around attacks; also shows live vision.  # 任务描述：单点挂机，不持续移动，仅单击换方向，攻击范围内最近怪直到消失，支持定时位移与定时转身攻击；实时推送带标注画面。测谎由独立服务值守，与本任务解耦。"
        self.icon = FluentIcon.FLAG  # 任务图标。
        self._held_key = None  # 当前持续按住的攻击键（近战或常规）；改为实例属性，独立测谎服务暂停任务时可读取并通过 pop_held_keys 释放它。
        for key in ("Character Feature", "Character Threshold", "Attack Key", "Melee Attack Key", "Melee Distance",  # 角色/怪物参数已搬到看板统一配置（见 src/dashboard_store.py），任务页不再展示。
                    "Attack Range X Min", "Attack Range X Max", "Attack Range Y Min", "Attack Range Y Max",
                    "Del Key Interval", "Monster Features", "Monster Threshold", "Monster Mirror Threshold"):
            self.default_config.pop(key, None)  # 移除默认值，框架加载旧配置时会自动清理已保存的旧键。
            self.config_description.pop(key, None)  # 同步移除帮助文本。
        self.default_config.update({  # 用户可在 GUI 编辑的配置项（仅保留动作节奏类，角色/怪物/测谎参数在看板页签配置）。
            "Move Interval": 30.0,  # 每隔该秒数停止全部状态做一次位移：停攻 1 秒→朝角色朝向反向移动 Move Away Seconds 秒→攻击一下→等 1 秒→再反方向移动 Move Back Seconds 秒，设为 0 表示禁用。
            "Move Away Seconds": 1.0,  # 位移第一段：朝角色朝向的反向按住方向键移动的秒数，支持 3 位小数（毫秒级）。
            "Move Back Seconds": 1.0,  # 位移第二段：攻击后停顿 1 秒再反方向（即朝向方向）按住方向键移动的秒数，支持 3 位小数（毫秒级）。
            "Turn Interval": 0.0,  # 每隔 x 秒做一次转身攻击：停止攻击等 1 秒→单击方向键转身→攻击一下→等 0.5 秒→再单击反方向键转身归位，设为 0 表示禁用。
            "Use Gray Scale": True,  # 是否转灰度匹配，对颜色差异更稳定。
            "Frame Interval": 0.05,  # 每帧处理之间的最小间隔秒数，控制检测节奏。
        })
        self.config_description.update({  # 各配置项的帮助文本。
            "Move Interval": "Seconds between reposition moves: stop attacking and wait 1s, move opposite to character facing for Move Away Seconds, attack once, wait 1s, then move back for Move Back Seconds; 0 disables it. 每隔该秒数停攻等 1 秒后位移一次（朝朝向反向移动、攻击一下、等 1 秒、再反方向移回），0 禁用。",
            "Move Away Seconds": "Hold seconds for the first leg, moving opposite to character facing, supports 3 decimals. 位移第一段：朝角色朝向反向移动的秒数，支持 3 位小数。",
            "Move Back Seconds": "Hold seconds for the second leg after attacking once and waiting 1s, moving in the opposite direction of the first leg, supports 3 decimals. 位移第二段：攻击一下停顿 1 秒后再反方向移动的秒数，支持 3 位小数。",
            "Turn Interval": "Every x seconds do a turn-around attack: stop attacking and wait 1s, tap direction key to turn, attack once, wait 0.5s, tap the opposite direction key to turn back; 0 disables it. 每隔 x 秒做一次转身攻击：停止攻击等 1 秒→单击方向键转身→攻击一下→等 0.5 秒→再单击反方向键转身归位，0 禁用。",
            "Use Gray Scale": "Match in grayscale, more robust to color differences. 是否转灰度匹配，对颜色差异更稳定。",
            "Frame Interval": "Minimum seconds between processed frames. 每帧处理之间的最小间隔秒数。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None（按键与近战距离校验已随配置项搬到看板）。
        return None  # 剩余配置项不做额外校验。

    def run(self):  # 任务运行入口。
        self.apply_shared_config()  # 从看板读取共享的角色/怪物/测谎参数覆盖任务配置（单一数据源）。
        char_name = self.config.get("Character Feature")  # 读取角色标注分类名。
        monster_names = self.parse_monster_names(self.config.get("Monster Features"))  # 解析逗号分隔的怪物分类名列表。
        frame = self.wait_frame()  # 先取到一帧画面，让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面时无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示取不到画面。
            return  # 直接结束任务。
        if not monster_names:  # 未配置任何怪物分类时无法运行。
            self.log_warning("No monster feature configured. 未配置怪物分类名，任务退出。")  # 提示配置缺失。
            return  # 直接结束任务。
        missing = [name for name in [char_name] + monster_names if not self.feature_ready(name)]  # 检查角色与全部怪物标注是否存在。
        if missing:  # 必要标注缺失时无法运行。
            self.log_warning(f"Template not ready, please annotate in the Template tab: {', '.join(missing)}. 模板未就绪，请先在模板页标注：{'、'.join(missing)}。")  # 提示用户去模板页标注。
            return  # 标注不可用时直接结束任务。
        attack_key = self.config.get("Attack Key")  # 读取常规攻击按键，横向距离超过近战距离时按住它。
        melee_key = self.config.get("Melee Attack Key")  # 读取近战攻击按键，横向距离在近战距离内时按住它。
        melee_distance = float(self.config.get("Melee Distance") or 0)  # 读取近战距离（像素），横向距离绝对值不超过该值用近战键。
        attack_x_min, attack_x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 读取攻击区域左右边界（符号化像素），填反时自动交换。
        attack_y_min, attack_y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 读取攻击区域上下边界（符号化像素），填反时自动交换。
        del_interval = float(self.config.get("Del Key Interval") or 0)  # 读取自动按 Del 键的间隔秒数，0 表示禁用。
        last_del_time = time.time()  # 上次按 Del 键的时间，从任务启动开始计时。
        move_interval = float(self.config.get("Move Interval") or 0)  # 读取定时位移的间隔秒数，0 表示禁用。
        last_move_time = time.time()  # 上次做位移的时间，从任务启动开始计时。
        move_away_seconds = float(self.config.get("Move Away Seconds") or 1)  # 读取位移第一段时长：朝朝向反向移动的秒数。
        move_back_seconds = float(self.config.get("Move Back Seconds") or 1)  # 读取位移第二段时长：反方向移回的秒数。
        turn_interval = float(self.config.get("Turn Interval") or 0)  # 读取转身攻击策略的间隔秒数 x，0 表示禁用。
        last_turn_time = time.time()  # 上次做转身攻击的时间，从任务启动开始计时。
        gpu = self.build_gpu_matcher(char_name, monster_names)  # 尝试构建 GPU 匹配器并注册全部模板，失败返回 None 走 CPU。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，只在需要换向时单击方向键。
        self._held_key = None  # 当前持续按住的攻击键（近战或常规），切换/换向/目标消失/退出时必须松开它。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        try:  # 包裹主循环，退出时兜底松开持续按住的攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                if del_interval > 0 and time.time() - last_del_time >= del_interval:  # 到达定时间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                if move_interval > 0 and time.time() - last_move_time >= move_interval:  # 到达位移间隔时停止全部状态做一次位移。
                    last_move_time = time.time()  # 重置位移计时。
                    if self._held_key is not None:  # 先松开持续按住的攻击键，位移期间不攻击。
                        self.send_key_up(self._held_key)  # 松开当前攻击键。
                        self._held_key = None  # 清空按住状态。
                    self.info_set("Status", "位移中")  # 在 GUI 显示位移状态。
                    self.sleep(1.0)  # 停止攻击后等待 1 秒再做位移，等攻击后摇结束避免位移被吞。
                    facing_assumed = False  # 本次位移的朝向是否来自假定，假定值不可信需在结束后清空。
                    if facing is None:  # 从未转身过导致朝向未知，先用角色模板匹配探测，失败才假定朝右。
                        facing = self.detect_facing(char_name)  # 原始模板命中=朝右、镜像命中=朝左。
                        if facing is None:  # 探测不到角色时退回假定。
                            facing = 1  # 假定角色当前朝右。
                            facing_assumed = True  # 标记朝向为假定值。
                            self.log_info("Facing unknown before reposition, assume facing right. 位移前朝向未知，假定角色朝右。")  # 记录朝向假定供排查。
                    away_key = MOVE_LEFT_KEY if facing == 1 else MOVE_RIGHT_KEY  # 第一段按键：角色朝向的反方向。
                    back_key = MOVE_RIGHT_KEY if facing == 1 else MOVE_LEFT_KEY  # 第二段按键：第一段的反方向，即朝向方向。
                    self.send_key(away_key, down_time=move_away_seconds)  # 按住方向键朝朝向反向移动 y 秒，首次按下会先转身再移动。
                    facing = -facing  # 第一段移动后角色已转身，朝向与原朝向相反。
                    self.send_key(attack_key, down_time=0.2)  # 短按一下常规攻击键攻击一次，位移后距离未知不用近战键。
                    self.sleep(1.0)  # 攻击后等 1 秒再执行第二段位移，等攻击后摇结束，避免第二段的方向键被吞。
                    self.send_key(back_key, down_time=move_back_seconds)  # 按住方向键反方向移动 z 秒，按下时先转回身再移动。
                    facing = -facing  # 第二段移动后再次转身，朝向恢复为位移前的原朝向。
                    if facing_assumed:  # 朝向为假定值时序列得到的朝向同样不可信。
                        facing = None  # 清空让下次触发重新探测，避免错误假定被长期沿用。
                if turn_interval > 0 and time.time() - last_turn_time >= turn_interval:  # 到达转身攻击间隔时停止攻击 1 秒，转身攻击一下再转身归位。
                    last_turn_time = time.time()  # 重置转身计时。
                    if self._held_key is not None:  # 先松开持续按住的攻击键，转身期间不攻击。
                        self.send_key_up(self._held_key)  # 松开当前按住的攻击键。
                        self._held_key = None  # 清空按住状态。
                    self.info_set("Status", "转身中")  # 在 GUI 显示转身状态。
                    self.sleep(1.0)  # 停止攻击后等待 1 秒再转身，等攻击后摇结束避免转身被吞。
                    facing_assumed = False  # 本次转身的朝向是否来自假定，假定值不可信需在结束后清空。
                    if facing is None:  # 从未转身过导致朝向未知，先用角色模板匹配探测，失败才假定朝右。
                        facing = self.detect_facing(char_name)  # 原始模板命中=朝右、镜像命中=朝左。
                        if facing is None:  # 探测不到角色时退回假定。
                            facing = 1  # 假定角色当前朝右。
                            facing_assumed = True  # 标记朝向为假定值。
                    turn_key = MOVE_LEFT_KEY if facing == 1 else MOVE_RIGHT_KEY  # 转身键：当前朝向的反方向，单击只触发转身动画。
                    self.send_key(turn_key, down_time=0.05)  # 短按一下方向键完成转身。
                    facing = -facing  # 转身后朝向与原朝向相反。
                    self.sleep(0.08)  # 等待转身动画生效后再攻击。
                    self.send_key(attack_key, down_time=0.2)  # 短按一下常规攻击键攻击一次，转身后距离未知不用近战键。
                    self.sleep(0.5)  # 攻击后等 0.5 秒再转身，等攻击后摇结束，避免归位的方向键被吞导致没有转回来。
                    back_key = MOVE_RIGHT_KEY if turn_key == MOVE_LEFT_KEY else MOVE_LEFT_KEY  # 归位键：转身键的反方向，转身后朝向已翻转，再按同一方向键会变成前行而不是转身。
                    self.send_key(back_key, down_time=0.05)  # 短按反方向键转回身归位。
                    facing = -facing  # 第二次转身后朝向恢复为转身前的原朝向。
                    if facing_assumed:  # 朝向为假定值时翻转得到的朝向同样不可信。
                        facing = None  # 清空让下次触发重新探测，避免错误假定被长期沿用。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时短暂等待后重试。
                    self.sleep(self.config.get("Frame Interval"))  # 等待一个帧间隔。
                    continue  # 进入下一帧处理。
                batch = None  # 本帧 CPU 并发匹配批次，GPU 路径不需要。
                if gpu is None:  # CPU 路径：本帧全部模板匹配彼此独立且只读同一帧画面，先一次性投递线程池并发执行。
                    batch = MatchBatch()  # 新建本帧匹配批次。
                    self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 提交角色与全部怪物的原始/镜像匹配。
                gm = None  # 本帧 GPU 匹配句柄，默认不可用。
                if gpu is not None:  # GPU 匹配器可用时才尝试批量匹配。
                    try:  # 上传/计算可能因显存等原因异常。
                        gm = gpu.match_frame(frame)  # 上传本帧做批量匹配。
                    except Exception as e:  # 本帧 GPU 处理异常。
                        gpu = None  # 永久回退 CPU，避免每帧重复报错。
                        self.log_warning(f"GPU match failed at runtime, fall back to CPU: {e}. GPU 匹配运行异常，已回退 CPU。")  # 记录回退原因。
                if gm is not None:  # GPU 路径：一帧内批量算完角色与全部怪物（含镜像）。
                    character = self.gpu_lookup_one(gm, char_name, self.config.get("Character Threshold"))  # 角色取最高分框，镜像命中带 flipped 标记。
                    monsters = []  # 收集本帧全部怪物匹配框。
                    for name in monster_names:  # 逐个怪物分类取全部达标框。
                        monsters.extend(self.gpu_lookup_all(gm, name, self.config.get("Monster Threshold"), self.config.get("Monster Mirror Threshold")))  # 原始与镜像各自阈值，合并后去重。
                else:  # CPU 路径：按与串行完全一致的顺序取回线程池里的匹配结果，行为与 GPU 路径一致。
                    if batch is None:  # 本帧 GPU 刚异常回退 CPU，批次还没建。
                        batch = MatchBatch()  # 补建批次。
                        self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 补提交角色与全部怪物匹配。
                    character, monsters = self.collect_char_monster_matches(batch, monster_names)  # 取回角色框与全部怪物框。
                if character is not None and monsters:  # 角色存在时先剔除压在角色身上的怪物框。
                    kept = []  # 过滤后的怪物框列表。
                    for monster in monsters:  # 逐只检查是否与角色框重叠。
                        if self.overlapping(character, monster):  # 低阈值时怪物模板容易误匹配到角色自身，中心重合就是典型特征。
                            self.log_warning(f"Drop monster box overlapping character: {monster.name} conf={monster.confidence:.2f} x={monster.x} y={monster.y}. 剔除与角色重叠的怪物框，疑似误匹配，建议调高 Template/Mirror Threshold。")  # 记录被剔除的框供排查。
                            continue  # 丢弃该框不参与距离与攻击判定。
                        kept.append(monster)  # 保留正常怪物框。
                    monsters = kept  # 用过滤后的列表替换原列表。
                nearest = None  # 距离角色最近的怪物框。
                if character is not None and monsters:  # 角色与怪物都找到时才计算距离。
                    nearest = min(monsters, key=lambda m: self.center_distance(character, m))  # 选中心点距离最近的怪物。
                    dx, dy = self.center_offset(character, nearest)  # 计算角色到怪物的中心点 xy 距离。
                    self.info_set("Distance", f"dx={dx} dy={dy}")  # 在 GUI 状态区显示 xy 距离。
                elif character is not None:  # 角色在但没匹配到任何怪物。
                    self.info_set("Distance", "-")  # 明确显示无怪，避免残留旧值造成误判。
                if time.time() - last_diag_time >= 1.0:  # 每秒限频输出一次诊断日志。
                    last_diag_time = time.time()  # 记录本次诊断时间。
                    char_desc = f"x={character.x} y={character.y} conf={character.confidence:.2f}" if character is not None else "None"  # 角色框位置与置信度。
                    mob_desc = "; ".join(f"{m.name}@({m.x},{m.y}) conf={m.confidence:.2f}" for m in monsters) or "None"  # 全部怪物框位置与置信度。
                    self.log_info(f"Match diag: CHAR[{char_desc}] MOBS[{mob_desc}] 匹配诊断：角色与怪物框坐标及置信度。")  # 输出诊断日志供排查误匹配。
                target = None  # 当前要攻击的目标怪物。
                if character is not None and monsters:  # 角色与怪物都存在时筛选攻击目标。
                    in_range = []  # 收集攻击区域内的全部怪物。
                    for monster in monsters:  # 逐只怪物检查是否在攻击区域内。
                        dx, dy = self.center_offset(character, monster)  # 计算角色到该怪物的 xy 偏移（符号化）。
                        if attack_x_min <= dx <= attack_x_max and attack_y_min <= dy <= attack_y_max:  # 偏移落在上下左右四条边界围成的区域内。
                            in_range.append(monster)  # 加入可攻击列表。
                    if in_range:  # 区域内有怪物时选定目标。
                        same_side = [m for m in in_range if facing is not None and (1 if self.center_offset(character, m)[0] > 0 else -1) == facing]  # 筛出当前朝向同侧的怪物。
                        pool = same_side if same_side else in_range  # 同侧还有怪就锁定该侧，清完才允许换侧，避免两侧反复转身。
                        target = min(pool, key=lambda m: self.center_distance(character, m))  # 取候选池中最近的一只，一直攻击直到它消失。
                og.my_app.update_vision(self.draw_overlay(frame, character, monsters, nearest, target))  # 把带标注画面推送给 UI 实时展示。
                if target is not None:  # 攻击范围内有怪物时原地攻击。
                    dx, dy = self.center_offset(character, target)  # 计算目标怪物相对角色的方向。
                    direction = 1 if dx > 0 else -1  # 1=怪在右侧，-1=怪在左侧。
                    want_key = melee_key if abs(dx) <= melee_distance else attack_key  # 横向距离在近战距离内用近战键，否则用常规攻击键，怪物走近走远时自动切换。
                    if self._held_key is not None and self._held_key != want_key:  # 换攻击键（近战/常规切换）时先松开旧键，避免两键同时按住。
                        self.send_key_up(self._held_key)  # 松开当前按住的攻击键。
                        self._held_key = None  # 清空按住状态。
                    if facing != direction:  # 朝向与怪物方向不一致时才单击方向键换向，绝不持续按住造成移动。
                        self.send_key(MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if self._held_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住近战或常规攻击键不放。
                        self._held_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "攻击中")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if self._held_key is not None:  # 目标消失时松开持续按住的攻击键。
                    self.send_key_up(self._held_key)  # 松开攻击键。
                    self._held_key = None  # 清空按住状态。
                self.info_set("Status", "原地待命" if character is not None else "未找到角色")  # 无目标时原地待命，显示当前状态。
                self.sleep(self.config.get("Frame Interval"))  # 等待一个帧间隔后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if self._held_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(self._held_key)  # 松开它。
                self._held_key = None  # 清空按住状态。

    def pop_held_keys(self):  # 上报并清空任务当前持有的按键，供独立测谎服务暂停任务后释放（纯属性操作，不调用执行器，线程安全）。
        keys = []  # 收集当前持有的按键。
        if self._held_key is not None:  # 有持续按住的攻击键。
            keys.append(self._held_key)  # 加入待释放列表。
            self._held_key = None  # 清空属性，任务恢复后 run() 会按需重新按下。
        return keys  # 返回持有的按键列表。

    def parse_monster_names(self, text):  # 把逗号分隔的怪物分类名文本解析成列表。
        return [name.strip() for name in str(text or '').split(',') if name.strip()]  # 去掉空白项与首尾空格。

    def draw_overlay(self, frame, character, monsters, nearest, target):  # 在一帧画面上绘制全部标注，返回新画面。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if character is not None:  # 匹配到角色时绘制角色标注。
            self.draw_target(canvas, character, (0, 255, 0), "CHAR" + ("-flip" if getattr(character, "flipped", False) else ""))  # 绿色框+十字延长线，镜像命中时标注 -flip。
            cx, cy = self.box_center(character)  # 攻击范围基准点：角色框中心，与 run() 中攻击判定公式一致。
            x_min, x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 左右边界，填反自动交换，与判定公式一致。
            y_min, y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 上下边界，填反自动交换，与判定公式一致。
            cv2.rectangle(canvas, (int(cx + x_min), int(cy + y_min)), (int(cx + x_max), int(cy + y_max)), (255, 0, 255), 2)  # 紫色矩形框标出攻击范围，怪物中心点落入框内才会被攻击。
            self.draw_text(canvas, "ATK RANGE", (int(cx + x_min), max(int(cy + y_min) - 6, 14)), (255, 0, 255))  # 范围框左上角标注文本。
            melee_distance = int(float(self.config.get("Melee Distance") or 0))  # 近战距离，与 run() 中近战/常规攻击键切换公式一致。
            if melee_distance > 0:  # 配置了近战距离时用蓝线框出近战范围。
                cv2.rectangle(canvas, (int(cx - melee_distance), int(cy + y_min)), (int(cx + melee_distance), int(cy + y_max)), (255, 0, 0), 2)  # 蓝色矩形框标出近战范围，怪物中心点落入框内用近战键。
                self.draw_text(canvas, f"MELEE {melee_distance}", (int(cx - melee_distance), min(int(cy + y_max) + 16, canvas.shape[0] - 6)), (255, 0, 0))  # 近战框下方标注近战距离，避免与上方的 ATK RANGE 标注重叠。
        for monster in monsters:  # 绘制每只怪物的标注。
            self.draw_target(canvas, monster, (0, 0, 255), "MOB" + ("-flip" if getattr(monster, "flipped", False) else ""))  # 红色框+十字延长线，镜像命中时标注 -flip，方便确认镜像匹配生效。
        if target is not None:  # 存在当前攻击目标时额外高亮。
            self.draw_target(canvas, target, (255, 255, 255), "TARGET", cross=False)  # 白色框标出正在攻击的怪物，消失后不再绘制。
        if character is not None and nearest is not None:  # 角色与最近怪物都存在时绘制距离信息。
            cx, cy = self.box_center(character)  # 角色中心点。
            mx, my = self.box_center(nearest)  # 怪物中心点。
            cv2.line(canvas, (int(cx), int(cy)), (int(mx), int(my)), (255, 255, 0), 1)  # 角色与怪物中心连线。
            dx, dy = self.center_offset(character, nearest)  # 计算 xy 距离。
            self.draw_text(canvas, f"dx={dx} dy={dy}", (int((cx + mx) / 2), int((cy + my) / 2)), (255, 255, 0))  # 在连线中点显示 xy 距离。
        return canvas  # 返回绘制完成的画面。

    def draw_target(self, canvas, box, color, label, cross=True):  # 绘制一个匹配框，可选十字延长线到画面边缘。
        x1, y1 = box.x, box.y  # 匹配框左上角。
        x2, y2 = box.x + box.width, box.y + box.height  # 匹配框右下角。
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)  # 绘制匹配框。
        cx, cy = self.box_center(box)  # 计算匹配框中心点。
        if cross:  # 移动目标需要十字线辅助观察位置。
            cv2.line(canvas, (0, int(cy)), (canvas.shape[1], int(cy)), color, 1)  # 以中心为原点画横向延长线到画面左右边缘。
            cv2.line(canvas, (int(cx), 0), (int(cx), canvas.shape[0]), color, 1)  # 以中心为原点画纵向延长线到画面上下边缘。
        if label:  # 有标签文本时画在框上方。
            self.draw_text(canvas, label, (x1, max(y1 - 6, 14)), color)  # 标签显示在匹配框左上角。

    def draw_text(self, canvas, text, position, color):  # 在画面上绘制带黑色描边的文字，保证可读性。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # 先画黑色描边。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  # 再画彩色文字。

    def wait_frame(self):  # 等待取到一帧画面，最多等 10 秒。
        for _ in range(100):  # 最多尝试 100 次，每次间隔 0.1 秒。
            frame = self.next_frame()  # 取最新一帧画面。
            if frame is not None:  # 取到画面就返回。
                return frame  # 返回画面矩阵。
            self.sleep(0.1)  # 未取到画面时短暂等待。
        return None  # 超时仍取不到画面返回 None。

    def feature_ready(self, feature_name):  # 检查模板页中是否存在指定分类名的标注。
        try:  # 标注文件损坏或编码异常时避免任务崩溃。
            return self.executor.feature_set.feature_exists(feature_name)  # 查询 FeatureSet 中是否加载到该标注。
        except Exception as e:  # 加载标注文件失败。
            self.log_warning(f"Failed to load template annotations: {e}")  # 记录加载失败日志。
            return False  # 标注不可用。

    def detect_facing(self, char_name):  # 用角色模板匹配探测朝向：原始模板命中视为朝右、镜像命中视为朝左，找不到角色返回 None，假定标注模板为朝右立绘。
        frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
        if frame is None:  # 取不到画面无法探测。
            return None  # 返回未知。
        box = self.find_one_feature(char_name, frame, self.config.get("Character Threshold"))  # 先原始朝向匹配，未命中再用镜像模板匹配。
        if box is None:  # 两种朝向都找不到角色。
            return None  # 返回未知。
        return -1 if getattr(box, "flipped", False) else 1  # 镜像命中=朝左，原始命中=朝右。

    def build_gpu_matcher(self, char_name, monster_names):  # 构建 GPU 匹配器并为角色与全部怪物注册原始/镜像模板，不可用时返回 None。
        # 显卡加速隐藏式启用（不设配置开关）：统一优先 GPU，无显卡/异常时自动降级 CPU。
        try:  # 延迟导入，未安装 cupy 时不影响任务加载。
            from src.gpu_match import GpuTemplateMatcher, gpu_available  # 导入项目 GPU 匹配模块。
            if not gpu_available():  # 无可用 NVIDIA 显卡或 CuPy 未装好。
                self.log_info("GPU match unavailable, use CPU. 显卡加速不可用，使用 CPU 匹配。")  # 提示回退原因。
                return None  # 回退 CPU。
            matcher = GpuTemplateMatcher(gray=bool(self.config.get("Use Gray Scale")))  # 按灰度配置创建匹配器。
            feature_set = self.executor.feature_set  # 取执行器的特征集。
            for name in [char_name] + monster_names:  # 逐个分类注册原始与镜像两份模板。
                feature_set.ensure_feature(name)  # 确保标注已加载。
                feature = feature_set.feature_dict.get(name)  # 取特征对象。
                if feature is None or getattr(feature, "mask", None) is not None:  # 标注缺失或带掩码时 GPU 路径无法等价复现。
                    self.log_info(f"Feature {name} not GPU-matchable, use CPU. 标注 {name} 无法 GPU 匹配，整体回退 CPU。")  # 说明回退原因。
                    return None  # 回退 CPU 保证行为一致。
                matcher.add_template(name, feature.mat)  # 注册原始朝向模板。
                matcher.add_template(name + "__flip", cv2.flip(feature.mat, 1))  # 注册水平镜像模板。
            self.log_info("GPU template match enabled. 已启用显卡模板匹配加速。")  # 提示加速已生效。
            return matcher  # 返回可用的匹配器。
        except Exception as e:  # 初始化任何环节异常都不影响任务运行。
            self.log_warning(f"GPU match init failed, use CPU: {e}. GPU 匹配初始化失败，使用 CPU。")  # 记录异常原因。
            return None  # 回退 CPU。

    def gpu_lookup_one(self, gm, name, threshold, mirror_threshold=None):  # 在 GPU 帧句柄中找一个目标的最佳框，镜像未单独给阈值时沿用主阈值。
        try:  # GPU 计算异常时返回 None，由调用方回退 CPU。
            x, y, score = gm.best(name)  # 原始朝向模板的最高分与位置。
            flipped = False  # 默认原始朝向命中。
            if score < threshold:  # 原始朝向未达标时改用镜像模板。
                x, y, score = gm.best(name + "__flip")  # 镜像模板的最高分与位置。
                flipped = True  # 标记为镜像命中。
                threshold = mirror_threshold or threshold  # 镜像使用独立阈值。
            if score < threshold:  # 两种朝向都未达标。
                return None  # 按未匹配处理。
            td = gm.matcher.templates[name + "__flip" if flipped else name]  # 取命中模板的尺寸。
            box = Box(int(x), int(y), td["w"], td["h"], confidence=float(score), name=name)  # 包装成 Box。
            box.flipped = flipped  # 镜像命中标记，供画面标注区分朝向。
            return box  # 返回匹配框。
        except Exception as e:  # GPU 计算异常。
            self.log_warning(f"GPU lookup failed for {name}: {e}. GPU 匹配异常。")  # 记录异常。
            return None  # 按未匹配处理，调用方可回退 CPU。

    def gpu_lookup_all(self, gm, name, threshold, mirror_threshold=None):  # 在 GPU 帧句柄中找一个目标的全部匹配框，含镜像并去重。
        boxes = []  # 收集两种朝向的全部匹配框。
        try:  # GPU 计算异常时返回空列表，不影响主流程。
            for row in gm.above(name, threshold):  # 原始朝向全部达标位置。
                boxes.append(Box(int(row[0]), int(row[1]), gm.matcher.templates[name]["w"], gm.matcher.templates[name]["h"], confidence=float(row[2]), name=name))  # 包装成 Box。
            for row in gm.above(name + "__flip", mirror_threshold or threshold):  # 镜像朝向全部达标位置。
                box = Box(int(row[0]), int(row[1]), gm.matcher.templates[name + "__flip"]["w"], gm.matcher.templates[name + "__flip"]["h"], confidence=float(row[2]), name=name)  # 包装成 Box。
                box.flipped = True  # 标记镜像命中。
                boxes.append(box)  # 加入结果。
        except Exception as e:  # GPU 计算异常。
            self.log_warning(f"GPU lookup-all failed for {name}: {e}. GPU 匹配异常。")  # 记录异常。
        return self.merge_boxes(boxes)  # 两种朝向的框合并后统一去重。

    def find_one_raw(self, feature_name, frame, threshold):  # 仅用原始朝向匹配单个模板（不做镜像回退），返回置信度最高的框或 None。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            return self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，目标会移动不能只在标注位置附近找。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。

    def find_all_raw(self, feature_name, frame, threshold):  # 仅用原始朝向匹配单个模板的全部出现位置（不做镜像回退），返回匹配框列表。
        return self.find_feature(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，返回全部匹配框。

    def find_one_feature(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配一个标注模板，返回置信度最高的框或 None，镜像未单独给阈值时沿用主阈值。
        box = self.find_one_raw(feature_name, frame, threshold)  # 先按原始朝向匹配。
        if box is not None:  # 原始朝向匹配成功。
            return box  # 直接返回结果。
        return self.find_flipped(feature_name, frame, threshold=mirror_threshold or threshold)  # 目标转向时精灵图会水平镜像，用翻转模板再匹配一次。

    def find_all_features(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配标注模板的全部出现位置，镜像未单独给阈值时沿用主阈值。
        boxes = list(self.find_all_raw(feature_name, frame, threshold))  # 原始朝向的全部匹配框。
        boxes.extend(self.find_flipped(feature_name, frame, find_all=True, threshold=mirror_threshold or threshold))  # 补充镜像朝向的匹配结果，目标转向后也能识别。
        return self.merge_boxes(boxes)  # 两种朝向的框合并后统一去重，避免同一目标出重复框。

    def submit_char_monster_matches(self, batch, frame, char_name, monster_names):  # 把本帧角色与全部怪物的原始/镜像匹配一次性提交线程池，各匹配彼此独立且只读同一帧画面，结果与串行执行完全一致。
        char_threshold = self.config.get("Character Threshold")  # 角色独立阈值，角色镜像沿用同一阈值。
        mob_threshold = self.config.get("Monster Threshold")  # 怪物阈值。
        mob_mirror_threshold = self.config.get("Monster Mirror Threshold") or mob_threshold  # 怪物镜像阈值，未配置时沿用怪物阈值，与串行实现一致。
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.config.get("Use Gray Scale") else None  # 整帧灰度图只转一次给全部镜像匹配共用，避免每个线程重复转换整帧。
        batch.submit("char", lambda: self.find_one_raw(char_name, frame, char_threshold))  # 角色原始朝向匹配。
        batch.submit("char_flip", lambda: self.find_flipped(char_name, frame, threshold=char_threshold, gray_frame=gray_frame))  # 角色镜像朝向匹配，只在原始未命中时才会被采用。
        for i, name in enumerate(monster_names):  # 逐个怪物分类提交，键带序号避免怪物名与角色名重名冲突。
            batch.submit(f"m{i}", lambda name=name: self.find_all_raw(name, frame, mob_threshold))  # 怪物原始朝向的全部匹配框。
            batch.submit(f"m{i}_flip", lambda name=name: self.find_flipped(name, frame, find_all=True, threshold=mob_mirror_threshold, gray_frame=gray_frame))  # 怪物镜像朝向的全部匹配框。

    def collect_char_monster_matches(self, batch, monster_names):  # 按串行执行时的原顺序取回并发匹配结果，返回（角色框或 None, 全部怪物框）。
        character = batch.get("char")  # 先取角色原始朝向结果。
        if character is None:  # 原始朝向未命中时才采用镜像结果，与 find_one_feature 的短路逻辑一致。
            character = batch.get("char_flip")  # 取角色镜像朝向结果。
        monsters = []  # 收集本帧全部怪物匹配框。
        for i in range(len(monster_names)):  # 逐个怪物分类合并两种朝向的结果。
            boxes = list(batch.get(f"m{i}") or [])  # 原始朝向的全部匹配框。
            boxes.extend(batch.get(f"m{i}_flip") or [])  # 追加镜像朝向的全部匹配框。
            monsters.extend(self.merge_boxes(boxes))  # 每个分类单独做非极大值抑制，与串行实现完全一致。
        return character, monsters  # 返回角色框与怪物框列表。

    def merge_boxes(self, boxes):  # 对重叠的匹配框做非极大值抑制，按置信度从高到低保留。
        merged = []  # 去重后的结果列表。
        for box in sorted(boxes, key=lambda b: -b.confidence):  # 按置信度从高到低遍历。
            if any(self.overlapping(box, kept) for kept in merged):  # 与已保留框重叠视为同一目标。
                continue  # 跳过低分的重复框。
            merged.append(box)  # 保留该框。
        return merged  # 返回去重后的全部匹配框。

    def overlapping(self, a, b):  # 判断两个框中心是否足够近，近则视为同一目标。
        return abs(a.x - b.x) < min(a.width, b.width) / 2 and abs(a.y - b.y) < min(a.height, b.height) / 2  # 左上角差距小于较小框一半尺寸即重叠。

    def find_flipped(self, feature_name, frame, find_all=False, threshold=0.8, gray_frame=None):  # 用水平镜像的模板在整帧匹配，返回 Box 或 Box 列表，阈值由调用方按目标类型传入；gray_frame 为调用方已转好的整帧灰度图，并发匹配时多个线程共用一份。
        try:  # 获取标注模板失败时不能影响主流程。
            feature_set = self.executor.feature_set  # 取执行器的特征集。
            feature_set.ensure_feature(feature_name)  # 确保该标注已加载进缓存。
            feature = feature_set.feature_dict.get(feature_name)  # 从缓存取出特征对象。
        except Exception as e:  # 标注加载异常。
            self.log_warning(f"Failed to load feature for flip match: {e}")  # 记录异常日志。
            return [] if find_all else None  # 异常时返回空结果。
        if feature is None:  # 标注不存在。
            return [] if find_all else None  # 返回空结果。
        template = cv2.flip(feature.mat, 1)  # 水平镜像模板，对应目标反朝向的精灵图。
        search = frame  # 待匹配画面。
        if self.config.get("Use Gray Scale"):  # 配置为灰度匹配时统一转灰度。
            search = gray_frame if gray_frame is not None else cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)  # 画面转灰度，调用方已转好则直接复用同一份。
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # 模板转灰度。
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)  # 整帧模板匹配。
        if not find_all:  # 只需最佳结果时。
            _, score, _, loc = cv2.minMaxLoc(result)  # 取最高分与位置。
            if score < threshold:  # 最高分低于阈值。
                return None  # 镜像方向也没找到。
            box = Box(loc[0], loc[1], template.shape[1], template.shape[0], confidence=score, name=feature_name)  # 包装成 Box。
            box.flipped = True  # 标记为镜像命中，供画面标注区分朝向。
            return box  # 返回镜像匹配框。
        boxes = []  # 收集全部镜像匹配框。
        positions = np.argwhere(result >= threshold)  # 所有达到阈值的位置（y, x）。
        for y, x in sorted(positions, key=lambda p: -result[p[0], p[1]]):  # 按得分从高到低遍历。
            if any(abs(x - b.x) < template.shape[1] / 2 and abs(y - b.y) < template.shape[0] / 2 for b in boxes):  # 与已选框距离过近视为同一目标。
                continue  # 跳过重复框实现简单非极大值抑制。
            box = Box(int(x), int(y), template.shape[1], template.shape[0], confidence=float(result[y, x]), name=feature_name)  # 构造镜像匹配框。
            box.flipped = True  # 标记为镜像命中，供画面标注区分朝向。
            boxes.append(box)  # 加入结果列表。
        return boxes  # 返回全部镜像匹配框。

    def box_center(self, box):  # 计算匹配框中心点坐标。
        return box.x + box.width / 2, box.y + box.height / 2  # 返回中心点 (x, y)。

    def center_distance(self, a, b):  # 计算两个匹配框中心点的直线距离，用于选最近目标。
        ax, ay = self.box_center(a)  # 第一个框的中心。
        bx, by = self.box_center(b)  # 第二个框的中心。
        return (ax - bx) ** 2 + (by - ay) ** 2  # 返回距离平方，比较大小无需开方。

    def center_offset(self, a, b):  # 计算从框 a 中心到框 b 中心的 xy 偏移（像素）。
        ax, ay = self.box_center(a)  # 第一个框的中心。
        bx, by = self.box_center(b)  # 第二个框的中心。
        return int(round(bx - ax)), int(round(by - ay))  # 返回 (dx, dy)。
