import math  # 导入标准库 math，用于解测谎鼠标追踪的步长计算。
import os  # 导入标准库 os，用于测谎报警音频的路径解析与存在性检查。
import time  # 导入标准库 time，用于定时位移、转身与攻击节奏计时。
import traceback  # 导入标准库 traceback，用于解测谎异常时记录完整堆栈定位问题。
from collections import deque  # 导入双端队列，用于解测谎时保留每条轨迹的最近中心尾迹。

import cv2  # 导入 OpenCV，用于镜像模板匹配和实时画面标注绘制。
import numpy as np  # 导入 NumPy，用于镜像匹配的阈值筛选。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from ok.feature.Box import Box  # 导入 Box 类，用于镜像匹配结果的包装。
from ok.task.exceptions import TaskDisabledException  # 导入任务被禁用异常，用户停任务时的正常退出信号不能误判为解测谎故障。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

from src.tasks.MyBaseTask import MyBaseTask  # 导入项目任务基类，导入时会同时生效标注文件 UTF-8 读取补丁。

try:  # 解测谎推理依赖可选：模块缺失时挂机照常，仅禁用自动解测谎。
    from src.liedetector.detector import TransparentShapeDetector  # 导入透明图形检测器（ONNX 会话懒加载，优先 CUDA）。
    from src.liedetector.solver import TransparentShapeSolver  # 导入透明图形求解器（跟踪+背景方向评分+光标预测）。
    LIE_SOLVER_AVAILABLE = True  # 推理流水线可用。
except ImportError:  # liedetector 模块缺失或依赖损坏。
    LIE_SOLVER_AVAILABLE = False  # 禁用自动解测谎，运行时日志提示。

MOVE_LEFT_KEY = "left"  # 左方向键：单击用于换方向，定时位移时按住用于移动。
MOVE_RIGHT_KEY = "right"  # 右方向键：单击用于换方向，定时位移时按住用于移动。

LIE_REGION_SHIFT_PIXELS = 4  # 【测谎坐标框】移动超过该像素数视为新一局（或窗口位移），重置求解器与轨迹。
LIE_TRAIL_LENGTH = 30  # 解测谎可视化每条轨迹保留最近 30 个中心尾迹，与测谎检验页签一致。
LIE_MOVE_MAX_STEP = 50  # 解测谎时每帧鼠标最多移动的像素数，分步追赶避免光标瞬移过大。
LIE_MAX_TICKS = 750  # 解测谎无结果兑底退出的帧数（约 25 秒 @30FPS），防止触发标注误匹配造成死循环。
LIE_FRAME_WAIT = 0.05  # 解测谎中取不到画面时的重试等待秒数。
CAPTURE_FPS = 30  # 截图采集固定帧率：解测谎取帧按该节拍。
CAPTURE_MIN_INTERVAL = 1.0 / CAPTURE_FPS  # 固定帧间隔秒数（约 0.0333）。


class MapleIdleTask(MyBaseTask):  # 定义冒险岛挂机任务，继承项目基类。

    def __init__(self, *args, **kwargs):  # 构造函数，先初始化父类再设置任务元数据。
        super().__init__(*args, **kwargs)  # 必须先调用父类构造。
        self.name = "Maple Idle"  # 任务显示名称。
        self.description = "Single-spot camping: stay in place, single-tap direction key only to turn, keep attacking the nearest monster within attack range until it dies; supports optional periodic reposition moves and turn-around attacks; also shows live vision; watches the lie detector trigger from the dashboard config and auto solves when triggered.  # 任务描述：单点挂机，不持续移动，仅单击换方向，攻击范围内最近怪直到消失，支持定时位移与定时转身攻击；按看板配置值守测谎触发并自动解测谎。"
        self.icon = FluentIcon.FLAG  # 任务图标。
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
            "GPU Match": True,  # 是否用 NVIDIA 显卡做模板匹配，需安装 cupy，不可用时自动回退 CPU。
            "Frame Interval": 0.05,  # 每帧处理之间的最小间隔秒数，控制检测节奏。
        })
        self.config_description.update({  # 各配置项的帮助文本。
            "Move Interval": "Seconds between reposition moves: stop attacking and wait 1s, move opposite to character facing for Move Away Seconds, attack once, wait 1s, then move back for Move Back Seconds; 0 disables it. 每隔该秒数停攻等 1 秒后位移一次（朝朝向反向移动、攻击一下、等 1 秒、再反方向移回），0 禁用。",
            "Move Away Seconds": "Hold seconds for the first leg, moving opposite to character facing, supports 3 decimals. 位移第一段：朝角色朝向反向移动的秒数，支持 3 位小数。",
            "Move Back Seconds": "Hold seconds for the second leg after attacking once and waiting 1s, moving in the opposite direction of the first leg, supports 3 decimals. 位移第二段：攻击一下停顿 1 秒后再反方向移动的秒数，支持 3 位小数。",
            "Turn Interval": "Every x seconds do a turn-around attack: stop attacking and wait 1s, tap direction key to turn, attack once, wait 0.5s, tap the opposite direction key to turn back; 0 disables it. 每隔 x 秒做一次转身攻击：停止攻击等 1 秒→单击方向键转身→攻击一下→等 0.5 秒→再单击反方向键转身归位，0 禁用。",
            "Use Gray Scale": "Match in grayscale, more robust to color differences. 是否转灰度匹配，对颜色差异更稳定。",
            "GPU Match": "Run template matching on NVIDIA GPU via CuPy for speed; falls back to CPU automatically when unavailable. 是否用 NVIDIA 显卡加速模板匹配，需安装 cupy，不可用时自动回退 CPU。",
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
        lie_enabled = bool(self.config.get("Lie Detector Auto Solve"))  # 读取看板测谎总开关，默认开启，全部任务默认支持。
        lie_trigger_name = str(self.config.get("Lie Detector Trigger Feature") or '').strip()  # 读取【测谎触发】标注分类名。
        lie_region_name = str(self.config.get("Lie Detector Region Feature") or '').strip()  # 读取【测谎坐标框】标注分类名。
        lie_threshold = float(self.config.get("Lie Detector Threshold") or 0.75)  # 读取测谎触发匹配阈值。
        alarm_sound = str(self.config.get("Lie Alarm Sound") or '').strip()  # 读取报警音频路径，留空表示不报警。
        if lie_enabled and not LIE_SOLVER_AVAILABLE:  # 开关打开但推理模块不可用。
            self.log_warning("Lie detector module unavailable, auto solve disabled. 测谎推理模块不可用，自动解测谎已禁用。")  # 提示并降级。
            lie_enabled = False  # 禁用后不影响挂机。
        for lie_name in (lie_trigger_name, lie_region_name):  # 两个测谎标注都必须在模板页标注过才能启用。
            if lie_enabled and (not lie_name or not self.feature_ready(lie_name)):  # 标注名为空或未标注。
                self.log_warning(f"Lie detector annotation missing, auto solve disabled: {lie_name}. 测谎标注缺失，自动解测谎已禁用：{lie_name}。")  # 提示并降级，不阻断挂机。
                lie_enabled = False  # 禁用。
        gpu = self.build_gpu_matcher(char_name, monster_names)  # 尝试构建 GPU 匹配器并注册全部模板，失败返回 None 走 CPU。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，只在需要换向时单击方向键。
        held_key = None  # 当前持续按住的攻击键（近战或常规），切换/换向/目标消失/退出时必须松开它。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        try:  # 包裹主循环，退出时兜底松开持续按住的攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                if del_interval > 0 and time.time() - last_del_time >= del_interval:  # 到达定时间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                if move_interval > 0 and time.time() - last_move_time >= move_interval:  # 到达位移间隔时停止全部状态做一次位移。
                    last_move_time = time.time()  # 重置位移计时。
                    if held_key is not None:  # 先松开持续按住的攻击键，位移期间不攻击。
                        self.send_key_up(held_key)  # 松开当前攻击键。
                        held_key = None  # 清空按住状态。
                    self.info_set("Status", "Moving")  # 在 GUI 显示位移状态。
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
                    if held_key is not None:  # 先松开持续按住的攻击键，转身期间不攻击。
                        self.send_key_up(held_key)  # 松开当前按住的攻击键。
                        held_key = None  # 清空按住状态。
                    self.info_set("Status", "Turning")  # 在 GUI 显示转身状态。
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
                if lie_enabled:  # 测谎值守已启用：每帧优先检查触发标注，命中则暂停攻击优先解测谎。
                    trigger_box = self.find_lie_box(lie_trigger_name, frame, lie_threshold)  # 全屏匹配【测谎触发】标注。
                    if trigger_box is not None:  # 识别到触发标注，代表测谎已触发。
                        if held_key is not None:  # 先松开当前按住的攻击键。
                            self.send_key_up(held_key)  # 松开攻击键。
                            held_key = None  # 清空按住状态。
                        self.play_lie_alarm(alarm_sound)  # 播放报警音频（未配置或文件不存在时自动跳过）。
                        lie_region_box = self.get_lie_region_box(lie_region_name)  # 直接采集【测谎坐标框】标注的坐标供画面框选，不做模板匹配。
                        og.my_app.update_vision(self.draw_lie_annotations(frame, trigger_box, lie_region_box))  # 触发瞬间立即把两个标注框选推送到实时画面。
                        try:  # 进入解测谎子循环，直到【测谎触发】标注消失才返回。
                            self.solve_lie_detector(frame, lie_trigger_name, lie_region_name, lie_threshold)  # 优先处理解测谎。
                        except TaskDisabledException:  # 用户手动停任务/任务被禁用是正常退出信号。
                            raise  # 透传给框架结束任务，绝不能当成解测谎故障禁用自动解测谎。
                        except Exception as e:  # 推理流水线异常不拖垮挂机。
                            self.log_warning(f"Lie solve error, auto solve disabled: {e}\n{traceback.format_exc()} 解测谎异常，已禁用自动解测谎（含完整堆栈）。")  # 记录异常原因与完整堆栈供定位。
                            lie_enabled = False  # 本次运行不再尝试解测谎。
                        facing = None  # 解测谎后重新探测朝向，避免沿用过期朝向。
                        self.sleep(0.5)  # 等弹窗关闭后画面稳定再继续挂机。
                        continue  # 重新取帧恢复挂机。
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
                else:  # CPU 路径：逐个模板调用 OpenCV 匹配，行为与 GPU 路径一致。
                    character = self.find_one_feature(char_name, frame, self.config.get("Character Threshold"))  # 用角色标注模板在本帧做匹配，角色用独立阈值。
                    monsters = []  # 收集本帧全部怪物匹配框。
                    for name in monster_names:  # 逐个怪物分类匹配，支持多个怪物。
                        monsters.extend(self.find_all_features(name, frame, self.config.get("Monster Threshold"), self.config.get("Monster Mirror Threshold")))  # 追加该分类的全部匹配框，怪物与怪物镜像各用独立阈值。
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
                    if held_key is not None and held_key != want_key:  # 换攻击键（近战/常规切换）时先松开旧键，避免两键同时按住。
                        self.send_key_up(held_key)  # 松开当前按住的攻击键。
                        held_key = None  # 清空按住状态。
                    if facing != direction:  # 朝向与怪物方向不一致时才单击方向键换向，绝不持续按住造成移动。
                        self.send_key(MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if held_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住近战或常规攻击键不放。
                        held_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Attacking")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if held_key is not None:  # 目标消失时松开持续按住的攻击键。
                    self.send_key_up(held_key)  # 松开攻击键。
                    held_key = None  # 清空按住状态。
                self.info_set("Status", "Camping" if character is not None else "Character not found")  # 无目标时原地待命，显示当前状态。
                self.sleep(self.config.get("Frame Interval"))  # 等待一个帧间隔后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if held_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(held_key)  # 松开它。
                held_key = None  # 清空按住状态。

    def find_lie_box(self, feature_name, frame, threshold):  # 全屏匹配单个测谎标注，返回置信度最高的框或 None。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            return self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，测谎弹窗位置不固定。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。

    def get_lie_region_box(self, region_name):  # 直接采集【测谎坐标框】标注记录的坐标框信息作为解测谎输入界面，不做模板匹配；未标注返回 None。
        try:  # 框架按标注名读取记录坐标，自动按当前画面分辨率缩放。
            box = self.get_box_by_name(region_name)  # 找不到标注时框架抛 ValueError。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按坐标框缺失处理。
        if box is None:  # 当前无画面帧时框架取不到标注坐标。
            return None  # 按坐标框缺失处理，等下一帧重试。
        return box  # 返回标注记录的坐标框。

    def get_lie_detector(self):  # 懒加载透明图形检测器并跨次运行复用（ONNX 会话首次推理时才创建）。
        if getattr(self, "_lie_detector", None) is None:  # 尚无缓存的检测器。
            self._lie_detector = TransparentShapeDetector()  # 优先 CUDA EP，失败自动回退 CPU（检测器内部处理）。
        return self._lie_detector  # 返回复用实例。

    def play_lie_alarm(self, sound_path):  # 播放测谎报警音频：异步播放不阻塞解测谎，失败只记日志不影响任务，返回是否已播放。
        if not sound_path:  # 未配置音频：不报警。
            return False  # 静默跳过。
        path = sound_path if os.path.isabs(sound_path) else os.path.join(os.getcwd(), sound_path)  # 相对路径按项目根目录解析。
        if not os.path.exists(path):  # 文件不存在：不报警并提示。
            self.log_warning(f"Lie alarm sound not found, skip: {sound_path}. 测谎报警音频不存在，跳过报警：{sound_path}。")  # 提醒检查路径。
            return False  # 未播放。
        try:  # 播放失败也不能拖垮解测谎。
            if path.lower().endswith('.wav'):  # WAV 用系统声音接口直接异步播放。
                import winsound  # Windows 系统声音模块。
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)  # 异步播放，不阻塞任务线程。
            else:  # mp3/wma 等格式走 Windows MCI 解码播放。
                import ctypes  # 调用系统 winmm.dll。
                winmm = ctypes.windll.winmm  # Windows 多媒体 API。
                alias = "lie_alarm"  # 固定设备别名，重复触发时先关旧实例再重开。
                winmm.mciSendStringW(f"close {alias}", None, 0, 0)  # 关闭上一次播放残留，避免占用。
                if winmm.mciSendStringW(f'open "{path}" alias {alias}', None, 0, 0) != 0:  # 自动识别格式打开失败。
                    if winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0) != 0:  # 再按 mpeg 音频显式打开。
                        self.log_warning(f"Lie alarm sound open failed: {sound_path}. 测谎报警音频无法打开（格式可能不支持）：{sound_path}。")  # 两种打开方式都失败。
                        return False  # 未播放。
                winmm.mciSendStringW(f"play {alias}", None, 0, 0)  # 异步播放，不等待播完。
            self.log_info(f"Lie alarm playing: {sound_path}. 测谎报警音频已播放：{sound_path}。")  # 记录报警供排查。
            return True  # 已播放。
        except Exception as e:  # 播放异常不中断解测谎。
            self.log_warning(f"Lie alarm play failed: {e}. 测谎报警播放失败：{e}。")  # 记录异常原因。
            return False  # 未播放。

    def solve_lie_detector(self, first_frame, trigger_name, region_name, threshold):  # 解测谎子循环：暂停其他功能，直到【测谎触发】标注从画面消失。
        self.log_info("Lie detector triggered, task paused, auto solving. 识别到【测谎触发】，暂停任务进入解测谎状态。")  # 记录进入解测谎供排查。
        self.info_set("Status", "Solving lie detector")  # 在 GUI 显示解测谎状态。
        self.ensure_in_front()  # 游戏窗口置顶：鼠标追踪依赖前台窗口接收鼠标事件。
        detector = self.get_lie_detector()  # 透明图形 YOLO 检测器，懒加载模型。
        solver = TransparentShapeSolver(fps=30)  # ByteTrack 跟踪+背景方向评分+光标预测求解器，参数与测谎检验页签一致。
        trails = {}  # 轨迹 ID -> 最近中心点尾迹，用于画面绘制。
        region = None  # 当前有效的【测谎坐标框】区域 (x, y, w, h)，未匹配到前保持 None。
        mouse_pos = None  # 上一帧鼠标目标点（画面坐标），用于分步平滑追赶。
        frame = first_frame  # 从触发帧开始求解。
        tick = 0  # 已处理帧数，用于兑底超时与诊断限频。
        last_diag_time = 0.0  # 上次诊断日志时间。
        while True:  # 循环直到触发标注消失或兑底超时。
            tick += 1  # 帧计数累加。
            if tick > LIE_MAX_TICKS:  # 长时间未结束，可能触发标注误匹配，退回任务。
                self.log_warning(f"Lie solve exceeded {LIE_MAX_TICKS} frames, resume task. 解测谎超过 {LIE_MAX_TICKS} 帧未结束，退回任务。")  # 记录兑底退出。
                break  # 退出子循环。
            trigger_box = self.find_lie_box(trigger_name, frame, threshold)  # 本帧匹配【测谎触发】标注，同时供画面红色框选。
            if trigger_box is None:  # 【测谎触发】标注不在页面，代表测谎已结束。
                self.log_info("Lie detector finished, resume task. 【测谎触发】标注消失，测谎已结束，解除测谎状态恢复任务。")  # 记录退出原因。
                break  # 退出子循环恢复任务。
            region_box = self.get_lie_region_box(region_name)  # 直接采集【测谎坐标框】标注记录的坐标框信息，不做模板匹配。
            new_region = (region_box.x, region_box.y, region_box.width, region_box.height) if region_box is not None else None  # 转为四元组，未采集到时为 None。
            if new_region is None:  # 本帧未采集到坐标框：沿用旧区域，从未有过则跳过求解。
                if region is None:  # 从未采集到坐标框。
                    self.sleep(LIE_FRAME_WAIT)  # 短等后重试（与固定 30FPS 节拍同数量级）。
                    new_frame = self.next_frame()  # 取新帧。
                    if new_frame is not None:  # 取到新帧才替换；numpy 数组不能用 or 判真值，必须显式判 None。
                        frame = new_frame  # 更新当前帧，取不到沿用旧帧。
                    continue  # 进入下一轮检查。
            elif region is None or abs(new_region[0] - region[0]) > LIE_REGION_SHIFT_PIXELS or abs(new_region[1] - region[1]) > LIE_REGION_SHIFT_PIXELS:  # 区域首次出现或位置明显变化。
                region = new_region  # 更新区域。
                solver = TransparentShapeSolver(fps=30)  # 重置跟踪器与求解器，避免新一局的轨迹污染。
                trails = {}  # 清空尾迹。
                mouse_pos = None  # 鼠标目标点重新校准。
            else:  # 区域位置稳定。
                region = new_region  # 刷新区域（尺寸可能微调）。
            crop = frame[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]  # 裁剪谎言检测区域供 YOLO 检测。
            detections = detector.detect(crop)  # 检测区域内全部透明图形（区域局部坐标）。
            cursor, target = solver.solve(region, detections)  # 跟踪+背景方向评分选目标+预测光标位置（画面坐标）。
            if cursor is not None:  # 求解器给出光标目标点时才移动鼠标。
                mouse_pos = self.move_mouse_toward(mouse_pos, cursor)  # 分步向预测点移动鼠标，模拟人眼平滑追踪。
            if time.time() - last_diag_time >= 1.0:  # 每秒限频输出一次诊断日志。
                last_diag_time = time.time()  # 记录本次诊断时间。
                target_desc = f"#{target.track_id}" if target is not None else "-"  # 当前锁定目标描述。
                self.log_info(f"Lie solve diag: detections={len(detections)} tracks={len(solver.tracker.tracked)} target={target_desc} cursor={cursor} 解测谎诊断：检测数/轨迹数/目标/光标位置。")  # 供排查检测为空等问题。
            og.my_app.update_vision(self.draw_lie_overlay(frame, region, detections, solver.tracker.tracked, trails, target, cursor, solver.bg_direction, trigger_box))  # 把目标框选、轨迹追踪与触发标注框选推送给 UI。
            self.sleep(CAPTURE_MIN_INTERVAL)  # 按固定 30FPS 节拍取帧，同时作为用户停止检查点（手动停止时在此抛出退出）。
            next_frame = self.next_frame()  # 取最新一帧画面。
            if next_frame is not None:  # 取到新帧才替换，取不到沿用上一帧继续求解。
                frame = next_frame  # 更新当前帧。

    def move_mouse_toward(self, current, target):  # 每帧向预测光标位置分步移动鼠标，返回移动后的位置。
        if current is None:  # 首帧无参照点，直接跳到预测点（游戏内光标也会瞬间到位）。
            position = target  # 目标位置即预测光标点。
        else:  # 已有参照点，按最大步长追赶。
            dx = target[0] - current[0]  # 到预测点的横向位移。
            dy = target[1] - current[1]  # 到预测点的纵向位移。
            distance = math.hypot(dx, dy)  # 直线距离。
            if distance <= LIE_MOVE_MAX_STEP:  # 距离在一个步长内，一步到位。
                position = target  # 直接到达预测点。
            else:  # 距离超过单步上限，按最大步长截断方向向量。
                position = (current[0] + dx / distance * LIE_MOVE_MAX_STEP, current[1] + dy / distance * LIE_MOVE_MAX_STEP)  # 沿目标方向移动一步。
        self.move(int(position[0]), int(position[1]))  # 画面坐标转为鼠标移动事件发给游戏窗口。
        return position  # 返回当前位置，供下一帧作参照。

    def draw_lie_annotations(self, frame, trigger_box, region_box):  # 在画面上框选测谎标注：【测谎触发】红色、【测谎坐标框】青色，未命中的不画。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if trigger_box is not None:  # 【测谎触发】标注命中时用红色框选。
            cv2.rectangle(canvas, (trigger_box.x, trigger_box.y), (trigger_box.x + trigger_box.width, trigger_box.y + trigger_box.height), (0, 0, 255), 2)  # 红色框标出触发标注位置。
            self.draw_text(canvas, "LIE TRIGGER", (trigger_box.x, max(trigger_box.y - 6, 14)), (0, 0, 255))  # 触发标注标签。
        if region_box is not None:  # 【测谎坐标框】标注命中时用青色框选。
            cv2.rectangle(canvas, (region_box.x, region_box.y), (region_box.x + region_box.width, region_box.y + region_box.height), (255, 255, 0), 2)  # 青色框标出谎言检测区域。
            self.draw_text(canvas, "LIE REGION", (region_box.x, max(region_box.y - 6, 14)), (255, 255, 0))  # 坐标框标签。
        return canvas  # 返回绘制完成的画面。

    def draw_lie_overlay(self, frame, region, detections, tracks, trails, target, cursor, bg_direction, trigger_box=None):  # 绘制解测谎叠加画面：触发标注框、区域框、目标框选、轨迹尾迹、背景方向箭头与预测光标十字，风格与测谎检验页签一致。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        rx, ry, rw, rh = region  # 谎言检测区域。
        if trigger_box is not None:  # 【测谎触发】标注在页面上时用红色框选，方便确认触发匹配位置。
            cv2.rectangle(canvas, (trigger_box.x, trigger_box.y), (trigger_box.x + trigger_box.width, trigger_box.y + trigger_box.height), (0, 0, 255), 2)  # 红色框标出触发标注。
            self.draw_text(canvas, "LIE TRIGGER", (trigger_box.x, max(trigger_box.y - 6, 14)), (0, 0, 255))  # 触发标注标签。
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色框标出测谎坐标框边界。
        self.draw_text(canvas, "LIE REGION", (rx, max(ry - 6, 14)), (255, 255, 0))  # 区域标签。
        for (x, y, w, h), score in detections:  # 全部检测框（区域局部坐标转画面坐标）。
            p1 = (rx + int(x), ry + int(y))  # 检测框左上角。
            p2 = (rx + int(x + w), ry + int(y + h))  # 检测框右下角。
            cv2.rectangle(canvas, p1, p2, (0, 220, 0), 1)  # 绿色检测框。
            cv2.putText(canvas, f"{score:.2f}", (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)  # 置信度标签。
        alive_ids = set()  # 本帧存活轨迹 ID 集合。
        for track in tracks:  # 每条轨迹：尾迹折线 + 卡尔曼速度箭头。
            tid = track.track_id  # 轨迹 ID。
            alive_ids.add(tid)  # 记录存活。
            center = (rx + int(track.rect[0] + track.rect[2] // 2), ry + int(track.rect[1] + track.rect[3] // 2))  # 轨迹中心（局部转画面坐标，取整避免浮点传入 cv2）。
            trail = trails.setdefault(tid, deque(maxlen=LIE_TRAIL_LENGTH))  # 取或建该轨迹的尾迹队列。
            trail.append(center)  # 追加本帧中心。
            if len(trail) >= 2:  # 至少两个点才能画折线。
                points = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)  # 转为折线点集。
                cv2.polylines(canvas, [points], False, (0, 200, 255), 1)  # 橙色尾迹折线。
            vx, vy = track.kalman_velocity  # 卡尔曼速度。
            cv2.arrowedLine(canvas, center, (center[0] + int(vx * 8), center[1] + int(vy * 8)), (0, 128, 255), 1, tipLength=0.3)  # 速度方向箭头。
        for tid in [t for t in trails if t not in alive_ids]:  # 清理已消亡轨迹的尾迹。
            del trails[tid]  # 删除死轨迹。
        if bg_direction[0] != 0.0 or bg_direction[1] != 0.0:  # 背景运动方向已估计时画大箭头。
            origin = (rx + 30, ry + 30)  # 箭头起点：区域左上角内侧。
            cv2.arrowedLine(canvas, origin, (origin[0] + int(bg_direction[0] * 60), origin[1] + int(bg_direction[1] * 60)), (255, 80, 255), 2, tipLength=0.25)  # 背景方向箭头。
        if target is not None:  # 当前锁定目标高亮框选。
            x, y, w, h = target.rect  # 目标轨迹框（局部坐标）。
            cv2.rectangle(canvas, (rx + int(x), ry + int(y)), (rx + int(x + w), ry + int(y + h)), (0, 0, 255), 2)  # 红色高亮框（取整避免浮点传入）。
            self.draw_text(canvas, f"TARGET #{target.track_id}", (rx + int(x), max(ry + int(y) - 6, 14)), (0, 0, 255))  # 目标标签。
        if cursor is not None:  # 预测光标位置画十字标记。
            cx, cy = int(cursor[0]), int(cursor[1])  # 光标画面坐标。
            cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 2)  # 横向十字。
            cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 2)  # 纵向十字。
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), 1)  # 中心小圆。
        self.draw_text(canvas, "LIE DETECTOR: SOLVING", (8, 22), (0, 255, 255))  # 左上角状态横幅。
        return canvas  # 返回绘制完成的画面。

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
        if not self.config.get("GPU Match"):  # 配置关闭 GPU 匹配。
            return None  # 直接用 CPU。
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

    def find_one_feature(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配一个标注模板，返回置信度最高的框或 None，镜像未单独给阈值时沿用主阈值。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            box = self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，目标会移动不能只在标注位置附近找。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。
        if box is not None:  # 原始朝向匹配成功。
            return box  # 直接返回结果。
        return self.find_flipped(feature_name, frame, threshold=mirror_threshold or threshold)  # 目标转向时精灵图会水平镜像，用翻转模板再匹配一次。

    def find_all_features(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配标注模板的全部出现位置，镜像未单独给阈值时沿用主阈值。
        boxes = list(self.find_feature(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1))  # variance=1 表示全屏搜索，返回全部匹配框。
        boxes.extend(self.find_flipped(feature_name, frame, find_all=True, threshold=mirror_threshold or threshold))  # 补充镜像朝向的匹配结果，目标转向后也能识别。
        return self.merge_boxes(boxes)  # 两种朝向的框合并后统一去重，避免同一目标出重复框。

    def merge_boxes(self, boxes):  # 对重叠的匹配框做非极大值抑制，按置信度从高到低保留。
        merged = []  # 去重后的结果列表。
        for box in sorted(boxes, key=lambda b: -b.confidence):  # 按置信度从高到低遍历。
            if any(self.overlapping(box, kept) for kept in merged):  # 与已保留框重叠视为同一目标。
                continue  # 跳过低分的重复框。
            merged.append(box)  # 保留该框。
        return merged  # 返回去重后的全部匹配框。

    def overlapping(self, a, b):  # 判断两个框中心是否足够近，近则视为同一目标。
        return abs(a.x - b.x) < min(a.width, b.width) / 2 and abs(a.y - b.y) < min(a.height, b.height) / 2  # 左上角差距小于较小框一半尺寸即重叠。

    def find_flipped(self, feature_name, frame, find_all=False, threshold=0.8):  # 用水平镜像的模板在整帧匹配，返回 Box 或 Box 列表，阈值由调用方按目标类型传入。
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
            search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)  # 画面转灰度。
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
