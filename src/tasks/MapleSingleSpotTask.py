import time  # 导入标准库 time，用于归位判定、卡住计时与攻击节奏。

import cv2  # 导入 OpenCV，用于画面标注绘制（小地图框、归位点与容差带标记线）。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

# 导入巡逻任务复用小地图定位、黄点检测、朝向校准、卡住脱困与攻击辅助能力；
# 测谎监控与自动解题由独立服务 src/liedetector/service.py 值守，任务只需通过 pop_held_keys 上报持有键供服务暂停时释放。
from src.tasks.MaplePatrolTask import MaplePatrolTask, MatchBatch, CAPTURE_MIN_INTERVAL, ATTACK_TO_MOVE_WAIT

HOME_WAIT_SECONDS = 10.0  # 启动时记录初始坐标比例的最长等待秒数，超时仍找不到黄点则任务退出。
RETURN_DONE_RATIO = 0.5  # 归位滞回系数：偏移回落到最大值的一半以内才算归位完成，避免在阈值边界上反复走停抖动。


class MapleSingleSpotTask(MaplePatrolTask):  # 定义冒险岛单点挂机任务，继承巡逻任务复用小地图与攻击能力。

    def __init__(self, *args, **kwargs):  # 构造函数，先复用父类全部配置再按单点场景裁剪。
        super().__init__(*args, **kwargs)  # 父类构造已填好小地图、黄点、Del 浮动等全部配置项与帮助文本。
        self.name = "Maple Single Spot"  # 任务显示名称。
        self.description = "Single-spot camping with minimap home anchor: record the character's horizontal minimap position percent at start, turn left/right in place and attack monsters within attack range without walking; monster collision may push the character away, so return to the recorded spot when no monster exists or the offset exceeds Return Offset Max Percent; character/monster parameters come from the dashboard config; live vision marks the home position and tolerance band. The lie detector is handled by an independent service, decoupled from this task.  # 任务描述：单点挂机，开始时记录角色在小地图的横向坐标比例，原地左右转向攻击不巡逻走动；怪物碰撞导致偏移超过阈值或场上无怪时自动走回记录点归位；角色/怪物参数统一从看板采集；实时画面标出归位点与容差带。测谎由独立服务值守，与本任务解耦。"
        self.icon = FluentIcon.PAUSE  # 任务图标。
        for key in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent",  # 巡逻边界与总开关由单点归位逻辑取代，裁剪掉对应配置。
                    "Stuck Seconds", "Resume Wait Seconds"):  # 卡住判定沿用巡逻默认值（归位走动同样需要脱困兜底），不在任务页展示。
            self.default_config.pop(key, None)  # 移除默认值。
            self.config_description.pop(key, None)  # 移除帮助文本。
        self.default_config.update({  # 单点挂机专属配置项。
            "Return Offset Max Percent": 5.0,  # 碰撞偏移归位max：黄点偏离初始坐标比例超过该百分比（地图区域宽度占比）时走回归位点。
        })
        self.config_description.update({  # 单点挂机专属配置项的帮助文本。
            "Return Offset Max Percent": "Max horizontal offset from the recorded start position, as percent of the map area width; the character walks back home when the offset exceeds it (or when no monster exists), and resumes attacking once the offset falls back within half of it. 碰撞偏移归位max：黄点偏离初始坐标比例超过该百分比时走回归位，回落到一半以内才恢复攻击。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
        error = super().validate_config(key, value)  # 复用父类的小地图/黄点/地图区域校验。
        if error is not None:  # 父类校验不通过时直接返回。
            return error  # 透传错误提示。
        if key == "Return Offset Max Percent":  # 归位阈值必须是合法百分比。
            try:  # 尝试按浮点数解析。
                percent = float(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非数字输入。
                return "Return offset percent must be a number. 碰撞偏移归位max 必须是数字。"  # 阻止保存并提示。
            if not 0 <= percent <= 100:  # 超出百分比范围。
                return "Return offset percent must be within 0-100. 碰撞偏移归位max 必须在 0-100 之间。"  # 阻止保存并提示。
        return None  # 其他配置项不做额外校验。

    def release_all_keys(self):  # 松开当前按住的全部按键（移动键+攻击键），进入归位/状态切换前调用。
        if self._held_attack_key is not None:  # 有按住的攻击键。
            self.send_key_up(self._held_attack_key)  # 松开攻击键。
            self._held_attack_key = None  # 清空按住状态。
            self._attack_released_at = time.time()  # 记录松键时间，攻击后摇会吞方向键，走动前需短暂等待。
        if self._held_move_key is not None:  # 有按住的移动键。
            self.send_key_up(self._held_move_key)  # 松开移动键。
            self._held_move_key = None  # 清空按住状态。

    def wait_home_percent(self, minimap_name, minimap_threshold, hue_min, hue_max, dot_min_pixels):  # 启动时记录角色在小地图的初始横向坐标比例，超时返回 None。
        deadline = time.time() + HOME_WAIT_SECONDS  # 最长等待时间。
        while time.time() < deadline:  # 在超时前反复尝试定位。
            frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
            if frame is not None:  # 取到画面才做检测。
                minimap = self.find_minimap(minimap_name, frame, minimap_threshold)  # 模板匹配定位小地图框。
                rect = self.map_rect(minimap) if minimap is not None else None  # 计算实际地图区域（画面坐标）。
                dot = self.detect_dot(frame, rect, hue_min, hue_max, dot_min_pixels) if rect is not None else None  # 在地图区域内检测角色黄点。
                if rect is not None and dot is not None:  # 小地图与黄点都就绪。
                    return (dot[0] - rect[0]) / rect[2] * 100  # 返回黄点横向位置相对地图区域宽度的百分比。
            self.sleep(0.1)  # 未就绪时短暂等待后重试。
        return None  # 超时仍未记录到初始坐标。

    def run(self):  # 任务运行入口：记录初始坐标比例后原地转向攻击，偏移超阈值或无怪时走回归位。
        self.apply_shared_config()  # 从看板读取共享的角色/怪物/测谎参数覆盖任务配置（单一数据源）。
        char_name = self.config.get("Character Feature")  # 读取角色标注分类名。
        monster_names = self.parse_monster_names(self.config.get("Monster Features"))  # 解析逗号分隔的怪物分类名列表。
        minimap_name = str(self.config.get("Minimap Feature") or '').strip()  # 读取小地图标注分类名。
        facing_left_name = str(self.config.get("Character Facing Left Feature") or '').strip()  # 读取角色左朝向模板标注分类名，留空禁用朝向校准。
        facing_right_name = str(self.config.get("Character Facing Right Feature") or '').strip()  # 读取角色右朝向模板标注分类名。
        frame = self.wait_frame()  # 先取到一帧画面，让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面时无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示取不到画面。
            return  # 直接结束任务。
        if not monster_names:  # 未配置任何怪物分类时无法运行。
            self.log_warning("No monster feature configured. 未配置怪物分类名，任务退出。")  # 提示配置缺失。
            return  # 直接结束任务。
        missing = [name for name in [minimap_name, char_name] + monster_names if not self.feature_ready(name)]  # 检查小地图、角色与全部怪物标注是否存在。
        if missing:  # 必要标注缺失时无法运行。
            self.log_warning(f"Template not ready, please annotate in the Template tab: {', '.join(missing)}. 模板未就绪，请先在模板页标注：{'、'.join(missing)}。")  # 提示用户去模板页标注。
            return  # 标注不可用时直接结束任务。
        if facing_left_name and not self.feature_ready(facing_left_name):  # 左朝向模板已配置但未在模板页标注。
            self.log_warning(f"Left facing template not annotated, facing calibration disabled: {facing_left_name}. 左朝向模板未标注，朝向校准已禁用：{facing_left_name}。")  # 提示并禁用校准，不阻断任务。
            facing_left_name = ''  # 清空后不再参与校准。
        if facing_right_name and not self.feature_ready(facing_right_name):  # 右朝向模板已配置但未在模板页标注。
            self.log_warning(f"Right facing template not annotated, facing calibration disabled: {facing_right_name}. 右朝向模板未标注，朝向校准已禁用：{facing_right_name}。")  # 提示并禁用校准，不阻断任务。
            facing_right_name = ''  # 清空后不再参与校准。
        facing_check = bool(facing_left_name and facing_right_name)  # 左右两个朝向模板都就绪才启用图像朝向校准。
        attack_key = self.config.get("Attack Key")  # 读取常规攻击按键，横向距离超过近战距离时按住它。
        melee_key = self.config.get("Melee Attack Key")  # 读取近战攻击按键，横向距离在近战距离内时按住它。
        melee_distance = float(self.config.get("Melee Distance") or 0)  # 读取近战距离（像素），横向距离绝对值不超过该值用近战键。
        attack_x_min, attack_x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 读取攻击区域左右边界（符号化像素），填反时自动交换。
        attack_y_min, attack_y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 读取攻击区域上下边界（符号化像素），填反时自动交换。
        del_interval = float(self.config.get("Del Key Interval") or 0)  # 读取自动按 Del 键的基础间隔秒数，0 表示禁用。
        del_variance = float(self.config.get("Del Key Interval Variance") or 0)  # 读取 Del 间隔的随机浮动量，0 表示固定间隔。
        last_del_time = time.time()  # 上次按 Del 键的时间，从任务启动开始计时。
        next_del_interval = self.next_del_interval(del_interval, del_variance)  # 首段间隔也带随机浮动，方法复用父类。
        minimap_threshold = float(self.config.get("Minimap Threshold"))  # 读取小地图匹配阈值。
        char_threshold = self.config.get("Character Threshold")  # 读取角色匹配阈值，朝向校准沿用同一阈值。
        hue_min, hue_max = sorted((int(self.config.get("Dot Hue Min")), int(self.config.get("Dot Hue Max"))))  # 读取黄点色相范围，填反时自动交换。
        dot_min_pixels = max(1, int(self.config.get("Dot Min Pixels")))  # 读取黄点最小面积，至少 1 像素。
        stuck_seconds = float(self.config.get("Stuck Seconds") or 0)  # 读取卡住判定时长（沿用巡逻默认值），0 表示禁用。
        resume_wait = float(self.config.get("Resume Wait Seconds") or 0)  # 读取脱困后的等待秒数。
        return_max_pct = float(self.config.get("Return Offset Max Percent") or 0)  # 读取碰撞偏移归位max（百分比），偏移超过该值时走回归位点。
        frame_interval = float(self.config.get("Frame Interval") or 0)  # 读取配置帧间隔，0 表示不额外限速。
        loop_interval = max(frame_interval, CAPTURE_MIN_INTERVAL)  # 截图采集固定 30FPS，配置更慢则尊重配置。
        home_percent = self.wait_home_percent(minimap_name, minimap_threshold, hue_min, hue_max, dot_min_pixels)  # 启动时记录角色在小地图的初始横向坐标比例。
        if home_percent is None:  # 超时仍记录不到初始坐标时无法判定偏移。
            self.log_warning("Cannot record start position on minimap, check Minimap Feature/Map Rect/Dot Hue. 未能在小地图上记录初始坐标比例，请检查小地图标注、地图区域与黄点色相配置，任务退出。")  # 提示排查方向。
            return  # 直接结束任务。
        self.log_info(f"Home position recorded: {home_percent:.1f}%, return offset max {return_max_pct:g}%. 已记录初始坐标比例 {home_percent:.1f}%，碰撞偏移归位max {return_max_pct:g}%。")  # 记录初始坐标供排查。
        gpu = self.build_gpu_matcher(char_name, monster_names)  # 尝试构建 GPU 匹配器并注册全部模板，失败返回 None 走 CPU。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，只在需要换向时单击方向键。
        self._held_move_key = None  # 当前持续按住的移动方向键，归位/换向/退出时必须松开它。
        self._held_attack_key = None  # 当前持续按住的攻击键（近战或常规），切换/目标消失/退出时必须松开它。
        self._attack_released_at = 0.0  # 上次松开攻击键的时间戳，攻击后摇会吞方向键，走动前需等待。
        returning = False  # 是否处于归位状态，带滞回：偏移超阈值进入，回落到阈值一半以内退出。
        anchor_x = None  # 归位卡住判定的位置锚点，黄点移动超过阈值时重置。
        anchor_time = time.time()  # 位置锚点上次更新时间。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        last_facing_check = 0.0  # 上次朝向校准的时间戳，0 表示启动后立即校准一次。
        loop_start = time.time()  # 本轮循环起点，用于把截图采集节拍钉在固定 30FPS。
        try:  # 包裹主循环，退出时兜底松开移动键与攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                wait = loop_interval - (time.time() - loop_start)  # 计算本轮剩余等待，把取帧节拍钉在固定 30FPS。
                if wait > 0:  # 未到下一帧时点时先等待，sleep 同时承担用户停止检查。
                    self.sleep(wait)  # 补齐帧间隔。
                loop_start = time.time()  # 记录本轮起点供下轮计算。
                if del_interval > 0 and time.time() - last_del_time >= next_del_interval:  # 到达本次随机间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                    next_del_interval = self.next_del_interval(del_interval, del_variance)  # 重新随机下一段间隔，避免固定节奏。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时等待下一帧时点再重试。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    loop_start = time.time()  # 重置本轮起点，避免下轮再补等待。
                    continue  # 进入下一帧处理。
                facing_due = facing_check and time.time() - last_facing_check >= 0.2  # 本轮是否需要用图像校准朝向，提前算好以便把朝向匹配一并提交并发。
                batch = MatchBatch()  # 本帧并发匹配批次：朝向与小地图匹配始终走 CPU，角色/怪物匹配仅在 GPU 不可用时一并提交。
                if facing_due:  # 朝向校准到期才提交，未到期不白烧两个核心。
                    batch.submit("face_l", lambda: self.find_one_raw(facing_left_name, frame, char_threshold))  # 左朝向模板匹配，朝向模板绝不能用镜像否则左右会互串。
                    batch.submit("face_r", lambda: self.find_one_raw(facing_right_name, frame, char_threshold))  # 右朝向模板匹配。
                batch.submit("minimap", lambda: self.find_minimap(minimap_name, frame, minimap_threshold))  # 小地图模板匹配定位小地图框。
                cpu_matches_submitted = gpu is None  # 角色/怪物匹配是否已随本批次一并提交，GPU 可用时改由显卡批量匹配。
                if cpu_matches_submitted:  # CPU 路径：角色与全部怪物匹配也一并投递，与朝向/小地图匹配和下面的归位判定并行跑。
                    self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 提交角色与全部怪物的原始/镜像匹配。
                if facing_due:  # 配置了朝向模板时定期用图像校准实际朝向。
                    last_facing_check = time.time()  # 记录本次校准时间。
                    actual_facing = self.resolve_template_facing(batch.get("face_l"), batch.get("face_r"))  # 左右朝向模板取置信度高者。
                    if actual_facing is not None and facing is None:  # 朝向未知（如任务刚启动）时用图像结果初始化。
                        facing = actual_facing  # 直接采信图像朝向。
                    elif actual_facing is not None and actual_facing != facing:  # 图像识别的朝向与系统记录不一致。
                        self.log_info(f"Facing corrected by template: {facing} -> {actual_facing}. 朝向已由图像校准修正：{facing} -> {actual_facing}。")  # 记录修正供排查。
                        facing = actual_facing  # 及时按图像修正，后续转身判定会自动补发转身键。
                minimap = batch.get("minimap")  # 取回小地图匹配结果。
                rect = self.map_rect(minimap) if minimap is not None else None  # 计算实际地图区域（画面坐标）。
                dot = self.detect_dot(frame, rect, hue_min, hue_max, dot_min_pixels) if rect is not None else None  # 在地图区域内直接检测角色黄点，不做跨帧追踪。
                percent = (dot[0] - rect[0]) / rect[2] * 100 if dot is not None else None  # 黄点当前横向位置百分比，检测不到为 None。
                offset = abs(percent - home_percent) if percent is not None else None  # 相对初始坐标比例的横向偏移（百分比）。
                if percent is not None:  # 黄点就绪时在 GUI 状态区显示当前位置与偏移。
                    self.info_set("Position", f"{percent:.1f}% (home {home_percent:.1f}%, offset {offset:.1f}%)")  # 当前位置、归位点与偏移量。
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
                    if not cpu_matches_submitted:  # 本帧刚走 GPU 分支异常回退 CPU，角色/怪物匹配还没提交。
                        self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 补提交本帧全部匹配。
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
                    pos_desc = f"{percent:.1f}% offset {offset:.1f}%" if percent is not None else "None"  # 黄点位置与偏移描述。
                    self.log_info(f"Match diag: POS[{pos_desc}] CHAR[{char_desc}] MOBS[{mob_desc}] 匹配诊断：黄点偏移与角色怪物框坐标及置信度。")  # 输出诊断日志供排查误匹配。
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
                if target is None:  # 本帧不攻击（无目标/无怪/归位中）时先松开攻击键。
                    if self._held_attack_key is not None:  # 有按住的攻击键。
                        self.send_key_up(self._held_attack_key)  # 松开攻击键。
                        self._held_attack_key = None  # 清空按住状态。
                        self._attack_released_at = time.time()  # 记录松键时间，供走动前等待攻击后摇结束。
                og.my_app.update_vision(self.draw_overlay(frame, minimap, rect, dot, home_percent, return_max_pct, character, monsters, nearest, target))  # 把带标注画面推送给 UI 实时展示。
                if target is not None and not returning:  # 攻击范围内有怪且未在归位时原地转向攻击，绝不持续按住方向键造成移动。
                    dx, dy = self.center_offset(character, target)  # 计算目标怪物相对角色的方向。
                    want_direction = 1 if dx > 0 else -1  # 1=怪在右侧，-1=怪在左侧。
                    want_key = melee_key if abs(dx) <= melee_distance else attack_key  # 横向距离在近战距离内用近战键，否则用常规攻击键，怪物走近走远时自动切换。
                    if self._held_attack_key is not None and self._held_attack_key != want_key:  # 换攻击键（近战/常规切换）时先松开旧键，避免两键同时按住。
                        self.send_key_up(self._held_attack_key)  # 松开当前按住的攻击键。
                        self._held_attack_key = None  # 清空按住状态。
                    if facing != want_direction:  # 朝向与怪物方向不一致时执行转身序列。
                        if self._held_attack_key is not None:  # 先松开攻击键：攻击后摇期间方向键输入会被游戏吞掉，导致转身失败后长时间空打。
                            self.send_key_up(self._held_attack_key)  # 松开攻击键。
                            self._held_attack_key = None  # 清空按住状态，转身完成后重新按住。
                            self.sleep(0.5)  # 等攻击后摇结束再发方向键，与巡逻任务验证过的防吞值一致。
                        self.send_key("right" if want_direction == 1 else "left", down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = want_direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if self._held_attack_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住近战或常规攻击键不放。
                        self._held_attack_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Attacking")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if percent is None:  # 小地图或黄点丢失时无法判定偏移：松开移动键原地待命，攻击照常。
                    if self._held_move_key is not None:  # 有按住的移动键。
                        self.send_key_up(self._held_move_key)  # 松开方向键，位置未知时走动有越界风险。
                        self._held_move_key = None  # 清空按住状态。
                    anchor_x = None  # 黄点丢失时卡住锚点失效。
                    self.info_set("Status", "Minimap/dot lost")  # 在 GUI 显示定位丢失状态。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    continue  # 定位恢复后自动继续归位判定。
                if not monsters and not returning:  # 场上没有怪物且不在归位中时立即进入归位，把碰撞偏移拉回记录点。
                    returning = True  # 切换为归位状态。
                    self.log_info(f"No monster, return home from {percent:.1f}% to {home_percent:.1f}%. 场上无怪，从 {percent:.1f}% 走回归位点 {home_percent:.1f}%。")  # 记录归位事件供排查。
                if offset > return_max_pct and not returning:  # 偏移超过碰撞偏移归位max 时进入归位。
                    returning = True  # 切换为归位状态。
                    self.log_info(f"Offset {offset:.1f}% exceeds max {return_max_pct:g}%, return home. 偏移 {offset:.1f}% 超过阈值 {return_max_pct:g}%，开始归位。")  # 记录归位事件供排查。
                if returning:  # 归位状态：按偏移方向走回记录点，带滞回避免边界抖动。
                    if offset <= return_max_pct * RETURN_DONE_RATIO:  # 偏移回落到阈值一半以内才算归位完成。
                        returning = False  # 退出归位状态，恢复原地攻击。
                        self.log_info(f"Home reached at {percent:.1f}% (home {home_percent:.1f}%). 已归位到 {percent:.1f}%（记录点 {home_percent:.1f}%），恢复攻击。")  # 记录归位完成供排查。
                    else:  # 尚未回到容差带内，继续朝记录点走动。
                        anchor_x, anchor_time = self.update_stuck_anchor(dot[0], anchor_x, anchor_time, rect[2])  # 更新卡住判定锚点。
                        walk_direction = 1 if percent < home_percent else -1  # 归位走动方向：记录点在右侧向右走，在左侧向左走。
                        if stuck_seconds > 0 and self._held_move_key is not None and time.time() - anchor_time >= stuck_seconds:  # 按住方向键但位置长期不动视为卡住。
                            self.recover_stuck(self._held_move_key, walk_direction, resume_wait)  # 反向短移脱困，脱困后本帧松键，下一帧按最新偏移继续归位。
                            self._held_move_key = None  # 恢复过程已松键。
                            anchor_x, anchor_time = None, time.time()  # 脱困后重置卡住锚点。
                        if time.time() < self._attack_released_at + ATTACK_TO_MOVE_WAIT:  # 攻击刚结束的等待期内只识图不移动，方向键会被攻击后摇吞掉。
                            self.info_set("Status", "Return wait")  # 在 GUI 显示归位等待状态。
                            self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                            continue  # 等待期结束后自动恢复走动。
                        want_key = "right" if walk_direction == 1 else "left"  # 归位方向需要的方向键。
                        if self._held_move_key != want_key:  # 换向时先松开旧键再按新键，避免两键同时按住。
                            if self._held_move_key is not None:  # 有旧键按住。
                                self.send_key_up(self._held_move_key)  # 松开旧方向键。
                            self.send_key_down(want_key)  # 持续按住新方向键走回归位点。
                            self._held_move_key = want_key  # 记录当前按住的键。
                            facing = walk_direction  # 走动时朝向与移动方向一致，供攻击换向判定直接使用。
                        self.info_set("Status", f"Returning {'right' if walk_direction == 1 else 'left'} (offset {offset:.1f}%)")  # 在 GUI 显示归位方向与当前偏移。
                        self.sleep(loop_interval)  # 按固定 30FPS 节拍等待后处理下一帧。
                        continue  # 归位中不进入原地待命分支。
                if self._held_move_key is not None:  # 归位完成或无需归位时松开移动键，绝不带着移动键待命。
                    self.send_key_up(self._held_move_key)  # 松开方向键。
                    self._held_move_key = None  # 清空按住状态。
                anchor_x = None  # 停止走动后卡住锚点失效。
                self.info_set("Status", "Camping" if character is not None else "Character not found")  # 无怪或偏移在容差内时原地待命，显示当前状态。
                self.sleep(loop_interval)  # 按固定 30FPS 节拍等待后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if self._held_attack_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(self._held_attack_key)  # 松开它。
            if self._held_move_key is not None:  # 有按住未松的方向键。
                self.send_key_up(self._held_move_key)  # 松开它。

    def draw_overlay(self, frame, minimap, rect, dot, home_percent, return_max_pct, character, monsters, nearest, target):  # 在一帧画面上绘制单点挂机与攻击的全部标注，返回新画面。
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
            self.draw_target(canvas, monster, (0, 0, 255), "MOB" + ("-flip" if getattr(monster, "flipped", False) else ""), cross=False)  # 红色框不带十字延长线，避免多只怪物时红线交叉刷屏，镜像命中时标注 -flip。
        if target is not None:  # 存在当前攻击目标时额外高亮。
            self.draw_target(canvas, target, (255, 255, 255), "TARGET", cross=False)  # 白色框标出正在攻击的怪物，消失后不再绘制。
        if character is not None and nearest is not None:  # 角色与最近怪物都存在时绘制距离信息。
            cx, cy = self.box_center(character)  # 角色中心点。
            mx, my = self.box_center(nearest)  # 怪物中心点。
            cv2.line(canvas, (int(cx), int(cy)), (int(mx), int(my)), (255, 255, 0), 1)  # 角色与怪物中心连线。
            dx, dy = self.center_offset(character, nearest)  # 计算 xy 距离。
            self.draw_text(canvas, f"dx={dx} dy={dy}", (int((cx + mx) / 2), int((cy + my) / 2)), (255, 255, 0))  # 在连线中点显示 xy 距离。
        if minimap is not None:  # 找到小地图时绘制小地图框标注。
            cv2.rectangle(canvas, (minimap.x, minimap.y), (minimap.x + minimap.width, minimap.y + minimap.height), (0, 255, 0), 2)  # 绿色框标出小地图。
            self.draw_text(canvas, "MINIMAP", (minimap.x, max(minimap.y - 6, 14)), (0, 255, 0))  # 小地图框上方标签。
        if rect is not None:  # 有地图区域时绘制区域框、归位点绿线与容差带边界黄线。
            rx, ry, rw, rh = rect  # 地图区域。
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色框标出实际地图区域。
            home_x = int(rx + rw * home_percent / 100)  # 归位点画面 x 坐标。
            height = canvas.shape[0]  # 画面高度，标记线贯穿全高方便观察。
            cv2.line(canvas, (home_x, 0), (home_x, height), (0, 255, 0), 2)  # 归位点纵向绿线，标出任务启动时记录的初始坐标比例。
            self.draw_text(canvas, f"HOME {home_percent:.1f}%", (home_x + 4, 20), (0, 255, 0))  # 归位点标签。
            for edge in (home_percent - return_max_pct, home_percent + return_max_pct):  # 容差带左右边界。
                edge_x = int(rx + rw * max(0.0, min(100.0, edge)) / 100)  # 边界画面 x 坐标，越界时夹到地图区域边缘。
                cv2.line(canvas, (edge_x, 0), (edge_x, height), (0, 255, 255), 1)  # 容差带边界纵向黄线，偏移超出后触发归位。
        if dot is not None:  # 检测到角色黄点时绘制黄点标注。
            cv2.circle(canvas, (dot[0], dot[1]), 6, (0, 255, 255), 2)  # 黄色圆圈标出角色位置。
            percent = (dot[0] - rect[0]) / rect[2] * 100  # 当前位置百分比。
            self.draw_text(canvas, f"DOT {percent:.1f}%", (dot[0] + 8, dot[1] - 8), (0, 255, 255))  # 黄点旁显示位置百分比。
        return canvas  # 返回绘制完成的画面。
