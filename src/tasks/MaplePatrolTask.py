import random  # 导入标准库 random，用于 Del 间隔的随机浮动。
import time  # 导入标准库 time，用于边界折返节奏、卡住计时与攻击节奏。

import cv2  # 导入 OpenCV，用于小地图模板匹配、黄点取色与画面标注绘制。
import numpy as np  # 导入 NumPy，用于 HSV 取色掩码运算。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

from src.tasks.MapleIdleTask import MapleIdleTask  # 导入挂机任务，复用其模板检测、GPU 匹配与攻击相关的全部辅助方法。

MOVE_LEFT_KEY = "left"  # 左方向键：向左巡逻时持续按住。
MOVE_RIGHT_KEY = "right"  # 右方向键：向右巡逻时持续按住。

DOT_SAT_MIN = 80  # HSV 取黄色的饱和度下限，过滤灰白色干扰。
DOT_VAL_MIN = 80  # HSV 取黄色的亮度下限，过滤暗色干扰。
DOT_MISSING_HOLD_SECONDS = 2.0  # 黄点丢失期间允许继续按住方向键的最大秒数，超时松键等黄点恢复，避免盲走越界。
STUCK_MOVE_SECONDS = 0.5  # 卡住恢复时反向移动按住的秒数。
STUCK_MIN_MOVE_PERCENT = 0.8  # 位置变化达到该百分比（地图区域宽度）才算“移动了”，否则累计卡住时长。
ATTACK_TO_MOVE_WAIT = 0.5  # 攻击结束后恢复巡逻按方向键前的等待秒数，过早按键会被攻击后摇动画吞掉（沿用挂机任务验证过的时序）。
FACING_CHECK_INTERVAL = 0.2  # 用朝向模板校准角色实际朝向的间隔秒数，兼顾及时修正与 CPU 开销。


class MaplePatrolTask(MapleIdleTask):  # 定义冒险岛小地图巡逻打怪任务，继承挂机任务复用检测与攻击能力。

    def __init__(self, *args, **kwargs):  # 构造函数，先复用父类全部配置再按巡逻场景裁剪。
        super().__init__(*args, **kwargs)  # 父类构造已填好角色/怪物/攻击等全部配置项与帮助文本。
        self.name = "Maple Patrol"  # 任务显示名称。
        self.description = "Minimap patrol with combat: locate the minimap by template and track the character with the yellow dot color, hold the direction key to walk back and forth between the left/right percent boundaries; when a monster enters the attack range, stop and attack the nearest one until it disappears; red vertical lines mark the boundaries in the live vision."  # 任务描述：小地图巡逻+打怪，黄点跟踪往返边界，攻击范围内有怪就停下打最近一只，边界用红色竖线标出。
        self.icon = FluentIcon.PLAY  # 任务图标。
        for key in ("Move Interval", "Move Away Seconds", "Move Back Seconds", "Turn Interval"):  # 定时位移与转身策略由小地图巡逻取代，裁剪掉对应配置。
            self.default_config.pop(key, None)  # 移除默认值。
            self.config_description.pop(key, None)  # 移除帮助文本。
        self.default_config.update({  # 巡逻专属配置项，父类没有需自行补充。
            "Del Key Interval Variance": 20.0,  # Del 间隔随机浮动量：每次按完后下一次间隔在基础值 ±该值内随机，设为 0 表示固定间隔。
            "Character Facing Left Feature": "",  # 角色左朝向模板：模板页标注的分类名，用于图像识别校准朝向，留空禁用朝向校准。
            "Character Facing Right Feature": "",  # 角色右朝向模板：模板页标注的分类名，用于图像识别校准朝向，两个朝向模板都配置才启用校准。
            "Minimap Feature": "完整小地图",  # 小地图：模板页标注的分类名，按分类匹配小地图位置。
            "Minimap Threshold": 0.8,  # 小地图匹配阈值：越高匹配越严格。
            "Map Rect": "",  # 小地图框内的实际地图区域（相对小地图框的百分比，格式 x,y,w,h 如 5,20,90,75），留空表示整个小地图框。
            "Patrol Left Percent": 10.0,  # 巡逻左边界：地图区域宽度的百分比，黄点小于等于该值时改为向右走。
            "Patrol Right Percent": 90.0,  # 巡逻右边界：地图区域宽度的百分比，黄点大于等于该值时改为向左走。
            "Dot Hue Min": 18,  # 黄点 HSV 色相下限（0-179），用于抓取角色黄点。
            "Dot Hue Max": 38,  # 黄点 HSV 色相上限（0-179），用于抓取角色黄点。
            "Dot Min Pixels": 4,  # 黄点最小像素面积：小于该面积的连通块丢弃。
            "Stuck Seconds": 8.0,  # 卡住判定时长：按住方向键但黄点位置持续无明显变化达该秒数时反向脱困，0 禁用。
            "Resume Wait Seconds": 1.0,  # 卡住脱困后恢复巡逻前的等待秒数。
        })
        self.config_description.update({  # 巡逻专属配置项的帮助文本。
            "Del Key Interval Variance": "Random ±variance applied to Del Key Interval after each press, e.g. 100 with 20 gives 80-120; 0 means fixed. Del 间隔随机浮动量：每次按完后下一次间隔在基础值 ±该值内随机，0 表示固定间隔。",
            "Character Facing Left Feature": "Category name annotated for the left-facing character template; empty disables facing calibration. 角色左朝向模板：模板页标注的分类名，留空禁用朝向校准。",
            "Character Facing Right Feature": "Category name annotated for the right-facing character template; both facing templates are required to enable calibration. 角色右朝向模板：模板页标注的分类名，两个朝向模板都配置才启用校准。",
            "Minimap Feature": "Category name annotated for the minimap in the Template tab. 小地图：在模板页标注的分类名。",
            "Minimap Threshold": "Template match threshold for the minimap, higher means stricter. 小地图匹配阈值，越高越严格。",
            "Map Rect": "Actual map area inside the minimap box as percents of the minimap box, format x,y,w,h e.g. 5,20,90,75; empty means the whole minimap box. 小地图框内的实际地图区域（相对小地图框的百分比，格式 x,y,w,h），留空表示整个小地图框。",
            "Patrol Left Percent": "Left patrol boundary as percent of the map area width; the character turns right when the dot reaches it. 巡逻左边界（地图区域宽度的百分比），黄点到达后向右折返。",
            "Patrol Right Percent": "Right patrol boundary as percent of the map area width; the character turns left when the dot reaches it. 巡逻右边界（地图区域宽度的百分比），黄点到达后向左折返。",
            "Dot Hue Min": "HSV hue lower bound (0-179) for picking the character's yellow dot. 黄点 HSV 色相下限（0-179）。",
            "Dot Hue Max": "HSV hue upper bound (0-179) for picking the character's yellow dot. 黄点 HSV 色相上限（0-179）。",
            "Dot Min Pixels": "Minimum pixel area of the yellow dot; smaller blobs are ignored. 黄点最小像素面积，更小的连通块丢弃。",
            "Stuck Seconds": "Seconds of no visible dot movement while holding a direction key before a reverse escape; 0 disables it. 按住方向键但黄点持续不动达该秒数时反向脱困，0 禁用。",
            "Resume Wait Seconds": "Seconds to wait after a stuck escape before resuming patrol. 卡住脱困后恢复巡逻前的等待秒数。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
        error = super().validate_config(key, value)  # 复用父类的攻击按键合法性校验。
        if error is not None:  # 父类校验不通过时直接返回。
            return error  # 透传错误提示。
        if key in ("Patrol Left Percent", "Patrol Right Percent"):  # 巡逻边界必须是合法百分比。
            try:  # 尝试按浮点数解析。
                percent = float(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非数字输入。
                return "Patrol percent must be a number. 巡逻占比必须是数字。"  # 阻止保存并提示。
            if not 0 <= percent <= 100:  # 超出百分比范围。
                return "Patrol percent must be within 0-100. 巡逻占比必须在 0-100 之间。"  # 阻止保存并提示。
        if key in ("Dot Hue Min", "Dot Hue Max"):  # 色相必须是合法的 OpenCV HSV 色相。
            try:  # 尝试按整数解析。
                hue = int(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非整数输入。
                return "Hue must be an integer. 色相必须是整数。"  # 阻止保存并提示。
            if not 0 <= hue <= 179:  # OpenCV 色相范围是 0-179。
                return "Hue must be within 0-179. 色相必须在 0-179 之间。"  # 阻止保存并提示。
        if key == "Map Rect" and str(value or '').strip():  # 地图区域非空时必须能解析为 4 个百分比。
            try:  # 复用运行时解析逻辑做校验。
                self.parse_map_rect(str(value))  # 解析失败会抛 ValueError。
            except ValueError:  # 格式不合法。
                return "Map Rect must be 4 comma-separated percents like x,y,w,h. 地图区域必须是 x,y,w,h 四个逗号分隔的百分比。"  # 阻止保存并提示。
        return None  # 其他配置项不做额外校验。

    def next_del_interval(self, base, variance):  # 生成下一次按 Del 键前的等待秒数。
        if base <= 0:  # 基础间隔为 0 表示禁用自动按 Del。
            return 0.0  # 返回 0，调用方不会触发按键。
        if variance <= 0:  # 未配置浮动量时保持固定间隔。
            return base  # 直接返回基础值。
        return max(0.1, base + random.uniform(-variance, variance))  # 基础值 ±浮动量内随机，下限 0.1 秒防止浮动过大导致连续快速按键。

    def find_one_raw(self, feature_name, frame, threshold):  # 仅用原始朝向匹配单个模板（不做镜像回退），返回置信度最高的框或 None。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            return self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，朝向模板不能用镜像否则左右会互串。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。

    def detect_template_facing(self, frame, left_name, right_name, threshold):  # 用左右朝向模板判定角色实际朝向：-1=朝左、1=朝右，都未命中返回 None。
        left_box = self.find_one_raw(left_name, frame, threshold)  # 匹配左朝向模板。
        right_box = self.find_one_raw(right_name, frame, threshold)  # 匹配右朝向模板。
        if left_box is None and right_box is None:  # 两个模板都未命中。
            return None  # 朝向未知，保持系统当前记录。
        if left_box is None:  # 只有右朝向命中。
            return 1  # 角色朝右。
        if right_box is None:  # 只有左朝向命中。
            return -1  # 角色朝左。
        return -1 if getattr(left_box, "confidence", 0) >= getattr(right_box, "confidence", 0) else 1  # 两个都命中时采信置信度更高的一侧。

    def run(self):  # 任务运行入口：小地图巡逻移动 + 攻击范围内停下打怪的双层循环。
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
        left_pct, right_pct = sorted((float(self.config.get("Patrol Left Percent")), float(self.config.get("Patrol Right Percent"))))  # 读取巡逻左右边界（百分比），填反时自动交换。
        if left_pct >= right_pct:  # 左右边界相同没有可巡逻区间。
            self.log_warning("Left and right patrol percents are the same, nothing to patrol. 左右巡逻占比相同，无可巡逻区间，任务退出。")  # 提示配置问题。
            return  # 直接结束任务。
        hue_min, hue_max = sorted((int(self.config.get("Dot Hue Min")), int(self.config.get("Dot Hue Max"))))  # 读取黄点色相范围，填反时自动交换。
        dot_min_pixels = max(1, int(self.config.get("Dot Min Pixels")))  # 读取黄点最小面积，至少 1 像素。
        stuck_seconds = float(self.config.get("Stuck Seconds") or 0)  # 读取卡住判定时长，0 表示禁用。
        resume_wait = float(self.config.get("Resume Wait Seconds") or 0)  # 读取脱困后的等待秒数。
        frame_interval = float(self.config.get("Frame Interval") or 0.05)  # 读取帧间隔。
        gpu = self.build_gpu_matcher(char_name, monster_names)  # 尝试构建 GPU 匹配器并注册全部模板，失败返回 None 走 CPU。
        direction = 1  # 巡逻方向：1=向右、-1=向左，默认先向右走。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，移动时与方向同步，攻击时用于换向判定。
        held_move_key = None  # 当前持续按住的移动方向键，换向/攻击/退出时必须松开它。
        held_attack_key = None  # 当前持续按住的攻击键（近战或常规），切换/目标消失/退出时必须松开它。
        anchor_x = None  # 卡住判定的位置锚点，黄点移动超过阈值时重置。
        anchor_time = time.time()  # 位置锚点上次更新时间。
        dot_missing_since = None  # 黄点开始丢失的时间戳，None 表示当前能检测到。
        patrol_resume_at = 0.0  # 允许重新按住移动键的时间点，攻击刚结束后短暂等待避免方向键被后摇吞掉。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        last_facing_check = 0.0  # 上次朝向校准的时间戳，0 表示启动后立即校准一次。
        try:  # 包裹主循环，退出时兜底松开移动键与攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                if del_interval > 0 and time.time() - last_del_time >= next_del_interval:  # 到达本次随机间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                    next_del_interval = self.next_del_interval(del_interval, del_variance)  # 重新随机下一段间隔，避免固定节奏。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时短暂等待后重试。
                    self.sleep(frame_interval)  # 等待一个帧间隔。
                    continue  # 进入下一帧处理。
                if facing_check and time.time() - last_facing_check >= FACING_CHECK_INTERVAL:  # 配置了朝向模板时定期用图像校准实际朝向。
                    last_facing_check = time.time()  # 记录本次校准时间。
                    actual_facing = self.detect_template_facing(frame, facing_left_name, facing_right_name, self.config.get("Character Threshold"))  # 左右朝向模板取置信度高者。
                    if actual_facing is not None and facing is None:  # 朝向未知（如任务刚启动）时用图像结果初始化。
                        facing = actual_facing  # 直接采信图像朝向。
                    elif actual_facing is not None and actual_facing != facing:  # 图像识别的朝向与系统记录不一致。
                        self.log_info(f"Facing corrected by template: {facing} -> {actual_facing}. 朝向已由图像校准修正：{facing} -> {actual_facing}。")  # 记录修正供排查。
                        facing = actual_facing  # 及时按图像修正，后续转身判定会自动补发转身键。
                minimap = self.find_minimap(minimap_name, frame, minimap_threshold)  # 模板匹配定位小地图框。
                rect = self.map_rect(minimap) if minimap is not None else None  # 计算实际地图区域（画面坐标）。
                dot = self.detect_dot(frame, rect, hue_min, hue_max, dot_min_pixels) if rect is not None else None  # 在地图区域内直接检测角色黄点，不做跨帧追踪。
                if minimap is None:  # 找不到小地图时停止巡逻移动，攻击判定照常进行。
                    if held_move_key is not None:  # 有按住的移动键。
                        self.send_key_up(held_move_key)  # 松开方向键停止移动。
                        held_move_key = None  # 清空按住状态。
                    anchor_x, dot_missing_since = None, None  # 清空卡住锚点与丢失计时。
                elif dot is None:  # 小地图在但黄点检测不到：短暂保持移动，长时间丢失则松键等待。
                    anchor_x = None  # 黄点丢失时卡住锚点失效。
                    if dot_missing_since is None:  # 刚开始丢失。
                        dot_missing_since = time.time()  # 开始丢失计时。
                    elif held_move_key is not None and time.time() - dot_missing_since >= DOT_MISSING_HOLD_SECONDS:  # 丢失超时仍按住方向键有盲走风险。
                        self.send_key_up(held_move_key)  # 松开方向键等黄点恢复。
                        held_move_key = None  # 清空按住状态。
                else:  # 小地图与黄点都就绪，执行巡逻判断。
                    dot_missing_since = None  # 黄点恢复，清空丢失计时。
                    percent = (dot[0] - rect[0]) / rect[2] * 100  # 黄点横向位置相对地图区域宽度的百分比。
                    self.info_set("Position", f"{percent:.1f}%")  # 在 GUI 状态区显示当前位置百分比。
                    if direction == 1 and percent >= right_pct:  # 向右移动到达右边界时折返向左。
                        direction = -1  # 切换巡逻方向为向左。
                        self.log_info(f"Reached right bound {right_pct:g}%, turn left. 到达右边界 {right_pct:g}%，改为向左巡逻。")  # 记录折返供排查。
                    elif direction == -1 and percent <= left_pct:  # 向左移动到达左边界时折返向右。
                        direction = 1  # 切换巡逻方向为向右。
                        self.log_info(f"Reached left bound {left_pct:g}%, turn right. 到达左边界 {left_pct:g}%，改为向右巡逻。")  # 记录折返供排查。
                    anchor_x, anchor_time = self.update_stuck_anchor(dot[0], anchor_x, anchor_time, rect[2])  # 更新卡住判定锚点。
                    if stuck_seconds > 0 and held_move_key is not None and time.time() - anchor_time >= stuck_seconds:  # 按住方向键但位置长期不动视为卡住。
                        direction = self.recover_stuck(held_move_key, direction, resume_wait)  # 反向短移脱困并翻转巡逻方向。
                        held_move_key = None  # 恢复过程已松键，随后重新按住新方向键。
                        anchor_x, anchor_time = None, time.time()  # 脱困后重置卡住锚点。
                    if held_move_key is not None:  # 确实在移动时朝向与移动方向一致。
                        facing = direction  # 同步朝向，供攻击换向判定直接使用，减少转身探测。
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
                        target = min(in_range, key=lambda m: self.center_distance(character, m))  # 优先攻击距离最近的目标，另一侧出现更近的怪立即转身换打，不做同侧锁定。
                og.my_app.update_vision(self.draw_overlay(frame, minimap, rect, dot, left_pct, right_pct, character, monsters, nearest, target))  # 把带标注画面推送给 UI 实时展示。
                if target is not None:  # 攻击范围内有怪物时停下巡逻原地攻击。
                    if held_move_key is not None:  # 进入攻击前先松开移动键，站着打不边走边打。
                        self.send_key_up(held_move_key)  # 松开方向键。
                        held_move_key = None  # 清空按住状态。
                    dx, dy = self.center_offset(character, target)  # 计算目标怪物相对角色的方向。
                    want_direction = 1 if dx > 0 else -1  # 1=怪在右侧，-1=怪在左侧。
                    want_key = melee_key if abs(dx) <= melee_distance else attack_key  # 横向距离在近战距离内用近战键，否则用常规攻击键，怪物走近走远时自动切换。
                    if held_attack_key is not None and held_attack_key != want_key:  # 换攻击键（近战/常规切换）时先松开旧键，避免两键同时按住。
                        self.send_key_up(held_attack_key)  # 松开当前按住的攻击键。
                        held_attack_key = None  # 清空按住状态。
                    if facing != want_direction:  # 朝向与怪物方向不一致时执行转身序列，绝不持续按住方向键造成移动。
                        if held_attack_key is not None:  # 先松开攻击键：攻击后摇期间方向键输入会被游戏吞掉，导致转身失败后长时间空打。
                            self.send_key_up(held_attack_key)  # 松开攻击键。
                            held_attack_key = None  # 清空按住状态，转身完成后重新按住。
                            self.sleep(0.5)  # 等攻击后摇结束再发方向键，与转身策略验证过的防吞值一致。
                        self.send_key(MOVE_RIGHT_KEY if want_direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = want_direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if held_attack_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住近战或常规攻击键不放。
                        held_attack_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Attacking")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if held_attack_key is not None:  # 目标消失时松开持续按住的攻击键并准备恢复巡逻。
                    self.send_key_up(held_attack_key)  # 松开攻击键。
                    held_attack_key = None  # 清空按住状态。
                    patrol_resume_at = time.time() + ATTACK_TO_MOVE_WAIT  # 攻击后摇会吞方向键输入，短暂等待后再恢复移动。
                if time.time() < patrol_resume_at:  # 攻击刚结束的等待期内只识图不移动。
                    self.info_set("Status", "Resuming patrol")  # 在 GUI 显示恢复巡逻等待状态。
                    self.sleep(frame_interval)  # 等待一个帧间隔。
                    continue  # 等待期结束后自动恢复巡逻。
                if minimap is not None:  # 小地图可用时按巡逻方向持续移动。
                    want_key = MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY  # 当前巡逻方向需要的方向键。
                    if held_move_key != want_key:  # 换向时先松开旧键再按新键，避免两键同时按住。
                        if held_move_key is not None:  # 有旧键按住。
                            self.send_key_up(held_move_key)  # 松开旧方向键。
                        self.send_key_down(want_key)  # 持续按住新方向键保持移动。
                        held_move_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Patrolling right" if direction == 1 else "Patrolling left")  # 在 GUI 显示当前巡逻方向。
                else:  # 小地图不可用时保持静止等待。
                    self.info_set("Status", "Minimap not found")  # 在 GUI 显示未找到小地图。
                self.sleep(frame_interval)  # 等待一个帧间隔后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if held_attack_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(held_attack_key)  # 松开它。
            if held_move_key is not None:  # 有按住未松的方向键。
                self.send_key_up(held_move_key)  # 松开它。

    def find_minimap(self, feature_name, frame, threshold):  # 在整帧画面模板匹配小地图，返回小地图框或 None。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            return self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=True, horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，小地图可能被拖动过位置。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。

    def map_rect(self, minimap):  # 计算实际地图区域（画面坐标），Map Rect 留空时取整个小地图框。
        x0, y0, w, h = minimap.x, minimap.y, minimap.width, minimap.height  # 小地图框左上角与尺寸。
        text = str(self.config.get("Map Rect") or '').strip()  # 读取地图区域配置文本。
        if not text:  # 未配置地图区域。
            return x0, y0, w, h  # 整个小地图框就是地图区域。
        try:  # 尝试解析四个百分比。
            px, py, pw, ph = self.parse_map_rect(text)  # 解析为相对小地图框的百分比。
        except ValueError:  # 格式非法时兜底整个小地图框。
            return x0, y0, w, h  # 不阻断巡逻。
        return x0 + int(w * px / 100), y0 + int(h * py / 100), max(1, int(w * pw / 100)), max(1, int(h * ph / 100))  # 百分比换算为画面坐标区域。

    @staticmethod
    def parse_map_rect(text):  # 把 "x,y,w,h" 百分比文本解析成四元浮点组，非法格式抛 ValueError。
        parts = str(text or '').replace('%', '').replace(';', ',').split(',')  # 去掉百分号统一分隔符后切分。
        if len(parts) != 4:  # 必须恰好四个数。
            raise ValueError("Map Rect needs 4 values")  # 数量不对。
        values = [float(part.strip()) for part in parts]  # 逐个解析为浮点数，非数字自动抛 ValueError。
        if any(value < 0 or value > 100 for value in values):  # 百分比必须在 0-100。
            raise ValueError("Map Rect value out of range")  # 越界。
        return tuple(values)  # 返回 (x, y, w, h) 百分比。

    def detect_dot(self, frame, rect, hue_min, hue_max, min_pixels):  # 在地图区域内检测角色黄点，返回画面坐标 (x, y, 面积) 或 None。
        x, y, w, h = rect  # 地图区域。
        roi = frame[y:y + h, x:x + w]  # 裁剪地图区域画面。
        if roi.size == 0:  # 区域无效。
            return None  # 无法检测。
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转 HSV 按色相取色。
        lower = np.array([hue_min, DOT_SAT_MIN, DOT_VAL_MIN], dtype=np.uint8)  # 取色下限：色相下限 + 饱和度下限 + 亮度下限。
        upper = np.array([hue_max, 255, 255], dtype=np.uint8)  # 取色上限。
        mask = cv2.inRange(hsv, lower, upper)  # 生成黄点掩码。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算去除孤立噪点。
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取全部外轮廓。
        candidates = []  # 达到面积要求的候选块列表。
        for contour in contours:  # 逐个轮廓计算面积与重心。
            area = cv2.contourArea(contour)  # 轮廓面积。
            if area < min_pixels:  # 太小不可能是角色黄点。
                continue  # 丢弃。
            moments = cv2.moments(contour)  # 计算几何矩求重心。
            if moments["m00"] == 0:  # 面积矩为 0 无法求重心。
                continue  # 丢弃。
            cx = int(moments["m10"] / moments["m00"]) + x  # 重心换算为画面坐标。
            cy = int(moments["m01"] / moments["m00"]) + y  # 重心换算为画面坐标。
            candidates.append((cx, cy, area))  # 加入候选。
        if not candidates:  # 地图区域内没有黄点。
            return None  # 返回未检测到。
        return max(candidates, key=lambda c: c[2])  # 小地图内只有角色一个黄点，直接采信面积最大块；偶发噪点面积小自然被排除，无需跨帧追踪。

    def update_stuck_anchor(self, x, anchor_x, anchor_time, width):  # 更新卡住判定锚点：位置变化超过阈值视为移动了，重置锚点与时间。
        move_px = max(2, width * STUCK_MIN_MOVE_PERCENT / 100)  # 变化阈值：2 像素与区域宽度 0.8% 取较大者。
        if anchor_x is None or abs(x - anchor_x) >= move_px:  # 位置变化足够大视为角色在移动。
            return x, time.time()  # 重置锚点位置与时间。
        return anchor_x, anchor_time  # 未移动，保留原锚点继续累计卡住时长。

    def recover_stuck(self, held_key, direction, resume_wait):  # 卡住恢复：松开当前键，反向短移脱困，等待后翻转巡逻方向。
        self.info_set("Status", "Stuck, recovering")  # 在 GUI 显示卡住恢复状态。
        self.log_warning("Position stuck while holding direction key, reverse escape. 按住方向键但位置停滞，反向脱困。")  # 记录卡住事件供排查。
        self.send_key_up(held_key)  # 松开当前方向键。
        back_key = MOVE_LEFT_KEY if direction == 1 else MOVE_RIGHT_KEY  # 脱困键：卡住方向的反方向。
        self.send_key(back_key, down_time=STUCK_MOVE_SECONDS)  # 反向短按移动，离开卡住点。
        if resume_wait > 0:  # 配置了恢复等待。
            self.sleep(resume_wait)  # 等角色状态稳定再恢复巡逻。
        return -direction  # 翻转巡逻方向继续巡逻。

    def draw_overlay(self, frame, minimap, rect, dot, left_pct, right_pct, character, monsters, nearest, target):  # 在一帧画面上绘制巡逻与攻击的全部标注，返回新画面。
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
        if rect is not None:  # 有地图区域时绘制区域框与两条巡逻边界红线。
            rx, ry, rw, rh = rect  # 地图区域。
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色框标出实际地图区域。
            left_x = int(rx + rw * left_pct / 100)  # 左边界画面 x 坐标。
            right_x = int(rx + rw * right_pct / 100)  # 右边界画面 x 坐标。
            height = canvas.shape[0]  # 画面高度，红线贯穿全高方便观察。
            cv2.line(canvas, (left_x, 0), (left_x, height), (0, 0, 255), 2)  # 左边界纵向红线。
            cv2.line(canvas, (right_x, 0), (right_x, height), (0, 0, 255), 2)  # 右边界纵向红线。
            self.draw_text(canvas, f"L {left_pct:g}%", (left_x + 4, 20), (0, 0, 255))  # 左边界百分比标签。
            self.draw_text(canvas, f"R {right_pct:g}%", (right_x + 4, 20), (0, 0, 255))  # 右边界百分比标签。
        if dot is not None:  # 检测到角色黄点时绘制黄点标注。
            cv2.circle(canvas, (dot[0], dot[1]), 6, (0, 255, 255), 2)  # 黄色圆圈标出角色位置。
            percent = (dot[0] - rect[0]) / rect[2] * 100  # 当前位置百分比。
            self.draw_text(canvas, f"DOT {percent:.1f}%", (dot[0] + 8, dot[1] - 8), (0, 255, 255))  # 黄点旁显示位置百分比。
        return canvas  # 返回绘制完成的画面。
