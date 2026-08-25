import time  # 导入标准库 time，用于边界切换节奏与卡住计时。

import cv2  # 导入 OpenCV，用于小地图模板匹配、黄点取色与画面标注绘制。
import numpy as np  # 导入 NumPy，用于 HSV 取色掩码运算。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

from src.tasks.MyBaseTask import MyBaseTask  # 导入项目任务基类，导入时会同时生效标注文件 UTF-8 读取补丁。

MOVE_LEFT_KEY = "left"  # 左方向键：向左巡逻时持续按住。
MOVE_RIGHT_KEY = "right"  # 右方向键：向右巡逻时持续按住。

DOT_MAX_JUMP_PERCENT = 30.0  # 黄点跟踪跳变阈值（地图区域对角线的百分比）：候选块与上一帧位置距离超过该值视为瞬移或严重干扰，回退采信面积最大块。
DOT_SAT_MIN = 80  # HSV 取黄色的饱和度下限，过滤灰白色干扰。
DOT_VAL_MIN = 80  # HSV 取黄色的亮度下限，过滤暗色干扰。
DOT_MISSING_HOLD_SECONDS = 2.0  # 黄点丢失期间允许继续按住方向键的最大秒数，超时松键等黄点恢复，避免盲走越界。
STUCK_MOVE_SECONDS = 0.5  # 卡住恢复时反向移动按住的秒数。
STUCK_MIN_MOVE_PERCENT = 0.8  # 位置变化达到该百分比（地图区域宽度）才算“移动了”，否则累计卡住时长。


class MaplePatrolTask(MyBaseTask):  # 定义冒险岛小地图巡逻任务，继承项目基类。

    def __init__(self, *args, **kwargs):  # 构造函数，先初始化父类再设置任务元数据。
        super().__init__(*args, **kwargs)  # 必须先调用父类构造。
        self.name = "Maple Patrol"  # 任务显示名称。
        self.description = "Minimap-only patrol: locate the minimap by template, track the character with the yellow dot color, hold the direction key to walk back and forth between the left/right percent boundaries; red vertical lines mark the boundaries in the live vision; attacking will be added later."  # 任务描述：仅依赖小地图巡逻，模板定位小地图后按黄点色号跟踪角色，按住方向键在左右占比边界间往返，边界在实时画面用红色竖线标出，打怪后续补充。
        self.icon = FluentIcon.PLAY  # 任务图标。
        self.default_config.update({  # 用户可在 GUI 编辑的配置项。
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
            "Frame Interval": 0.05,  # 每帧处理之间的最小间隔秒数，控制检测节奏。
        })
        self.config_description.update({  # 各配置项的帮助文本。
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
            "Frame Interval": "Minimum seconds between processed frames. 每帧处理之间的最小间隔秒数。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
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

    def run(self):  # 任务运行入口。
        minimap_name = str(self.config.get("Minimap Feature") or '').strip()  # 读取小地图标注分类名。
        frame = self.wait_frame()  # 先取到一帧画面，让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面时无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示取不到画面。
            return  # 直接结束任务。
        if not minimap_name or not self.feature_ready(minimap_name):  # 小地图标注缺失时无法运行。
            self.log_warning(f"Minimap template not ready, please annotate '{minimap_name}' in the Template tab. 小地图模板未就绪，请先在模板页标注“{minimap_name}”。")  # 提示用户去模板页标注。
            return  # 标注不可用时直接结束任务。
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
        direction = 1  # 巡逻方向：1=向右、-1=向左，默认先向右走。
        held_key = None  # 当前持续按住的方向键，换向/退出时必须松开它。
        last_dot = None  # 上一帧黄点位置（画面坐标），用于跟踪连续性。
        anchor_x = None  # 卡住判定的位置锚点，黄点移动超过阈值时重置。
        anchor_time = time.time()  # 位置锚点上次更新时间。
        dot_missing_since = None  # 黄点开始丢失的时间戳，None 表示当前能检测到。
        try:  # 包裹主循环，退出时兜底松开持续按住的方向键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时短暂等待后重试。
                    self.sleep(frame_interval)  # 等待一个帧间隔。
                    continue  # 进入下一帧处理。
                minimap = self.find_minimap(minimap_name, frame, minimap_threshold)  # 模板匹配定位小地图框。
                rect = self.map_rect(minimap) if minimap is not None else None  # 计算实际地图区域（画面坐标）。
                dot = self.detect_dot(frame, rect, last_dot, hue_min, hue_max, dot_min_pixels) if rect is not None else None  # 在地图区域内检测角色黄点。
                if minimap is None:  # 找不到小地图时停止移动等待重新出现。
                    if held_key is not None:  # 有按住的方向键。
                        self.send_key_up(held_key)  # 松开方向键停止移动。
                        held_key = None  # 清空按住状态。
                    last_dot, anchor_x, dot_missing_since = None, None, None  # 清空全部跟踪状态。
                    self.info_set("Status", "Minimap not found")  # 在 GUI 显示未找到小地图。
                elif dot is None:  # 小地图在但黄点检测不到：短暂保持移动，长时间丢失则松键等待。
                    last_dot, anchor_x = None, None  # 黄点丢失时跟踪连续性中断。
                    if dot_missing_since is None:  # 刚开始丢失。
                        dot_missing_since = time.time()  # 开始丢失计时。
                    elif held_key is not None and time.time() - dot_missing_since >= DOT_MISSING_HOLD_SECONDS:  # 丢失超时仍按住方向键有盲走风险。
                        self.send_key_up(held_key)  # 松开方向键等黄点恢复。
                        held_key = None  # 清空按住状态。
                    self.info_set("Status", "Dot not found")  # 在 GUI 显示未找到黄点。
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
                    if stuck_seconds > 0 and held_key is not None and time.time() - anchor_time >= stuck_seconds:  # 按住方向键但位置长期不动视为卡住。
                        direction = self.recover_stuck(held_key, direction, resume_wait)  # 反向短移脱困并翻转巡逻方向。
                        held_key = None  # 恢复过程已松键，随后重新按住新方向键。
                        anchor_x, anchor_time = None, time.time()  # 脱困后重置卡住锚点。
                    want_key = MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY  # 当前巡逻方向需要的方向键。
                    if held_key != want_key:  # 换向时先松开旧键再按新键，避免两键同时按住。
                        if held_key is not None:  # 有旧键按住。
                            self.send_key_up(held_key)  # 松开旧方向键。
                        self.send_key_down(want_key)  # 持续按住新方向键保持移动。
                        held_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Patrolling right" if direction == 1 else "Patrolling left")  # 在 GUI 显示当前巡逻方向。
                    last_dot = dot  # 记录本帧黄点位置供下一帧跟踪。
                og.my_app.update_vision(self.draw_overlay(frame, minimap, rect, dot, left_pct, right_pct))  # 把带标注画面推送给 UI 实时展示。
                self.sleep(frame_interval)  # 等待一个帧间隔后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if held_key is not None:  # 有按住未松的方向键。
                self.send_key_up(held_key)  # 松开它。

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

    def detect_dot(self, frame, rect, last_dot, hue_min, hue_max, min_pixels):  # 在地图区域内检测角色黄点，返回画面坐标 (x, y, 面积) 或 None。
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
        if last_dot is not None:  # 有上一帧位置时优先采信离上一帧最近的候选块，保证跟踪连续性。
            max_jump = ((w ** 2 + h ** 2) ** 0.5) * DOT_MAX_JUMP_PERCENT / 100  # 跳变阈值：地图区域对角线的 30%。
            nearest = min(candidates, key=lambda c: (c[0] - last_dot[0]) ** 2 + (c[1] - last_dot[1]) ** 2)  # 离上一帧最近的候选块。
            if (nearest[0] - last_dot[0]) ** 2 + (nearest[1] - last_dot[1]) ** 2 <= max_jump ** 2:  # 距离在阈值内视为正常移动。
                return nearest  # 采信最近块。
        return max(candidates, key=lambda c: c[2])  # 首帧或发生瞬移/严重干扰时，采信面积最大块。

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

    def draw_overlay(self, frame, minimap, rect, dot, left_pct, right_pct):  # 在一帧画面上绘制全部标注，返回新画面。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
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
