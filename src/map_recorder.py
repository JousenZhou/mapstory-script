# 路线录制工具算法层 + 后台录制线程：把"边走边看实时小地图"的导航过程反向产出为
# MapleRouteTask 可直接消费的地图资产（全局拼接底图 map.png + 彩色指令路线图 routeN.png）。
#
# 核心思想（与消费端 src/map_route.py / MapleRouteTask 完全对称，保证录出来的图一定能被定位）：
#   1. 模板匹配（Minimap Feature）在游戏画面里定位小地图框 → 按 Map Rect 百分比裁出实时小地图内容 live；
#   2. HSV 取色在小地图内容里检测角色黄点 dot（画面坐标）；
#   3. 复用消费端同一个 locate_on_global_map(map.png, live, last_cam) 得到相机左上角 cam —— 由于用的是同一个
#      SQDIFF 函数，录制端与回放端对"cam 的定义"天然一致；
#   4. 把 live 原样贴回全局画布的 cam 处，实现增量拼接（贴的是 1:1 实时小地图像素，故 map.png 与实时小地图同比例）；
#   5. 角色全局坐标 loc = cam + (dot - live左上角)，与消费端 loc_global 公式一模一样；
#      以"当前指令色"把 loc 用线段连笔描到与画布同尺寸的路线图上（通道顺序与 route_canvas 手绘完全一致，
#      经 map_store 的 PNG 无损往返后仍能被 build_color_map/nearest_color 精确反查）。
#   6. 停止时把画布按"已覆盖区域"的最小外接矩形裁剪，map.png 与 routeN.png 同步裁，尺寸一致可直接落盘。
#
# 上色方式两套（由 meta 的 Record Use Keys 切换）：
#   * 键盘捕获（默认）：被动监听真实按键，每拍把"按住的方向键 + 有效窗内的点按技能键"推导成
#     "左右 上下 动作"三元指令，反查色表落色；色表没有的组合自动分配新色并在结束时写回该地图 meta，
#     同时把 (时刻, 全局坐标, 指令) 存入 routeN.keys.json sidecar，以后改粗细/改色表可离线重绘而不用重走地图。
#   * 手点画笔（兼容旧行为）：用 GUI 当前画笔色，换动作需要停下来点色块。
#
# 分层：detect_yellow_dot / compute_map_rect / parse_rect_percent / command_from_keys / render_route_from_events /
#       MapStitcher 均为纯计算，可单测；
#       MapRecorder 是后台守护线程，仅负责按节拍截图 + 调用框框模板匹配 + 驱动 MapStitcher，不碰 Qt。
import threading  # 后台录制守护线程
import time  # 取帧节拍与限频计时

import cv2  # 线段/圆点落笔与 HSV 取色
import numpy as np  # 画布矩阵运算

from ok import Logger  # 线程日志器（不依赖执行器，与测谎服务同源）

from src import route_palette  # 指令 <-> 色表反查与自动扩色
from src.key_capture import KeyCapture  # 全局按键捕获（只读真实键盘，不拦截）
from src.keynames import parse_key_list  # 键位配置串解析（支持逗号分隔多键同义）
from src.map_route import locate_on_global_map  # 复用消费端同款全局定位函数，确保录/放几何一致

logger = Logger.get_logger(__name__)  # 模块日志器

RECORD_FPS = 15  # 录制取帧帧率（比运行时低即可，移动慢；降低截图/匹配压力）
RECORD_INTERVAL = 1.0 / RECORD_FPS  # 每帧节拍秒数
DOT_SAT_MIN = 200  # 黄点 HSV 饱和度下限：角色标记是纯黄（S≈255），小地图地形的橙黄/暗黄通常低于此值
DOT_VAL_MIN = 180  # 黄点 HSV 亮度下限：同上，把偏暗的地形像素挡在掩码外
DOT_SAT_MIN_LOOSE = 150  # 降级档饱和度下限（角色标记被半透明遮罩压暗时使用）
DOT_VAL_MIN_LOOSE = 150  # 降级档亮度下限
DOT_TIERS = ((DOT_SAT_MIN, DOT_VAL_MIN), (DOT_SAT_MIN_LOOSE, DOT_VAL_MIN_LOOSE))  # 由严到宽依次尝试的阈值档
DOT_PURITY_WEIGHT = 0.5  # 候选块打分中颜色纯度的权重：score = 面积 * (权重 + 纯度)，纯度越高分越高
LOCATE_SCORE_MAX = 0.6  # 录制期定位可接受的最大 SQDIFF 得分（比运行期宽松，因底图尚不完整）
STATIC_DOT_TICKS = 10  # 超过这么多拍落点都在同一个像素，就判定黄点没锁到角色（或角色没走），结束时告警
DEFAULT_TAP_WINDOW = 0.25  # 点按动作有效窗默认值（秒）

# 键位角色与 meta 配置键的对应关系（与 MAP_META_DEFAULTS 的 Record * Keys 一一对应）。
RECORD_KEY_META_KEYS = {
    'left': 'Record Left Keys',
    'right': 'Record Right Keys',
    'up': 'Record Up Keys',
    'down': 'Record Down Keys',
    'jump': 'Record Jump Keys',
    'teleport': 'Record Teleport Keys',
}


def bindings_from_meta(meta):  # 从地图 meta 解析键位绑定 {角色: [规范键名...]}，未配置的为空列表。
    meta = meta or {}  # 允许传 None
    return {role: parse_key_list(meta.get(meta_key)) for role, meta_key in RECORD_KEY_META_KEYS.items()}


def tap_window_from_meta(meta):  # 从 meta 读点按动作有效窗秒数，非法回退默认值。
    try:
        value = float(meta.get('Record Action Tap Window'))
    except (TypeError, ValueError):  # 未配置或非数字
        return DEFAULT_TAP_WINDOW
    return value if value > 0 else DEFAULT_TAP_WINDOW  # 非正数视为无效


def _pick_held_role(held, role_keys):  # 从按住的键里取指令分量：后按下的优先（与游戏后按优先一致），无命中返回 none。
    for name in reversed(list(held or [])):  # held 按按下时刻升序，倒序遍历即"后按优先"
        for role, keys in role_keys:  # 逐个角色比对
            if name in (keys or ()):  # 该角色绑了这个键
                return role
    return 'none'


def _pick_tapped_action(taps, role_keys, window, now):  # 取动作分量：按"最近点按落在有效窗内"判定，优先级 teleport > jump（与 apply_cmd 一致）。
    taps = taps or {}  # {键名: 按下时刻}
    for role, keys in role_keys:  # teleport 在前，故优先
        latest = None  # 该角色绑定键中最近一次按下时刻
        for name in (keys or ()):  # 多个同义键取最新的那个
            ts = taps.get(name)
            if ts is not None and (latest is None or ts > latest):
                latest = ts
        if latest is not None and (now - latest) <= window:  # 落在有效窗内
            return role
    return 'none'


def command_from_keys(held, taps, bindings, window, now=None):  # 实时按键 -> (左右, 上下, 动作)；全 none（站着不动）返回 None。
    bindings = bindings or {}  # {角色: [键名...]}
    ref = time.time() if now is None else float(now)  # 参考时刻
    move_x = _pick_held_role(held, [('left', bindings.get('left')), ('right', bindings.get('right'))])
    move_y = _pick_held_role(held, [('up', bindings.get('up')), ('down', bindings.get('down'))])
    action = _pick_tapped_action(taps, [('teleport', bindings.get('teleport')), ('jump', bindings.get('jump'))], window, ref)
    if move_x == 'none' and move_y == 'none' and action == 'none':  # 没有任何指令
        return None  # 调用方按 gap 处理（不描线，重绘时在此断线）
    return (move_x, move_y, action)  # 三元指令


def render_route_from_events(events, w, h, thickness, color_lookup):  # 事件流离线重绘路线图，返回 (图像, 告警列表)。
    # events 每行为 [时刻, x, y, 指令或 None]；cmd 为 None 是 gap 断点（不连线，从本点重新起笔）。
    canvas = np.zeros((int(h), int(w), 3), dtype=np.uint8)  # 黑底：无指令
    warnings = []  # 跳过的事件说明
    prev = None  # 上一点，None 表示折线已断开
    width = max(1, int(thickness))  # 线宽
    radius = max(0, width // 2)  # 起笔圆点半径
    for index, item in enumerate(events or []):  # 逐事件
        if item is None or len(item) < 4:  # 行结构不合法
            warnings.append(f'event #{index} malformed')  # 记录
            continue
        x, y, cmd = int(item[1]), int(item[2]), item[3]  # 坐标与指令
        if not (0 <= x < w and 0 <= y < h):  # 越界（事件流与画布尺寸不匹配）
            warnings.append(f'event #{index} out of bounds')  # 记录
            continue
        if cmd is None:  # gap：断开折线，该点本身不描色
            prev = None
            continue
        rgb = color_lookup.get(route_palette.command_text(*route_palette.split_command(cmd)))  # 指令反查色
        if rgb is None:  # 色表里没有这个指令（如恢复了默认色表）
            warnings.append(f'event #{index} unknown command {cmd}')  # 提示但不断线
            continue
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))  # 通道序与录制端一致
        if prev is not None:  # 连笔成线段
            cv2.line(canvas, prev, (x, y), color, width, cv2.LINE_8)
        else:  # 新段起点画实心点，保证单拍段也留下像素
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_8)
        prev = (x, y)  # 推进上一点
    return canvas, warnings


def parse_rect_percent(text):  # 解析 Map Rect 文本 "x,y,w,h"（相对小地图框的百分比）为四元浮点组，非法抛 ValueError，空返回 None。
    raw = str(text or '').strip()  # 去空白
    if not raw:  # 空串表示整块小地图框
        return None  # 供调用方按"整个框"处理
    parts = raw.replace('%', '').replace(';', ',').split(',')  # 去百分号统一分隔符后切分
    if len(parts) != 4:  # 必须恰好四个数
        raise ValueError("Map Rect needs 4 values")  # 数量不对
    values = [float(p.strip()) for p in parts]  # 逐个转浮点，非数字自动抛 ValueError
    if any(v < 0 or v > 100 for v in values):  # 百分比越界
        raise ValueError("Map Rect value out of range")  # 非法
    return tuple(values)  # 返回 (x, y, w, h) 百分比


def compute_map_rect(box_x, box_y, box_w, box_h, map_rect_text):  # 由小地图框与 Map Rect 百分比算出实际地图区域（画面坐标 (rx,ry,rw,rh)），留空取整个框。
    parsed = parse_rect_percent(map_rect_text)  # 解析百分比
    if parsed is None:  # 未配置区域：整框即地图
        return box_x, box_y, box_w, box_h  # 直接返回小地图框
    px, py, pw, ph = parsed  # 四个百分比
    rx = box_x + int(box_w * px / 100)  # 区域左上角画面 x
    ry = box_y + int(box_h * py / 100)  # 区域左上角画面 y
    rw = max(1, int(box_w * pw / 100))  # 区域宽（至少 1 像素）
    rh = max(1, int(box_h * ph / 100))  # 区域高（至少 1 像素）
    return rx, ry, rw, rh  # 返回实际地图区域


def _best_yellow_dot(hsv, hue_min, hue_max, sat_min, val_min, min_pixels, offset):  # 在给定阈值下取「最像角色标记」的黄色块，返回 (得分, x, y, 面积) 或 None。
    lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)  # 取色下限
    upper = np.array([hue_max, 255, 255], dtype=np.uint8)  # 取色上限
    mask = cv2.inRange(hsv, lower, upper)  # 黄点掩码（不做开运算：角色标记是十字/菱形细块，3x3 开运算会整块抹掉）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取外轮廓
    best = None  # 本档最优候选
    for contour in contours:  # 逐个轮廓求面积与颜色纯度
        area = cv2.contourArea(contour)  # 轮廓面积
        if area < min_pixels:  # 太小不可能是角色黄点
            continue  # 丢弃
        moments = cv2.moments(contour)  # 几何矩求重心
        if moments["m00"] == 0:  # 面积矩为 0 无法求重心
            continue  # 丢弃
        inside = np.zeros(mask.shape, dtype=np.uint8)  # 轮廓填充图
        cv2.drawContours(inside, [contour], -1, 255, -1)  # 填充以取内部像素
        picked = inside > 0  # 轮廓内掩码
        purity = (float(hsv[:, :, 1][picked].mean()) + float(hsv[:, :, 2][picked].mean())) / 510.0  # 饱和度+亮度均值归一到 0~1
        score = area * (DOT_PURITY_WEIGHT + purity)  # 面积为主、颜色越接近纯黄加权越高（地形大块又暗又浊，抢不过标记）
        cx = int(moments["m10"] / moments["m00"]) + offset[0]  # 重心换算画面坐标
        cy = int(moments["m01"] / moments["m00"]) + offset[1]  # 重心换算画面坐标
        if best is None or score > best[0]:  # 取分数最高者
            best = (score, cx, cy, area)  # 记录候选
    return best  # 无达标块则为 None


def detect_yellow_dot_detail(frame, rect, hue_min, hue_max, min_pixels, sat_min=None, val_min=None):  # 同 detect_yellow_dot，额外返回实际生效的阈值档，供「试定位」回显。
    x, y, w, h = rect  # 地图区域
    if w <= 0 or h <= 0:  # 非法区域
        return None  # 无法检测
    roi = frame[y:y + h, x:x + w]  # 裁剪地图区域画面
    if roi.size == 0:  # 区域越界为空
        return None  # 无法检测
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转 HSV 按色相取色
    tiers = list(DOT_TIERS)  # 默认由严到宽
    if sat_min is not None or val_min is not None:  # 用户在参数里指定过阈值：把它放最前优先尝试
        custom = (int(sat_min if sat_min is not None else DOT_SAT_MIN), int(val_min if val_min is not None else DOT_VAL_MIN))  # 自定义档
        tiers = [custom] + [tier for tier in tiers if tier != custom]  # 自定义档失败仍降级兜底
    for sat, val in tiers:  # 逐档尝试，命中即返回
        best = _best_yellow_dot(hsv, hue_min, hue_max, sat, val, max(1, int(min_pixels)), (x, y))  # 本档最优块
        if best is not None:  # 命中
            return best[1], best[2], best[3], (sat, val)  # (画面 x, 画面 y, 面积, 生效阈值档)
    return None  # 各档都没黄点


def detect_yellow_dot(frame, rect, hue_min, hue_max, min_pixels, sat_min=None, val_min=None):  # 在地图区域内检测角色黄点，返回画面坐标 (x, y, 面积) 或 None。
    detail = detect_yellow_dot_detail(frame, rect, hue_min, hue_max, min_pixels, sat_min, val_min)  # 走同一套判据
    return None if detail is None else detail[:3]  # 隐藏阈值档，保持三元组兼容


class MapStitcher:  # 增量拼接器：维护全局底图画布 + 同尺寸路线图，按定位得到的 cam 贴小地图、描指令轨迹，停止时按覆盖区外接矩形裁成最终资产。
    # 坐标约定与消费端严格对齐：map 画布存 BGR 实时小地图像素；route 画布以 ndarray 通道顺序写指令色三元组，
    # 经 map_store 的 PNG 无损往返后 pixel_key 仍能反查 build_color_map 的键（与 route_canvas 手绘同一条通道约定）。

    def __init__(self, canvas_w, canvas_h, crop_w, crop_h, match_radius=128):  # 画布尺寸（预分配上限）与实时小地图裁剪尺寸、定位局部搜索半径。
        self.map = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # 全局底图画布（BGR，黑=尚未覆盖）
        self.route = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # 指令路线画布，与底图同尺寸
        self.covered = np.zeros((canvas_h, canvas_w), dtype=bool)  # 已覆盖像素掩码，供停止时裁剪到内容外接矩形
        self.crop_w = int(crop_w)  # 单帧小地图内容宽
        self.crop_h = int(crop_h)  # 单帧小地图内容高
        self.match_radius = max(8, int(match_radius))  # 定位局部搜索半径
        self.cam = None  # 上一帧相机左上角（画布坐标），None 表示尚未种下第一块
        self.prev_point = None  # 上一落点（画布坐标），用于把离散落点连成线段
        self.crop_offset = (0, 0)  # finalize 时的裁剪起点（画布坐标 -> 最终图坐标的平移量）
        self.pasted = 0  # 已贴帧计数（诊断用）
        self.located = 0  # 成功定位并落点计数（诊断用）
        self.lost = 0  # 定位失败/丢帧计数（诊断用）

    def locate_cam(self, live):  # 用实时小地图在已拼接底图上定位相机左上角；首帧直接种在画布中心，返回 (cam(x,y) 或 None, score)。
        if live is None or live.size == 0:  # 无裁剪内容
            return None, 1.0  # 定位失败
        if live.shape[0] != self.crop_h or live.shape[1] != self.crop_w:  # 小地图框尺寸与首帧不符（窗口被缩放），拒绝避免错位拼接
            return None, 1.0  # 视为本帧不可用
        if self.cam is None:  # 首帧：把 live 种在画布中心，作为坐标系原点，之后向四周生长
            cx = max(0, (self.map.shape[1] - self.crop_w) // 2)  # 中心 x
            cy = max(0, (self.map.shape[0] - self.crop_h) // 2)  # 中心 y
            return (cx, cy), 0.0  # 首帧得分视为完美（自匹配）
        cam, score = locate_on_global_map(self.map, live, self.cam, radius=self.match_radius, score_max=LOCATE_SCORE_MAX)  # 复用消费端同款定位
        if cam is None or score > LOCATE_SCORE_MAX:  # 底图不完整或匹配失败：本帧丢弃，保留旧 cam 供下帧重试
            return None, score  # 未定位
        return cam, score  # 定位成功

    def place_and_paste(self, cam, live):  # 把 live 原样贴回底图 cam 处（1:1 同比例），并标记覆盖区；cam 会被夹到画布边界内。
        if cam is None or live is None:  # 无定位结果
            return None  # 不贴
        x, y = cam  # 相机左上角
        x = max(0, min(self.map.shape[1] - self.crop_w, int(x)))  # 横向夹到界内
        y = max(0, min(self.map.shape[0] - self.crop_h, int(y)))  # 纵向夹到界内
        self.map[y:y + self.crop_h, x:x + self.crop_w] = live  # 贴小地图内容（通道顺序原样保留）
        self.covered[y:y + self.crop_h, x:x + self.crop_w] = True  # 标记已覆盖
        self.cam = (x, y)  # 记录相机供下帧局部搜索
        self.pasted += 1  # 计数
        return (x, y)  # 返回实际贴合坐标

    def trace(self, loc, color, thickness=2):  # 以指令色 color=(r,g,b)（ndarray 通道顺序）在路线图把全局落点 loc 连笔描下，与上一落点用线段相连。
        if loc is None:  # 无落点
            return  # 不描
        x, y = int(loc[0]), int(loc[1])  # 全局落点
        if not (0 <= x < self.route.shape[1] and 0 <= y < self.route.shape[0]):  # 越界（画出预分配画布）
            self.lost += 1  # 计入丢失
            return  # 跳过该点，避免越界写入异常
        col = (int(color[0]), int(color[1]), int(color[2]))  # 指令色三元组（直接写通道，与 route_canvas 同约定）
        radius = max(0, thickness // 2)  # 落笔半径
        if self.prev_point is not None:  # 已有上一点：连线段保证快速移动不断点
            cv2.line(self.route, self.prev_point, (x, y), col, max(1, thickness), cv2.LINE_8)  # 粗线段
        else:  # 首点画圆点
            cv2.circle(self.route, (x, y), radius, col, -1, cv2.LINE_8)  # 实心点
        self.prev_point = (x, y)  # 更新上一点
        self.located += 1  # 成功落点计数

    def break_trace(self):  # 断开连笔（站着不动或定位失败时调用），避免下一拍跨空隙拉出一条直线。
        self.prev_point = None  # 下一拍重新以实心点起笔

    def finalize(self):  # 按已覆盖区最小外接矩形裁出最终 (map, route)，两图同步裁剪保证尺寸一致；无覆盖内容时返回 (None, None)。
        if not self.covered.any():  # 从没贴过任何帧
            return None, None  # 无资产
        ys, xs = np.where(self.covered)  # 覆盖区所有行列坐标
        y0, y1 = int(ys.min()), int(ys.max()) + 1  # 纵向外接（含右端点）
        x0, x1 = int(xs.min()), int(xs.max()) + 1  # 横向外接
        self.crop_offset = (x0, y0)  # 记下裁剪起点，供事件流坐标平移
        cropped_map = self.map[y0:y1, x0:x1].copy()  # 裁底图
        cropped_route = self.route[y0:y1, x0:x1].copy()  # 裁路线图（同矩形，尺寸必然一致）
        return cropped_map, cropped_route  # 返回最终资产


class MapRecorder:  # 后台录制线程：按节拍截图 + 框架模板匹配定位小地图 + 驱动 MapStitcher 拼接描线，停止后落盘地图资产。
    # 线程模型完全对齐 LieDetectorService：截图 device_manager.capture_method.get_frame（内部加锁、线程安全），
    # 模板匹配 executor.feature_set.find_feature（自带锁），只读当前选择色通过回调原子获取，不触碰任何 Qt 对象。

    def __init__(self, map_name, route_file, meta, color_getter, on_finished=None,
                 bindings=None, use_keys=True, auto_goal=True, tap_window=None):  # 目标地图名、目标路线图文件名、地图 meta（定位与色表参数）、当前指令色取用回调、完成回调、键位绑定、是否键盘捕获、是否自动补终点、点按有效窗。
        self.map_name = map_name  # 录制写入的地图目录名
        self.route_file = route_file  # 录制写入的路线图文件名
        self.meta = dict(meta or {})  # 定位参数快照（Minimap Feature/Threshold/Map Rect/色相等）
        self._color_getter = color_getter  # 无参回调，返回当前指令色三元组 (r,g,b)（仅手点画笔模式用）
        self._on_finished = on_finished  # 可选：录制结束时以 (map, route) 回调（在工作线程调用，勿直接碰 Qt）
        self.use_keys = bool(use_keys)  # 键盘捕获开关：关闭时完全退回旧行为（用画笔色、不写 sidecar）
        self.auto_goal = bool(auto_goal)  # 停止时在末点自动补 goal 色
        self.bindings = dict(bindings) if bindings is not None else bindings_from_meta(self.meta)  # {角色: [规范键名...]}
        self.tap_window = float(tap_window) if tap_window else tap_window_from_meta(self.meta)  # 点按动作有效窗秒数
        self.thickness = max(1, int(self.meta.get('Record Trace Thickness') or 2))  # 描线粗细（同样写入 sidecar 供重绘）
        self.capture = KeyCapture() if self.use_keys else None  # 按键捕获器（工作线程内启停）
        self.events = []  # 事件流 [[时刻s, x, y, 指令串或 None], ...]，坐标为画布坐标，落盘前平移至裁剪坐标
        self.current_cmd = None  # 当前拍推导出的指令（供 GUI 展示）
        self.current_rgb = None  # 当前拍实际落下的色三元组（供 GUI 展示，与 current_cmd 同拍更新）
        self.last_loc = None  # 最近一次有效落点，定位失败时用它标记 gap
        self.trace_box = None  # 已描线落点的外接矩形 [min_x, min_y, max_x, max_y]，用于判断全程是否动过
        self.trace_ticks = 0  # 成功描线的拍数（与 trace_box 一起构成「落点没动」的判据）
        self._t0 = None  # 录制起始时刻，用于算事件相对时间
        self.color_additions = []  # 本局自动新增的色 [(表名, "r,g,b", 指令串), ...]
        self.errors = []  # 录制过程中的非致命问题（色池耗尽等），供 GUI 展示
        self.additions_saved = False  # 自动新增色是否已写回 meta
        self._thread = None  # 工作线程句柄
        self._stop_event = threading.Event()  # 停止请求标志
        self.finished = False  # 线程是否已结束（供 GUI 轮询）
        self.saved = False  # 是否已成功落盘
        self.error = None  # 结束原因/错误信息（供 GUI 展示）
        self.status = 'idle'  # 当前状态文本（idle/locating/lost/saving）
        self.stitcher = None  # 拼接器实例（首帧拿到小地图尺寸后创建）
        self.stats = {'pasted': 0, 'located': 0, 'lost': 0}  # 诊断统计快照
        # 色表内存副本：录制中新增指令色写进这里，使同一组合后续拍复用同色，结束时合并回 meta。
        self._main_map = dict(self.meta.get('Color Code') or {})
        self._ud_map = dict(self.meta.get('Color Code Up Down') or {})
        self._used = route_palette.used_colors(self._main_map, self._ud_map)

    def start(self):  # 启动录制线程；已启动则忽略。
        if self._thread is not None:  # 已启动
            return  # 幂等
        self._thread = threading.Thread(target=self._run, name='MapRecorder', daemon=True)  # 守护线程随进程退出
        self._thread.start()  # 启动
        logger.info(f'Map recorder started for map={self.map_name} route={self.route_file}. 路线录制已启动：地图 {self.map_name}，路线 {self.route_file}。')  # 记录

    def stop(self):  # 请求停止录制（非阻塞，线程在下一拍检查到停止后收尾落盘）。
        self._stop_event.set()  # 置停止标志

    def wait(self, timeout=5.0):  # 等待录制线程结束（供 GUI 在关闭页签/切换地图前确保已落盘），返回是否在超时内结束。
        if self._thread is None:  # 未启动
            return True  # 视为已结束
        return self._thread.join(timeout) is None  # join 返回 None 表示线程已结束

    # ------------------------------------------------------------------ 工作线程

    def _capture(self):  # 采集一帧游戏画面，失败返回 None（与测谎服务同款取帧路径）。
        from ok import og  # 延迟导入全局对象，避免模块级依赖执行器
        device_manager = getattr(og, 'device_manager', None)  # 取设备管理器
        method = getattr(device_manager, 'capture_method', None) if device_manager is not None else None  # 取截图方法
        if method is None:  # 未选择窗口等
            return None  # 无画面
        try:  # 截图异常不能让线程崩溃
            return method.get_frame()  # 取一帧（内部加锁，线程安全）
        except Exception:  # 截图失败
            return None  # 按无画面处理

    def _match_minimap(self, frame, feature_name, threshold):  # 全屏模板匹配小地图框，返回置信度最高框或 None。
        from ok import og  # 延迟导入取执行器特征集
        executor = getattr(og, 'executor', None)  # 取执行器
        feature_set = getattr(executor, 'feature_set', None) if executor is not None else None  # 取特征集
        if feature_set is None:  # 特征集不可用
            return None  # 无法匹配
        try:  # 标注缺失会抛 ValueError
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)  # variance=1 全屏搜索，灰度匹配与任务一致
        except ValueError:  # 未标注该分类
            return None  # 未匹配
        except Exception as e:  # 匹配异常
            logger.warning(f'Map recorder minimap match failed: {e}. 录制小地图匹配失败：{e}。')  # 记录
            return None  # 未匹配
        if not boxes:  # 无达标框
            return None  # 未匹配
        return max(boxes, key=lambda b: getattr(b, 'confidence', 0))  # 取最高分框

    def _run(self):  # 线程主循环：截图→定位小地图→裁 live→检测黄点→定位 cam→贴图画布→算全局坐标→描路线，直到停止后落盘。
        feature_name = str(self.meta.get('Minimap Feature') or '').strip()  # 小地图模板名
        minimap_threshold = float(self.meta.get('Minimap Threshold') or 0.8)  # 小地图匹配阈值
        map_rect_text = str(self.meta.get('Map Rect') or '').strip()  # 实际地图区域百分比
        hue_min = int(self.meta.get('Dot Hue Min') or 18)  # 黄点色相下限
        hue_max = int(self.meta.get('Dot Hue Max') or 38)  # 黄点色相上限
        dot_min = max(1, int(self.meta.get('Dot Min Pixels') or 4))  # 黄点最小面积
        dot_sat = int(self.meta.get('Dot Sat Min') or DOT_SAT_MIN)  # 黄点饱和度下限（调低可捞出偏暗的角色标记）
        dot_val = int(self.meta.get('Dot Val Min') or DOT_VAL_MIN)  # 黄点亮度下限
        canvas_w, canvas_h = self._parse_canvas_size()  # 预分配画布尺寸
        if self.capture is not None:  # 键盘捕获模式：钩子在本线程装，退出时必停
            if not self.capture.start():  # 启动失败（无 pynput / 权限不足）
                self.errors.append(f'key capture unavailable: {self.capture.error}')  # 记录原因
                self.status = 'key hook failed'  # GUI 可见
        try:  # 包裹主循环，任何异常都要收尾落盘并置 finished
            while not self._stop_event.is_set():  # 录制循环直到停止
                frame = self._capture()  # 采集一帧
                if frame is None:  # 无画面（窗口未就绪）
                    self.status = 'no frame'  # 记录状态
                    self._sleep(RECORD_INTERVAL)  # 短等重试
                    continue  # 下一拍
                box = self._match_minimap(frame, feature_name, minimap_threshold)  # 定位小地图框
                if box is None:  # 没找到小地图
                    self.status = 'minimap lost'  # 记录
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                rx, ry, rw, rh = compute_map_rect(box.x, box.y, box.width, box.height, map_rect_text)  # 实际地图区域
                rx = max(0, rx); ry = max(0, ry)  # 夹到画面内
                rw = min(rw, frame.shape[1] - rx); rh = min(rh, frame.shape[0] - ry)  # 防越界
                if rw <= 0 or rh <= 0:  # 区域退化
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                live = frame[ry:ry + rh, rx:rx + rw]  # 实时小地图内容裁剪
                if self.stitcher is None:  # 首帧确定裁剪尺寸后建画布
                    self.stitcher = MapStitcher(canvas_w, canvas_h, rw, rh)  # 创建拼接器
                dot = detect_yellow_dot(frame, (rx, ry, rw, rh), hue_min, hue_max, dot_min, dot_sat, dot_val)  # 检测角色黄点（画面坐标）
                cam, score = self.stitcher.locate_cam(live)  # 定位相机左上角
                if cam is None:  # 定位失败
                    self.lost_tick()  # 累计丢失
                    self.status = f'locate lost s={score:.2f}'  # 记录得分供排查
                    self._mark_gap()  # 断点：不让下一拍跨这段空隙拉直线
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                self.stitcher.place_and_paste(cam, live)  # 贴小地图到画布（增量拼接）
                if dot is not None:  # 有黄点才能算全局落点
                    loc = (cam[0] + (dot[0] - rx), cam[1] + (dot[1] - ry))  # 与消费端一模一样的 loc_global 公式
                    self.status = f'@({loc[0]},{loc[1]}) s={score:.2f}'  # 记录当前全局坐标
                    self._trace_tick(loc)  # 按当前模式推导指令并落色描线
                else:  # 贴了图但没检测到黄点
                    self.status = f'dot lost s={score:.2f}'  # 记录
                    self._mark_gap()  # 无落点同样断线
                self._refresh_stats()  # 更新统计快照
                self._sleep(RECORD_INTERVAL)  # 按节拍下帧
            self._finish_and_save()  # 停止请求：裁剪落盘
        except Exception as e:  # 循环异常兜底
            self.error = str(e)  # 记录错误
            logger.warning(f'Map recorder loop error: {e}. 路线录制循环异常：{e}。')  # 记录
            try:  # 异常也尽量落盘已录内容
                self._finish_and_save()  # 收尾
            except Exception:  # 落盘再失败则忽略
                pass
        finally:  # 无论成败都停钩子并置 finished 供 GUI 轮询
            if self.capture is not None:  # 释放全局钩子
                self.capture.stop()
            self.finished = True  # 标记线程结束

    def _trace_tick(self, loc):  # 一有落点的拍：键盘模式推导指令并自动落色，否则沿用 GUI 画笔色（旧行为）。
        now = time.time()  # 本拍时刻
        if self._t0 is None:  # 首拍定下时间基准
            self._t0 = now
        self.last_loc = loc  # 记下落点，供后续定位失败时标 gap
        if self.capture is None:  # 手点画笔模式
            color = self._current_color()  # 取当前指令色
            if color is not None:  # 有有效色才描线（橡皮/无选择色时跳过）
                self.stitcher.trace(loc, color, self.thickness)  # 描指令轨迹
                self._note_trace_point(loc)  # 统计落点范围
            return
        held = self.capture.held_ordered()  # 当前按住的键（按下时刻升序）
        taps = self.capture.taps_snapshot()  # 各键最近一次按下时刻
        cmd = command_from_keys(held, taps, self.bindings, self.tap_window, now)  # 实时按键 -> 三元指令
        self.current_cmd = cmd  # 供 GUI 显示当前指令
        self.current_rgb = None  # 先清空落色，下面描线成功时才回填
        if cmd is None:  # 站着不动：不描线（否则原地糊成一片）
            self._mark_gap(loc)
            return
        cmd_text = route_palette.command_text(*cmd)  # 指令串（与色表值同格式）
        rgb = self._color_for_command(cmd_text)  # 反查色表，没有那么自动扩色
        if rgb is None:  # 色池耗尽
            self.errors.append(f'color pool exhausted: {cmd_text}')  # 记一次，本拍不描线
            self._mark_gap(loc)
            return
        self.stitcher.trace(loc, rgb, self.thickness)  # 落色描线
        self._note_trace_point(loc)  # 统计落点范围，结束时用于判断「全程没动」
        self.current_rgb = rgb  # 供 GUI 显示本拍落色
        self.events.append([round(now - self._t0, 3), int(loc[0]), int(loc[1]), cmd_text])  # 记入事件流

    def _note_trace_point(self, loc):  # 记下一个已描线落点的外接矩形与拍数（O(1)，不改线程外部状态）。
        self.trace_ticks += 1  # 描线拍计数
        x, y = int(loc[0]), int(loc[1])  # 画布坐标
        if self.trace_box is None:  # 首点直接初始化
            self.trace_box = [x, y, x, y]  # [min_x, min_y, max_x, max_y]
            return
        box = self.trace_box  # 已有范围
        box[0] = min(box[0], x); box[1] = min(box[1], y)  # 扩张左上
        box[2] = max(box[2], x); box[3] = max(box[3], y)  # 扩张右下

    def _trace_quality_warning(self, cropped_route):  # 落点全程不动 / 一像素都没描出时给出可执行原因（黄点锁错、角色未走），避免静默交出一张看似空白的图。
        if self.trace_ticks >= STATIC_DOT_TICKS and self.trace_box is not None \
                and self.trace_box[2] - self.trace_box[0] <= 1 and self.trace_box[3] - self.trace_box[1] <= 1:  # 拍数够多但落点从未离开一个像素
            self.errors.append('落点全程未移动：黄点可能锁到地形，或角色根本没走')  # GUI 状态行直接可见
            logger.warning(
                f'Map recorder traced {self.trace_ticks} ticks at the same point {(self.trace_box[0], self.trace_box[1])}. '
                f'落点全程停在同一个像素（只会留下一个点而不是线）：请用「试定位当前帧」核对黄点坐标，必要时调黄点色相/饱和/亮度下限。')  # 记录详细原因与处置
            return
        if int(np.count_nonzero(cropped_route.any(axis=2))) > 0:  # 描出了东西，不算空路线
            return
        self.errors.append(f'未描出任何路线（有效描线 {self.trace_ticks} 拍）：检查黄点检测与键位绑定')  # 记为非致命问题

    def _mark_gap(self, loc=None):  # 记下断点（无指令/无落点），重绘时据此断开折线。
        self.current_cmd = None  # GUI 显示为无指令
        if self.capture is None:  # 手点画笔模式不产出事件流
            return
        point = loc if loc is not None else self.last_loc  # 本拍无落点时沿用上一点坐标作断点
        if point is None or self.stitcher is None:  # 从未有过落点，无需断线
            return
        if self._t0 is None:
            self._t0 = time.time()
        self.events.append([round(time.time() - self._t0, 3), int(point[0]), int(point[1]), None])  # cmd=None 即 gap
        self.stitcher.break_trace()  # 断开连笔

    def _color_for_command(self, cmd_text):  # 指令串 -> 落色三元组；色表没定义时从未占用色池分配新色并登记写回 meta。
        rgb, addition = route_palette.resolve_color(cmd_text, self._main_map, self._ud_map, self._used)  # 先反查，未命中则扩色
        if rgb is not None and addition and route_palette.apply_addition(self._main_map, self._ud_map, addition):
            self._used.add(rgb)  # 新色归档为已占用，下一种组合不会重色
            self.color_additions.append(addition)  # 结束时合并回该地图 meta
            logger.info(f'Route palette auto-extended: {addition[1]} -> {addition[2]} in {addition[0]}. 色表自动新增：{addition[1]} = {addition[2]}（{addition[0]}）。')  # 记录
        return rgb

    def _finish_and_save(self):  # 停止后：裁剪最终资产并写入 map_store（覆盖底图与目标路线图），仅在有覆盖内容时落盘。
        if self.stitcher is None:  # 一帧都没录
            self.error = self.error or 'no frames captured'  # 记原因
            return  # 不落盘
        cropped_map, cropped_route = self.stitcher.finalize()  # 裁到内容外接矩形
        if cropped_map is None:  # 无覆盖内容
            self.error = self.error or 'nothing stitched'  # 记原因
            return  # 不落盘
        from src import map_store  # 延迟导入存取层，落盘资产
        self._trace_quality_warning(cropped_route)  # 先于补终点判质量：终点色也会占像素，补完就不算全黑了
        if self.capture is not None:  # 键盘捕获模式才补终点与写事件流（手点模式保持旧行为）
            self._append_goal_mark(cropped_route)  # 在末点补 goal 色（直改裁剪后的路线图）
        map_store.save_map_image(self.map_name, cropped_map)  # 覆盖写 map.png
        map_store.save_route_image(self.map_name, self.route_file, cropped_route)  # 写目标路线图
        if self.capture is not None:  # 事件流坐标平移后落 sidecar，供离线重绘
            self._save_events(map_store, cropped_route.shape)
        self._merge_color_additions(map_store)  # 本局自动新增的指令色写回该地图 meta
        self.saved = True  # 标记已落盘
        self._refresh_stats()  # 末次刷新统计
        self.status = f'saved {cropped_map.shape[1]}x{cropped_map.shape[0]}'  # 记录最终尺寸
        if self.errors:  # 非致命问题一并展示
            self.status = f'{self.status} warn={len(self.errors)}'
        logger.info(f'Map recorder saved map={self.map_name} size={cropped_map.shape[1]}x{cropped_map.shape[0]}. 路线录制已保存：地图 {self.map_name}，尺寸 {cropped_map.shape[1]}x{cropped_map.shape[0]}。')  # 记录
        if self._on_finished is not None:  # 回调（在工作线程，调用方勿直接操作 Qt）
            try:  # 回调异常不影响落盘结果
                self._on_finished(cropped_map, cropped_route)  # 通知
            except Exception as e:  # 回调异常
                logger.warning(f'Map recorder on_finished callback error: {e}. 录制完成回调异常：{e}。')  # 记录

    def _append_goal_mark(self, cropped_route):  # 在最后一个落点画 goal 色实心点，省去手动标终点；goal 色取色表定义，缺失则用默认黄。
        if not self.auto_goal:  # 用户关了自动补终点
            return
        point = self.stitcher.prev_point if self.stitcher.prev_point is not None else self.last_loc  # 末点（画布坐标）
        if point is None:  # 全程没描过线
            return
        x0, y0 = self.stitcher.crop_offset  # 裁剪起点
        x, y = int(point[0]) - int(x0), int(point[1]) - int(y0)  # 平移到裁剪坐标
        h, w = cropped_route.shape[:2]  # 最终图尺寸
        if not (0 <= x < w and 0 <= y < h):  # 末点落在裁剪区外（理论上不会出现）
            return
        goal_text = route_palette.command_text('none', 'none', 'goal')  # 终点指令串
        rgb = route_palette.build_reverse(self._main_map, self._ud_map).get(goal_text) or (255, 255, 0)  # 色表定义的终点色，没定义则默认黄
        cv2.circle(cropped_route, (x, y), max(1, self.thickness), (int(rgb[0]), int(rgb[1]), int(rgb[2])), -1, cv2.LINE_8)  # 实心点
        self.events.append([round(time.time() - (self._t0 or time.time()), 3), int(point[0]), int(point[1]), goal_text])  # 重绘时要重现终点（坐标保画布坐标，与其余事件同一次平移）

    def _save_events(self, map_store, shape):  # 把事件流的画布坐标平移到裁剪坐标后写入 routeN.keys.json；无事件则不写。
        if not self.events:  # 一个有效拍都没录
            return
        x0, y0 = self.stitcher.crop_offset  # 裁剪起点
        shifted = [[item[0], int(item[1]) - int(x0), int(item[2]) - int(y0), item[3]] for item in self.events]  # 只平移坐标，时刻与指令原样保留
        h, w = int(shape[0]), int(shape[1])  # 最终图尺寸（重绘时的画布尺寸）
        try:  # sidecar 写失败不应影响已落盘的图像
            map_store.save_route_events(self.map_name, self.route_file, shifted, canvas=(w, h), thickness=self.thickness)  # 落盘
        except Exception as e:  # 写入异常
            self.errors.append(f'events save failed: {e}')  # 记为非致命问题
            logger.warning(f'Map recorder events save failed: {e}. 事件流写入失败：{e}。')  # 记录

    def _merge_color_additions(self, map_store):  # 把本局自动新增的指令色合并写回该地图 meta（已有同色/同指令则不重复动）。
        if not self.color_additions:  # 没扩过色
            return
        try:  # meta 读写异常不能掩盖已落盘的图像
            meta = map_store.load_meta(self.map_name)  # 重读最新 meta（录制期间保存按钮已禁用，无并发写冲突）
            for table_key, rgb_key, cmd in self.color_additions:  # 逐条新增
                table = meta.get(table_key)  # 目标色表
                if not isinstance(table, dict):  # 表被弄坏了，重建
                    table = {}
                if rgb_key in table or cmd in table.values():  # 已被其他途径定义过，不覆盖用户数据
                    continue
                table[rgb_key] = cmd  # 登记新色
                meta[table_key] = table  # 放回 meta
            map_store.save_meta(self.map_name, meta)  # 写回
            self.additions_saved = True  # 标记已合并
        except Exception as e:  # 写回失败
            self.errors.append(f'color additions save failed: {e}')  # 记为非致命问题（重绘时会因缺色而告警）
            logger.warning(f'Map recorder color additions save failed: {e}. 色表新增写回失败：{e}。')  # 记录

    def _current_color(self):  # 通过回调取当前指令色三元组，非法/None 返回 None。
        try:  # 回调可能异常
            color = self._color_getter()  # 取色
        except Exception:  # 回调失败
            return None  # 无色
        if color is None:  # 未选色
            return None  # 跳过
        try:  # 规整为三元整数
            return (int(color[0]), int(color[1]), int(color[2]))  # 三元组
        except (TypeError, ValueError, IndexError):  # 非法色
            return None  # 跳过

    def _parse_canvas_size(self):  # 解析 meta 的预分配画布尺寸 "w,h"，非法回退 1600x1200。
        text = str(self.meta.get('Record Canvas Size') or '').strip()  # 尺寸文本
        try:  # 解析两数
            w, h = [int(p.strip()) for p in text.replace('×', ',').replace('x', ',').split(',')]  # 兼容 x/×分隔
        except (TypeError, ValueError):  # 非法
            w, h = 1600, 1200  # 默认值
        return max(200, w), max(200, h)  # 下限保护，至少能容一块小地图

    def lost_tick(self):  # 定位失败计数（写进 stitcher 若已建，否则忽略）。
        if self.stitcher is not None:  # 已有拼接器
            self.stitcher.lost += 1  # 累加

    def _refresh_stats(self):  # 把拼接器计数同步到对外 stats 快照，供 GUI 轮询展示。
        if self.stitcher is not None:  # 已创建
            self.stats = {'pasted': self.stitcher.pasted, 'located': self.stitcher.located, 'lost': self.stitcher.lost}  # 拷贝计数

    def _sleep(self, seconds):  # 分片短等，随时响应停止事件，避免停止后还卡在整拍。
        end = time.time() + seconds  # 结束时间
        while not self._stop_event.is_set():  # 未停止时循环
            remaining = end - time.time()  # 剩余
            if remaining <= 0:  # 已到点
                return  # 返回
            time.sleep(min(remaining, 0.05))  # 每片最多 0.05 秒，保证及时停止
