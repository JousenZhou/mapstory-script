# 路线跟随纯算法层：把参考库 MapleStoryAutoLevelUp 的"全局小地图定位 + 彩色指令解码"移植为无状态函数，
# 不依赖 Qt / ok 执行器，既供 MapleRouteTask 运行时调用，也可被 tests/TestMapRoute.py 直接单测。
#
# 关键约定（与地图页签 MapTab / route_canvas 完全一致）：
#   - 路线图 routeN.png 是"合成指令图"，指令色只是标签，录制端把 meta 的 "r,g,b" 键元组原样写入像素通道，
#     PNG 无损往返后读取到的数组通道顺序与写入时相同；因此解码端用同一通道顺序反查键即可精确命中，
#     物理上的红/蓝是否互换不影响正确性（只要录制与回放约定一致）。
#   - 黑色像素 (0,0,0) 表示该处无指令，跳过。
import cv2  # 模板匹配定位与像素绘制。
import numpy as np  # 图像矩阵运算。


def rgb_to_key(rgb):  # 把三元组/字符串颜色统一成 meta 里使用的 "r,g,b" 字符串键。
    if isinstance(rgb, str):
        return ','.join(str(int(p.strip())) for p in rgb.split(','))  # 已是逗号串则规范化。
    return ','.join(str(int(c)) for c in rgb)  # 三元组拼接。


def key_to_tuple(key):  # "r,g,b" 字符串键 -> 整数三元组（数组通道顺序，与写入端一致）。
    return tuple(int(p.strip()) for p in str(key).split(','))


def build_color_map(meta):  # 由地图 meta 构造 (主指令色表, 上下指令色表)：均为 {通道三元组: "左右 上下 动作"}。
    main_map = {}  # 主指令色表。
    ud_map = {}  # 上下专用色表。
    for key, command in (meta.get('Color Code') or {}).items():  # 遍历主表。
        try:  # 非法颜色串跳过，保证运行期查表字典一定可用。
            main_map[key_to_tuple(key)] = str(command)  # 键为通道三元组。
        except (TypeError, ValueError):
            continue  # 忽略非法行。
    for key, command in (meta.get('Color Code Up Down') or {}).items():  # 遍历上下表。
        try:
            ud_map[key_to_tuple(key)] = str(command)  # 键为通道三元组。
        except (TypeError, ValueError):
            continue  # 忽略非法行。
    return main_map, ud_map  # 返回两张查表字典。


def pixel_key(route_img, x, y):  # 取路线图 (x,y) 像素的通道三元组键（与 build_color_map 的键同一顺序）。
    p = route_img[y, x]  # 注意 numpy 索引为 [行=y, 列=x]。
    return (int(p[0]), int(p[1]), int(p[2]))  # 直接按数组通道顺序，反查即命中录制端写入的键。


def nearest_color(route_img, loc, search_range, main_map, ud_map):  # 以角色全局坐标 loc 为中心，在 ±search_range 方形内按曼哈顿距离找最近的主色点与上下色点。
    x0, y0 = int(loc[0]), int(loc[1])  # 角色在当前路线图上的全局坐标。
    h, w = route_img.shape[:2]  # 路线图尺寸（与 map.png 同尺寸）。
    xmin, xmax = max(0, x0 - search_range), min(w, x0 + search_range)  # 搜索窗横向裁剪到图内。
    ymin, ymax = max(0, y0 - search_range), min(h, y0 + search_range)  # 搜索窗纵向裁剪到图内。
    nearest = None  # 最近主色点 {pixel,command,distance}。
    nearest_ud = None  # 最近上下色点。
    min_dist = float('inf')  # 主色最小曼哈顿距离。
    min_dist_ud = float('inf')  # 上下色最小曼哈顿距离。
    for y in range(ymin, ymax):  # 逐行扫描搜索窗。
        for x in range(xmin, xmax):  # 逐列扫描。
            key = pixel_key(route_img, x, y)  # 当前像素通道键。
            if key == (0, 0, 0):  # 黑色=无指令，跳过。
                continue  # 下一像素。
            dist = abs(x - x0) + abs(y - y0)  # 到角色的曼哈顿距离。
            if key in main_map and dist < min_dist:  # 命中主指令色且更近。
                nearest = {'pixel': (x, y), 'command': main_map[key], 'distance': dist}  # 记录。
                min_dist = dist  # 更新最近距离。
            if key in ud_map and dist < min_dist_ud:  # 命中上下指令色且更近。
                nearest_ud = {'pixel': (x, y), 'command': ud_map[key], 'distance': dist}  # 记录。
                min_dist_ud = dist  # 更新最近距离。
    return nearest, nearest_ud  # 都找不到时返回 (None, None)。


def combine_cmd(nearest, nearest_ud):  # 主色与上下色互补合成 (move_x, move_y, action)，移植参考库 update_cmd_by_route 的双色互补逻辑。
    move_x, move_y, action = 'none', 'none', 'none'  # 默认无指令（保持当前，调用方据此决定松键）。
    if nearest and nearest_ud:  # 两张表都找到最近点。
        if nearest['distance'] < nearest_ud['distance']:  # 主色更近：以主色为主导。
            move_x, move_y, action = nearest['command'].split()  # 主色三元组。
            _, ud_y, _ = nearest_ud['command'].split()  # 取上下色的上下分量。
            if move_y == 'none':  # 主色没有上下动作时，用上下色补上，保证爬梯/下梯平滑（等效参考库在梯上的互补）。
                move_y = ud_y  # 互补上下。
        else:  # 上下色更近：以上下色为主导。
            move_x, move_y, action = nearest_ud['command'].split()  # 上下色三元组。
            main_x, _, _ = nearest['command'].split()  # 取主色的左右分量。
            if move_x == 'none':  # 上下色没有左右动作时，用主色补左右，避免梯口停顿。
                move_x = main_x  # 互补左右。
    elif nearest:  # 只有主色。
        move_x, move_y, action = nearest['command'].split()  # 直接采用。
    elif nearest_ud:  # 只有上下色。
        move_x, move_y, action = nearest_ud['command'].split()  # 直接采用。
    return move_x, move_y, action  # 返回三元指令。


def locate_on_global_map(map_img, live_crop, last_loc=None, radius=60, score_max=0.5):  # 用实时小地图裁剪块在全局拼接图上做 SQDIFF 模板匹配，返回 (相机左上角(x,y) 或 None, 归一化得分)。
    if map_img is None or live_crop is None:  # 缺图无法定位。
        return None, 1.0  # 得分最差。
    m = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY) if map_img.ndim == 3 else map_img  # 全局图转灰度更稳健。
    t = cv2.cvtColor(live_crop, cv2.COLOR_BGR2GRAY) if live_crop.ndim == 3 else live_crop  # 实时小地图转灰度。
    th, tw = t.shape[:2]  # 模板（实时小地图）尺寸。
    h, w = m.shape[:2]  # 全局图尺寸。
    if th <= 0 or tw <= 0 or th >= h or tw >= w:  # 模板不小于搜索图时无法匹配（用户底图尺寸/比例不对）。
        return None, 1.0  # 定位失败。
    if last_loc is not None and score_max > 0.0:  # 有上一帧位置时先做局部搜索加速（移植参考库 find_pattern_sqdiff 的局部优先）。
        x0 = max(0, last_loc[0] - radius)  # 局部窗左界。
        y0 = max(0, last_loc[1] - radius)  # 局部窗上界。
        x1 = min(w, last_loc[0] + radius + tw)  # 局部窗右界（含模板宽）。
        y1 = min(h, last_loc[1] + radius + th)  # 局部窗下界（含模板高）。
        roi = m[y0:y1, x0:x1]  # 局部ROI。
        if roi.shape[0] >= th and roi.shape[1] >= tw:  # ROI 足够大才匹配。
            res = cv2.matchTemplate(roi, t, cv2.TM_SQDIFF_NORMED)  # 归一化平方差匹配（越小越好）。
            res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)  # 去除数值异常。
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)  # 取最优。
            if min_val < score_max:  # 局部命中。
                return (x0 + int(min_loc[0]), y0 + int(min_loc[1])), float(min_val)  # 返回全局坐标。
    res = cv2.matchTemplate(m, t, cv2.TM_SQDIFF_NORMED)  # 全局回退匹配。
    res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)  # 去数值异常。
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)  # 取最优。
    return (int(min_loc[0]), int(min_loc[1])), float(min_val)  # 相机左上角在全局图上的坐标与得分。


def is_near_edge(route_img, loc, edge_key, box_w=20, box_h=10):  # 判断角色全局坐标附近是否出现平台边缘标记色，用于坠落保护（移植参考库 edge_teleport）。
    if route_img is None or not edge_key:  # 未配置边缘色或无路线图时不保护。
        return False  # 不在边缘。
    edge = key_to_tuple(edge_key) if isinstance(edge_key, str) else tuple(int(c) for c in edge_key)  # 边缘色三元组。
    if edge == (0, 0, 0):  # 黑色不作为边缘色。
        return False  # 忽略。
    x0, y0 = int(loc[0]), int(loc[1])  # 角色全局坐标。
    h, w = route_img.shape[:2]  # 路线图尺寸。
    bx, by = max(1, box_w // 2), max(1, box_h // 2)  # 半宽半高。
    xmin, xmax = max(0, x0 - bx), min(w, x0 + bx)  # 检测窗横向。
    ymin, ymax = max(0, y0 - by), min(h, y0 + by)  # 检测窗纵向。
    for y in range(ymin, ymax):  # 逐行。
        for x in range(xmin, xmax):  # 逐列。
            if pixel_key(route_img, x, y) == edge:  # 命中边缘标记色。
                return True  # 判定接近边缘。
    return False  # 窗内无边缘色。
