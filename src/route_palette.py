# 指令色表工具：把"左右 上下 动作"三元指令与路线图像素色互查，并在指令没有对应色时自动扩色。
#
# 背景：录制端按键盘实时推导出指令，但地图 meta 的色表（Color Code / Color Code Up Down）只收录了有限的组合，
# 玩家做出表里没有的组合（如"左+上"同时按住）时必须给一个色，否则那一段就描不出来。这里的做法是：
#   1) 先反查两张表；2) 查不到就从固定色池 ALLOC_POOL 取一个未占用色；3) 按 combine_cmd 的语义决定进哪张表，
#   由调用方（MapRecorder）在录制结束时把新增项写回该地图 meta，回放端 build_color_map 立刻能反查。
#
# 通道约定与 src/map_route.pixel_key 一致：色三元组就是路线图 ndarray 的通道顺序（录制与手绘同约定），
# 因此这里的三元组一律是 "r,g,b" 语义的整数元组，不做任何 BGR 交换。
import numpy as np  # 只在 unknown_route_colors 里做像素去重

from src.map_store import DEFAULT_EDGE_COLOR  # 边缘保护色也要避开，防止新增指令色与边缘标记撞色

MAIN_TABLE = 'Color Code'  # 主指令色表 meta 键。
UD_TABLE = 'Color Code Up Down'  # 上下专用指令色表 meta 键。

# 自动扩色池：肉眼可辨、互不相同的 RGB，且刻意避开内置 12+2 指令色、边缘色与黑/近灰。
ALLOC_POOL = [
    (0, 255, 0), (255, 255, 255), (0, 127, 255), (255, 63, 127),
    (127, 255, 127), (63, 127, 255), (255, 127, 255), (127, 255, 255),
    (127, 127, 0), (0, 0, 127), (127, 0, 0), (0, 127, 127),
    (255, 165, 0), (148, 0, 211), (0, 215, 255), (34, 139, 34),
    (220, 20, 60), (255, 20, 147), (0, 191, 255), (154, 205, 50),
    (255, 215, 0), (72, 61, 139), (240, 128, 128), (102, 255, 102),
]

BLACK = (0, 0, 0)  # 黑色=该像素无指令，任何指令色都不能用它。


def rgb_from_key(key):  # "r,g,b" 串 -> 整数三元组，非法返回 None。
    parts = str(key or '').replace('，', ',').split(',')  # 兼容中文逗号
    if len(parts) != 3:  # 必须三段
        return None
    try:  # 逐个转整数
        values = tuple(int(p.strip()) for p in parts)
    except ValueError:  # 非数字
        return None
    if any(v < 0 or v > 255 for v in values):  # 越界
        return None
    return values  # 合法三元组


def key_from_rgb(rgb):  # 整数三元组 -> "r,g,b" 串（写回 meta 的键格式）。
    return ','.join(str(int(c)) for c in rgb)


def command_text(move_x, move_y, action):  # 三段指令 -> meta 色表里的指令串（缺省补 none，多余段丢弃）。
    return ' '.join([str(move_x or 'none').strip().lower(), str(move_y or 'none').strip().lower(),
                     str(action or 'none').strip().lower()])


def split_command(command):  # 指令串 -> (move_x, move_y, action)，段数不足补 none，多余忽略。
    tokens = str(command or '').split()  # 按空白切分
    while len(tokens) < 3:  # 补齐
        tokens.append('none')
    return tokens[0], tokens[1], tokens[2]


def build_reverse(main_map, ud_map):  # {指令串: rgb三元组}：主表优先，重复指令保留主表色（与人工编辑习惯一致）。
    reverse = {}  # 结果字典
    for color_map in (main_map or {}, ud_map or {}):  # 先主表再上下表，配合 setdefault 让主表色胜出
        for key, command in color_map.items():  # 遍历 {rgb串: 指令}
            rgb = rgb_from_key(key)  # 解析颜色
            if rgb is None:  # 非法色串跳过
                continue
            cmd = command_text(*split_command(command))  # 规整指令串，避免大小写/空格差异导致查不到
            reverse.setdefault(cmd, rgb)  # 同一指令只认首个写入者，即主表
    return reverse


def target_table(command):  # 该指令应写进哪张色表：含左右或动作归主表，纯上下归上下表（与 combine_cmd 互补语义一致）。
    move_x, move_y, action = split_command(command)  # 三段指令
    if move_x != 'none' or action != 'none':  # 有横向或有动作
        return MAIN_TABLE
    return UD_TABLE  # 只靠上下分量


def used_colors(main_map, ud_map, extra=()):  # 收集不可占用的颜色集合：两张表全部色 + 黑 + 边缘色 + 调用方附加色。
    used = {BLACK}  # 黑色保留给"无指令"
    try:  # 边缘色解析失败不影响扩色
        edge = rgb_from_key(DEFAULT_EDGE_COLOR)
        if edge is not None:
            used.add(edge)
    except Exception:
        pass
    for color_map in (main_map or {}, ud_map or {}):  # 两张已有色表
        for key in color_map.keys():  # 只看颜色键
            rgb = rgb_from_key(key)
            if rgb is not None:
                used.add(rgb)
    for item in extra or ():  # 调用方附加（如本局已分配过的色）
        rgb = item if isinstance(item, tuple) else rgb_from_key(item)
        if rgb is not None:
            used.add(rgb)
    return used


def allocate_color(used):  # 从 ALLOC_POOL 取首个未占用色；池耗尽返回 None（调用方按"本拍不描线"处理）。
    for rgb in ALLOC_POOL:  # 按池内顺序，保证同样输入下分配结果稳定可复现
        if rgb not in used:  # 未被占用
            return rgb
    return None


def resolve_color(command, main_map, ud_map, used=None):  # 指令 -> (rgb, 新增项或 None)；未命中色表时自动扩色。
    cmd = command_text(*split_command(command))  # 规整指令串
    rgb = build_reverse(main_map, ud_map).get(cmd)  # 先反查两张表
    if rgb is not None:  # 已有定义，直接落色
        return rgb, None
    occupied = used if used is not None else used_colors(main_map, ud_map)  # 未占用色集合
    new_rgb = allocate_color(occupied)  # 分配一个新色
    if new_rgb is None:  # 色池耗尽
        return None, None
    addition = (target_table(cmd), key_from_rgb(new_rgb), cmd)  # (写哪张表, 颜色键, 指令串)
    return new_rgb, addition


def apply_addition(main_map, ud_map, addition):  # 把新增色写进内存色表副本，使同一指令后续帧复用同色。
    if not addition:  # 无新增
        return False
    table_key, rgb_key, cmd = addition  # 三元组
    target = main_map if table_key == MAIN_TABLE else ud_map  # 目标表
    target[rgb_key] = cmd  # 登记
    return True


def unknown_route_colors(route_img, main_map, ud_map, edge_rgb=None):  # 路线图上出现但色表未定义的色串列表（升序），用于提示色表被清掉。
    if route_img is None or route_img.size == 0:  # 无图
        return []
    defined = used_colors(main_map, ud_map, extra=(edge_rgb,))  # 已定义色（含黑、边缘色）
    flat = route_img.reshape(-1, route_img.shape[-1])  # 展平成 Nx3
    unique = np.unique(flat, axis=0) if (flat.shape[-1] == 3) else flat  # 去重后的所有像素色
    result = []  # 未定义色
    for row in unique:  # 逐个检查
        rgb = (int(row[0]), int(row[1]), int(row[2]))  # 三元组
        if rgb in defined:  # 已定义或为黑/边缘色
            continue
        result.append(key_from_rgb(rgb))
    return sorted(result)
