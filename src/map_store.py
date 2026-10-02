# 地图资产存取层：路线挂机（地图跟随）所需的全局小地图与彩色路线图的单一数据源。
#
# 目录结构（maps/）：
#   maps/_index.json            地图顺序与当前默认地图
#   maps/<地图名>/meta.json     该地图全部配置（导航/定位/打怪参数，utf-8）
#   maps/<地图名>/map.png       全局小地图拼接图
#   maps/<地图名>/routeN.png    与 map.png 同尺寸的彩色路线图（纯指令色像素画在黑底上）
#   maps/<地图名>/route_rest.png 休息路线（可选）
#
# 本页签只管资产与配置；将来 MapleRouteTask 启动时读取 maps/<默认地图>/meta.json + 路线图消费，
# 与看板（dashboard_store）解耦：看板负责角色/怪物/测谎共享参数，地图页负责地图与路线。
import json  # 读写 meta/index JSON。
import os  # 路径拼接与目录扫描。
import shutil  # 删除/重命名地图目录。

import cv2  # 读写 PNG 路线图与地图。
import numpy as np  # 图像矩阵。

from ok.util.logger import Logger  # 框架日志器。

logger = Logger.get_logger(__name__)

MAPS_ROOT = 'maps'  # 地图资产根目录（相对项目根，随 exe 工作目录一致）。
MAP_INDEX_FILE = os.path.join(MAPS_ROOT, '_index.json')  # 索引文件路径。
MAP_META_FILE = 'meta.json'  # 地图配置文件名（相对地图目录）。
MAP_IMAGE_FILE = 'map.png'  # 全局小地图文件名。
ROUTE_PREFIX = 'route'  # 路线图文件名前缀（route1.png / route_rest.png）。
ROUTE_EVENTS_SUFFIX = '.keys.json'  # 按键事件流 sidecar 后缀（routeN.png -> routeN.keys.json）。

# 主指令色表（RGB -> "左右 上下 动作"），移植自参考库 MapleStoryAutoLevelUp 的 config_default.yaml。
# 三元组语义：左右 ∈ {left,right,none}，上下 ∈ {up,down,none}，动作 ∈ {none,jump,teleport,goal,stop}。
DEFAULT_COLOR_CODE = {
    '255,0,0': 'left none none',        # 红：向左走
    '0,0,255': 'right none none',       # 蓝：向右走
    '255,127,0': 'left none jump',      # 橙：向左跳
    '0,255,255': 'right none jump',     # 青：向右跳
    '127,255,0': 'none down jump',      # 绿：下跳
    '255,0,255': 'none none jump',      # 品红：原地上跳
    '0,255,127': 'stop stop stop',      # 浅绿：停止（保留字，暂作占位）
    '255,255,0': 'none none goal',      # 黄：路线终点，切换下一条路线
    '255,0,127': 'none up teleport',    # 粉：上瞬移
    '127,0,255': 'none down teleport',  # 紫：下瞬移
    '0,127,0': 'left none teleport',    # 深绿：向左瞬移
    '139,69,19': 'right none teleport',  # 棕：向右瞬移
}

# 上下专用色表：与主表互补，用于保证爬梯/下梯动作平滑（防止角色卡在梯子顶端）。
DEFAULT_COLOR_CODE_UP_DOWN = {
    '127,127,127': 'none up none',      # 灰：爬梯向上
    '255,255,127': 'none down none',    # 浅黄：下梯
}

# 平台边缘色（RGB）：路线图标记该色像素，角色接近时瞬移/jump 回拉，防止坠落。
DEFAULT_EDGE_COLOR = '255,127,127'

# 地图配置默认值：键名将与将来 MapleRouteTask 采集的配置键保持一致。
MAP_META_DEFAULTS = {
    # —— 显示名 ——
    'Display Name': '',                 # 中文显示名，留空则用目录名（目录名建议英文/数字，避免路径编码坑）。
    # —— 导航参数 ——
    'Search Range': 10,                 # 以角色全局坐标为中心，搜索最近路线色像素的半径（像素）。
    'Color Code': dict(DEFAULT_COLOR_CODE),            # 主指令色表 {RGB: "左右 上下 动作"}。
    'Color Code Up Down': dict(DEFAULT_COLOR_CODE_UP_DOWN),  # 上下专用色表。
    'Edge Color': DEFAULT_EDGE_COLOR,   # 平台边缘标记色（RGB 逗号串），留空禁用边缘保护。
    'Use Teleport To Walk': False,      # 是否行走时也持续瞬移加速（法师向）。
    'Teleport Cooldown': 1.0,           # 瞬移技能冷却秒数。
    # —— 定位参数 ——
    'Minimap Feature': '完整小地图',     # 小地图模板分类名（沿用巡逻任务，模板页标注），用于在游戏画面里定位小地图框。
    'Minimap Threshold': 0.8,           # 小地图模板匹配阈值。
    # —— 小地图几何与黄点检测（录制与运行时共用，保证录出来的图一定能被定位）——
    'Map Rect': '',                     # 小地图框内实际地图区域（相对小地图框百分比 x,y,w,h，如 5,20,90,75），留空=整框。
    'Dot Hue Min': 18,                  # 角色黄点 HSV 色相下限（0-179）。
    'Dot Hue Max': 38,                  # 角色黄点 HSV 色相上限（0-179）。
    'Dot Min Pixels': 4,                # 黄点最小连通块面积（像素），更小视为噪点。
    'Dot Sat Min': 200,                 # 黄点 HSV 饱和度下限：角色标记是纯黄，调高可排除地形的橙黄块。
    'Dot Val Min': 180,                 # 黄点 HSV 亮度下限：调高可排除偏暗地形，标记发暗时再调低。
    # —— 录制参数（仅"地图"页签的路线录制工具使用）——
    'Record Canvas Size': '1600,1200',  # 录制期预分配拼接画布尺寸 "宽,高"（像素），停止时按内容外接矩形裁剪。
    'Record Trace Thickness': 2,        # 录制描线粗细（画布像素），越大轨迹越粗、 nearest_color 越易命中。
    # —— 键盘捕获录制参数（开着"键盘捕获"时录制才推导指令，否则退回手点画笔色）——
    'Record Use Keys': True,            # 是否被动监听真实按键自动上色（关闭则用 GUI 当前画笔色）。
    'Record Auto Goal': True,           # 停止录制时在末点自动补一个 goal 色，省得手动标记终点。
    'Record Left Keys': 'left',         # 左移键位，逗号分隔可配多个（如 a, left）。
    'Record Right Keys': 'right',       # 右移键位。
    'Record Up Keys': 'up',             # 上（爬梯）键位。
    'Record Down Keys': 'down',         # 下（下梯）键位。
    'Record Jump Keys': 'space',        # 跳跃键位。
    'Record Teleport Keys': '',         # 瞬移技能键位，留空表示该地图不用瞬移。
    'Record Action Tap Window': 0.25,   # 点按动作有效窗（秒）：录制帧拍推导时，该时长内的点按仍算生效。
    # —— 打怪参数（默认从看板继承，特殊地图可覆盖）——
    'Monster Features': '',             # 该地图怪物标注分类名，英文逗号分隔；留空表示运行时取看板共享值。
}


def _read_json_utf8(path):  # 按 UTF-8(带 BOM 兼容) 读取 JSON，失败返回 None。
    try:
        with open(path, encoding='utf-8-sig') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json_atomic(path, data):  # 原子写 JSON：先写临时文件再替换，进程中途被杀不会写坏现有文件。
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def _imread_unicode(path):  # 读取图像，兼容中文/空格路径（cv2.imread 在 Windows 下对非 ASCII 路径失败）。
    try:
        buf = np.fromfile(path, dtype=np.uint8)  # 先按字节读出，绕过 cv2 的路径编码问题。
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # 解码为 BGR 三通道。
    return img


def _imwrite_unicode(path, img):  # 写入 PNG，兼容非 ASCII 路径。
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ok, buf = cv2.imencode('.png', img)  # 内存编码为 PNG 字节。
    if not ok:
        return False
    buf.tofile(path)  # 落盘。
    return True


# ---------------------------------------------------------------- 目录 / 索引

def map_dir(name):  # 地图目录路径。
    return os.path.join(MAPS_ROOT, name)


def meta_path(name):  # 某地图 meta.json 路径。
    return os.path.join(map_dir(name), MAP_META_FILE)


def map_image_path(name):  # 某地图 map.png 路径。
    return os.path.join(map_dir(name), MAP_IMAGE_FILE)


def load_index():  # 读取索引 {order:[...], default:name}；文件缺失时按磁盘扫描结果自动生成并落盘。
    data = _read_json_utf8(MAP_INDEX_FILE)
    if not isinstance(data, dict) or 'order' not in data:
        data = {'order': _scan_map_dirs(), 'default': ''}
        _write_json_atomic(MAP_INDEX_FILE, data)
    return data


def save_index(index):  # 保存索引。
    _write_json_atomic(MAP_INDEX_FILE, index)


def _scan_map_dirs():  # 扫描 maps/ 下含 map.png 的子目录名，按名称排序。
    if not os.path.isdir(MAPS_ROOT):
        return []
    result = []
    for entry in sorted(os.listdir(MAPS_ROOT)):
        if os.path.isfile(os.path.join(MAPS_ROOT, entry, MAP_IMAGE_FILE)):
            result.append(entry)
    return result


def list_maps():  # 返回全部地图目录名：以索引顺序为准，磁盘上存在但不在索引里的追加到末尾。
    index = load_index()
    on_disk = set(_scan_map_dirs())
    order = [name for name in index.get('order', []) if name in on_disk]  # 去掉磁盘已不存在的残留项。
    for name in _scan_map_dirs():
        if name not in order:
            order.append(name)
    return order


def get_default_map():  # 返回当前默认地图名，无效时回退到列表首个（可能为空字符串）。
    index = load_index()
    default = index.get('default')
    maps = list_maps()
    if default in maps:
        return default
    return maps[0] if maps else ''


def set_default_map(name):  # 设为默认地图。
    index = load_index()
    index['default'] = name
    save_index(index)


def _unique_dir_name(base):  # 若目录重名，追加序号生成唯一安全目录名。
    if not os.path.exists(map_dir(base)):
        return base
    i = 2
    while os.path.exists(map_dir(f'{base}_{i}')):
        i += 1
    return f'{base}_{i}'


# ---------------------------------------------------------------- 地图 meta

def load_meta(name):  # 读取地图配置，缺失键补默认值（Color Code 单独深拷贝，避免污染默认常量）。
    data = _read_json_utf8(meta_path(name))
    if not isinstance(data, dict):
        data = {}
    for key, value in MAP_META_DEFAULTS.items():
        if key not in data:
            data[key] = dict(value) if isinstance(value, dict) else value
    return data


def save_meta(name, meta):  # 保存地图配置。
    _write_json_atomic(meta_path(name), meta)


def create_map(name, image):  # 新建地图：建目录、写 map.png 与默认 meta、登记到索引。返回实际使用的目录名。
    name = _unique_dir_name(str(name).strip())
    os.makedirs(map_dir(name), exist_ok=True)
    if image is None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)  # 占位空地图，稍后由用户导入截图覆盖。
    _imwrite_unicode(map_image_path(name), image)
    meta = {k: (dict(v) if isinstance(v, dict) else v) for k, v in MAP_META_DEFAULTS.items()}
    meta['Display Name'] = ''
    save_meta(name, meta)
    index = load_index()
    if name not in index.get('order', []):
        index.setdefault('order', []).append(name)
        save_index(index)
    if not index.get('default'):
        set_default_map(name)
    return name


def delete_map(name):  # 删除地图目录并从索引移除。
    target = map_dir(name)
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    index = load_index()
    index['order'] = [n for n in index.get('order', []) if n != name]
    if index.get('default') == name:
        index['default'] = ''
    save_index(index)


def rename_map(old_name, new_name):  # 重命名地图目录：仅改目录名，meta 内 Display Name 由页签另行处理。返回新目录名或 None。
    new_name = str(new_name).strip()
    if not new_name or new_name == old_name:
        return None
    if os.path.exists(map_dir(new_name)):
        new_name = _unique_dir_name(new_name)
    shutil.move(map_dir(old_name), map_dir(new_name))
    index = load_index()
    index['order'] = [new_name if n == old_name else n for n in index.get('order', [])]
    if index.get('default') == old_name:
        index['default'] = new_name
    save_index(index)
    return new_name


# ---------------------------------------------------------------- 路线文件

def list_routes(name):  # 返回该地图全部路线图文件名（不含 route_rest.png），排序稳定。
    target = map_dir(name)
    if not os.path.isdir(target):
        return []
    routes = []
    for entry in os.listdir(target):
        if not entry.startswith(ROUTE_PREFIX) or not entry.endswith('.png'):
            continue
        if entry == 'route_rest.png':
            continue
        routes.append(entry)
    return sorted(routes, key=lambda s: _route_sort_key(s))


def _route_sort_key(route_file):  # routeN.png -> N，其它（如 route.png）排到末尾且稳定。
    stem = os.path.splitext(route_file)[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10 ** 9, stem)


def route_path(name, route_file):  # 某路线图完整路径。
    return os.path.join(map_dir(name), route_file)


def next_route_name(name):  # 生成下一条路线图文件名 routeN.png（N 取现有最大 +1）。
    max_n = 0
    for route_file in list_routes(name):
        stem = os.path.splitext(route_file)[0]
        digits = ''.join(ch for ch in stem if ch.isdigit())
        if digits:
            max_n = max(max_n, int(digits))
    return f'{ROUTE_PREFIX}{max_n + 1}.png'


def add_route(name):  # 新建一条与 map.png 同尺寸的全黑路线图，返回文件名。
    map_img = load_map_image(name)
    if map_img is None:
        return None
    h, w = map_img.shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)  # 黑底：黑色表示该像素无指令。
    file_name = next_route_name(name)
    _imwrite_unicode(route_path(name, file_name), blank)
    return file_name


def delete_route(name, route_file):  # 删除一条路线图及其按键事件流 sidecar（禁止删到只剩零条由调用方保证）。
    path = route_path(name, route_file)
    if os.path.isfile(path):
        os.remove(path)
    events = route_events_path(name, route_file)  # 同步清理 sidecar，避免残留事件流与新路线对不上。
    if os.path.isfile(events):
        os.remove(events)
    backup = path + '.bak'  # 重绘前的旧图备份也一并清掉，不让它泄露在列表之外。
    if os.path.isfile(backup):
        os.remove(backup)


def load_route_image(name, route_file):  # 读取路线图 BGR 矩阵。
    return _imread_unicode(route_path(name, route_file))


def save_route_image(name, route_file, img):  # 保存路线图。
    return _imwrite_unicode(route_path(name, route_file), img)


def load_map_image(name):  # 读取 map.png BGR 矩阵。
    return _imread_unicode(map_image_path(name))


def save_map_image(name, img):  # 保存 map.png（导入截图覆盖时使用）。
    return _imwrite_unicode(map_image_path(name), img)


# ---------------------------------------------------------------- 按键事件流 sidecar
#
# 录制时除了描出 PNG，还把每拍的 (时刻, 全局坐标, 指令) 存下来：改描线粗细/改色表/自动扩色后，
# 可以离线重绘路线图而不必重走地图；cmd 为 None 表示那一拍没有指令（定位失败/站着不动），重绘时在此断线。

def route_events_path(name, route_file):  # routeN.png 对应的事件流路径 routeN.keys.json。
    stem = os.path.splitext(os.path.basename(route_file))[0]
    return os.path.join(map_dir(name), stem + ROUTE_EVENTS_SUFFIX)


def has_route_events(name, route_file):  # 该路线图是否带事件流 sidecar。
    return os.path.isfile(route_events_path(name, route_file))


def backup_route_image(name, route_file):  # 重绘覆盖前备份旧图为 routeN.png.bak，返回备份路径；无原图返回 None。
    path = route_path(name, route_file)
    if not os.path.isfile(path):
        return None
    backup = path + '.bak'
    shutil.copyfile(path, backup)
    return backup


def save_route_events(name, route_file, events, canvas=None, thickness=None):  # 写事件流，返回落盘路径或 None。
    payload = {
        'version': 1,  # 格式版本，后续加字段可据此兼容读取。
        'canvas': [int(canvas[0]), int(canvas[1])] if canvas else None,  # 事件坐标所处画布尺寸。
        'thickness': int(thickness) if thickness else None,  # 录制时的描线粗细。
        'events': [[round(float(e[0]), 3), int(e[1]), int(e[2]), e[3]] for e in (events or [])],  # [时刻s, x, y, 指令或 null]
    }
    path = route_events_path(name, route_file)
    _write_json_atomic(path, payload)
    return path


def load_route_events(name, route_file):  # 读事件流，返回 {version, canvas, thickness, events}；缺失或损坏返回 None。
    data = _read_json_utf8(route_events_path(name, route_file))
    if not isinstance(data, dict) or not isinstance(data.get('events'), list):
        return None
    events = []  # 归一化每行：旧数据多余元素丢弃、不足补 null，保证调用方可安全解包。
    for item in data['events']:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        cmd = item[3] if len(item) > 3 else None
        events.append([float(item[0]), int(item[1]), int(item[2]), cmd])
    return {'version': int(data.get('version') or 1), 'canvas': data.get('canvas'),
            'thickness': data.get('thickness'), 'events': events}


def validate_routes(name):  # 校验所有路线图与 map.png 尺寸一致，返回 [(route_file, 错误说明 或 None)]。
    map_img = load_map_image(name)
    if map_img is None:
        return [(r, 'map.png 缺失') for r in list_routes(name)]
    map_h, map_w = map_img.shape[:2]
    result = []
    for route_file in list_routes(name):
        route_img = load_route_image(name, route_file)
        if route_img is None:
            result.append((route_file, '无法读取'))
            continue
        h, w = route_img.shape[:2]
        if (h, w) != (map_h, map_w):
            result.append((route_file, f'尺寸 {w}x{h} 与地图 {map_w}x{map_h} 不一致'))
        else:
            result.append((route_file, None))
    return result


def map_size(name):  # 返回 (w, h) 或 None。
    img = load_map_image(name)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)
