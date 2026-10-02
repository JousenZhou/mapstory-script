# 按键名规范化工具：统一"看板/地图 meta 里配置的按键名"与"pynput 回调报出的按键名"两种写法，
# 供解测谎急停键匹配与路线录制的键盘捕获做同口径比较。
# 原先这段逻辑内联在 src/liedetector/service.py，因录制侧（src/key_capture.py）同样需要，抽到独立模块避免互相牵依赖。
try:  # 复用框框 Pynput 交互的按键名映射（如 lshift -> shift_l），保证配置键与任务按键用同一套命名。
    from ok.device.interaction_methods.pynput import PynputInteraction as _PynputInteraction
    _KEY_NAME_MAP = _PynputInteraction.KEY_MAP  # ok-script 按键名 -> pynput 按键名（只收录需要改名的键）。
except Exception:  # 框框结构变化时降级为空映射，按键名仍按原名比较。
    _KEY_NAME_MAP = {}

# pynput 监听回调报出的键名 -> 规范名：pynput 不区分左右修饰键（Key.shift_l.name 也是 'shift'），
# 且命名风格与看板配置不同（page_up vs pageup），故两侧都归一到同一规范名再比较。
_KEY_NAME_ALIAS = {
    'shift_l': 'shift', 'shift_r': 'shift', 'ctrl_l': 'ctrl', 'ctrl_r': 'ctrl',
    'alt_l': 'alt', 'alt_r': 'alt', 'alt_gr': 'alt', 'cmd_l': 'cmd', 'cmd_r': 'cmd',
    'page_up': 'pageup', 'page_down': 'pagedown', 'caps_lock': 'capslock',
    'num_lock': 'numlock', 'scroll_lock': 'scrolllock', 'print_screen': 'printscreen',
    'return': 'enter',
}


def normalize_key_name(value):  # 把配置的按键名或 pynput 报出的按键名归一到同一规范名，供按键匹配比较。
    name = str(value or '').strip().lower()  # 统一小写并去空格。
    if not name:  # 未配置。
        return ''
    name = _KEY_NAME_MAP.get(name, name)  # 先按框框映射转成 pynput 名（表里没有的键原样保留，如 f8）。
    return _KEY_NAME_ALIAS.get(name, name)  # 再收敛左右修饰键与命名风格差异。


def parse_key_list(text):  # 解析 "left, a" 这类逗号（或分号/空格）分隔的多键位串，返回去重且保序的规范键名列表。
    raw = str(text or '').strip()  # 原文本。
    if not raw:  # 留空表示未绑定。
        return []
    names = raw.replace(';', ',').replace(' ', ',').split(',')  # 统一分隔符后切分，允许空格分隔。
    result = []  # 保序去重结果。
    for item in names:  # 逐个规范化。
        name = normalize_key_name(item)  # 规范键名。
        if name and name not in result:  # 跳过空串与重复项。
            result.append(name)
    return result  # 返回列表。
