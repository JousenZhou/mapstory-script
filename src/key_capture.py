# 全局按键捕获器：被动监听真实键盘的按下/松开（不拦截、不吞键），维护"当前按住"与"最近点按"两张表，
# 供路线录制把玩家的实时操作反推成"左右 上下 动作"三元指令（见 src/map_recorder.command_from_keys）。
#
# 为什么用钩子而不是轮询按键状态：录制循环只有 15FPS（约 66ms 一拍），而跳跃/瞬移是一次 50~100ms 的点按，
# 轮询会整拍漏掉；边沿事件把按下时刻记进 _taps，之后只要落在有效窗内就能被判定，不依赖采样时机。
#
# 线程模型沿用 src/liedetector/service.py 的急停按键监听：pynput 回调运行在监听线程，只做名称解析与加锁改表，
# 任何异常都在回调内吞掉，避免钩子异常拖垮录制线程。
import threading  # 状态锁与监听线程句柄
import time  # 按下/松开时间戳

from collections import OrderedDict  # 记录按键的先后顺序，供"左右同时按住取后按下"的规则

from src.keynames import normalize_key_name  # 按键名归一，配置键与 pynput 报出的键同口径比较

try:  # pynput 为可选依赖：缺失时本模块仍可导入，start() 返回 False 由调用方降级
    from pynput import keyboard as _pynput_keyboard
except Exception:  # 无 pynput 环境
    _pynput_keyboard = None  # 标记不可用


def key_event_name(key):  # 把 pynput 回调的键对象解析成规范键名：字符键取 char，功能/修饰键取 name。
    try:  # 解析失败（未知键对象）按空名处理
        name = getattr(key, 'char', None) or getattr(key, 'name', '') or ''
    except Exception:  # 极端情况下 pynput 键对象访问属性抛错
        return ''
    return normalize_key_name(name)  # 归一为规范名（如 shift_l -> shift）


class KeyCapture:  # 全局键盘状态捕获：held 记录当前按住的键（保持按下顺序），taps 记录每个键最近一次按下时刻。

    def __init__(self):  # 初始化空状态，不装钩子（钩子在 start() 时装）。
        self._lock = threading.Lock()  # 保护 _held/_taps/_last_event_ts 的读改
        self._held = OrderedDict()  # 规范键名 -> 首次按下时刻（重复的自动连发不覆盖，保证顺序稳定）
        self._taps = {}  # 规范键名 -> 最近一次按下时刻（含自动连发，供点按动作判定）
        self._listener = None  # pynput 监听线程句柄，None 表示未启动
        self._last_event_ts = None  # 最近一次收到任何按键事件的时刻，供"按键自检"判断钩子是否收得到
        self._seen = OrderedDict()  # 本会话出现过的键名（只记顺序不记次数），供自检回显"收到过哪些键"
        self.error = None  # 启动失败原因（权限/依赖缺失），供 GUI 提示

    # ------------------------------------------------------------------ 生命周期

    def start(self):  # 启动全局监听；已启动则幂等返回，pynput 不可用或启动异常返回 False。
        if self._listener is not None:  # 已在监听
            return True
        if _pynput_keyboard is None:  # 依赖缺失
            self.error = 'pynput unavailable'  # 记录原因
            return False
        try:  # 钩子注册可能因权限/平台限制失败
            listener = _pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            listener.daemon = True  # 守护线程，随进程退出不阻塞关闭
            listener.start()
            self._listener = listener  # 保存句柄供停用
            self.error = None  # 清掉上一次的失败原因
            return True
        except Exception as e:  # 启动异常
            self.error = str(e)  # 记录原因
            self._listener = None
            return False

    def stop(self):  # 停止监听并清空状态，异常一律吞掉（停不掉不影响后续流程）。
        listener = self._listener
        self._listener = None  # 先摘引用避免重复停
        if listener is not None:  # 有监听器才停
            try:
                listener.stop()
            except Exception:
                pass
        self.reset()  # 清空按住状态，避免残留键位影响下次录制

    def is_active(self):  # 是否处于监听态。
        return self._listener is not None

    # ------------------------------------------------------------------ 状态写入（也是单测注入口）

    def press(self, name, now=None):  # 记录一次按下：首次按住才写入 held（保持顺序），taps 每次都刷新时刻。
        key = normalize_key_name(name)  # 归一化后的键名
        if not key:  # 空名忽略
            return
        ts = time.time() if now is None else float(now)  # 事件时刻
        with self._lock:  # 与读取方互斥
            if key not in self._held:  # 新按住：记录按下顺序的起点
                self._held[key] = ts
            self._taps[key] = ts  # 点按时刻（含长按自动连发）
            self._seen[key] = True  # 自检回显用
            self._last_event_ts = ts  # 自检用

    def release(self, name, now=None):  # 记录一次松开：从 held 摘除，taps 保留供动作窗判定。
        key = normalize_key_name(name)  # 归一化后的键名
        if not key:  # 空名忽略
            return
        ts = time.time() if now is None else float(now)  # 事件时刻
        with self._lock:  # 与读取方互斥
            self._held.pop(key, None)  # 未在按住列表中则忽略（松开事件可能先于按下事件到达）
            self._last_event_ts = ts  # 自检用

    def reset(self):  # 清空全部按键状态（开始/停止录制时调用，避免跨会话残留）。
        with self._lock:
            self._held.clear()
            self._taps.clear()
            self._seen.clear()

    # ------------------------------------------------------------------ pynput 回调

    def _on_press(self, key):  # 按下回调：运行在监听线程，务必轻量并吞异常。
        try:
            self.press(key_event_name(key))
        except Exception:
            pass

    def _on_release(self, key):  # 松开回调：同上。
        try:
            self.release(key_event_name(key))
        except Exception:
            pass

    # ------------------------------------------------------------------ 状态读取

    def held_ordered(self):  # 当前按住的键名列表，按按下时刻升序（后按的在末尾）。
        with self._lock:
            return sorted(self._held.keys(), key=lambda k: self._held[k])  # 顺序即按下先后，供"后按优先"规则取值

    def seen_keys(self):  # 本次启动以来收到过的键名（按首次出现顺序），供自检确认钩子确实收到了按键。
        with self._lock:
            return list(self._seen.keys())

    def taps_snapshot(self):  # {键名: 最近按下时刻} 的副本，供纯函数 command_from_keys 使用。
        with self._lock:
            return dict(self._taps)

    def tapped_within(self, name, window, now=None):  # 某键是否在 window 秒内被按下过（点按动作判定）。
        key = normalize_key_name(name)  # 归一化
        if not key or window <= 0:  # 未绑定或窗口非正
            return False
        ts = self.taps_snapshot().get(key)  # 最近按下时刻
        if ts is None:  # 从未按下
            return False
        ref = time.time() if now is None else float(now)  # 参考时刻
        return (ref - ts) <= window  # 落在有效窗内

    def since_last_event(self, now=None):  # 距最近一次收到按键事件的秒数；从未收到过返回 None（自检据此判断钩子是否失效）。
        with self._lock:
            ts = self._last_event_ts
        if ts is None:
            return None
        return (time.time() if now is None else float(now)) - ts
