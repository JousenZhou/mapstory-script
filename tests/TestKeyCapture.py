# 全局按键捕获器（src/key_capture.py）单元测试。
# 只测状态机与键名解析，不装真实钩子：press/release 是公开方法，既是 pynput 回调的入口也是单测注入口，
# 因此全部用例都用注入时刻（now=...）来摆脱真实时间依赖；start/stop 用假 listener 覆盖，绝不触碰系统键盘。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 离屏平台设置，避免任何 Qt 初始化触碰屏幕。

import unittest

import src.key_capture as kc  # 被测按键捕获模块。


class FakeKeyCode:  # 模拟 pynput 的键对象：字符键带 char，功能键带 name。

    def __init__(self, char=None, name=None):  # 二选一。
        self._char = char  # 字符键内容。
        self._name = name  # 功能/修饰键名称。

    @property
    def char(self):  # pynput 的 Key.char：功能键为 None。
        return self._char

    @property
    def name(self):  # pynput 的 Key.name：字符键为 None。
        return self._name


class ExplodingKey:  # 任何属性访问都抛错的键对象，用于验证回调吞异常。

    def __getattr__(self, item):  # 取任意属性都炸。
        raise RuntimeError('boom')


class TestKeyNameParsing(unittest.TestCase):  # 键名解析与归一。

    def test_char_key_and_named_key(self):  # 字符键取 char，功能键取 name 并归一。
        self.assertEqual('a', kc.key_event_name(FakeKeyCode(char='a')))  # 字母键。
        self.assertEqual('A', kc.key_event_name(FakeKeyCode(char='A')).upper())  # 大写字母仍解析得到。
        self.assertEqual('space', kc.key_event_name(FakeKeyCode(name='space')))  # 空格。
        self.assertEqual('shift', kc.key_event_name(FakeKeyCode(name='shift_l')))  # 左 shift 归一到 shift。

    def test_normalize_left_right_modifiers(self):  # 修饰键不区分左右，配置写 shift 也能匹配 shift_l。
        self.assertEqual('shift', kc.key_event_name(FakeKeyCode(name='shift_r')))  # 右 shift。
        self.assertEqual('ctrl', kc.key_event_name(FakeKeyCode(name='ctrl_l')))  # 左 ctrl。
        self.assertEqual('enter', kc.key_event_name(FakeKeyCode(name='return')))  # return -> enter。

    def test_unparsable_key_yields_empty(self):  # 解析不出名字的键返回空串，调用方忽略。
        self.assertEqual('', kc.key_event_name(FakeKeyCode()))  # 无 char 无 name。


class TestCaptureState(unittest.TestCase):  # 按住/点按状态机（注入时刻，不依赖真实时间）。

    def test_held_order_is_press_order(self):  # held 按按下时刻升序返回，供"后按优先"规则取值。
        cap = kc.KeyCapture()  # 未装钩子的纯状态机。
        cap.press('left', now=100.0)  # 先按左。
        cap.press('right', now=101.0)  # 后按右。
        self.assertEqual(['left', 'right'], cap.held_ordered())  # 顺序即按下先后。

    def test_auto_repeat_does_not_reorder_held(self):  # 长按自动连发不改变按住顺序（否则方向会漂）。
        cap = kc.KeyCapture()
        cap.press('left', now=100.0)  # 先按住左。
        cap.press('right', now=101.0)  # 再按住右。
        cap.press('left', now=102.0)  # 左的自动连发。
        self.assertEqual(['left', 'right'], cap.held_ordered())  # 首次按下时刻才算顺序。
        self.assertEqual(102.0, cap.taps_snapshot()['left'])  # 但点按时刻要刷新到最新。

    def test_release_removes_held_but_keeps_tap(self):  # 松开后不再算"按住"，但点按时刻保留供动作窗判定。
        cap = kc.KeyCapture()
        cap.press('space', now=50.0)  # 点一下跳。
        cap.release('space', now=50.05)  # 松开。
        self.assertEqual([], cap.held_ordered())  # 已不在按住表。
        self.assertEqual(50.0, cap.taps_snapshot()['space'])  # 点按时刻仍在。

    def test_tapped_within_window(self):  # 动作窗内判定：窗口内为真、过期为假、未绑定的键为假。
        cap = kc.KeyCapture()
        cap.press('space', now=200.0)  # 按下时刻固定。
        self.assertTrue(cap.tapped_within('space', 0.25, now=200.2))  # 落在窗内。
        self.assertFalse(cap.tapped_within('space', 0.25, now=200.5))  # 超过窗口。
        self.assertFalse(cap.tapped_within('z', 0.25, now=200.1))  # 从未按下。
        self.assertFalse(cap.tapped_within('', 0.25, now=200.1))  # 未绑定键名。
        self.assertFalse(cap.tapped_within('space', 0, now=200.1))  # 窗口非正。

    def test_seen_keys_and_since_last_event(self):  # 自检要用的两读法：收到过哪些键、距上次事件多久。
        cap = kc.KeyCapture()  # 新建。
        self.assertIsNone(cap.since_last_event(now=1000.0))  # 从未有事件 → None（自检据此判钩子失效）。
        cap.press('w', now=10.0)  # 收到一个键。
        cap.press('a', now=11.0)  # 又一个。
        cap.release('w', now=12.0)  # 松开也计事件。
        self.assertEqual(['w', 'a'], cap.seen_keys())  # 按首次出现顺序去重。
        self.assertEqual(2.0, cap.since_last_event(now=14.0))  # 距最近事件 2 秒。

    def test_reset_clears_state(self):  # 复位清空按住与点按，避免跨会话残留键位。
        cap = kc.KeyCapture()
        cap.press('left', now=1.0)  # 按住左。
        cap.reset()  # 复位。
        self.assertEqual([], cap.held_ordered())  # 按住清空。
        self.assertEqual({}, cap.taps_snapshot())  # 点按清空。
        self.assertEqual([], cap.seen_keys())  # 自检记录也清空。

    def test_empty_key_name_ignored(self):  # 空键名的按下/松开直接忽略，不污染状态。
        cap = kc.KeyCapture()
        cap.press('', now=1.0)  # 空名按下。
        cap.release(None, now=1.0)  # None 松开。
        self.assertEqual([], cap.held_ordered())  # 未写入。
        self.assertEqual({}, cap.taps_snapshot())  # 未写入。


class TestCallbacks(unittest.TestCase):  # pynput 回调：只做解析与改表，异常必须吞掉。

    def test_on_press_and_on_release(self):  # 回调走通到状态表。
        cap = kc.KeyCapture()
        cap._on_press(FakeKeyCode(name='up'))  # 按下回调。
        self.assertEqual(['up'], cap.held_ordered())  # 进入按住表。
        cap._on_release(FakeKeyCode(name='up'))  # 松开回调。
        self.assertEqual([], cap.held_ordered())  # 移出按住表。

    def test_callback_swallows_exception(self):  # 键对象解析抛错时回调不得外传（否则监听线程会拖垮录制）。
        cap = kc.KeyCapture()
        cap._on_press(ExplodingKey())  # 不该抛。
        cap._on_release(ExplodingKey())  # 不该抛。
        self.assertEqual([], cap.held_ordered())  # 状态未变。

    def test_release_before_press_is_safe(self):  # 松开事件早于按下事件到达时不报错（钩子启动瞬间可能出现）。
        cap = kc.KeyCapture()
        cap.release('left', now=1.0)  # 无对应按下。
        self.assertEqual([], cap.held_ordered())  # 仍为空。


class FakeListener:  # 假 pynput 监听器，记录启停调用。

    def __init__(self, on_press=None, on_release=None):  # 保存回调供测试手动触发。
        self.on_press = on_press  # 按下回调。
        self.on_release = on_release  # 松开回调。
        self.started = False  # 是否启动过。
        self.stopped = False  # 是否停止过。

    def start(self):  # 模拟启动。
        self.started = True

    def stop(self):  # 模拟停止。
        self.stopped = True


class FakeKeyboardModule:  # 假 pynput.keyboard，用来替换真钩子。

    def __init__(self, listener=None, error=None):  # 可注入监听器实例或启动异常。
        self._listener = listener  # 返回的监听器。
        self._error = error  # 构造时抛出的异常。
        self.created = []  # 造过的监听器列表。

    class _Listener:  # 供 keyboard.Listener(...) 调用的类。
        pass

    def Listener(self, on_press=None, on_release=None):  # 构造监听器。
        if self._error is not None:  # 模拟注册钩子失败。
            raise self._error
        listener = self._listener if self._listener is not None else FakeListener(on_press, on_release)  # 注入或新建。
        self.created.append(listener)  # 记录。
        return listener


class TestLifecycle(unittest.TestCase):  # start/stop 的幂等与失败处理。

    def setUp(self):  # 备份真模块引用。
        self._orig = kc._pynput_keyboard  # 原 pynput.keyboard 引用。

    def tearDown(self):  # 还原，避免影响其他用例。
        kc._pynput_keyboard = self._orig

    def test_start_stop_and_is_active(self):  # 启动后处于监听态，停止后复位且幂等。
        fake = FakeKeyboardModule()  # 假键盘模块。
        kc._pynput_keyboard = fake  # 替换依赖。
        cap = kc.KeyCapture()  # 新建捕获器。
        self.assertTrue(cap.start())  # 启动成功。
        self.assertTrue(cap.is_active())  # 监听态。
        self.assertTrue(cap.start())  # 重复启动幂等。
        self.assertEqual(1, len(fake.created))  # 只装了一个钩子。
        cap.stop()  # 停止。
        self.assertFalse(cap.is_active())  # 已停用。
        self.assertTrue(fake.created[0].stopped)  # 真调用了监听器 stop。
        cap.stop()  # 重复停止不抛。

    def test_start_failure_records_error(self):  # 钩子注册失败时返回 False 并记录原因，供 GUI 提示。
        kc._pynput_keyboard = FakeKeyboardModule(error=OSError('hook denied'))  # 模拟权限失败。
        cap = kc.KeyCapture()  # 新建。
        self.assertFalse(cap.start())  # 启动失败。
        self.assertFalse(cap.is_active())  # 未进入监听态。
        self.assertIn('hook denied', str(cap.error))  # 原因可见。

    def test_start_without_pynput_returns_false(self):  # 依赖缺失（_pynput_keyboard=None）时不抛异常。
        kc._pynput_keyboard = None  # 模拟无 pynput 环境。
        cap = kc.KeyCapture()  # 新建。
        self.assertFalse(cap.start())  # 直接失败。
        self.assertIn('pynput', str(cap.error))  # 原因说明依赖缺失。


if __name__ == '__main__':
    unittest.main()
