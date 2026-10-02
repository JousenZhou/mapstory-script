# 地图页签键盘捕获录制相关 UI 接线（src/ui/MapTab.py）无头回归测试。
#
# 设计要点：
#   - QT_QPA_PLATFORM=offscreen 须在导入 PySide6 前设置，仅构造控件并直接调用槽函数，不起录制线程、不取画面、不弹模态框；
#   - MAPS_ROOT 与索引路径重定向到临时目录，绝不触碰仓库里真实的 maps/；
#   - 覆盖：录制参数（键盘捕获开关/键位/动作窗/自动补终点）的加载与保存往返、按键自检的成功与"钩子收不到"两条提示、
#     录制期间控件锁定、用按键事件流离线重绘路线（覆盖前备份 .bak）与缺事件流时的提示、概要里的可重绘数量与未知色提示。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import shutil
import tempfile
import unittest

import numpy as np  # 构造测试底图与断言像素。

from PySide6.QtWidgets import QApplication

import src.key_capture as key_capture  # 用假 pynput 模块替换真钩子，测试绝不监听系统键盘。
import src.map_store as map_store  # 地图资产层（重定向根目录、写事件流 sidecar）。


class FakeListener:  # 假 pynput 监听器：记录启停，不装系统钩子。

    def __init__(self, on_press=None, on_release=None):  # 保存回调供手动触发。
        self.on_press = on_press  # 按下回调。
        self.on_release = on_release  # 松开回调。
        self.stopped = False  # 是否停止过。

    def start(self):  # 模拟启动。
        pass

    def stop(self):  # 模拟停止。
        self.stopped = True


class FakeKeyboardModule:  # 假 pynput.keyboard，把监听器实例交给被测代码。

    def __init__(self):  # 初始化。
        self.created = []  # 造过的监听器列表。

    def Listener(self, on_press=None, on_release=None):  # 构造并记录监听器。
        listener = FakeListener(on_press, on_release)
        self.created.append(listener)
        return listener


class MapTabTestCase(unittest.TestCase):  # 公共脚手架：临时 maps 根目录 + 离屏 MapTab。

    @classmethod
    def setUpClass(cls):  # 全部用例共享一个离屏应用实例。
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):  # 建临时地图目录，创建一张底图与一条路线，再构造页签。
        self.temp_dir = tempfile.mkdtemp(prefix='map_tab_test_')
        self._orig_root = map_store.MAPS_ROOT
        self._orig_index = map_store.MAP_INDEX_FILE
        self._orig_keyboard = key_capture._pynput_keyboard
        map_store.MAPS_ROOT = self.temp_dir
        map_store.MAP_INDEX_FILE = os.path.join(self.temp_dir, '_index.json')
        map_store.create_map('ui_map', np.full((60, 90, 3), 40, dtype=np.uint8))  # 90x60 底图。
        self.route_file = map_store.add_route('ui_map')
        self.tab = self._make_tab()

    def _make_tab(self):  # 构造页签并挂清理（停轮询定时器、释放控件）。
        from src.ui.MapTab import MapTab
        tab = MapTab()
        self.addCleanup(tab._rec_timer.stop)  # 避免测试结束后定时器回调打到已销毁控件。
        return tab

    def tearDown(self):  # 还原全局常量与假依赖，清理临时目录。
        map_store.MAPS_ROOT = self._orig_root
        map_store.MAP_INDEX_FILE = self._orig_index
        key_capture._pynput_keyboard = self._orig_keyboard
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestRecordingParams(MapTabTestCase):  # 键盘捕获参数加载与保存。

    def test_widgets_loaded_from_meta_defaults(self):  # 默认 meta 能回填开关、键位、动作窗。
        tab = self.tab
        self.assertTrue(tab.record_keys_switch.isChecked())  # 默认开键盘捕获。
        self.assertTrue(tab.auto_goal_switch.isChecked())  # 默认自动补终点。
        self.assertEqual('left', tab.rec_left_edit.text())  # 默认键位。
        self.assertEqual('space', tab.rec_jump_edit.text())  # 跳跃键。
        self.assertEqual('', tab.rec_teleport_edit.text())  # 瞬移键留空。
        self.assertAlmostEqual(0.25, tab.tap_window_spin.value())  # 动作窗默认值。

    def test_save_writes_recording_keys_back_to_meta(self):  # 改键位/开关/动作窗后保存，磁盘 meta 同步更新。
        tab = self.tab
        tab.rec_right_edit.setText('d, right')  # 多键同义写法。
        tab.rec_teleport_edit.setText('z')
        tab.auto_goal_switch.setChecked(False)
        tab.record_keys_switch.setChecked(False)
        tab.tap_window_spin.setValue(0.4)
        tab.save()
        meta = map_store.load_meta('ui_map')  # 直接读盘校验。
        self.assertEqual('d, right', meta['Record Right Keys'])  # 原样保存（解析时再规整）。
        self.assertEqual('z', meta['Record Teleport Keys'])  # 瞬移键。
        self.assertFalse(meta['Record Auto Goal'])  # 关自动补终点。
        self.assertFalse(meta['Record Use Keys'])  # 关键盘捕获（退回手点画笔）。
        self.assertAlmostEqual(0.4, meta['Record Action Tap Window'])  # 动作窗。

    def test_key_bindings_roundtrip_through_recorder(self):  # 界面填的键位串能被录制端解析成规范键名列表。
        from src.map_recorder import bindings_from_meta
        tab = self.tab
        tab.rec_left_edit.setText('A, left')  # 大写与多键。
        tab.save()
        bindings = bindings_from_meta(map_store.load_meta('ui_map'))  # 录制端解析。
        self.assertEqual(['a', 'left'], bindings['left'])  # 归一后保序去重。

    def test_dot_thresholds_roundtrip(self):  # 黄点饱和/亮度下限能按默认回填并随保存写回 meta（黄点锁到地形时靠它调）。
        from src.map_recorder import DOT_SAT_MIN, DOT_VAL_MIN  # 模块常量是默认值的唯一来源。
        tab = self.tab
        self.assertEqual(DOT_SAT_MIN, tab.dot_sat_spin.value())  # 默认回填严格档。
        self.assertEqual(DOT_VAL_MIN, tab.dot_val_spin.value())  # 亮度默认。
        tab.dot_sat_spin.setValue(230)  # 用户调高饱和下限排除地形。
        tab.dot_val_spin.setValue(210)
        tab.save()
        meta = map_store.load_meta('ui_map')  # 读盘校验。
        self.assertEqual(230, meta['Dot Sat Min'])  # 新键已写入。
        self.assertEqual(210, meta['Dot Val Min'])  # 同上。
        tab._load_params(map_store.load_meta('ui_map'))  # 重新回填。
        self.assertEqual(230, tab.dot_sat_spin.value())  # 往返一致。
        self.assertEqual(210, tab.dot_val_spin.value())  # 往返一致。

    def test_idle_hint_lists_bindings(self):  # 非录制态提示行回显当前键位，让用户走图前先看一眼。
        tab = self.tab
        tab.rec_right_edit.setText('d')
        tab.save()
        tab._refresh_idle_hint()  # 手动刷新提示行。
        self.assertIn('right:d', tab.record_hint.text())  # 键位摘要可见。
        self.assertIn('键盘捕获', tab.record_hint.text())  # 模式说明可见。

    def test_idle_hint_for_manual_brush_mode(self):  # 关掉键盘捕获后提示行改说手描模式。
        tab = self.tab
        tab.record_keys_switch.setChecked(False)
        tab._refresh_idle_hint()
        self.assertIn('手描模式', tab.record_hint.text())  # 提示切到手描。


class TestRecordingLock(MapTabTestCase):  # 录制期间控件锁定。

    def test_lock_and_unlock_controls(self):  # 锁定时保存/调色板/路线切换/自检/重绘都不可用，解锁后恢复。
        tab = self.tab
        tab._set_recording_locked(True)
        self.assertFalse(tab.save_button.isEnabled())  # 录制中禁止保存，避免与线程写 meta 冲突。
        self.assertFalse(tab.route_combo.isEnabled())  # 禁止切路线。
        self.assertFalse(tab.key_test_btn.isEnabled())  # 钩子已被录制线程占用。
        self.assertFalse(tab.replay_btn.isEnabled())  # 禁止重绘覆盖正在录的路线。
        self.assertFalse(tab.eraser_btn.isEnabled())  # 禁止改画笔。
        self.assertTrue(all(not b.isEnabled() for b in tab._swatch_buttons))  # 色块全禁。
        tab._set_recording_locked(False)
        self.assertTrue(tab.save_button.isEnabled())  # 恢复。
        self.assertTrue(tab.replay_btn.isEnabled())  # 恢复。
        self.assertTrue(all(b.isEnabled() for b in tab._swatch_buttons))  # 恢复。


class TestKeySelfTest(MapTabTestCase):  # 按键自检（假 pynput，绝不装真钩子）。

    def test_receives_keys_reports_ok(self):  # 自检期间收到键 → 状态提示 OK 并回显键名。
        fake = FakeKeyboardModule()
        key_capture._pynput_keyboard = fake  # 替换依赖。
        tab = self.tab
        tab.on_key_self_test()
        self.assertIsNotNone(tab._key_test)  # 已持有临时捕获器。
        self.assertEqual(1, len(fake.created))  # 只装了一个假监听器。
        self.assertFalse(tab.key_test_btn.isEnabled())  # 防重复点。
        tab._key_test.press('a')  # 模拟收到一个键。
        tab._key_test.press('space')  # 又一个。
        tab._finish_key_self_test()  # 等价于 3 秒到点。
        self.assertIsNone(tab._key_test)  # 已释放。
        self.assertTrue(tab.key_test_btn.isEnabled())  # 按钮恢复。
        self.assertTrue(fake.created[0].stopped)  # 钩子已停。
        status = tab.record_status.text()
        self.assertIn('按键自检 OK', status)  # 结论。
        self.assertIn('a', status)  # 回显键名。

    def test_silent_hook_reports_privilege_hint(self):  # 一个键都没收到 → 提示改用管理员启动或关闭键盘捕获。
        key_capture._pynput_keyboard = FakeKeyboardModule()
        tab = self.tab
        tab.on_key_self_test()
        tab._finish_key_self_test()  # 期间没按任何键。
        status = tab.record_status.text()
        self.assertIn('一个键都没收到', status)  # 明确失败原因。
        self.assertIn('管理员', status)  # 给出可操作建议。

    def test_self_test_refused_while_recording(self):  # 录制中钩子已占用，拒绝再叠一个自检。
        tab = self.tab
        key_capture._pynput_keyboard = FakeKeyboardModule()
        recorder_started = []  # 记录被请求启动的假录制线程。

        class DummyRecorder:  # 只用来把页签置成"录制中"状态。
            finished = False

            def start(self):  # 空实现。
                recorder_started.append(True)

            def stop(self):  # 空实现。
                pass

        tab._recorder = DummyRecorder()
        self.addCleanup(setattr, tab, '_recorder', None)
        tab.on_key_self_test()
        self.assertIsNone(tab._key_test)  # 没启动自检钩子。
        self.assertIn('录制进行中', tab.record_status.text())  # 提示。


class TestReplayRoute(MapTabTestCase):  # 事件流离线重绘。

    def _write_events(self, events):  # 给当前路线写一份事件流 sidecar。
        map_store.save_route_events('ui_map', self.route_file, events,
                                    canvas=(90, 60), thickness=2)

    def test_replay_redraws_from_events_and_backs_up(self):  # 重绘用当前色表与界面粗细，覆盖前生成 .bak。
        blue = map_store.MAP_META_DEFAULTS['Color Code'].get('0,0,255')  # 默认蓝=向右。
        self.assertEqual('right none none', blue)  # 前置确认色表定义。
        events = [[0.0, 10, 30, 'right none none'], [0.1, 60, 30, 'right none none']]  # 一段水平线。
        self._write_events(events)
        tab = self.tab
        tab.current_route = self.route_file  # 选中这条路线。
        tab.route_combo.setCurrentText(self.route_file)
        tab.trace_thickness_spin.setValue(5)  # 改粗后重绘，验证无需重走地图。
        tab.on_replay_route()
        self.assertIn('已按事件流重绘', tab.record_status.text())  # 状态反馈。
        self.assertIn('route1.png.bak', tab.record_status.text())  # 告知备份文件名。
        route_img = map_store.load_route_image('ui_map', self.route_file)  # 读回重绘结果。
        self.assertEqual((60, 90), route_img.shape[:2])  # 尺寸与底图一致。
        self.assertEqual((0, 0, 255), tuple(int(v) for v in route_img[30, 35]))  # 线中间被加粗覆盖。
        backup = os.path.join(map_store.map_dir('ui_map'), self.route_file + '.bak')  # 备份文件。
        self.assertTrue(os.path.isfile(backup))  # 已生成。

    def test_replay_reports_unknown_command_skipped(self):  # 恢复默认色表后旧路线里的指令缺色，重绘要提示跳过条数。
        events = [[0.0, 10, 30, 'left up none'], [0.1, 20, 30, 'left up none']]  # 色表里没有的组合。
        self._write_events(events)
        tab = self.tab
        tab.current_route = self.route_file
        tab.on_replay_route()
        self.assertIn('跳过', tab.record_status.text())  # 明确告知有事件未画出。
        route_img = map_store.load_route_image('ui_map', self.route_file)  # 读回。
        self.assertTrue((route_img == 0).all())  # 全黑：未知指令不落成乱色。

    def test_replay_without_events_prompts(self):  # 手描模式录的（或旧的）路线没有事件流，提示无法重绘。
        tab = self.tab
        tab.current_route = self.route_file
        tab.on_replay_route()
        self.assertIn('没有按键事件流', tab.record_status.text())  # 提示。
        self.assertFalse(os.path.isfile(os.path.join(map_store.map_dir('ui_map'), self.route_file + '.bak')))  # 未动旧图。

    def test_preview_info_counts_redrawable_routes(self):  # 概要卡显示可重绘路线数量。
        self._write_events([[0.0, 10, 30, 'right none none']])  # 只有当前这条有 sidecar。
        tab = self.tab
        tab.current_route = self.route_file
        tab._update_preview_info()
        self.assertIn('可重绘（含按键事件流）: 1/1 条', tab.preview_info.text())  # 计数可见。

    def test_preview_info_warns_on_undefined_route_color(self):  # 路线上出现色表未定义的色要在概要里提示。
        img = np.zeros((60, 90, 3), dtype=np.uint8)  # 黑底路线图。
        img[30, 20] = (9, 9, 9)  # 写一个色表里没有的颜色。
        map_store.save_route_image('ui_map', self.route_file, img)
        tab = self.tab
        tab.current_route = self.route_file
        tab._update_preview_info()
        self.assertIn('色表未定义的色', tab.preview_info.text())  # 提示存在未知色。
        self.assertIn('9,9,9', tab.preview_info.text())  # 具体颜色可见。


if __name__ == '__main__':
    unittest.main()
