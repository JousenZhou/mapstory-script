# DashboardTab 无头冒烟测试：验证怪物多选下拉（MultiSelectComboBox）的数据逻辑、逗号分隔文本解析，
# 以及角色三字段改为单选下拉、怪物改为多选下拉后看板页签能正确构造并按类别刷新标注选项。
# 仅做只读构造与控件状态检查，不调用 save() 以免写盘；构造依赖仓库内既有的 Dashboard.json 与 coco_annotations.json。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import unittest

from PySide6.QtWidgets import QApplication


class TestMultiSelectComboBox(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # 全部用例共享一个离屏应用实例。

    def test_parse_feature_list(self):
        from src.ui.DashboardTab import parse_feature_list
        self.assertEqual([], parse_feature_list(''))  # 空串解析为空列表。
        self.assertEqual([], parse_feature_list(None))  # None 解析为空列表。
        self.assertEqual(['a', 'b'], parse_feature_list(' a , , b '))  # 去空白项与首尾空格。

    def test_default_placeholder_and_value(self):
        from src.ui.DashboardTab import PLACEHOLDER, MultiSelectComboBox
        box = MultiSelectComboBox()
        self.assertEqual(PLACEHOLDER, box.text())  # 未选任何项时显示占位文本。
        self.assertEqual([], box.value())  # 已选集合为空。
        self.assertEqual('', box.value_text())  # 逗号拼接为空串。

    def test_set_value_and_text(self):
        from src.ui.DashboardTab import MultiSelectComboBox
        box = MultiSelectComboBox()
        box.set_options(['slime', 'bat', 'bee'])
        box.set_value(['slime'])
        self.assertEqual(['slime'], box.value())  # 单选取值。
        self.assertEqual('slime', box.value_text())  # 单项拼接即自身。
        self.assertEqual('slime', box.text())  # 单项直接显示名称。
        box.set_value(['slime', 'bat'])
        self.assertEqual('slime,bat', box.value_text())  # 多项以英文逗号拼接。
        self.assertIn('2', box.text())  # 多项时按钮显示数量摘要。
        self.assertIn('、', box.toolTip())  # 悬浮提示列出全部已选项。

    def test_toggle_updates_selection(self):
        from src.ui.DashboardTab import MultiSelectComboBox
        box = MultiSelectComboBox()
        box.set_options(['slime', 'bat'])
        box._on_toggle('slime', True)  # 勾选。
        self.assertEqual(['slime'], box.value())
        box._on_toggle('bat', True)
        self.assertEqual(['slime', 'bat'], box.value())
        box._on_toggle('slime', True)  # 重复勾选不产生重复项。
        self.assertEqual(['slime', 'bat'], box.value())
        box._on_toggle('slime', False)  # 取消勾选。
        self.assertEqual(['bat'], box.value())
        box._on_toggle('ghost', False)  # 取消未选项不报错。
        self.assertEqual(['bat'], box.value())

    def test_menu_items_keep_historical_selection(self):
        from src.ui.DashboardTab import MultiSelectComboBox
        box = MultiSelectComboBox()
        box.set_value(['ghost'])  # 历史配置里的分类名。
        box.set_options(['slime', 'bat'])  # 刷新标注后 ghost 不在可选项里。
        items = box._menu_items()
        self.assertIn('ghost', items)  # 历史选中值仍在菜单里，保证能取消勾选。
        self.assertIn('slime', items)
        self.assertEqual(['ghost'], box.value())  # 刷新可选项不会丢已选值。


class TestDashboardTabWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_character_combos_and_monster_multiselect(self):
        from qfluentwidgets import EditableComboBox
        from src.ui.DashboardTab import DashboardTab, MultiSelectComboBox
        tab = DashboardTab()
        try:
            # 角色特征/左朝向/右朝向都应是单选下拉。
            self.assertIsInstance(tab.char_feature_combo, EditableComboBox)
            self.assertIsInstance(tab.char_facing_left_combo, EditableComboBox)
            self.assertIsInstance(tab.char_facing_right_combo, EditableComboBox)
            # 怪物特征应是多选下拉。
            self.assertIsInstance(tab.monster_multi, MultiSelectComboBox)
            self.assertIsInstance(tab.monster_multi.value(), list)  # 取值为列表。
            # 按类别刷新标注选项不应报错，且刷新后取值类型不变。
            tab.reload_annotations()
            self.assertIsInstance(tab.monster_multi.value_text(), str)  # 保存用逗号字符串。
        finally:
            tab.timer.stop()  # 停掉画面刷新定时器，避免测试结束后回调已销毁控件。

    def test_precision_and_delay_controls_removed(self):
        # lievision 移植后精度五档与触发延迟全面废弃：看板不再暴露相关控件，不再读写相关配置键，旧残留键静默忽略。
        from unittest.mock import MagicMock, patch
        from src.dashboard_store import DASHBOARD_DEFAULTS
        from src.ui.DashboardTab import DashboardTab
        self.assertNotIn('Lie Detector Precision', DASHBOARD_DEFAULTS)  # 默认配置字典已删精度键。
        self.assertNotIn('Lie Detector Trigger Delay', DASHBOARD_DEFAULTS)  # 默认配置字典已删延迟键。
        tab = DashboardTab()
        try:
            self.assertFalse(hasattr(tab, 'lie_delay_spin'))  # 触发延迟输入框已移除。
            self.assertFalse(hasattr(tab, 'lie_precision_combo'))  # 精度下拉已移除。
            with patch('src.ui.DashboardTab.load_dashboard_config', return_value={'Lie Detector Precision': 'ultra', 'Lie Detector Trigger Delay': 5.0}):
                tab.load_config()  # 旧 Dashboard.json 残留键：加载不报错即可（静默忽略）。
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()) as save:
                tab.save()
            saved = save.call_args[0][0]
            self.assertNotIn('Lie Detector Precision', saved)  # 保存不再写回精度键。
            self.assertNotIn('Lie Detector Trigger Delay', saved)  # 保存不再写回延迟键。
        finally:
            tab.timer.stop()  # 同样停掉定时器。

    def test_no_gpu_switch_exposed(self):
        # 显卡加速已改为隐藏式（不设开关）：看板不再暴露 GPU Match 控件，也不再写入相关配置键。
        from unittest.mock import MagicMock, patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            self.assertFalse(hasattr(tab, 'lie_gpu_switch'))  # 测谎显卡开关已移除。
            self.assertFalse(hasattr(tab, 'auto_login_gpu_switch'))  # 自动登录显卡开关已移除。
            tab.load_config()
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()) as save:
                tab.save()
            saved = save.call_args[0][0]  # 取出组装落盘的配置字典。
            self.assertNotIn('Lie Detector GPU Match', saved)  # 不再写入测谎显卡开关键。
            self.assertNotIn('Auto Login GPU Match', saved)  # 不再写入自动登录显卡开关键。
        finally:
            tab.timer.stop()  # 同样停掉定时器。

    def test_lie_backend_label_shows_service_backend(self):
        # 实战服务已就绪时，「后端」标签直接显服务解析出的测谎后端（只读，不可调、不落盘）。
        from unittest.mock import MagicMock, patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            lie_service = MagicMock()
            lie_service._backend_desc.return_value = "GPU · TensorRT INT8（缓存引擎）"
            with patch('src.ui.DashboardTab.og') as og_mock:  # 伪框架全局：my_app.lie_service 指向假服务。
                og_mock.my_app.lie_service = lie_service
                tab._refresh_lie_backend()
            self.assertEqual("后端: GPU · TensorRT INT8（缓存引擎）", tab.lie_backend_label.text())
            lie_service._backend_desc.return_value = "装配中"
            with patch('src.ui.DashboardTab.og') as og_mock, \
                    patch('src.ui.DashboardTab.QTimer.singleShot') as single_shot:  # 装配中瞬态必须自调度补刷新，就绪后才能自动换成生效后端。
                og_mock.my_app.lie_service = lie_service
                tab._refresh_lie_backend()
            self.assertEqual("后端: 装配中", tab.lie_backend_label.text())  # 后台线程未装配完时给预期提示。
            single_shot.assert_called_once_with(3000, tab, tab._refresh_lie_backend)
        finally:
            tab.timer.stop()

    def test_lie_backend_label_falls_back_to_probe(self):
        # 服务未启动（无窗口/单测）时按 GPU 三探测给预期后端；探测异常不影响看板。
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            with patch('src.ui.DashboardTab.og') as og_mock, \
                    patch('src.liedetector.vision.device.gpu_option_state', return_value=SimpleNamespace(enabled=True, reason="")):
                og_mock.my_app = None  # 没起实战服务：走探测分支。
                tab._refresh_lie_backend()
            self.assertIn("GPU · TensorRT INT8", tab.lie_backend_label.text())  # GPU 可用给待装配预期。
            with patch('src.ui.DashboardTab.og') as og_mock, \
                    patch('src.liedetector.vision.device.gpu_option_state', return_value=SimpleNamespace(enabled=False, reason="无 NVIDIA 显卡\n细节")):
                og_mock.my_app = None
                tab._refresh_lie_backend()
            self.assertTrue(tab.lie_backend_label.text().startswith("后端: CPU（无 NVIDIA 显卡"))  # GPU 不可用给 CPU 与首行原因。
            with patch('src.ui.DashboardTab.og') as og_mock, \
                    patch('src.liedetector.vision.device.gpu_option_state', side_effect=RuntimeError("探测炸")):
                og_mock.my_app = None
                tab._refresh_lie_backend()  # 异常不外溢。
            self.assertEqual("后端: --", tab.lie_backend_label.text())  # 异常兜底占位。
        finally:
            tab.timer.stop()

    def test_lie_abort_key_wired_and_validated(self):
        # 测谎栏应有「急停按键」输入框：能从看板配置读写往返，非法按键名被 save() 拦截，留空允许（停用）。
        from unittest.mock import MagicMock, patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            self.assertTrue(hasattr(tab, 'lie_abort_edit'))  # 急停按键输入框已创建。
            with patch('src.ui.DashboardTab.load_dashboard_config', return_value={'Lie Detector Abort Key': 'f8'}):
                tab.load_config()
            self.assertEqual('f8', tab.lie_abort_edit.text())  # 配置值填入输入框。
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()) as save:
                tab.save()
            self.assertEqual('f8', save.call_args[0][0]['Lie Detector Abort Key'])  # 保存写回所选按键。
            tab.lie_abort_edit.setText('not_a_key')  # 非法按键名。
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()) as save_bad:
                tab.save()
            save_bad.assert_not_called()  # 校验不通过，阻止落盘。
            tab.lie_abort_edit.setText('')  # 留空表示停用急停。
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()) as save_empty:
                tab.save()
            self.assertEqual('', save_empty.call_args[0][0]['Lie Detector Abort Key'])  # 留空允许保存。
        finally:
            tab.timer.stop()

    def test_sync_abort_notice_flips_switch_off(self):
        # 服务发出急停通知后，看板同步应把「自动解测谎」开关置为关；无通知时不动开关。
        from unittest.mock import MagicMock, patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            fake_service = MagicMock()
            fake_service.consume_abort_notice.return_value = True  # 服务报告已发生急停。
            tab.lie_auto_switch.setChecked(True)  # 假设开关本来是开的。
            with patch('src.ui.DashboardTab.og') as og:
                og.my_app.lie_service = fake_service
                tab._sync_abort_notice()
            self.assertFalse(tab.lie_auto_switch.isChecked())  # 开关被同步为关。
            fake_service.consume_abort_notice.return_value = False  # 无通知时不动开关。
            tab.lie_auto_switch.setChecked(True)
            with patch('src.ui.DashboardTab.og') as og:
                og.my_app.lie_service = fake_service
                tab._sync_abort_notice()
            self.assertTrue(tab.lie_auto_switch.isChecked())  # 无通知则保持原状。
        finally:
            tab.timer.stop()

    def test_buttons_removed_and_autosave_wired(self):
        # 底部「刷新标注」「保存」按钮已移除（保留状态提示）；变动经防抖定时器调度保存，加载守卫期间不调度。
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            self.assertFalse(hasattr(tab, 'save_button'))  # 保存按钮已移除。
            self.assertFalse(hasattr(tab, 'refresh_button'))  # 刷新标注按钮已移除。
            self.assertTrue(hasattr(tab, 'status_label'))  # 状态提示保留。
            self.assertTrue(tab._save_timer.isSingleShot())  # 防抖定时器单次触发。
            tab._loading = False  # 非加载期。
            tab._save_timer.stop()
            tab._schedule_save()  # 模拟控件变动。
            self.assertTrue(tab._save_timer.isActive())  # 已调度一次防抖保存。
            tab._save_timer.stop()
            tab._loading = True  # 加载期（程序回填）。
            tab._schedule_save()
            self.assertFalse(tab._save_timer.isActive())  # 守卫屏蔽，不调度。
        finally:
            tab._loading = False
            tab.timer.stop()
            tab._save_timer.stop()

    def test_control_change_schedules_save(self):
        # 改动数字框/开关/怪物多选都经信号触发防抖保存调度（无需点保存按钮）。
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            tab._save_timer.stop()
            tab.lie_threshold_spin.setValue(0.42)  # 改阈值（与默认 0.75 不同，必触发 valueChanged）。
            self.assertTrue(tab._save_timer.isActive())  # valueChanged -> _schedule_save。
            tab._save_timer.stop()
            tab.lie_auto_switch.setChecked(not tab.lie_auto_switch.isChecked())  # 翻转开关。
            self.assertTrue(tab._save_timer.isActive())  # checkedChanged -> _schedule_save。
            tab._save_timer.stop()
            tab.monster_multi._on_toggle('__test_mon__', True)  # 勾选怪物多选。
            self.assertTrue(tab._save_timer.isActive())  # on_changed -> _schedule_save。
        finally:
            tab.timer.stop()
            tab._save_timer.stop()

    def test_poll_reloads_annotations_on_coco_change(self):
        # 标注文件指纹变化时，轮询自动 reload_annotations（代替原「刷新标注」按钮）。
        from unittest.mock import patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            with patch.object(tab, 'reload_annotations') as reload_mock, \
                    patch('src.ui.DashboardTab.coco_fingerprint', return_value=('stale', 0)):
                tab._last_coco_fp = ('current', 1)  # 基线与当前指纹不一致，模拟标注被改。
                tab._poll_external_changes()
            reload_mock.assert_called_once()  # 触发了标注刷新。
        finally:
            tab.timer.stop()
            tab._save_timer.stop()

    def test_poll_reloads_config_unless_editing(self):
        # 配置被服务端写回（指纹变）时轮询自动 load_config 回填 UI；但用户正在文本框输入时暂缓，避免冲掉未提交内容。
        from unittest.mock import patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            with patch.object(tab, 'load_config') as load_mock, \
                    patch.object(tab, '_editing_text', return_value=False), \
                    patch('src.ui.DashboardTab.config_fingerprint', return_value=('stale', 0)):
                tab._last_config_fp = ('current', 1)  # 基线与当前不一致，模拟服务端写回。
                tab._poll_external_changes()
            load_mock.assert_called_once()  # 未编辑：回填 UI。
            with patch.object(tab, 'load_config') as load_mock2, \
                    patch.object(tab, '_editing_text', return_value=True), \
                    patch('src.ui.DashboardTab.config_fingerprint', return_value=('stale2', 0)):
                tab._last_config_fp = ('current2', 1)
                tab._poll_external_changes()
            load_mock2.assert_not_called()  # 编辑中：暂缓回填。
        finally:
            tab.timer.stop()
            tab._save_timer.stop()

    def test_save_updates_config_fingerprint_baseline(self):
        # save() 写盘后更新配置指纹基线，避免轮询把自己的写当成外部改动又回填。
        from unittest.mock import MagicMock, patch
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            tab._last_config_fp = ('old', 0)
            with patch('src.ui.DashboardTab.save_dashboard_config', MagicMock()), \
                    patch('src.ui.DashboardTab.config_fingerprint', return_value=('new', 1)):
                tab.save()
            self.assertEqual(('new', 1), tab._last_config_fp)  # 基线更新为写盘后的指纹。
        finally:
            tab.timer.stop()
            tab._save_timer.stop()


if __name__ == "__main__":
    unittest.main()
