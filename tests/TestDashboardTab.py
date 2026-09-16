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

    def test_lie_trigger_delay_spin_wired(self):
        # 测谎栏应有「触发延迟」输入框：范围 0~60 秒、一位小数，能从看板配置填值。
        from src.dashboard_store import DASHBOARD_DEFAULTS
        from src.ui.DashboardTab import DashboardTab
        tab = DashboardTab()
        try:
            self.assertEqual((0.0, 60.0), (tab.lie_delay_spin.minimum(), tab.lie_delay_spin.maximum()))
            self.assertEqual(1, tab.lie_delay_spin.decimals())  # 一位小数，与保存时的 round(..., 1) 一致。
            self.assertEqual(5.0, DASHBOARD_DEFAULTS['Lie Detector Trigger Delay'])  # 看板默认延迟 5 秒。
            tab.load_config()  # 读 configs/Dashboard.json，缺失键由 load_dashboard_config 补默认值。
            self.assertGreaterEqual(tab.lie_delay_spin.value(), 0.0)
            self.assertLessEqual(tab.lie_delay_spin.value(), 60.0)
        finally:
            tab.timer.stop()  # 同样停掉定时器。


if __name__ == "__main__":
    unittest.main()
