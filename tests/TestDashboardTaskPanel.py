# DashboardTaskPanel 无头构造冒烟测试：在 offscreen QApplication 下验证运行日志面板、手风琴任务卡与容器
# 能正确构造与交互，覆盖日志解析/渲染/筛选/清空、按类型建控件、配置写回与按钮状态刷新等纯 UI 逻辑。
# 不依赖完整框架初始化：og.app 为 None 时 _tr 回退原文，og.executor 为 None 时容器只建日志面板。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import unittest

from PySide6.QtWidgets import QApplication


class _StubTask:
    """满足 TaskAccordionCard 构造与刷新所需的最小任务替身。"""

    def __init__(self):
        self.icon = None
        self.name = "Smoke Task 冒烟任务"
        self.default_config = {
            "Flag": True,  # bool -> SwitchButton（必须先于 int 判断）
            "Count": 5,  # int -> SpinBox
            "Ratio": 1.5,  # float -> DoubleSpinBox
            "Label": "abc",  # str -> LineEdit
            "_hidden": "x",  # 下划线开头，网格应跳过
        }
        self.config_description = {"Count": "an int", "Ratio": "a float"}
        self.config = dict(self.default_config)  # 普通字典即可满足控件读写（get/__setitem__/__getitem__）。
        self.enabled = False
        self.paused = False
        self.running = False
        self.info = {"Status": "Idle"}
        self.first_run_alert = None

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def disable(self):
        self.enabled = False


class TestDashboardTaskPanel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # 全部用例共享一个离屏应用实例。

    def test_run_log_panel_render_filter_clear(self):
        from src.ui.DashboardTaskPanel import RunLogPanel
        panel = RunLogPanel()
        try:
            warn = panel._make_record("2026-09-10 01:00:00,000 WARNING something happened")
            self.assertEqual("WARNING", warn["level"])  # 行首级别被正确解析。
            cont = panel._make_record("    continuation without level")
            self.assertEqual("INFO", cont["level"])  # 无级别续行默认归 INFO。
            panel._records.append(warn)
            panel._records.append(cont)
            panel._render()  # ALL 级别全量渲染。
            self.assertIn("something happened", panel.view.toPlainText())
            panel.level_combo.setCurrentIndex(panel.level_combo.findText("WARNING"))  # 切到 WARNING 及以上。
            panel._level_changed()
            text = panel.view.toPlainText()
            self.assertIn("something happened", text)  # WARNING 保留。
            self.assertNotIn("continuation without level", text)  # INFO 续行被过滤。
            panel._clear()
            self.assertEqual(0, len(panel._records))  # 记录清空。
            self.assertEqual("", panel.view.toPlainText())  # 视图清空。
        finally:
            panel._timer.stop()

    def test_decode_records_keeps_partial_line(self):
        from src.ui.DashboardTaskPanel import RunLogPanel
        panel = RunLogPanel()
        try:
            data = "2026-09-10 01:00:00,000 INFO line1\n2026-09-10 01:00:01,000 INFO lin".encode("utf-8")
            records = panel._decode_records(data)
            self.assertEqual(1, len(records))  # 只有 line1 完整成行。
            self.assertEqual("line1", records[0]["text"].split()[-1])
            self.assertTrue(panel._pending.endswith("lin"))  # 半行留到下次拼接。
        finally:
            panel._timer.stop()

    def test_append_incremental_and_poll(self):
        # 覆盖定时器 _poll 走到的增量追加路径：该路径用 QTextCursor.MoveOperation.End，
        # 无事件循环时不会自动触发，必须直接调用才能提前暴露枚举误用。
        from src.ui.DashboardTaskPanel import LOG_FILE, RunLogPanel
        panel = RunLogPanel()
        try:
            records = [panel._make_record("2026-09-10 02:00:00,000 INFO inc line A"),
                       panel._make_record("2026-09-10 02:00:01,000 ERROR inc line B")]
            panel._records.extend(records)
            panel._append_incremental(records)  # 关键：增量追加走 cursor 移到末尾再 insertHtml。
            text = panel.view.toPlainText()
            self.assertIn("inc line A", text)
            self.assertIn("inc line B", text)
            panel._poll()  # 增量轮询路径（读文件 + _file_id_static + 解码）不报错。
            if LOG_FILE.exists():
                with open(LOG_FILE, "rb") as file:
                    self.assertEqual(2, len(panel._file_id_static(file)))  # 文件标识是 (dev, inode/ctime) 二元组。
        finally:
            panel._timer.stop()

    def test_log_visibility_toggle(self):
        # 日志列表默认隐藏；打开开关后视图/筛选/清空控件可见并渲染已累积记录，关闭后重新隐藏。
        from src.ui.DashboardTaskPanel import RunLogPanel
        panel = RunLogPanel()
        try:
            self.assertFalse(panel._log_visible)  # 默认不显示日志列表。
            self.assertFalse(panel.view.isVisibleTo(panel))  # 视图初始隐藏。
            self.assertFalse(panel.level_combo.isVisibleTo(panel))  # 筛选下拉同步隐藏。
            self.assertFalse(panel.clear_button.isVisibleTo(panel))  # 清空按钮同步隐藏。
            panel._records.append(panel._make_record("2026-09-10 03:00:00,000 INFO toggle line"))  # 隐藏期间累积一条记录。
            panel.show_switch.setChecked(True)  # 打开显示开关。
            self.assertTrue(panel._log_visible)  # 可见状态置位。
            self.assertTrue(panel.view.isVisibleTo(panel))  # 视图显示。
            self.assertIn("toggle line", panel.view.toPlainText())  # 打开时全量渲染已累积记录。
            panel.show_switch.setChecked(False)  # 关闭显示开关。
            self.assertFalse(panel._log_visible)  # 可见状态复位。
            self.assertFalse(panel.view.isVisibleTo(panel))  # 视图重新隐藏。
        finally:
            panel._timer.stop()

    def test_build_control_by_type(self):
        from qfluentwidgets import DoubleSpinBox, LineEdit, SpinBox, SwitchButton
        from src.ui import DashboardTaskPanel as mod
        task = _StubTask()
        self.assertIsInstance(mod._build_control(task, "Flag"), SwitchButton)  # bool -> 开关。
        self.assertIsInstance(mod._build_control(task, "Count"), SpinBox)  # int -> 整数框。
        self.assertIsInstance(mod._build_control(task, "Ratio"), DoubleSpinBox)  # float -> 小数框。
        self.assertIsInstance(mod._build_control(task, "Label"), LineEdit)  # str -> 输入框。

    def test_write_config_and_str(self):
        from src.ui import DashboardTaskPanel as mod
        task = _StubTask()
        mod._write_config(task, "Count", 9)  # 写回整数。
        self.assertEqual(9, task.config["Count"])
        edit = mod.LineEdit()
        edit.setText("hello")
        mod._write_str_config(task, "Label", edit)  # 写回字符串，配置生效则不回退。
        self.assertEqual("hello", task.config["Label"])
        self.assertEqual("hello", edit.text())

    def test_task_accordion_card(self):
        from src.ui import DashboardTaskPanel as mod
        task = _StubTask()
        card = mod.TaskAccordionCard(task)
        try:
            card.refresh()  # 未启用：只显示 Start。
            self.assertTrue(card.start_button.isVisibleTo(card))
            self.assertFalse(card.stop_button.isVisibleTo(card))
            self.assertFalse(card.pause_button.isVisibleTo(card))
            task.enabled = True
            task.running = True
            card.refresh()  # 运行中：显示 Pause + Stop。
            self.assertTrue(card.pause_button.isVisibleTo(card))
            self.assertTrue(card.stop_button.isVisibleTo(card))
            text, _color = card._status_text()
            self.assertIn("Running", text)  # 状态文本含 Running。
            card.setExpand(True)  # 展开/收起不报错。
            card.setExpand(False)
        finally:
            mod.communicate.task.disconnect(card._on_task_event)  # 断开全局信号，避免回调已销毁控件。

    def test_control_panel_without_executor(self):
        from src.ui import DashboardTaskPanel as mod
        old = getattr(mod.og, "executor", None)
        mod.og.executor = None
        try:
            panel = mod.TaskControlPanel()
            self.assertEqual(0, len(panel.cards))  # 无 executor 时不建任务卡。
            self.assertIsNotNone(panel.log_panel)  # 日志面板始终存在。
            panel._status_timer.stop()
            panel.log_panel._timer.stop()
        finally:
            mod.og.executor = old


if __name__ == "__main__":
    unittest.main()
