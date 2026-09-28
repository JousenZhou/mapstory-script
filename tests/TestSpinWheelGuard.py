# 数字微调框滚轮守卫的单元测试。
# 验证：滚轮不再抢焦点（StrongFocus）、未聚焦时滚轮不改值、聚焦后滚轮照常改值，
# 以及给框架既有控件打补丁的 disable_wheel_until_focused 同样能拦截未聚焦滚轮。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import unittest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication


def _make_wheel(dy=120):
    """构造一次真实鼠标滚轮 notch 事件（angleDelta.y=dy，NoScrollPhase）。"""
    return QWheelEvent(
        QPointF(5, 5), QPointF(5, 5),  # 局部/全局坐标。
        QPoint(0, 0), QPoint(0, dy),  # pixelDelta 空、angleDelta 竖直一格。
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)


class TestSpinWheelGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # 全部用例共享一个离屏应用实例。

    def test_focus_policy_is_strong_focus(self):
        # 守卫子类必须去掉滚轮抢焦点：只有点击/Tab 能聚焦，杜绝「悬浮即改值」。
        from src.ui.spin_wheel_guard import DoubleSpinBox, SpinBox
        for widget_cls in (SpinBox, DoubleSpinBox):
            spin = widget_cls()
            self.assertEqual(Qt.StrongFocus, spin.focusPolicy())  # 焦点策略为 StrongFocus（不含滚轮）。
            spin.deleteLater()

    def test_wheel_ignored_without_focus(self):
        # 未聚焦（未点击、无光标）时滚动滚轮：数值保持不变。
        from src.ui.spin_wheel_guard import DoubleSpinBox, SpinBox
        spin = SpinBox()
        spin.setRange(0, 100)
        spin.setSingleStep(1)
        spin.setValue(5)
        self.assertFalse(spin.hasFocus())  # 未显示即未聚焦。
        QApplication.sendEvent(spin, _make_wheel(120))  # 向上滚一格。
        self.assertEqual(5, spin.value())  # 整数框：未聚焦滚轮被忽略，值不变。
        spin.deleteLater()

        dspin = DoubleSpinBox()
        dspin.setRange(0.0, 100.0)
        dspin.setSingleStep(0.5)
        dspin.setValue(5.0)
        QApplication.sendEvent(dspin, _make_wheel(120))
        self.assertEqual(5.0, dspin.value())  # 小数框：未聚焦滚轮被忽略，值不变。
        dspin.deleteLater()

    def test_wheel_works_with_focus(self):
        # 已聚焦（点击出现光标）后滚动滚轮：数值照常改变，功能不被破坏。
        from src.ui.spin_wheel_guard import SpinBox

        class _ForceFocusSpinBox(SpinBox):  # 模拟已点击聚焦：强制 hasFocus 返回 True。
            def hasFocus(self):
                return True

        spin = _ForceFocusSpinBox()
        spin.setRange(0, 100)
        spin.setSingleStep(1)
        spin.setValue(5)
        QApplication.sendEvent(spin, _make_wheel(120))  # 聚焦后向上滚一格。
        self.assertNotEqual(5, spin.value())  # 值发生变化，证明聚焦后滚轮生效。
        spin.deleteLater()

    def test_disable_wheel_until_focused_patches_existing_widget(self):
        # 通用补丁：给无法继承的既有数字框（如框架内部控件）也能拦截未聚焦滚轮。
        from qfluentwidgets import SpinBox as FluentSpinBox
        from src.ui.spin_wheel_guard import disable_wheel_until_focused
        spin = FluentSpinBox()
        spin.setRange(0, 100)
        spin.setSingleStep(1)
        spin.setValue(5)
        disable_wheel_until_focused(spin)  # 打补丁。
        self.assertEqual(Qt.StrongFocus, spin.focusPolicy())  # 焦点策略被改为 StrongFocus。
        self.assertFalse(spin.hasFocus())
        QApplication.sendEvent(spin, _make_wheel(120))  # 未聚焦滚动。
        self.assertEqual(5, spin.value())  # 事件过滤器拦截滚轮，值不变。
        spin.deleteLater()

    def test_dashboard_task_panel_builds_guarded_controls(self):
        # 任务配置网格用 _build_control 建的整数/小数框必须是守卫子类（滚轮不误改）。
        from src.ui import spin_wheel_guard
        from tests.TestDashboardTaskPanel import _StubTask
        from src.ui.DashboardTaskPanel import _build_control
        task = _StubTask()
        int_control = _build_control(task, "Count")  # int -> 守卫整数框。
        float_control = _build_control(task, "Ratio")  # float -> 守卫小数框。
        self.assertIsInstance(int_control, spin_wheel_guard.SpinBox)
        self.assertIsInstance(float_control, spin_wheel_guard.DoubleSpinBox)
        self.assertEqual(Qt.StrongFocus, int_control.focusPolicy())
        self.assertEqual(Qt.StrongFocus, float_control.focusPolicy())
        int_control.deleteLater()
        float_control.deleteLater()


if __name__ == "__main__":
    unittest.main()
