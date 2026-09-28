# 数字微调框滚轮误触防护。
#
# 背景：qfluentwidgets 的 SpinBox / DoubleSpinBox 默认焦点策略是 WheelFocus——鼠标只要悬浮在
#       控件上滚动滚轮就会直接改值。滚动页面时经过这些数字框，极易把阈值、间隔等参数误改掉。
# 目标：统一改成「必须先点击控件、出现输入光标（获得焦点）之后，滚轮才能改值」；未聚焦时忽略滚轮，
#       并让它向上冒泡给外层滚动区域，页面照常滚动、数值保持不变。
#
# 用法：
#   1) 本项目自建的数字框：改用这里的 SpinBox / DoubleSpinBox 子类（替换 qfluentwidgets 同名类）。
#   2) 框架内部已创建、无法继承的数字框：调用 disable_wheel_until_focused(widget) 打补丁。
from PySide6.QtCore import QEvent, QObject, Qt
from qfluentwidgets import DoubleSpinBox as _FluentDoubleSpinBox
from qfluentwidgets import SpinBox as _FluentSpinBox


class _WheelGuardMixin:
    """滚轮守卫混入：仅当控件已获得焦点时才响应滚轮，否则忽略并向上冒泡给父级滚动区域。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # StrongFocus = 点击 + Tab 聚焦，但不含滚轮聚焦：滚轮因此不会再抢焦点，杜绝「悬浮即改值」。
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):  # Qt 虚函数，命名固定。
        if self.hasFocus():  # 已点击聚焦（有光标）：正常响应滚轮改值。
            super().wheelEvent(event)
        else:  # 未聚焦：忽略滚轮，事件向上冒泡，外层滚动区域照常滚动页面。
            event.ignore()


class SpinBox(_WheelGuardMixin, _FluentSpinBox):
    """整数微调框：滚轮需先点击聚焦才生效，避免滚动页面时误改数值。"""


class DoubleSpinBox(_WheelGuardMixin, _FluentDoubleSpinBox):
    """小数微调框：滚轮需先点击聚焦才生效，避免滚动页面时误改数值。"""


class _WheelBlocker(QObject):
    """事件过滤器：给无法继承的既有数字框（如框架内部控件）拦截未聚焦时的滚轮。"""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():  # 未聚焦时的滚轮。
            event.ignore()  # 标记忽略，阻止悬浮改值。
            return True  # 拦截该事件，不再投递给数字框。
        return super().eventFilter(watched, event)


_wheel_blocker = _WheelBlocker()  # 进程级单例过滤器，生命周期覆盖所有被过滤控件。


def disable_wheel_until_focused(widget):
    """给任意既有数字框打「聚焦后才响应滚轮」补丁（用于框架内部、无法继承的控件）。"""
    widget.setFocusPolicy(Qt.StrongFocus)  # 去掉滚轮抢焦点，必须点击/Tab 聚焦。
    widget.installEventFilter(_wheel_blocker)  # 失焦时拦截滚轮改值。
    return widget
