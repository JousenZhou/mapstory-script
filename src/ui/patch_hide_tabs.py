# 屏蔽主窗口的【脚本】【开发工具】【运行代码】【关于】【任务】五个默认菜单。
#
# 框架 MainWindow 会按 debug 模式 / 自定义脚本情况添加 DebugTab、RunCodeTab、EditTaskTab、
# AboutTab 等页签，本项目不使用这些功能；此外【任务】页（OneTimeTaskTab）的挂机/巡逻两个脚本
# 已迁移进看板页签的任务控制栏（见 src/ui/DashboardTaskPanel.py），原导航项一并屏蔽：
# 包装 MainWindow.addSubInterface，导航项文本命中屏蔽名单时跳过加入导航。
# 页签对象本身仍被框架正常构造（如 MainWindow 后续还引用 self.about_tab.update_card），
# 只是不进导航栏，无其他副作用。
#
# 采用运行时猴子补丁（与 patch_window_selector.py 模式一致），不直接改 site-packages，
# 重装依赖后仍生效。只需在启动 GUI 之前 import 本模块。

from qfluentwidgets import NavigationItemPosition

from ok.ui.qt.MainWindow import MainWindow

# 需要屏蔽的导航项文本：同时覆盖 zh_CN 与默认英文两种语言下的文案。
_HIDDEN_TAB_TEXTS = {
    '脚本', 'Script',
    '开发工具', 'Debug',
    '运行代码', 'Run Code',
    '关于', 'About',
    '任务', 'Tasks',  # 挂机/巡逻脚本已迁移到看板页签的任务控制栏，屏蔽原【任务】导航项。
}

_original_add_sub_interface = MainWindow.addSubInterface


def _patched_add_sub_interface(self, interface, icon, text,
                               position=NavigationItemPosition.TOP, parent=None, isTransparent=False):
    if str(text) in _HIDDEN_TAB_TEXTS:  # 命中文案即不进导航，页签对象仍由框架构造供后续引用。
        return None
    return _original_add_sub_interface(self, interface, icon, text,
                                       position=position, parent=parent, isTransparent=isTransparent)


MainWindow.addSubInterface = _patched_add_sub_interface
