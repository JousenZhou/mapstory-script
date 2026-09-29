# 看板任务控制栏：把原【任务】页的两个脚本迁移进 Dashboard，做成手风琴卡片 + 紧凑配置网格 + 内嵌运行日志。
#
# 结构（自上而下）：
#   1. 每个任务一张 TaskAccordionCard（继承 qfluentwidgets.ExpandSettingCard）：
#      - 卡片头：任务图标 + 任务名 + 实时状态 + 启动/暂停/停止按钮 + 展开箭头（点击头部手风琴展开/收起）；
#      - 卡片体：角色配置那种尺寸的紧凑网格（每行两对「标签 + 控件」），控件直接读写 task.config，改动即校验落盘。
#   2. 一个 RunLogPanel：内嵌读取 logs/ok-script.log 尾部，带级别筛选与清空，自动跟随滚动。
#
# 任务实例通过 og.executor 精确按类取（type is cls，避免 MaplePatrolTask 继承 MapleIdleTask 造成误匹配）；
# 启停复用框架 TaskCard 的语义：启动走 og.app.start_controller.start(task)，停止走 task.disable()+unpause()，
# 暂停走 task.pause()；状态变化由 communicate.task 信号驱动刷新（该信号可能传 None，需兼容）。
import codecs  # 增量解码日志字节流，跨行 UTF-8 不会被截断成乱码。
import html  # 转义日志文本，防止 HTML 注入破坏渲染。
import os  # 拼接日志文件路径。
import re  # 解析日志行首的级别字段。
from collections import deque  # 有界队列保存日志记录，超出上限自动丢弃最旧的。
from pathlib import Path  # 日志文件路径对象。

from PySide6.QtCore import Qt, QTimer  # 定时器与对齐常量。
from PySide6.QtGui import QFont, QTextCursor  # 日志等宽字体与文本光标移动枚举。
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QTextEdit, QVBoxLayout, QWidget)  # 布局与日志视图控件。
from qfluentwidgets import (BodyLabel, ComboBox, ExpandSettingCard, FluentIcon,  # Fluent 控件。
                            LineEdit, PrimaryPushButton, PushButton, SwitchButton, isDarkTheme)

from ok import Logger, og  # 全局对象与日志器。
from src.ui.spin_wheel_guard import DoubleSpinBox, SpinBox  # 数字框改用滚轮守卫子类：需点击聚焦后滚轮才生效，避免滚动页面误改数值。
from ok.core.events import communicate  # 应用事件总线：订阅任务状态与配置校验事件。

from src.tasks.MapleIdleTask import MapleIdleTask  # 挂机任务类，用于按类取实例。
from src.tasks.MaplePatrolTask import MaplePatrolTask  # 巡逻任务类，用于按类取实例。
from src.tasks.MapleSingleSpotTask import MapleSingleSpotTask  # 单点挂机任务类，用于按类取实例。

logger = Logger.get_logger(__name__)  # 本模块日志器。

LOG_FILE = Path(os.getcwd()) / "logs" / "ok-script.log"  # 运行日志文件，与框架 LogWindow 同源。
LOG_MAX_RECORDS = 1500  # 日志面板最多保留的记录条数，超出丢弃最旧的，约束内存与渲染量。
LOG_POLL_MS = 150  # 日志轮询间隔毫秒数，兼顾实时性与开销。
LOG_TAIL_CHUNK = 256 * 1024  # 首次加载日志尾部时最多回读的字节数。
LOG_INITIAL_LINES = 200  # 面板打开时预加载的历史日志行数，提供上下文。
STATUS_POLL_MS = 600  # 任务实时状态文本的轮询间隔毫秒数（info_set 不发信号，只能轮询读取）。

# 日志行首级别解析：与框架 LogWindow 用同一条正则，识别 DEBUG/INFO/WARNING/ERROR/CRITICAL。
LOG_LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+")

# 级别到筛选阈值的映射，用于「只显示某级别及以上」。
LOG_LEVELS = {"ALL": 0, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# 浮点配置项的显示规格：键 -> (小数位, 下限, 上限, 步进)。未列出的键用通用默认值。
# 与 MyBaseTask 对 LabelAndDoubleSpinBox 的补丁保持一致：位移时长要 3 位小数，Del 间隔类要放大上限。
_FLOAT_SPEC = {
    "Move Interval": (1, 0.0, 86400.0, 1.0),
    "Move Away Seconds": (3, 0.0, 999.0, 0.05),
    "Move Back Seconds": (3, 0.0, 999.0, 0.05),
    "Turn Interval": (1, 0.0, 86400.0, 1.0),
    "Frame Interval": (3, 0.0, 10.0, 0.01),
    "Minimap Threshold": (2, 0.01, 1.0, 0.05),
    "Patrol Left Percent": (1, 0.0, 100.0, 1.0),
    "Patrol Right Percent": (1, 0.0, 100.0, 1.0),
    "Return Offset Max Percent": (1, 0.0, 100.0, 0.5),
    "Stuck Seconds": (1, 0.0, 999.0, 0.5),
    "Resume Wait Seconds": (1, 0.0, 999.0, 0.5),
}

# 整数配置项的取值范围：键 -> (下限, 上限, 步进)。范围与任务 validate_config 的校验一致，非法值根本无法输入。
_INT_SPEC = {
    "Dot Hue Min": (0, 179, 1),  # OpenCV HSV 色相范围 0-179。
    "Dot Hue Max": (0, 179, 1),
    "Dot Min Pixels": (1, 100000, 1),  # 面积至少 1 像素。
}

# 字符串配置项的占位提示，引导用户填对格式。
_STR_PLACEHOLDER = {
    "Minimap Feature": "标注分类名",
    "Map Rect": "x,y,w,h 如 5,20,90,75",
}

# 任务配置项的简短中文标签：项目 gettext 目录未翻译这些键，直接用英文键名对用户不友好，
# 这里按含义给出简短易懂的中文（未列出的键回退 _tr(key)），完整双语说明仍保留在悬浮提示里。
_CONFIG_LABEL_ZH = {
    # —— 挂机任务（MapleIdleTask）——
    "Move Interval": "位移间隔(秒)",
    "Move Away Seconds": "远离移动(秒)",
    "Move Back Seconds": "返回移动(秒)",
    "Turn Interval": "转身间隔(秒)",
    "Use Gray Scale": "灰度匹配",
    "Frame Interval": "帧间隔(秒)",
    # —— 巡逻任务（MaplePatrolTask）——
    "Patrol Enabled": "启用巡逻",
    "Minimap Feature": "小地图标注",
    "Minimap Threshold": "小地图阈值",
    "Map Rect": "地图区域(%)",
    "Patrol Left Percent": "左边界(%)",
    "Patrol Right Percent": "右边界(%)",
    "Dot Hue Min": "黄点色相下限",
    "Dot Hue Max": "黄点色相上限",
    "Dot Min Pixels": "黄点最小面积",
    "Stuck Seconds": "卡住判定(秒)",
    "Resume Wait Seconds": "恢复等待(秒)",
    # —— 单点挂机任务（MapleSingleSpotTask）——
    "Return Offset Max Percent": "碰撞偏移归位max(%)",
}

# 任务控制栏统一隐藏的底层配置键：这些变量对日常挂机没有调整价值，一律不在配置网格中渲染，
# 运行时沿用任务 default_config 的默认值（灰度匹配、帧间隔、地图区域与黄点取色/面积阈值）。
_HIDDEN_KEYS = frozenset({
    "Use Gray Scale",  # 灰度匹配。
    "Frame Interval",  # 帧间隔(秒)。
    "Map Rect",  # 地图区域(%)。
    "Dot Hue Min",  # 黄点色相下限。
    "Dot Hue Max",  # 黄点色相上限。
    "Dot Min Pixels",  # 黄点最小面积。
})


def _tr(text):  # 翻译文本：优先用 app 的 gettext 目录（任务名与配置键都在其中），app 尚未就绪时原样返回，与框架 start_controller.tr 的防御式写法一致。
    app = getattr(og, "app", None)  # 全局 app，页签构造期可能尚未赋值。
    return app.tr(text) if app is not None else text  # 有 app 用其翻译，否则回退原文。


def _write_config(task, key, value):  # 把控件值写回任务配置：Config.__setitem__ 会自动校验并落盘，校验失败时保持旧值。
    try:  # 写入异常（如配置校验抛错）不能让 UI 崩溃。
        task.config[key] = value  # 触发框架的校验 + 保存；非法值会被拒绝并由 MainWindow 弹出提示。
    except Exception as e:  # 兜底记录，避免信号回调里抛异常。
        logger.warning(f"write task config failed {key}={value}: {e}")  # 记录写入失败供排查。


def _build_control(task, key):  # 按配置值类型为单个键创建紧凑控件，并绑定读写；bool 必须先于 int 判断（bool 是 int 子类）。
    value = task.config.get(key)  # 当前配置值，决定控件类型与初值。
    if isinstance(value, bool):  # 布尔开关。
        control = SwitchButton()  # 与看板其他开关一致的滑动按钮。
        control.setChecked(bool(value))  # 设初值。
        control.checkedChanged.connect(lambda checked, k=key: _write_config(task, k, bool(checked)))  # 变更即写回。
        return control
    if isinstance(value, int):  # 整数输入。
        control = SpinBox()  # 整数微调框。
        low, high, step = _INT_SPEC.get(key, (0, 99999999, 1))  # 取该键范围，缺省用通用范围。
        control.setRange(low, high)  # 限制范围使非法值无法输入。
        control.setSingleStep(step)  # 步进。
        control.setValue(int(value))  # 设初值。
        control.valueChanged.connect(lambda v, k=key: _write_config(task, k, int(v)))  # 变更即写回。
        return control
    if isinstance(value, float):  # 浮点输入。
        control = DoubleSpinBox()  # 小数微调框。
        decimals, low, high, step = _FLOAT_SPEC.get(key, (2, 0.0, 999999.0, 0.05))  # 取该键的精度与范围。
        control.setDecimals(decimals)  # 小数位数。
        control.setRange(low, high)  # 范围。
        control.setSingleStep(step)  # 步进。
        control.setValue(float(value))  # 设初值。
        control.valueChanged.connect(lambda v, k=key: _write_config(task, k, float(v)))  # 变更即写回。
        return control
    control = LineEdit()  # 其余按字符串处理，用单行输入框。
    control.setText(str(value or ''))  # 设初值。
    if key in _STR_PLACEHOLDER:  # 有格式引导的键显示占位提示。
        control.setPlaceholderText(_STR_PLACEHOLDER[key])  # 占位文本。
    control.editingFinished.connect(lambda k=key, edit=control: _write_str_config(task, k, edit))  # 编辑结束写回并在被拒时回退。
    return control


def _write_str_config(task, key, edit):  # 字符串控件写回：校验被拒（如 Map Rect 格式错）时把输入框回退到配置里的实际值，避免显示未生效的文本。
    text = edit.text().strip()  # 用户输入的文本。
    _write_config(task, key, text)  # 尝试写回。
    actual = task.config.get(key)  # 写回后配置里的实际值。
    if str(actual if actual is not None else '') != text:  # 实际值与输入不一致，说明被校验拒绝。
        edit.setText(str(actual if actual is not None else ''))  # 回退输入框到真实生效值，MainWindow 已弹出校验失败提示。


class TaskAccordionCard(ExpandSettingCard):  # 单个任务的手风琴卡片：头部含状态与启停按钮，展开体是紧凑配置网格。

    def __init__(self, task, parent=None):  # 构造卡片。
        super().__init__(task.icon or FluentIcon.INFO, _tr(task.name), None, parent)  # 头部：图标 + 任务名，content 留空使头部保持 50px 紧凑高度。
        self.task = task  # 绑定的任务实例。
        self._last_status = None  # 上次显示的状态文本，变化时才刷新标签，避免无谓重排。
        self.viewLayout.setContentsMargins(16, 10, 16, 14)  # 展开体内边距。
        self.viewLayout.setSpacing(6)  # 展开体内控件间距。
        self._build_header_tail()  # 头部尾部：状态标签 + 启停按钮。
        self._build_config_grid()  # 展开体：紧凑配置网格。
        self.setExpand(False)  # 默认收起，保持看板紧凑。
        self.refresh()  # 按当前任务状态初始化按钮与状态文本。
        communicate.task.connect(self._on_task_event)  # 订阅任务状态事件，任意任务启停都刷新本卡。

    def _build_header_tail(self):  # 在头部展开箭头左侧放状态标签与启动/暂停/停止按钮。
        tail = QWidget()  # 尾部容器。
        layout = QHBoxLayout(tail)  # 横向排列。
        layout.setContentsMargins(0, 0, 0, 0)  # 去边距。
        layout.setSpacing(8)  # 控件间距。
        self.status_label = BodyLabel("")  # 实时状态文本。
        self.pause_button = PushButton(FluentIcon.PAUSE, "暂停")  # 暂停按钮。
        self.pause_button.clicked.connect(self.pause_clicked)  # 绑定暂停。
        self.stop_button = PushButton(FluentIcon.CANCEL, "停止")  # 停止按钮。
        self.stop_button.clicked.connect(self.stop_clicked)  # 绑定停止。
        self.start_button = PrimaryPushButton(FluentIcon.PLAY, "启动")  # 启动/恢复按钮（主色）。
        self.start_button.clicked.connect(self.start_clicked)  # 绑定启动。
        layout.addWidget(self.status_label)  # 状态在最左。
        layout.addWidget(self.pause_button)  # 暂停。
        layout.addWidget(self.stop_button)  # 停止。
        layout.addWidget(self.start_button)  # 启动在最右（紧邻展开箭头）。
        self.addWidget(tail)  # 交给 ExpandSettingCard 插入头部尾部（展开按钮之前）。

    def _build_config_grid(self):  # 用角色配置那种尺寸的紧凑网格渲染本任务的全部可编辑配置项：每行两对「标签 + 控件」。
        grid_widget = QWidget()  # 网格容器。
        grid = QGridLayout(grid_widget)  # 四列网格：标签、控件、标签、控件。
        grid.setContentsMargins(0, 0, 0, 0)  # 去外边距。
        grid.setHorizontalSpacing(10)  # 列间距。
        grid.setVerticalSpacing(8)  # 行间距。
        row, col = 0, 0  # 当前写入的行列（col 取 0/1 表示本行第几对）。
        for key in self.task.default_config:  # 遍历任务默认配置的键（已在任务构造时剔除搬到看板的共享键）。
            if key.startswith('_'):  # 下划线开头是框架内部键，不展示。
                continue
            if key in _HIDDEN_KEYS:  # 灰度匹配/帧间隔/地图区域/黄点取色等底层变量统一隐藏，运行时沿用默认值。
                continue
            label = BodyLabel(_CONFIG_LABEL_ZH.get(key) or _tr(key))  # 标签优先用简短中文，未收录的键回退翻译/原文。
            control = _build_control(self.task, key)  # 按类型创建绑定控件。
            desc = (self.task.config_description or {}).get(key)  # 该键的双语帮助文本。
            if desc:  # 有帮助文本时作为悬浮提示。
                label.setToolTip(desc)  # 标签悬浮显示完整说明。
                control.setToolTip(desc)  # 控件同样显示，方便对照。
            grid.addWidget(label, row, col * 2)  # 标签放偶数列。
            grid.addWidget(control, row, col * 2 + 1)  # 控件放奇数列。
            col += 1  # 下一对。
            if col == 2:  # 一行放满两对。
                col = 0  # 回到第一对。
                row += 1  # 换行。
        grid.setColumnStretch(1, 1)  # 第一对控件列可伸展。
        grid.setColumnStretch(3, 1)  # 第二对控件列可伸展。
        self.viewLayout.addWidget(grid_widget)  # 网格加入展开体。

    # ------------------------------------------------------------------ 启停控制（复用框架 TaskCard 语义）

    def start_clicked(self):  # 启动或恢复任务。
        self.setExpand(False)  # 启动即收起卡片，把空间留给画面与日志（与 TaskCard 行为一致）。
        if self.task.enabled and self.task.paused:  # 已启用且处于暂停：本次是恢复。
            self.task.unpause()  # 恢复运行。
            return  # 结束。
        first_run_alert = getattr(self.task, 'first_run_alert', None)  # 部分任务有首次运行确认弹窗。
        if first_run_alert and not self.task.config.get('_first_run_alert'):  # 需要确认且尚未确认过。
            from qfluentwidgets import Dialog  # 延迟导入对话框。
            dialog = Dialog(_tr('Alert'), _tr(first_run_alert), self.window())  # 构造确认框。
            dialog.yesButton.setText("确认")  # 确认按钮文案。
            dialog.cancelButton.setText("取消")  # 取消按钮文案。
            dialog.setContentCopyable(True)  # 允许复制内容。
            if dialog.exec():  # 用户确认。
                self.task.config['_first_run_alert'] = first_run_alert  # 记录已确认，下次不再弹。
            else:  # 用户取消。
                return  # 不启动。
        og.app.start_controller.start(self.task)  # 交给框架启动控制器：刷新设备、连接窗口、启用任务。

    def stop_clicked(self):  # 停止任务。
        self.task.disable()  # 禁用任务，执行器会移除它。
        self.task.unpause()  # 清除暂停态，确保执行器不再等待。

    def pause_clicked(self):  # 暂停任务。
        self.task.pause()  # 暂停当前任务。

    # ------------------------------------------------------------------ 状态刷新

    def _on_task_event(self, task=None):  # communicate.task 回调：事件可能针对本任务、其他任务或 None（全部结束）。
        if task is None or task is self.task:  # 只在与本卡相关时刷新，减少无关重绘。
            self.refresh()  # 刷新按钮与状态。

    def refresh(self):  # 按任务当前状态刷新启停按钮可见性与状态文本。
        self._refresh_buttons()  # 刷新按钮。
        self._refresh_status()  # 刷新状态文本。

    def _refresh_buttons(self):  # 按钮可见性规则与框架 TaskCard.update_buttons 一致。
        if self.task.enabled:  # 任务已启用。
            if self.task.paused:  # 暂停中：显示「恢复 + 停止」。
                self.start_button.setText("恢复")  # 启动按钮变恢复。
                self.start_button.setVisible(True)  # 显示。
                self.pause_button.setVisible(False)  # 隐藏暂停。
                self.stop_button.setVisible(True)  # 显示停止。
            elif self.task.running:  # 运行中：显示「暂停 + 停止」。
                self.start_button.setVisible(False)  # 隐藏启动。
                self.pause_button.setVisible(True)  # 显示暂停。
                self.stop_button.setVisible(True)  # 显示停止。
            else:  # 已启用但排队等待：只显示「停止」。
                self.start_button.setVisible(False)  # 隐藏启动。
                self.pause_button.setVisible(False)  # 隐藏暂停。
                self.stop_button.setVisible(True)  # 显示停止。
        else:  # 未启用：只显示「启动」。
            self.start_button.setText("启动")  # 按钮文案复位为启动。
            self.start_button.setVisible(True)  # 显示启动。
            self.pause_button.setVisible(False)  # 隐藏暂停。
            self.stop_button.setVisible(False)  # 隐藏停止。

    def _status_text(self):  # 依据任务状态与实时 info 计算状态文本（含运行中任务用 info_set 写入的 Status）。
        if self.task.enabled:  # 已启用。
            if self.task.paused:  # 暂停。
                return "已暂停", "#e6a23c"  # 文案 + 琥珀色。
            if self.task.running:  # 运行中：附带任务实时状态（如 Attacking / Patrolling right）。
                live = self.task.info.get("Status") if isinstance(self.task.info, dict) else None  # 读实时状态。
                text = "运行中" + (f" · {live}" if live else "")  # 拼接实时状态。
                return text, "#67c23a"  # 绿色。
            return "排队中", "#409eff"  # 排队中，蓝色。
        return "已停止", "#909399"  # 未启动，灰色。

    def _refresh_status(self):  # 状态文本变化时才更新标签与颜色，避免每次轮询都触发重排。
        text, color = self._status_text()  # 计算最新状态。
        if text == self._last_status:  # 与上次相同则跳过。
            return  # 不刷新。
        self._last_status = text  # 记录本次文本。
        self.status_label.setText(text)  # 更新文本。
        self.status_label.setStyleSheet(f"color: {color};")  # 按状态着色。


class RunLogPanel(QWidget):  # 内嵌运行日志面板：增量读取日志尾部，级别着色，支持筛选与清空，自动跟随滚动。

    def __init__(self, parent=None):  # 构造面板。
        super().__init__(parent)  # 初始化父类。
        self._records = deque(maxlen=LOG_MAX_RECORDS)  # 有界日志记录队列，每条 {"level", "text"}。
        self._position = 0  # 已读取到的文件字节位置，用于增量 seek。
        self._file_id = None  # 文件标识（设备号 + inode/创建时间），轮转时用于识别文件被替换。
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")  # 增量 UTF-8 解码器。
        self._pending = ""  # 尚未凑成整行的残余文本。
        self._follow = True  # 是否自动跟随滚动到底部，用户上滚查看历史时暂停跟随。
        self._updating = False  # 程序化滚动期间置位，避免被误判为用户滚动。
        self._log_visible = False  # 日志列表默认隐藏，由工具栏开关控制是否显示。
        self._build_ui()  # 构建工具栏与日志视图。
        self._load_initial_tail()  # 预加载历史日志尾部。
        self._render()  # 首次渲染。
        self._timer = QTimer(self)  # 轮询定时器。
        self._timer.setInterval(LOG_POLL_MS)  # 轮询间隔。
        self._timer.timeout.connect(self._poll)  # 绑定轮询。
        self._timer.start()  # 启动轮询。

    def _build_ui(self):  # 构建级别筛选下拉 + 清空按钮 + 只读日志视图。
        layout = QVBoxLayout(self)  # 纵向布局。
        layout.setContentsMargins(0, 0, 0, 0)  # 去外边距。
        layout.setSpacing(6)  # 间距。
        toolbar = QWidget()  # 工具栏容器。
        bar_layout = QHBoxLayout(toolbar)  # 横向排列。
        bar_layout.setContentsMargins(0, 0, 0, 0)  # 去边距。
        bar_layout.setSpacing(8)  # 间距。
        bar_layout.addWidget(BodyLabel("运行日志"))  # 标题。
        self.show_switch = SwitchButton()  # 日志列表显示开关，默认关闭（列表隐藏）。
        self.show_switch.setOffText("显示日志")  # 关闭态文案。
        self.show_switch.setOnText("显示日志")  # 开启态文案，状态由滑块位置区分。
        self.show_switch.setChecked(False)  # 默认不显示日志列表。
        self.show_switch.checkedChanged.connect(self._toggle_log)  # 切换显示/隐藏。
        bar_layout.addWidget(self.show_switch)  # 开关紧跟标题。
        self.level_combo = ComboBox()  # 级别筛选下拉。
        self.level_combo.addItems(["全部", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])  # 全部 + 各级别。
        self.level_combo.currentIndexChanged.connect(self._level_changed)  # 切换级别即重渲染。
        self.clear_button = PushButton(FluentIcon.DELETE, "清空")  # 清空按钮。
        self.clear_button.clicked.connect(self._clear)  # 绑定清空。
        bar_layout.addWidget(self.level_combo)  # 下拉。
        bar_layout.addStretch(1)  # 弹性空隙把清空按钮推到右侧。
        bar_layout.addWidget(self.clear_button)  # 清空。
        layout.addWidget(toolbar)  # 工具栏入布局。
        self.view = QTextEdit()  # 日志视图。
        self.view.setReadOnly(True)  # 只读。
        self.view.setLineWrapMode(QTextEdit.NoWrap)  # 不自动换行，长行横向滚动。
        self.view.setFixedHeight(170)  # 固定紧凑高度。
        font = QFont("Consolas")  # 等宽字体。
        font.setStyleHint(QFont.Monospace)  # 回退到系统等宽字体。
        font.setPointSize(9)  # 小字号保持紧凑。
        self.view.setFont(font)  # 应用字体。
        scrollbar = self.view.verticalScrollBar()  # 纵向滚动条。
        scrollbar.rangeChanged.connect(self._on_range_changed)  # 内容高度变化时按需跟随到底。
        scrollbar.sliderReleased.connect(self._sync_follow)  # 用户拖动释放后同步跟随状态。
        scrollbar.actionTriggered.connect(lambda _: QTimer.singleShot(0, self._sync_follow))  # 用户点击/滚轮后同步跟随状态。
        layout.addWidget(self.view)  # 视图入布局。
        self._apply_log_visibility(False)  # 初始按默认（隐藏日志列表）应用可见性。

    # ------------------------------------------------------------------ 日志读取

    def _make_record(self, line):  # 把一行日志解析成记录：匹配到级别用真实级别，续行归入上一条的级别由渲染层处理，这里默认 INFO。
        match = LOG_LINE_PATTERN.match(line)  # 解析行首级别。
        return {"level": match.group("level") if match else "INFO", "text": line}  # 返回记录。

    def _load_initial_tail(self):  # 打开面板时预加载日志尾部若干行，提供上下文，并把读取位置定位到文件末尾。
        if not LOG_FILE.exists():  # 日志文件还不存在。
            return  # 无历史可加载。
        try:  # 读文件可能因轮转/占用失败。
            with open(LOG_FILE, "rb") as file:  # 以二进制打开。
                file.seek(0, os.SEEK_END)  # 定位到末尾。
                end = file.tell()  # 文件大小。
                size = min(LOG_TAIL_CHUNK, end)  # 回读字节数。
                file.seek(end - size)  # 定位到回读起点。
                data = file.read(size)  # 读取尾部字节。
                self._position = end  # 记录已读位置。
                self._file_id = self._file_id_static(file)  # 记录文件标识。
        except OSError:  # 读取失败。
            return  # 放弃预加载。
        lines = data.decode("utf-8", errors="replace").splitlines()  # 解码并切行。
        if size < end and lines:  # 从文件中间起读时首行可能是残行。
            lines = lines[1:]  # 丢弃首个残行。
        for line in lines[-LOG_INITIAL_LINES:]:  # 只保留最近若干行。
            self._records.append(self._make_record(line))  # 入队。

    @staticmethod
    def _file_id_static(file):  # 计算文件标识：设备号 + inode（无 inode 时用创建时间），单独命名避免与实例属性 self._file_id 冲突。
        stat = os.fstat(file.fileno())  # 文件状态。
        inode = getattr(stat, "st_ino", 0)  # inode。
        return stat.st_dev, inode or getattr(stat, "st_ctime_ns", stat.st_ctime)  # 设备号 + inode/创建时间。

    def _poll(self):  # 定时轮询：增量读取新写入的日志字节，解析成记录并按当前级别追加/重渲染。
        if not LOG_FILE.exists():  # 文件暂不存在（如刚轮转）。
            return  # 本轮跳过。
        try:  # 读取可能因轮转短暂失败。
            with open(LOG_FILE, "rb") as file:  # 打开日志。
                file_id = self._file_id_static(file)  # 当前文件标识。
                file.seek(0, os.SEEK_END)  # 定位末尾。
                size = file.tell()  # 当前大小。
                if file_id != self._file_id or size < self._position:  # 文件被替换或被截断（轮转）。
                    self._position = 0  # 从头重读。
                    self._file_id = file_id  # 更新标识。
                    self._decoder.reset()  # 重置解码器。
                    self._pending = ""  # 清空残行。
                if size == self._position:  # 没有新内容。
                    return  # 本轮无更新。
                file.seek(self._position)  # 定位到上次读取处。
                data = file.read()  # 读取新增字节。
                self._position = file.tell()  # 更新读取位置。
        except OSError:  # 读取失败（占用/轮转）。
            return  # 本轮跳过。
        new_records = self._decode_records(data)  # 解码为记录列表。
        if not new_records:  # 无完整新行。
            return  # 结束。
        self._records.extend(new_records)  # 入队（超出上限自动丢弃最旧）。
        if not self._log_visible:  # 日志列表隐藏时只累积记录，跳过渲染，打开时再一次性重绘。
            return  # 不做无谓的 HTML 渲染。
        if self._selected_level() == "ALL":  # 未筛选级别时增量追加，避免整屏重渲染。
            self._append_incremental(new_records)  # 增量渲染新行。
        else:  # 已筛选级别时按筛选结果整体重渲染。
            self._render()  # 全量重渲染。

    def _decode_records(self, data):  # 把新增字节解码并按行切成记录，残行留到下次拼接。
        text = self._pending + self._decoder.decode(data, final=False)  # 拼接残行后增量解码。
        lines = text.splitlines()  # 切行。
        if text.endswith(("\n", "\r")):  # 以换行结尾说明没有残行。
            self._pending = ""  # 清空残行。
        else:  # 结尾是半行。
            self._pending = lines.pop() if lines else text  # 弹出半行留待下次。
        return [self._make_record(line) for line in lines]  # 逐行解析为记录。

    # ------------------------------------------------------------------ 日志渲染

    def _selected_level(self):  # 当前筛选级别名，索引 0 表示 ALL。
        index = self.level_combo.currentIndex()  # 当前下拉索引。
        return "ALL" if index <= 0 else self.level_combo.currentText()  # 返回级别名。

    @staticmethod
    def _palette():  # 按当前主题返回级别配色，深色/浅色各一套，与框架 LogWindow 观感一致。
        if isDarkTheme():  # 深色主题。
            return {"background": "#1e1f22", "text": "#dfe1e5", "DEBUG": "#8ab4f8", "INFO": "#dfe1e5",
                    "WARNING": "#f5c16c", "ERROR": "#ff867f", "CRITICAL": "#ff5f56"}  # 深色调色板。
        return {"background": "#ffffff", "text": "#202124", "DEBUG": "#1a73e8", "INFO": "#202124",
                "WARNING": "#9a6700", "ERROR": "#d93025", "CRITICAL": "#b3261e"}  # 浅色调色板。

    def _html_line(self, record, palette):  # 把一条记录渲染成带颜色的 HTML 行（转义文本防注入）。
        color = palette.get(record["level"], palette["INFO"])  # 该级别的颜色。
        return f'<span style="color:{color};">{html.escape(record["text"])}</span><br>'  # 返回一行 HTML。

    def _append_incremental(self, records):  # 增量把新记录追加到视图末尾，超过上限则整体重渲染裁剪。
        palette = self._palette()  # 当前配色。
        fragment = "".join(self._html_line(record, palette) for record in records)  # 拼接新行 HTML。
        self._updating = True  # 标记程序化更新，避免触发跟随误判。
        try:  # 保证标记复位。
            cursor = self.view.textCursor()  # 取文本光标。
            cursor.movePosition(QTextCursor.MoveOperation.End)  # 移到末尾（PySide6 用作用域枚举，不能用 cursor.End）。
            cursor.insertHtml(fragment)  # 插入新行。
        finally:  # 结束更新。
            self._updating = False  # 复位标记。
        if self.view.document().blockCount() > LOG_MAX_RECORDS:  # 视图块数超过上限。
            self._render()  # 整体重渲染，用有界队列裁掉最旧的。
            return  # 重渲染已处理滚动。
        self._scroll_tail()  # 跟随到底部。

    def _render(self):  # 按当前级别筛选全量重渲染日志视图。
        min_level = LOG_LEVELS[self._selected_level()]  # 筛选阈值。
        palette = self._palette()  # 当前配色。
        fragments = [self._html_line(record, palette) for record in self._records  # 逐条渲染。
                     if LOG_LEVELS.get(record["level"], 0) >= min_level]  # 只保留达到筛选级别的记录。
        self._updating = True  # 标记程序化更新。
        try:  # 保证标记复位。
            self.view.setHtml("".join(fragments))  # 一次性写入全部可见日志。
        finally:  # 结束更新。
            self._updating = False  # 复位标记。
        self._scroll_tail()  # 跟随到底部。

    def _scroll_tail(self):  # 若处于跟随状态则滚动到底部。
        if not self._follow or self._updating:  # 未跟随或正在程序化更新时不抢滚动。
            return  # 跳过。
        self._updating = True  # 标记，避免滚动回调反过来改跟随状态。
        try:  # 保证标记复位。
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())  # 滚到底。
        finally:  # 结束。
            self._updating = False  # 复位。

    def _sync_follow(self):  # 依据滚动条位置同步跟随状态：贴底才跟随，用户上滚查看历史时停止跟随。
        if self._updating:  # 程序化滚动期间不改变跟随状态。
            return  # 跳过。
        scrollbar = self.view.verticalScrollBar()  # 纵向滚动条。
        self._follow = scrollbar.value() >= scrollbar.maximum()  # 处于底部则跟随。

    def _on_range_changed(self, _min, _max):  # 内容高度变化：跟随状态下自动贴底。
        if self._follow and not self._updating:  # 跟随且非程序化更新。
            self._updating = True  # 标记。
            try:  # 保证复位。
                self.view.verticalScrollBar().setValue(_max)  # 贴底。
            finally:  # 结束。
                self._updating = False  # 复位。

    # ------------------------------------------------------------------ 交互

    def _level_changed(self):  # 级别筛选变化：按新级别重渲染。
        self._render()  # 全量重渲染。

    def _clear(self):  # 清空：丢弃全部记录并清空视图。
        self._records.clear()  # 清空记录队列。
        self._pending = ""  # 清空残行。
        self.view.clear()  # 清空视图。

    def _toggle_log(self, checked):  # 显示开关切换：控制日志列表（及其筛选/清空控件）显隐。
        self._apply_log_visibility(bool(checked))  # 应用可见性。

    def _apply_log_visibility(self, visible):  # 按开关状态显隐日志视图与相关控件；显示时全量重绘当前记录。
        self._log_visible = visible  # 记录当前可见状态，_poll 据此决定是否渲染。
        self.view.setVisible(visible)  # 日志列表本体。
        self.level_combo.setVisible(visible)  # 级别筛选下拉仅在有列表时才有意义。
        self.clear_button.setVisible(visible)  # 清空按钮同上。
        if visible:  # 打开列表时把已累积的记录一次性渲染出来。
            self._render()  # 全量重渲染。


class TaskControlPanel(QWidget):  # 看板任务控制栏容器：两张任务手风琴卡 + 一个内嵌运行日志面板。

    def __init__(self, parent=None):  # 构造容器并装配任务卡与日志面板。
        super().__init__(parent)  # 初始化父类。
        layout = QVBoxLayout(self)  # 纵向堆叠。
        layout.setContentsMargins(0, 0, 0, 0)  # 去外边距，交由外层卡片控制留白。
        layout.setSpacing(8)  # 卡片间距。
        self.cards = []  # 任务卡列表，供状态轮询遍历。
        for task in self._collect_tasks():  # 逐个任务建卡。
            card = TaskAccordionCard(task)  # 手风琴任务卡。
            self.cards.append(card)  # 记录。
            layout.addWidget(card)  # 入布局。
        self.log_panel = RunLogPanel()  # 内嵌运行日志面板。
        layout.addWidget(self.log_panel)  # 日志放最下。
        self._status_timer = QTimer(self)  # 实时状态轮询定时器。
        self._status_timer.setInterval(STATUS_POLL_MS)  # 轮询间隔。
        self._status_timer.timeout.connect(self._poll_status)  # 绑定轮询。
        self._status_timer.start()  # 启动轮询。

    @staticmethod
    def _collect_tasks():  # 从执行器按类精确取出要迁移的两个任务实例，取不到的跳过。
        executor = getattr(og, "executor", None)  # 全局执行器，构造期已由 do_init 赋值。
        if executor is None:  # 执行器不可用（异常场景）。
            return []  # 返回空，面板只剩日志。
        tasks = []  # 收集结果。
        for cls in (MapleIdleTask, MaplePatrolTask, MapleSingleSpotTask):  # 目标三个任务类，按挂机、巡逻、单点挂机顺序。
            for candidate in executor.onetime_tasks:  # 遍历已注册的一次性任务。
                if type(candidate) is cls:  # 精确匹配类，避免 MaplePatrolTask 因继承被 MapleIdleTask 命中。
                    tasks.append(candidate)  # 命中。
                    break  # 每个类只取一个。
        return tasks  # 返回任务实例列表。

    def _poll_status(self):  # 轮询刷新运行中任务的实时状态文本（info_set 不发信号，只能定时读取）。
        for card in self.cards:  # 遍历任务卡。
            card._refresh_status()  # 只刷新状态文本，变化才更新标签。
