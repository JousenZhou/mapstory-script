# 看板页签：全局共享配置的唯一编辑入口 + 实时画面展示 + 任务控制。
#
# 纵向五栏：
#   1. 视图区：实时显示任务推送的带标注画面（含解测谎运算过程叠加），无任务时显示原始截图；
#   2. 任务控制栏：原【任务】页的挂机/巡逻两个脚本迁移至此，手风琴卡片 + 紧凑配置网格 + 内嵌运行日志；
#   3. 测谎栏：区域/触发两个按类别过滤的标注选择、裁剪预览、匹配阈值、报警音频与总开关；
#   4. 角色栏：角色特征/阈值、朝向模板、攻击键、近战键与距离、攻击范围、Del 间隔；
#   5. 怪物栏：怪物特征、阈值与镜像阈值。
# 保存写入 configs/Dashboard.json，任务启动时由 MyBaseTask.apply_shared_config() 采集（单一数据源）。
import cv2  # 导入 OpenCV，用于画面转换与区域裁剪预览。
from PySide6.QtCore import QPoint, Qt, QTimer  # 导入 Qt 定时器、对齐常量与坐标点（多选菜单弹出定位）。
from PySide6.QtGui import QImage, QPixmap  # 导入图像对象，用于把画面矩阵转成图片显示。
from PySide6.QtWidgets import (QFormLayout, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget)  # 导入布局与控件。
from qfluentwidgets import (BodyLabel, CheckBox, DoubleSpinBox, EditableComboBox, FluentIcon, LineEdit,  # 导入 Fluent 控件。
                            MenuAnimationType, PrimaryPushButton, PushButton, RoundMenu, SpinBox, SwitchButton)

from ok import og  # 导入全局对象，用于读取任务线程推送的画面与截图设备。
from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。

from src.dashboard_store import (DASHBOARD_DEFAULTS, SUPER_CHARACTER, SUPER_LIE_REGION, SUPER_LIE_TRIGGER,  # 看板共享配置存取层。
                                 SUPER_MONSTER, load_annotations_by_supercategory, load_dashboard_config,
                                 save_dashboard_config, validate_key_name)
from src.ui.DashboardTaskPanel import TaskControlPanel  # 任务控制栏：挂机/巡逻脚本手风琴卡 + 内嵌运行日志。

CAPTURE_FPS = 30  # 截图采集固定帧率：实时画面刷新与无任务时的截图取帧都按该节拍。
CAPTURE_INTERVAL_MS = round(1000 / CAPTURE_FPS)  # 帧间隔毫秒数（33ms）。
VISION_MAX_AGE = 5.0  # 带标注画面的最长可用秒数：任务循环偶尔超过 1 秒时继续沿用上一帧标注画面，而不是立即闪回原始截图，消除两种画面交替闪烁。
PLACEHOLDER = '(未选择)'  # 下拉框空选项占位文本，保存时映射回空字符串。


def to_pixmap(frame):  # 把 OpenCV 的 BGR 画面矩阵转换成 Qt 图片，传入的画面应已缩放到目标显示尺寸。
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 转 RGB 供 Qt 显示。
    height, width, channel = rgb.shape  # 取画面尺寸与通道数。
    image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)  # 用矩阵数据构造图片（仅引用 rgb 内存，不拷贝）。
    return QPixmap.fromImage(image.copy())  # 复制图片数据后转成 QPixmap，避免 rgb 被回收后引用失效。


class VisionLabel(QLabel):  # 定义自适应缩放的画面标签。

    def __init__(self, parent=None):  # 构造函数。
        super().__init__(parent)  # 初始化父类。
        self.source = None  # 保存原始尺寸的 BGR 画面矩阵，窗口缩放时重新按比例缩放。
        self.setAlignment(Qt.AlignCenter)  # 画面居中显示。
        self.setMinimumSize(640, 360)  # 设置最小显示尺寸。
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许随窗口拉伸。
        self.setStyleSheet("background-color: #1e1e1e;")  # 深色背景更接近游戏画面观感。
        self.setText("No frame. 暂无画面")  # 初始占位文字。

    def set_frame(self, frame):  # 接收一帧新的 BGR 画面矩阵并按当前标签尺寸缩放显示。
        self.source = frame  # 保存原始画面供窗口缩放时重绘（任务推送的是 draw_overlay 的副本，不会被覆写）。
        self.rescale()  # 立即按当前尺寸刷新显示。

    def clear_frame(self, text):  # 清空画面并显示占位提示文字。
        self.source = None  # 清除缓存画面。
        self.clear()  # 清除当前图片。
        self.setText(text)  # 显示提示文字。

    def rescale(self):  # 按标签当前尺寸等比缩放画面：先用 OpenCV 缩到目标尺寸再转 Qt 图片，避开“整帧转图片 + Qt 再缩放”的双重全尺寸开销。
        if self.source is None:  # 没有画面时不处理。
            return  # 直接返回。
        height, width = self.source.shape[:2]  # 原始画面尺寸。
        box_w, box_h = max(1, self.width()), max(1, self.height())  # 标签当前尺寸，至少 1 像素避免除零。
        scale = min(box_w / width, box_h / height)  # 等比缩放比例，取宽高两个方向较小者保证画面完整显示。
        target_w, target_h = max(1, int(width * scale)), max(1, int(height * scale))  # 缩放后的目标尺寸。
        if (target_w, target_h) == (width, height):  # 尺寸刚好一致时不必缩放。
            resized = self.source  # 直接用原始画面。
        else:  # 需要缩放。
            resized = cv2.resize(self.source, (target_w, target_h),  # 缩到目标尺寸，cv2.resize 对大图会自行多核并行。
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)  # 缩小用 INTER_AREA 避免标注框与文字锯齿，放大用 INTER_LINEAR 更快。
        self.setPixmap(to_pixmap(resized))  # 转成图片显示，已是目标尺寸无需 Qt 再缩放。

    def resizeEvent(self, event):  # 标签尺寸变化时重新缩放画面。
        super().resizeEvent(event)  # 先执行父类逻辑。
        self.rescale()  # 按新尺寸刷新画面。


def parse_feature_list(text):  # 把英文逗号分隔的分类名文本解析成列表，去掉空白项与首尾空格。
    return [name.strip() for name in str(text or '').split(',') if name.strip()]


class MultiSelectComboBox(PushButton):  # 下拉多选控件：点击弹出带复选框的菜单，可连续勾选多项，选中结果以英文逗号拼接为字符串（与怪物特征配置的存储格式一致）。

    def __init__(self, parent=None):  # 构造多选下拉。
        super().__init__(parent)  # 初始化按钮。
        self._options = []  # 当前可选项（来自模板标注的分类名）。
        self._selected = []  # 已选项（保持勾选顺序，含不在选项里的历史值，避免刷新标注后丢配置）。
        self._menu = None  # 当前弹出的菜单引用，防止被垃圾回收。
        self.clicked.connect(self._show_menu)  # 点击弹出多选菜单。
        self._update_text()  # 初始化按钮文案。

    def set_options(self, options):  # 更新可选项列表（保留已选值）。
        self._options = [str(o) for o in options if str(o)]  # 去空并转字符串。
        self._update_text()  # 选项变化可能影响摘要展示，刷新文案。

    def set_value(self, names):  # 设置已选项（names 为分类名列表）。
        self._selected = []  # 先清空。
        for name in (names or []):  # 逐项加入。
            name = str(name).strip()  # 去首尾空格。
            if name and name not in self._selected:  # 非空且未重复。
                self._selected.append(name)  # 保序加入。
        self._update_text()  # 刷新文案。

    def value(self):  # 返回已选分类名列表。
        return list(self._selected)  # 拷贝返回，避免外部改动内部状态。

    def value_text(self):  # 返回以英文逗号拼接的字符串（怪物特征配置的存储格式）。
        return ','.join(self._selected)  # 与任务 parse_monster_names 的逗号分隔解析一致。

    def _menu_items(self):  # 菜单要展示的项：全部可选项 + 已选但不在选项里的历史值。
        items = list(self._options)  # 先放可选项。
        for name in self._selected:  # 补上历史选中值。
            if name not in items:  # 不在可选项里。
                items.append(name)  # 追加，保证能取消勾选。
        return items  # 返回菜单项列表。

    def _show_menu(self):  # 弹出带复选框的下拉菜单。
        menu = RoundMenu(parent=self)  # 每次弹出新建菜单，避免复用陈旧复选框。
        items = self._menu_items()  # 取菜单项。
        if not items:  # 无任何可选标注时给出提示。
            menu.addWidget(BodyLabel('无「怪物」类别标注，请先在模板页标注'), selectable=False)  # 不可选，仅展示。
        for name in items:  # 逐项加复选框。
            box = CheckBox(name)  # 复选框，文本即分类名。
            box.setChecked(name in self._selected)  # 已选项默认勾选。
            box.toggled.connect(lambda checked, n=name: self._on_toggle(n, checked))  # 勾选变化即更新已选集合。
            menu.addWidget(box, selectable=False)  # selectable=False 让点击落在复选框上、菜单不自动关闭，可连续多选。
        self._menu = menu  # 持有引用防回收。
        menu.exec(self.mapToGlobal(QPoint(0, self.height())), aniType=MenuAnimationType.DROP_DOWN)  # 在按钮下方弹出。

    def _on_toggle(self, name, checked):  # 复选框状态变化：同步已选集合并刷新文案。
        if checked:  # 勾选。
            if name not in self._selected:  # 尚未记录。
                self._selected.append(name)  # 追加。
        elif name in self._selected:  # 取消勾选。
            self._selected.remove(name)  # 移除。
        self._update_text()  # 刷新按钮文案。

    def _update_text(self):  # 按已选项刷新按钮文案与悬浮提示。
        if not self._selected:  # 未选任何项。
            self.setText(PLACEHOLDER)  # 显示占位文本。
            self.setToolTip('未选择怪物标注')  # 悬浮提示。
        elif len(self._selected) == 1:  # 只选一项。
            self.setText(self._selected[0])  # 直接显示该项。
            self.setToolTip(self._selected[0])  # 悬浮提示。
        else:  # 选了多项。
            self.setText(f'{self._selected[0]} 等 {len(self._selected)} 项')  # 首项 + 数量摘要。
            self.setToolTip('、'.join(self._selected))  # 悬浮显示全部已选项。


class DashboardTab(CustomTab):  # 定义看板页签：视图区 + 测谎栏 + 角色栏 + 怪物栏。

    def __init__(self):  # 构造函数。
        super().__init__()  # 初始化父类。
        self.icon = FluentIcon.VIEW  # 页签图标。
        self._last_frame = None  # 缓存最近一帧画面，供测谎区域裁剪预览复用。
        self._rendered_frame = None  # 上一次已渲染的画面对象，任务还没推送新帧时跳过重复的缩放与转图。
        self._rendered_region = None  # 上一次已渲染的测谎区域分类名，区域选择变了即使同一帧也要重画预览。

        # —— 第 1 栏：视图区（原 Vision 页签迁移至此）——
        self.image_label = VisionLabel()  # 创建画面显示标签。
        self.add_card("Realtime Vision 实时识图画面", self.image_label, stretch=1)  # 画面标签占满剩余空间。

        # —— 第 2 栏：任务控制栏（原【任务】页的挂机/巡逻脚本迁移至此）——
        self.add_card("Tasks 任务控制", TaskControlPanel())  # 手风琴任务卡 + 紧凑配置网格 + 内嵌运行日志面板。

        # —— 第 3 栏：测谎栏 ——
        lie_widget = QWidget()  # 测谎栏容器。
        lie_layout = QHBoxLayout(lie_widget)  # 左右两列：预览 + 表单。
        lie_layout.setContentsMargins(0, 0, 0, 0)
        self.region_preview = VisionLabel()  # 测谎区域裁剪预览：时刻采集所选标注区域画面。
        self.region_preview.setMinimumSize(320, 180)  # 预览尺寸略小于主画面。
        self.region_preview.setText("No region selected. 未选择测谎区域")  # 初始占位文字。
        lie_form = QFormLayout()  # 测谎参数表单。
        self.lie_auto_switch = SwitchButton()  # 测谎总开关：开启后全部任务运行时值守并自动解测谎。
        self.lie_region_combo = EditableComboBox()  # 测谎区域标注选择（类别为「测谎」的标注单选，可手工输入未标注名）。
        self.lie_trigger_combo = EditableComboBox()  # 测谎触发标注选择（类别为「测谎触发」的标注单选）。
        self.lie_threshold_spin = DoubleSpinBox()  # 测谎触发匹配阈值。
        self.lie_threshold_spin.setRange(0.01, 1.0)  # 阈值范围 (0, 1]。
        self.lie_threshold_spin.setSingleStep(0.05)  # 步长。
        self.lie_delay_spin = DoubleSpinBox()  # 匹配到测谎触发后延迟多少秒才开始解测谎。
        self.lie_delay_spin.setRange(0.0, 60.0)  # 延迟范围 0~60 秒，0 表示不延迟立即解题。
        self.lie_delay_spin.setSingleStep(0.5)  # 步长。
        self.lie_delay_spin.setDecimals(1)  # 保留一位小数。
        self.lie_delay_spin.setSuffix(" s")  # 单位后缀，一眼看出是秒数。
        self.lie_alarm_edit = LineEdit()  # 报警音频路径输入框。
        self.lie_alarm_edit.setPlaceholderText("alarm.mp3")  # 占位提示。
        lie_form.addRow("Auto Solve 自动解测谎", self.lie_auto_switch)
        lie_form.addRow("Region 测谎区域标注", self.lie_region_combo)
        lie_form.addRow("Trigger 测谎触发标注", self.lie_trigger_combo)
        lie_form.addRow("Threshold 匹配阈值", self.lie_threshold_spin)
        lie_form.addRow("Trigger Delay 触发延迟", self.lie_delay_spin)
        lie_form.addRow("Alarm Sound 报警音频", self.lie_alarm_edit)
        lie_layout.addWidget(self.region_preview, 1)  # 预览占左半。
        lie_layout.addLayout(lie_form, 1)  # 表单占右半。
        self.add_card("Lie Detector 测谎", lie_widget)

        # —— 第 4 栏：角色栏 ——
        char_widget = QWidget()  # 角色栏容器。
        char_grid = QGridLayout(char_widget)  # 两列表单减少纵向占用。
        char_grid.setContentsMargins(0, 0, 0, 0)
        self.char_feature_combo = EditableComboBox()  # 角色标注分类名（类别为「角色」的标注单选，可手工输入未标注名）。
        self.char_threshold_spin = DoubleSpinBox()  # 角色匹配阈值。
        self.char_threshold_spin.setRange(0.01, 1.0)
        self.char_threshold_spin.setSingleStep(0.05)
        self.char_facing_left_combo = EditableComboBox()  # 左朝向模板分类名（类别为「角色」的标注单选），留空禁用朝向校准。
        self.char_facing_right_combo = EditableComboBox()  # 右朝向模板分类名（类别为「角色」的标注单选），两个都配置才启用校准。
        self.attack_key_edit = LineEdit()  # 常规攻击按键。
        self.melee_key_edit = LineEdit()  # 近战攻击按键。
        self.melee_distance_spin = SpinBox()  # 近战距离（像素）。
        self.melee_distance_spin.setRange(0, 10000)
        self.range_x_min_spin = SpinBox()  # 攻击区域左边界（符号化像素）。
        self.range_x_min_spin.setRange(-10000, 10000)
        self.range_x_max_spin = SpinBox()  # 攻击区域右边界。
        self.range_x_max_spin.setRange(-10000, 10000)
        self.range_y_min_spin = SpinBox()  # 攻击区域上边界。
        self.range_y_min_spin.setRange(-10000, 10000)
        self.range_y_max_spin = SpinBox()  # 攻击区域下边界。
        self.range_y_max_spin.setRange(-10000, 10000)
        self.del_interval_spin = DoubleSpinBox()  # 自动按 Del 键间隔（秒），0 禁用。
        self.del_interval_spin.setRange(0.0, 86400.0)
        self.del_interval_spin.setDecimals(1)
        char_grid.addWidget(BodyLabel("Feature 角色特征"), 0, 0)
        char_grid.addWidget(self.char_feature_combo, 0, 1)
        char_grid.addWidget(BodyLabel("Threshold 阈值"), 0, 2)
        char_grid.addWidget(self.char_threshold_spin, 0, 3)
        char_grid.addWidget(BodyLabel("Facing Left 左朝向"), 1, 0)
        char_grid.addWidget(self.char_facing_left_combo, 1, 1)
        char_grid.addWidget(BodyLabel("Facing Right 右朝向"), 1, 2)
        char_grid.addWidget(self.char_facing_right_combo, 1, 3)
        char_grid.addWidget(BodyLabel("Attack Key 攻击键"), 2, 0)
        char_grid.addWidget(self.attack_key_edit, 2, 1)
        char_grid.addWidget(BodyLabel("Melee Key 近战键"), 2, 2)
        char_grid.addWidget(self.melee_key_edit, 2, 3)
        char_grid.addWidget(BodyLabel("Melee Distance 近战距离"), 3, 0)
        char_grid.addWidget(self.melee_distance_spin, 3, 1)
        char_grid.addWidget(BodyLabel("Del Interval Del间隔"), 3, 2)
        char_grid.addWidget(self.del_interval_spin, 3, 3)
        char_grid.addWidget(BodyLabel("Range X 攻击范围X"), 4, 0)
        range_x = QWidget()  # X 范围两个输入框拼一行。
        range_x_layout = QHBoxLayout(range_x)
        range_x_layout.setContentsMargins(0, 0, 0, 0)
        range_x_layout.addWidget(self.range_x_min_spin)
        range_x_layout.addWidget(QLabel("~"))
        range_x_layout.addWidget(self.range_x_max_spin)
        char_grid.addWidget(range_x, 4, 1)
        char_grid.addWidget(BodyLabel("Range Y 攻击范围Y"), 4, 2)
        range_y = QWidget()  # Y 范围两个输入框拼一行。
        range_y_layout = QHBoxLayout(range_y)
        range_y_layout.setContentsMargins(0, 0, 0, 0)
        range_y_layout.addWidget(self.range_y_min_spin)
        range_y_layout.addWidget(QLabel("~"))
        range_y_layout.addWidget(self.range_y_max_spin)
        char_grid.addWidget(range_y, 4, 3)
        self.add_card("Character 角色配置", char_widget)

        # —— 第 5 栏：怪物栏 ——
        monster_widget = QWidget()  # 怪物栏容器。
        monster_form = QFormLayout(monster_widget)
        monster_form.setContentsMargins(0, 0, 0, 0)
        self.monster_multi = MultiSelectComboBox()  # 怪物标注分类名多选（类别为「怪物」的标注），选中项以英文逗号拼接存储。
        self.monster_threshold_spin = DoubleSpinBox()  # 怪物匹配阈值。
        self.monster_threshold_spin.setRange(0.01, 1.0)
        self.monster_threshold_spin.setSingleStep(0.05)
        self.monster_mirror_spin = DoubleSpinBox()  # 怪物镜像匹配阈值。
        self.monster_mirror_spin.setRange(0.01, 1.0)
        self.monster_mirror_spin.setSingleStep(0.05)
        monster_form.addRow("Features 怪物特征(多选)", self.monster_multi)
        monster_form.addRow("Threshold 阈值", self.monster_threshold_spin)
        monster_form.addRow("Mirror Threshold 镜像阈值", self.monster_mirror_spin)
        self.add_card("Monster 怪物配置", monster_widget)

        # —— 底部操作行：刷新标注 + 保存 ——
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.refresh_button = PushButton(FluentIcon.SYNC, "Refresh Annotations 刷新标注")  # 重新读取模板页标注填充下拉框。
        self.refresh_button.clicked.connect(self.reload_annotations)
        self.save_button = PrimaryPushButton(FluentIcon.SAVE, "Save 保存")  # 校验并保存看板共享配置。
        self.save_button.clicked.connect(self.save)
        self.status_label = BodyLabel("")  # 保存结果提示。
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        action_layout.addWidget(self.refresh_button)
        action_layout.addWidget(self.save_button)
        action_layout.addWidget(self.status_label, 1)
        self.add_card("Actions 操作", action_widget)

        self.reload_annotations()  # 首次加载标注选项。
        self.load_config()  # 首次加载看板共享配置填充表单。

        self.timer = QTimer(self)  # 创建刷新定时器。
        self.timer.timeout.connect(self.refresh)  # 定时刷新画面与区域预览。
        self.timer.start(CAPTURE_INTERVAL_MS)  # 按固定 30FPS 节拍刷新，截图采集速率由此钉死。

    @property
    def name(self):  # 页签显示名称。
        return "Dashboard"  # 返回页签名。

    # ------------------------------------------------------------------ 配置读写

    def load_config(self):  # 读取看板共享配置填充全部表单控件。
        data = load_dashboard_config()  # 文件不存在时自动迁移既有任务配置并落盘。
        self.lie_auto_switch.setChecked(bool(data.get('Lie Detector Auto Solve')))
        self._set_combo_value(self.lie_region_combo, str(data.get('Lie Detector Region Feature') or ''))
        self._set_combo_value(self.lie_trigger_combo, str(data.get('Lie Detector Trigger Feature') or ''))
        self.lie_threshold_spin.setValue(float(data.get('Lie Detector Threshold') or 0.75))
        self.lie_delay_spin.setValue(float(data.get('Lie Detector Trigger Delay', DASHBOARD_DEFAULTS['Lie Detector Trigger Delay'])))
        self.lie_alarm_edit.setText(str(data.get('Lie Alarm Sound') or ''))
        self._set_combo_value(self.char_feature_combo, str(data.get('Character Feature') or ''))
        self.char_threshold_spin.setValue(float(data.get('Character Threshold') or 0.8))
        self._set_combo_value(self.char_facing_left_combo, str(data.get('Character Facing Left Feature') or ''))
        self._set_combo_value(self.char_facing_right_combo, str(data.get('Character Facing Right Feature') or ''))
        self.attack_key_edit.setText(str(data.get('Attack Key') or ''))
        self.melee_key_edit.setText(str(data.get('Melee Attack Key') or ''))
        self.melee_distance_spin.setValue(int(float(data.get('Melee Distance') or 0)))
        self.range_x_min_spin.setValue(int(data.get('Attack Range X Min', DASHBOARD_DEFAULTS['Attack Range X Min'])))
        self.range_x_max_spin.setValue(int(data.get('Attack Range X Max', DASHBOARD_DEFAULTS['Attack Range X Max'])))
        self.range_y_min_spin.setValue(int(data.get('Attack Range Y Min', DASHBOARD_DEFAULTS['Attack Range Y Min'])))
        self.range_y_max_spin.setValue(int(data.get('Attack Range Y Max', DASHBOARD_DEFAULTS['Attack Range Y Max'])))
        self.del_interval_spin.setValue(float(data.get('Del Key Interval') or 0))
        self.monster_multi.set_value(parse_feature_list(data.get('Monster Features')))
        self.monster_threshold_spin.setValue(float(data.get('Monster Threshold') or 0.65))
        self.monster_mirror_spin.setValue(float(data.get('Monster Mirror Threshold') or 0.65))

    def _set_combo_value(self, combo, value):  # 把配置值设到下拉框，选项不存在时补充进去保证不丢用户配置。
        value = str(value or '')  # 统一转字符串。
        if value and combo.findText(value) < 0:  # 选项列表里没有该值（如标注被删后配置仍保留）。
            combo.addItem(value)  # 补充一个选项。
        combo.setCurrentText(value if value else PLACEHOLDER)  # 选中目标值，空值选占位项。

    def _combo_value(self, combo):  # 读取下拉框当前值，占位项映射回空字符串。
        text = combo.currentText().strip()  # 取当前文本。
        return '' if text == PLACEHOLDER else text  # 占位项按空处理。

    def reload_annotations(self):  # 重新读取模板页标注，按类别填充测谎/角色下拉框与怪物多选框（保留当前选择）。
        annotations = load_annotations_by_supercategory()  # {类别: {分类名: 标注信息}}。
        for combo, super_name in ((self.lie_region_combo, SUPER_LIE_REGION),  # 区域框按「测谎」类别过滤。
                                  (self.lie_trigger_combo, SUPER_LIE_TRIGGER),  # 触发框按「测谎触发」类别过滤。
                                  (self.char_feature_combo, SUPER_CHARACTER),  # 角色特征按「角色」类别过滤。
                                  (self.char_facing_left_combo, SUPER_CHARACTER),  # 左朝向按「角色」类别过滤。
                                  (self.char_facing_right_combo, SUPER_CHARACTER)):  # 右朝向按「角色」类别过滤。
            current = self._combo_value(combo)  # 记录当前选择，刷新后尽量保持。
            combo.blockSignals(True)  # 填充期间不触发信号。
            combo.clear()  # 清空旧选项。
            combo.addItem(PLACEHOLDER)  # 首项占位代表不选择。
            combo.addItems(sorted((annotations.get(super_name) or {}).keys()))  # 该类别下的全部分类名。
            combo.blockSignals(False)  # 恢复信号。
            self._set_combo_value(combo, current)  # 还原之前的选择。
        self.monster_multi.set_options(sorted((annotations.get(SUPER_MONSTER) or {}).keys()))  # 怪物多选按「怪物」类别刷新可选项，已选值保留。

    def save(self):  # 校验并保存看板共享配置到 configs/Dashboard.json。
        for label, key_edit in (("Attack Key 攻击键", self.attack_key_edit),  # 按键类字段必须与框架按键校验规则一致。
                                ("Melee Key 近战键", self.melee_key_edit)):
            value = key_edit.text().strip()  # 取按键文本。
            if value and not validate_key_name(value):  # 非空且不在键盘按键名单内。
                self.status_label.setText(f"Invalid key {value}. 按键 {value} 不存在，请重新填写。")  # 阻止保存并提示。
                return  # 中止保存。
        data = {  # 收集全部共享配置项。
            'Lie Detector Auto Solve': bool(self.lie_auto_switch.isChecked()),
            'Lie Detector Region Feature': self._combo_value(self.lie_region_combo),
            'Lie Detector Trigger Feature': self._combo_value(self.lie_trigger_combo),
            'Lie Detector Threshold': round(float(self.lie_threshold_spin.value()), 2),
            'Lie Detector Trigger Delay': round(float(self.lie_delay_spin.value()), 1),
            'Lie Alarm Sound': self.lie_alarm_edit.text().strip(),
            'Character Feature': self._combo_value(self.char_feature_combo),
            'Character Threshold': round(float(self.char_threshold_spin.value()), 2),
            'Character Facing Left Feature': self._combo_value(self.char_facing_left_combo),
            'Character Facing Right Feature': self._combo_value(self.char_facing_right_combo),
            'Attack Key': self.attack_key_edit.text().strip(),
            'Melee Attack Key': self.melee_key_edit.text().strip(),
            'Melee Distance': int(self.melee_distance_spin.value()),
            'Attack Range X Min': int(self.range_x_min_spin.value()),
            'Attack Range X Max': int(self.range_x_max_spin.value()),
            'Attack Range Y Min': int(self.range_y_min_spin.value()),
            'Attack Range Y Max': int(self.range_y_max_spin.value()),
            'Del Key Interval': round(float(self.del_interval_spin.value()), 1),
            'Monster Features': self.monster_multi.value_text(),
            'Monster Threshold': round(float(self.monster_threshold_spin.value()), 2),
            'Monster Mirror Threshold': round(float(self.monster_mirror_spin.value()), 2),
        }
        save_dashboard_config(data)  # 落盘，任务下次启动时采集生效。
        lie_service = getattr(og.my_app, 'lie_service', None) if og.my_app is not None else None  # 取独立测谎监控服务（由 Globals 持有）。
        if lie_service is not None:  # 服务已就绪时通知它立即重读配置。
            lie_service.reload_config()  # 测谎参数热更新，无需重启任务或等待轮询间隔。
        self.status_label.setText("Saved. Lie detector config applies immediately; task config on next start. 已保存，测谎配置立即生效，任务配置下次启动时生效。")  # 提示保存成功。
        self.logger.info('dashboard shared config saved 看板共享配置已保存')  # 记录日志供排查。

    # ------------------------------------------------------------------ 画面刷新

    def refresh(self):  # 定时刷新：优先显示任务推送的带标注画面（含解测谎运算叠加），任务停止后才回退到原始截图。
        frame = None  # 待显示的画面。
        try:  # 读取共享画面失败时不应导致 UI 崩溃。
            if og.my_app is not None and hasattr(og.my_app, 'get_vision'):  # 全局对象提供画面共享接口时才读取。
                frame = og.my_app.get_vision(max_age=VISION_MAX_AGE)  # 读取 VISION_MAX_AGE 秒内的最新带标注画面，任务循环偶尔变慢时沿用上一帧而不是闪回原始截图。
            if frame is None:  # 任务未运行（或已停止超过 VISION_MAX_AGE 秒）时才直接截取游戏画面。
                frame = self.capture_raw()  # 从截图设备取一帧原始画面。
        except Exception as e:  # 任何读取异常都按无画面处理。
            self.logger.warning(f'DashboardTab refresh failed: {e}')  # 记录异常日志。
            frame = None  # 按无画面处理。
        if frame is None:  # 仍无画面时显示占位提示。
            self._last_frame = None  # 清空缓存帧。
            self._rendered_frame, self._rendered_region = None, None  # 清空渲染记录，下一帧拿到画面时必须重新渲染。
            self.image_label.clear_frame("No frame, please connect a window and start the task. 暂无画面，请先连接游戏窗口并启动任务。")  # 提示用户操作步骤。
            return  # 结束本次刷新。
        self._last_frame = frame  # 缓存供区域裁剪预览。
        region_name = self._combo_value(self.lie_region_combo)  # 当前选择的测谎区域分类名。
        if frame is self._rendered_frame and region_name == self._rendered_region:  # 任务推送频率低于 30Hz 刷新节拍，同一个帧对象重复缩放转图纯属白耗 GUI 线程。
            return  # 画面与区域选择都未变，直接跳过本次渲染。
        self._rendered_frame, self._rendered_region = frame, region_name  # 记下本次渲染的帧与区域选择。
        self.image_label.set_frame(frame)  # 把画面交给标签，由它按当前尺寸预缩放后转图片显示。
        self.refresh_region_preview(frame, region_name)  # 同步刷新测谎区域裁剪预览。

    def refresh_region_preview(self, frame, region_name=None):  # 按所选测谎区域标注从当前画面裁剪预览，坐标按画面分辨率等比缩放。
        region_name = self._combo_value(self.lie_region_combo) if region_name is None else region_name  # 未传入时自行读取当前选择。
        if not region_name:  # 未选择区域时显示占位。
            self.region_preview.clear_frame("No region selected. 未选择测谎区域")  # 占位提示。
            return  # 结束。
        try:  # 标注文件读取异常时不能拖垮刷新。
            info = (load_annotations_by_supercategory().get(SUPER_LIE_REGION) or {}).get(region_name)  # 取标注坐标与源图尺寸。
        except Exception as e:  # 读取异常。
            self.logger.warning(f'read annotation failed: {e}')  # 记录日志。
            info = None  # 按缺失处理。
        if info is None:  # 所选分类名没有「测谎」类别标注。
            self.region_preview.clear_frame(f"Annotation not found: {region_name}. 未找到标注：{region_name}")  # 提示去模板页标注。
            return  # 结束。
        height, width = frame.shape[:2]  # 当前画面尺寸。
        src_w = info.get('img_w') or width  # 标注源图宽，缺失时按当前画面不缩放。
        src_h = info.get('img_h') or height  # 标注源图高。
        sx, sy = width / src_w, height / src_h  # 当前画面相对标注源图的缩放比例。
        x = int(info['x'] * sx)  # 缩放后区域左上角 x。
        y = int(info['y'] * sy)  # 缩放后区域左上角 y。
        w = max(1, int(info['w'] * sx))  # 缩放后区域宽。
        h = max(1, int(info['h'] * sy))  # 缩放后区域高。
        x = max(0, min(x, width - 1))  # 防越界。
        y = max(0, min(y, height - 1))  # 防越界。
        crop = frame[y:min(y + h, height), x:min(x + w, width)]  # 裁剪区域画面。
        if crop.size == 0:  # 区域完全在画面外。
            self.region_preview.clear_frame(f"Region out of frame. 区域超出画面：{region_name}")  # 提示。
            return  # 结束。
        self.region_preview.set_frame(crop)  # 显示裁剪预览，同样由标签自行预缩放。

    def capture_raw(self):  # 任务未运行时直接从截图设备取原始画面，失败返回 None。
        device_manager = getattr(og, 'device_manager', None)  # 取设备管理器。
        if device_manager is None or device_manager.capture_method is None:  # 尚未连接游戏窗口。
            return None  # 返回无画面。
        return device_manager.capture_method.get_frame()  # 返回最新一帧截图。
