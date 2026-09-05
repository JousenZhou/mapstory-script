# 看板页签：全局共享配置的唯一编辑入口 + 实时画面展示。
#
# 纵向四栏：
#   1. 视图区：实时显示任务推送的带标注画面（含解测谎运算过程叠加），无任务时显示原始截图；
#   2. 测谎栏：区域/触发两个按类别过滤的标注选择、裁剪预览、匹配阈值、报警音频与总开关；
#   3. 角色栏：角色特征/阈值、朝向模板、攻击键、近战键与距离、攻击范围、Del 间隔；
#   4. 怪物栏：怪物特征、阈值与镜像阈值。
# 保存写入 configs/Dashboard.json，任务启动时由 MyBaseTask.apply_shared_config() 采集（单一数据源）。
import cv2  # 导入 OpenCV，用于画面转换与区域裁剪预览。
from PySide6.QtCore import Qt, QTimer  # 导入 Qt 定时器与对齐常量。
from PySide6.QtGui import QImage, QPixmap  # 导入图像对象，用于把画面矩阵转成图片显示。
from PySide6.QtWidgets import (QFormLayout, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget)  # 导入布局与控件。
from qfluentwidgets import (BodyLabel, DoubleSpinBox, EditableComboBox, FluentIcon, LineEdit,  # 导入 Fluent 控件。
                            PrimaryPushButton, PushButton, SpinBox, SwitchButton)

from ok import og  # 导入全局对象，用于读取任务线程推送的画面与截图设备。
from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。

from src.dashboard_store import (DASHBOARD_DEFAULTS, SUPER_LIE_REGION, SUPER_LIE_TRIGGER,  # 看板共享配置存取层。
                                 load_annotations_by_supercategory, load_dashboard_config,
                                 save_dashboard_config, validate_key_name)

CAPTURE_FPS = 30  # 截图采集固定帧率：实时画面刷新与无任务时的截图取帧都按该节拍。
CAPTURE_INTERVAL_MS = round(1000 / CAPTURE_FPS)  # 帧间隔毫秒数（33ms）。
PLACEHOLDER = '(未选择)'  # 下拉框空选项占位文本，保存时映射回空字符串。


class VisionLabel(QLabel):  # 定义自适应缩放的画面标签。

    def __init__(self, parent=None):  # 构造函数。
        super().__init__(parent)  # 初始化父类。
        self.source = None  # 保存原始尺寸画面图片，窗口缩放时重新按比例缩放。
        self.setAlignment(Qt.AlignCenter)  # 画面居中显示。
        self.setMinimumSize(640, 360)  # 设置最小显示尺寸。
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许随窗口拉伸。
        self.setStyleSheet("background-color: #1e1e1e;")  # 深色背景更接近游戏画面观感。
        self.setText("No frame. 暂无画面")  # 初始占位文字。

    def set_frame(self, pixmap):  # 接收一张新画面并按当前标签尺寸缩放显示。
        self.source = pixmap  # 保存原始画面供窗口缩放时重绘。
        self.rescale()  # 立即按当前尺寸刷新显示。

    def clear_frame(self, text):  # 清空画面并显示占位提示文字。
        self.source = None  # 清除缓存画面。
        self.clear()  # 清除当前图片。
        self.setText(text)  # 显示提示文字。

    def rescale(self):  # 按标签当前尺寸等比缩放画面。
        if self.source is None:  # 没有画面时不处理。
            return  # 直接返回。
        scaled = self.source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 等比缩放到标签大小。
        self.setPixmap(scaled)  # 显示缩放后的画面。

    def resizeEvent(self, event):  # 标签尺寸变化时重新缩放画面。
        super().resizeEvent(event)  # 先执行父类逻辑。
        self.rescale()  # 按新尺寸刷新画面。


class DashboardTab(CustomTab):  # 定义看板页签：视图区 + 测谎栏 + 角色栏 + 怪物栏。

    def __init__(self):  # 构造函数。
        super().__init__()  # 初始化父类。
        self.icon = FluentIcon.VIEW  # 页签图标。
        self._last_frame = None  # 缓存最近一帧画面，供测谎区域裁剪预览复用。

        # —— 第 1 栏：视图区（原 Vision 页签迁移至此）——
        self.image_label = VisionLabel()  # 创建画面显示标签。
        self.add_card("Realtime Vision 实时识图画面", self.image_label, stretch=1)  # 画面标签占满剩余空间。

        # —— 第 2 栏：测谎栏 ——
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
        self.lie_alarm_edit = LineEdit()  # 报警音频路径输入框。
        self.lie_alarm_edit.setPlaceholderText("alarm.mp3")  # 占位提示。
        lie_form.addRow("Auto Solve 自动解测谎", self.lie_auto_switch)
        lie_form.addRow("Region 测谎区域标注", self.lie_region_combo)
        lie_form.addRow("Trigger 测谎触发标注", self.lie_trigger_combo)
        lie_form.addRow("Threshold 匹配阈值", self.lie_threshold_spin)
        lie_form.addRow("Alarm Sound 报警音频", self.lie_alarm_edit)
        lie_layout.addWidget(self.region_preview, 1)  # 预览占左半。
        lie_layout.addLayout(lie_form, 1)  # 表单占右半。
        self.add_card("Lie Detector 测谎", lie_widget)

        # —— 第 3 栏：角色栏 ——
        char_widget = QWidget()  # 角色栏容器。
        char_grid = QGridLayout(char_widget)  # 两列表单减少纵向占用。
        char_grid.setContentsMargins(0, 0, 0, 0)
        self.char_feature_edit = LineEdit()  # 角色标注分类名。
        self.char_threshold_spin = DoubleSpinBox()  # 角色匹配阈值。
        self.char_threshold_spin.setRange(0.01, 1.0)
        self.char_threshold_spin.setSingleStep(0.05)
        self.char_facing_left_edit = LineEdit()  # 左朝向模板分类名，留空禁用朝向校准。
        self.char_facing_right_edit = LineEdit()  # 右朝向模板分类名，两个都配置才启用校准。
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
        char_grid.addWidget(self.char_feature_edit, 0, 1)
        char_grid.addWidget(BodyLabel("Threshold 阈值"), 0, 2)
        char_grid.addWidget(self.char_threshold_spin, 0, 3)
        char_grid.addWidget(BodyLabel("Facing Left 左朝向"), 1, 0)
        char_grid.addWidget(self.char_facing_left_edit, 1, 1)
        char_grid.addWidget(BodyLabel("Facing Right 右朝向"), 1, 2)
        char_grid.addWidget(self.char_facing_right_edit, 1, 3)
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

        # —— 第 4 栏：怪物栏 ——
        monster_widget = QWidget()  # 怪物栏容器。
        monster_form = QFormLayout(monster_widget)
        monster_form.setContentsMargins(0, 0, 0, 0)
        self.monster_features_edit = LineEdit()  # 怪物标注分类名，英文逗号分隔支持多个。
        self.monster_threshold_spin = DoubleSpinBox()  # 怪物匹配阈值。
        self.monster_threshold_spin.setRange(0.01, 1.0)
        self.monster_threshold_spin.setSingleStep(0.05)
        self.monster_mirror_spin = DoubleSpinBox()  # 怪物镜像匹配阈值。
        self.monster_mirror_spin.setRange(0.01, 1.0)
        self.monster_mirror_spin.setSingleStep(0.05)
        monster_form.addRow("Features 怪物特征(逗号分隔)", self.monster_features_edit)
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
        self.lie_alarm_edit.setText(str(data.get('Lie Alarm Sound') or ''))
        self.char_feature_edit.setText(str(data.get('Character Feature') or ''))
        self.char_threshold_spin.setValue(float(data.get('Character Threshold') or 0.8))
        self.char_facing_left_edit.setText(str(data.get('Character Facing Left Feature') or ''))
        self.char_facing_right_edit.setText(str(data.get('Character Facing Right Feature') or ''))
        self.attack_key_edit.setText(str(data.get('Attack Key') or ''))
        self.melee_key_edit.setText(str(data.get('Melee Attack Key') or ''))
        self.melee_distance_spin.setValue(int(float(data.get('Melee Distance') or 0)))
        self.range_x_min_spin.setValue(int(data.get('Attack Range X Min', DASHBOARD_DEFAULTS['Attack Range X Min'])))
        self.range_x_max_spin.setValue(int(data.get('Attack Range X Max', DASHBOARD_DEFAULTS['Attack Range X Max'])))
        self.range_y_min_spin.setValue(int(data.get('Attack Range Y Min', DASHBOARD_DEFAULTS['Attack Range Y Min'])))
        self.range_y_max_spin.setValue(int(data.get('Attack Range Y Max', DASHBOARD_DEFAULTS['Attack Range Y Max'])))
        self.del_interval_spin.setValue(float(data.get('Del Key Interval') or 0))
        self.monster_features_edit.setText(str(data.get('Monster Features') or ''))
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

    def reload_annotations(self):  # 重新读取模板页标注，按类别填充测谎两个下拉框（保留当前选择）。
        annotations = load_annotations_by_supercategory()  # {类别: {分类名: 标注信息}}。
        for combo, super_name in ((self.lie_region_combo, SUPER_LIE_REGION),  # 区域框按「测谎」类别过滤。
                                  (self.lie_trigger_combo, SUPER_LIE_TRIGGER)):  # 触发框按「测谎触发」类别过滤。
            current = self._combo_value(combo)  # 记录当前选择，刷新后尽量保持。
            combo.blockSignals(True)  # 填充期间不触发信号。
            combo.clear()  # 清空旧选项。
            combo.addItem(PLACEHOLDER)  # 首项占位代表不选择。
            combo.addItems(sorted((annotations.get(super_name) or {}).keys()))  # 该类别下的全部分类名。
            combo.blockSignals(False)  # 恢复信号。
            self._set_combo_value(combo, current)  # 还原之前的选择。

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
            'Lie Alarm Sound': self.lie_alarm_edit.text().strip(),
            'Character Feature': self.char_feature_edit.text().strip(),
            'Character Threshold': round(float(self.char_threshold_spin.value()), 2),
            'Character Facing Left Feature': self.char_facing_left_edit.text().strip(),
            'Character Facing Right Feature': self.char_facing_right_edit.text().strip(),
            'Attack Key': self.attack_key_edit.text().strip(),
            'Melee Attack Key': self.melee_key_edit.text().strip(),
            'Melee Distance': int(self.melee_distance_spin.value()),
            'Attack Range X Min': int(self.range_x_min_spin.value()),
            'Attack Range X Max': int(self.range_x_max_spin.value()),
            'Attack Range Y Min': int(self.range_y_min_spin.value()),
            'Attack Range Y Max': int(self.range_y_max_spin.value()),
            'Del Key Interval': round(float(self.del_interval_spin.value()), 1),
            'Monster Features': self.monster_features_edit.text().strip(),
            'Monster Threshold': round(float(self.monster_threshold_spin.value()), 2),
            'Monster Mirror Threshold': round(float(self.monster_mirror_spin.value()), 2),
        }
        save_dashboard_config(data)  # 落盘，任务下次启动时采集生效。
        self.status_label.setText("Saved. Tasks pick up on next start. 已保存，任务下次启动时生效。")  # 提示保存成功。
        self.logger.info('dashboard shared config saved 看板共享配置已保存')  # 记录日志供排查。

    # ------------------------------------------------------------------ 画面刷新

    def refresh(self):  # 定时刷新：优先显示任务推送的带标注画面（含解测谎运算叠加），无任务时显示原始截图。
        frame = None  # 待显示的画面。
        try:  # 读取共享画面失败时不应导致 UI 崩溃。
            if og.my_app is not None and hasattr(og.my_app, 'get_vision'):  # 全局对象提供画面共享接口时才读取。
                frame = og.my_app.get_vision(max_age=1.0)  # 读取 1 秒内的最新带标注画面。
            if frame is None:  # 任务未运行或画面过期时尝试直接截取游戏画面。
                frame = self.capture_raw()  # 从截图设备取一帧原始画面。
        except Exception as e:  # 任何读取异常都按无画面处理。
            self.logger.warning(f'DashboardTab refresh failed: {e}')  # 记录异常日志。
            frame = None  # 按无画面处理。
        if frame is None:  # 仍无画面时显示占位提示。
            self._last_frame = None  # 清空缓存帧。
            self.image_label.clear_frame("No frame, please connect a window and start the task. 暂无画面，请先连接游戏窗口并启动任务。")  # 提示用户操作步骤。
            return  # 结束本次刷新。
        self._last_frame = frame  # 缓存供区域裁剪预览。
        self.image_label.set_frame(self.to_pixmap(frame))  # 把画面转成图片并显示。
        self.refresh_region_preview(frame)  # 同步刷新测谎区域裁剪预览。

    def refresh_region_preview(self, frame):  # 按所选测谎区域标注从当前画面裁剪预览，坐标按画面分辨率等比缩放。
        region_name = self._combo_value(self.lie_region_combo)  # 当前选择的测谎区域分类名。
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
        self.region_preview.set_frame(self.to_pixmap(crop))  # 显示裁剪预览。

    def capture_raw(self):  # 任务未运行时直接从截图设备取原始画面，失败返回 None。
        device_manager = getattr(og, 'device_manager', None)  # 取设备管理器。
        if device_manager is None or device_manager.capture_method is None:  # 尚未连接游戏窗口。
            return None  # 返回无画面。
        return device_manager.capture_method.get_frame()  # 返回最新一帧截图。

    def to_pixmap(self, frame):  # 把 OpenCV 的 BGR 画面矩阵转换成 Qt 图片。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 转 RGB 供 Qt 显示。
        height, width, channel = rgb.shape  # 取画面尺寸与通道数。
        image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)  # 用矩阵数据构造图片。
        return QPixmap.fromImage(image.copy())  # 复制图片数据后转成 QPixmap，避免引用失效。
