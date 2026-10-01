# 地图页签：管理路线挂机的地图资产与配置（全局小地图底图 + 彩色指令路线图）。
#
# 纵向五卡片（对齐看板页签的单一数据源风格，本只管地图与路线，角色/怪物/测谎仍在看板）：
#   1. 地图列表：新建/重命名/删除/设为默认/刷新/从当前画面或文件导入底图；
#   2. 资产预览与校验：地图尺寸、路线数量、默认地图与路线图尺寸一致性提示；
#   3. 路线编辑：路线图列表 + 指令调色板 + 笔刷/橡皮/撤销，直接在地图上落笔绘制并即时落盘；
#   4. 指令表编辑：主指令色表与上下专用色表两张可编辑表格 + 恢复默认；
#   5. 参数与保存：搜索半径、边缘色、瞬移、小地图定位、怪物特征；保存写入 maps/<地图名>/meta.json。
#
# 消费方式：将来 MapleRouteTask 启动时读取 maps/<默认地图>/meta.json + 路线图，本页面不参与运行时逻辑。
import os  # 路径拼接与打开目录。
import subprocess  # 调用系统文件管理器打开地图目录。

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView,
                               QInputDialog, QMessageBox, QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, ComboBox, EditableComboBox, FluentIcon,
                            LineEdit, PrimaryPushButton, PushButton, SwitchButton)

from ok import og  # 读取当前截图设备画面，用于从实时画面导入地图底图与录制时定位小地图。

from src.map_store import (  # 地图资产存取层。
    DEFAULT_COLOR_CODE, DEFAULT_COLOR_CODE_UP_DOWN, MAP_META_DEFAULTS,
    add_route, create_map, delete_map, delete_route, get_default_map, list_maps,
    list_routes, load_map_image, load_meta, load_route_image, map_size, rename_map,
    save_map_image, save_meta, save_route_image, set_default_map, validate_routes)
from src.map_recorder import MapRecorder, compute_map_rect, detect_yellow_dot  # 路线录制线程与实时定位纯函数。
from src.dashboard_store import SUPER_MONSTER, load_annotations_by_supercategory  # 复用标注按类别读取。
from src.ui.route_canvas import RouteCanvas  # 内嵌路线图绘制控件。
from src.ui.spin_wheel_guard import DoubleSpinBox, SpinBox  # 需点击聚焦后才响应滚轮的数框。
from ok.gui.widget.CustomTab import CustomTab  # 自定义页签基类。

PLACEHOLDER = '(未选择)'  # 下拉框空占位。


class ColorSwatchButton(QPushButton):  # 指令色块按钮：自绘纯色背景，点击选为画笔色，不受 Fluent 主题样式覆盖。

    def __init__(self, rgb, parent=None):  # rgb 为 (R,G,B) 指令色。
        super().__init__(parent)
        self.rgb = tuple(int(c) for c in rgb)  # 保存颜色。
        self.picked = None  # 由外部赋值为回调 (rgb)。
        self.setFixedSize(26, 26)  # 小色块尺寸。
        self.setCursor(Qt.PointingHandCursor)  # 手型光标。
        self._selected = False  # 是否当前选中。
        self.clicked.connect(lambda: self.picked and self.picked(self.rgb))  # 点击回调。
        self.setToolTip(f'RGB {self.rgb}')  # 悬浮显示 RGB。

    def set_selected(self, selected):  # 设置选中态（重绘边框）。
        self._selected = bool(selected)
        self.update()

    def paintEvent(self, event):  # 自绘：填充纯色 + 边框（选中时高亮描边）。
        painter = QPainter(self)
        rect = self.rect().adjusted(2, 2, -2, -2)  # 内缩留边框空间。
        painter.setBrush(QColor(*self.rgb))  # 纯色填充。
        if self._selected:
            pen = QPen(QColor(0, 255, 0))  # 选中：绿色高亮边框。
            pen.setWidth(3)
            painter.setPen(pen)
        else:
            painter.setPen(QPen(QColor(128, 128, 128)))  # 常态：灰色细边框。
        painter.drawRect(rect)
        painter.end()

    def color_label_text(self):  # 供状态提示展示当前色。
        return f'RGB {self.rgb}'


def parse_rgb(text):  # 解析 "r,g,b" 文本为三元组，非法返回 None。
    try:
        parts = [int(x.strip()) for x in str(text).split(',')]
    except (TypeError, ValueError):
        return None
    if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
        return None
    return tuple(parts)


def fill_color_table(table, color_map):  # 用 {RGB串: 指令} 填充两张列的表格。
    table.setRowCount(0)
    for rgb_str, command in color_map.items():
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(str(rgb_str)))
        table.setItem(row, 1, QTableWidgetItem(str(command)))


def read_color_table(table):  # 从表格读回 {RGB串: 指令}，跳过非法行（RGB 非法或指令非三段）。
    result = {}
    for row in range(table.rowCount()):
        rgb_item = table.item(row, 0)
        cmd_item = table.item(row, 1)
        rgb_text = (rgb_item.text().strip() if rgb_item else '')
        cmd_text = (cmd_item.text().strip() if cmd_item else '')
        rgb = parse_rgb(rgb_text)
        tokens = cmd_text.split()
        if rgb is None or len(tokens) != 3:
            continue  # 非法行忽略，保证写出的表一定可被任务解析。
        result[','.join(str(c) for c in rgb)] = ' '.join(tokens)
    return result


def make_two_col_table():  # 构造 RGB / 指令 两列表格。
    table = QTableWidget(0, 2)
    table.setHorizontalHeaderLabels(['颜色 RGB', '指令(左右 上下 动作)'])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)
    table.setMinimumHeight(160)
    return table


class MapTab(CustomTab):  # 地图资产与配置管理页签。

    def __init__(self):  # 构造函数：搭建五卡片并加载首个地图。
        super().__init__()
        self.icon = FluentIcon.GLOBE  # 页签图标。
        self.current_map = ''  # 当前选中地图目录名。
        self.current_route = ''  # 当前编辑的路线图文件名。
        self._loading = False  # 加载中标志，抑制控件信号触发的写盘。
        self._brush_rgb = (255, 0, 0)  # 当前画笔指令色。
        self._swatch_buttons = []  # 调色板按钮列表。
        self._recorder = None  # 当前运行中的 MapRecorder 线程，None 表示未在录制。
        self._rec_timer = QTimer(self)  # 录制期轮询线程状态/收尾的定时器（不跨线程调 Qt，仅读线程属性）。
        self._rec_timer.setInterval(300)  # 每 300ms 轮询一次。
        self._rec_timer.timeout.connect(self._poll_recorder)  # 轮询回调。

        self._build_list_card()  # 卡片 1。
        self._build_preview_card()  # 卡片 2。
        self._build_route_card()  # 卡片 3。
        self._build_record_card()  # 卡片 3.5：实时路线录制。
        self._build_table_card()  # 卡片 4。
        self._build_param_card()  # 卡片 5。

        self.refresh_map_list()  # 首次填充地图列表。

    @property
    def name(self):  # 页签显示名称。
        return "地图"

    # ------------------------------------------------------------------ 卡片 1：地图列表

    def _build_list_card(self):  # 地图选择与增删操作栏。
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.map_combo = ComboBox()  # 地图下拉。
        self.map_combo.currentIndexChanged.connect(self._on_map_selected)  # 切换地图。
        new_btn = PushButton(FluentIcon.ADD, "新建")
        new_btn.clicked.connect(self.on_new_map)
        import_btn = PushButton(FluentIcon.PHOTO, "导入底图")
        import_btn.clicked.connect(self.on_import_map_image)
        capture_btn = PushButton(FluentIcon.CONNECT, "用当前画面")
        capture_btn.clicked.connect(self.on_capture_map)
        rename_btn = PushButton(FluentIcon.EDIT, "重命名")
        rename_btn.clicked.connect(self.on_rename_map)
        delete_btn = PushButton(FluentIcon.DELETE, "删除")
        delete_btn.clicked.connect(self.on_delete_map)
        default_btn = PushButton(FluentIcon.PIN, "设为默认")
        default_btn.clicked.connect(self.on_set_default)
        folder_btn = PushButton(FluentIcon.FOLDER, "打开目录")
        folder_btn.clicked.connect(self.on_open_folder)
        refresh_btn = PushButton(FluentIcon.SYNC, "刷新")
        refresh_btn.clicked.connect(self.refresh_map_list)
        for btn in (self.map_combo, new_btn, import_btn, capture_btn, rename_btn,
                    delete_btn, default_btn, folder_btn, refresh_btn):
            layout.addWidget(btn)
        layout.addWidget(BodyLabel("默认:"), 0)
        self.default_label = BodyLabel("-")  # 显示当前默认地图。
        layout.addWidget(self.default_label, 1)
        self.add_card("地图列表", widget)

    def refresh_map_list(self):  # 重新扫描地图目录填充下拉，并保持/选中默认地图。
        self._loading = True
        self.map_combo.clear()
        maps = list_maps()
        self.map_combo.addItems(maps)  # 目录名即选项。
        self._loading = False
        if not maps:
            self.current_map = ''
            self.default_label.setText('无')
            self._reload_assets()  # 清空预览与编辑区。
            return
        default = get_default_map()
        prefer = default if default in maps else maps[0]
        self.map_combo.setCurrentText(prefer)  # 选中优先地图。
        self._on_map_selected()  # 主动加载一次，避免重复值不触发信号。

    def _on_map_selected(self):  # 下拉切换地图。
        if self._loading:
            return
        if self._is_recording():  # 切换地图前先停掉旧录制，避免线程向旧地图写资产。
            self._stop_recorder()  # 请求停止。
            if self._recorder is not None:  # 仍有线程。
                self._recorder.wait()  # 等其完成落盘再切，串行化文件写入。
                self._recorder = None  # 释放。
                self._rec_timer.stop()  # 停轮询。
                self.record_btn.setEnabled(True)  # 恢复按钮。
                self.record_btn.setText("开始录制")  # 恢复文案。
                self.record_route_check.setEnabled(True)  # 解锁。
        name = str(self.map_combo.currentText() or '').strip()
        self.current_map = name
        self.default_label.setText(get_default_map() or '-')
        self._reload_assets()  # 重新加载该地图的路线、表格与参数。

    def on_new_map(self):  # 新建地图：输入名称，用当前画面（无则空图）作底图。
        name, ok = QInputDialog.getText(self, "新建地图", "地图目录名(建议英文/数字):")
        if not ok or not str(name).strip():
            return
        image = self._capture_current_frame()  # 尝试用实时画面当底图。
        created = create_map(name.strip(), image)
        self.refresh_map_list()
        self._select_map(created)
        self.preview_info.setText(f"已创建地图 {created}，请导入正确的全局小地图底图并绘制路线。")

    def _select_map(self, name):  # 选中指定地图（不存在于列表时先刷新）。
        if self.map_combo.findText(name) >= 0:
            self.map_combo.setCurrentText(name)
            self._on_map_selected()

    def on_import_map_image(self):  # 从图片文件导入为当前地图底图。
        if not self.current_map:
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入地图底图", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        from src.map_store import _imread_unicode  # 复用兼容非 ASCII 路径读取。
        img = _imread_unicode(path)
        if img is None:
            self.preview_info.setText("导入失败：无法读取图片。")
            return
        save_map_image(self.current_map, img)
        self._reload_assets()
        self.preview_info.setText("底图已导入，路线图尺寸若与底图不符会被重置。")

    def on_capture_map(self):  # 用当前游戏画面（截图设备最新帧）覆盖当前地图底图。
        if not self.current_map:
            return
        frame = self._capture_current_frame()
        if frame is None:
            self.preview_info.setText("无画面：请先连接游戏窗口再导入。")
            return
        save_map_image(self.current_map, frame)
        self._reload_assets()

    def _capture_current_frame(self):  # 从截图设备取一帧，失败返回 None。
        device_manager = getattr(og, 'device_manager', None)
        if device_manager is None or getattr(device_manager, 'capture_method', None) is None:
            return None
        try:
            return device_manager.capture_method.get_frame()
        except Exception as e:  # 取帧异常按无画面处理。
            self.logger.warning(f'capture frame failed: {e}')
            return None

    def on_rename_map(self):  # 重命名当前地图目录。
        if not self.current_map:
            return
        name, ok = QInputDialog.getText(self, "重命名地图", "新目录名:", text=self.current_map)
        if not ok or not str(name).strip():
            return
        new_name = rename_map(self.current_map, name.strip())
        self.refresh_map_list()
        if new_name:
            self._select_map(new_name)

    def on_delete_map(self):  # 删除当前地图（弹确认）。
        if not self.current_map:
            return
        confirm, ok = QInputDialog.getText(
            self, "删除地图", f"输入地图名以确认删除 {self.current_map}:")
        if ok and str(confirm).strip() == self.current_map:
            delete_map(self.current_map)
            self.refresh_map_list()

    def on_set_default(self):  # 设当前地图为默认。
        if not self.current_map:
            return
        set_default_map(self.current_map)
        self.default_label.setText(self.current_map)
        self.logger.info(f'default map set to {self.current_map}')

    def on_open_folder(self):  # 用系统文件管理器打开当前地图目录。
        if not self.current_map:
            return
        from src import map_store
        target = os.path.abspath(map_store.map_dir(self.current_map))
        try:
            if os.name == 'nt':
                subprocess.Popen(['explorer', target])
            else:
                subprocess.Popen(['xdg-open', target])
        except Exception as e:
            self.logger.warning(f'open folder failed: {e}')

    # ------------------------------------------------------------------ 卡片 2：预览与校验

    def _build_preview_card(self):  # 地图尺寸、路线数与校验信息。
        self.preview_info = BodyLabel("未选择地图")  # 概要信息。
        self.preview_info.setWordWrap(True)
        self.preview_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.add_card("资产预览与校验", self.preview_info)

    # ------------------------------------------------------------------ 卡片 3：路线编辑

    def _build_route_card(self):  # 路线列表 + 调色板 + 画布。
        container = QWidget()
        v = QVBoxLayout(container)  # 纵向：控制行 + 调色板 + 画布。
        v.setContentsMargins(0, 0, 0, 0)

        ctrl = QWidget()
        row = QHBoxLayout(ctrl)
        row.setContentsMargins(0, 0, 0, 0)
        self.route_combo = ComboBox()  # 当前地图的路线文件。
        self.route_combo.currentIndexChanged.connect(self._on_route_selected)
        add_route_btn = PushButton(FluentIcon.ADD, "新建路线")
        add_route_btn.clicked.connect(self.on_add_route)
        del_route_btn = PushButton(FluentIcon.DELETE, "删除路线")
        del_route_btn.clicked.connect(self.on_delete_route)
        row.addWidget(BodyLabel("路线:"))
        row.addWidget(self.route_combo)
        row.addWidget(add_route_btn)
        row.addWidget(del_route_btn)
        row.addSpacing(12)

        # 画笔粗细 + 橡皮 + 撤销 + 清空。
        row.addWidget(BodyLabel("笔刷:"))
        self.brush_size_spin = SpinBox()
        self.brush_size_spin.setRange(1, 12)
        self.brush_size_spin.setValue(2)
        self.brush_size_spin.valueChanged.connect(self._apply_brush_size)
        row.addWidget(self.brush_size_spin)
        self.eraser_btn = PushButton(FluentIcon.REMOVE, "橡皮")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.toggled.connect(self._on_eraser_toggled)
        row.addWidget(self.eraser_btn)
        undo_btn = PushButton(FluentIcon.HISTORY, "撤销")
        undo_btn.clicked.connect(self.on_undo)
        row.addWidget(undo_btn)
        clear_btn = PushButton(FluentIcon.DELETE, "清空路线")
        clear_btn.clicked.connect(self.on_clear_route)
        row.addWidget(clear_btn)
        row.addStretch(1)
        v.addWidget(ctrl)

        # 调色板行：按当前 meta 的 color_code 生成色块按钮。
        self.palette_widget = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_widget)
        self.palette_layout.setContentsMargins(0, 4, 0, 4)
        self.palette_layout.addWidget(BodyLabel("指令调色板:"))
        self.brush_status = BodyLabel("")  # 当前画笔色与指令提示。
        self.palette_layout.addWidget(self.brush_status)
        self.palette_layout.addStretch(1)
        v.addWidget(self.palette_widget)

        # 画布。
        self.canvas = RouteCanvas()
        self.canvas.setMinimumHeight(360)
        self.canvas.route_changed.connect(self._on_canvas_changed)  # 落笔结束即保存到当前路线文件。
        v.addWidget(self.canvas, 1)

        self.add_card("路线编辑", container)

    def _reload_assets(self):  # 依据 current_map 重新加载路线列表、调色板、表格与参数。
        if not self.current_map:
            self._loading = True
            self.route_combo.clear()
            self._loading = False
            self.canvas.set_images(None, None)
            self.preview_info.setText("未选择地图")
            return
        meta = load_meta(self.current_map)
        self._rebuild_palette(meta)  # 先建调色板，保证画笔默认色有效。
        self._reload_route_list()
        self._load_tables(meta)
        self._load_params(meta)
        self._update_preview_info()

    def _reload_route_list(self):  # 填充路线下拉并加载首条到画布。
        self._loading = True
        self.route_combo.clear()
        routes = list_routes(self.current_map)
        self.route_combo.addItems(routes)
        self._loading = False
        if routes:
            self.route_combo.setCurrentIndex(0)
            self._on_route_selected()  # 主动加载第一条。
        else:
            self.current_route = ''
            self.canvas.set_images(load_map_image(self.current_map), None)  # 仅有底图，无路线。

    def _on_route_selected(self):  # 切换编辑的路线图。
        if self._loading or not self.current_map:
            return
        name = str(self.route_combo.currentText() or '').strip()
        self.current_route = name
        map_img = load_map_image(self.current_map)
        route_img = load_route_image(self.current_map, name) if name else None
        self.canvas.set_images(map_img, route_img)  # 载入到底图与路线。
        self._update_preview_info()

    def on_add_route(self):  # 新建一条全黑路线图并切换过去。
        if not self.current_map:
            return
        file_name = add_route(self.current_map)
        if file_name:
            self._reload_route_list()
            idx = self.route_combo.findText(file_name)
            if idx >= 0:
                self.route_combo.setCurrentIndex(idx)

    def on_delete_route(self):  # 删除当前路线图。
        if not self.current_map or not self.current_route:
            return
        delete_route(self.current_map, self.current_route)
        self._reload_route_list()

    def on_undo(self):  # 撤销画布一步。
        self.canvas.undo()

    def on_clear_route(self):  # 清空当前路线绘制。
        self.canvas.clear_route()

    def _on_canvas_changed(self):  # 画布内容变化：落盘到当前路线文件。
        if not self.current_map or not self.current_route:
            return
        img = self.canvas.route_image()
        if img is None:
            return
        save_route_image(self.current_map, self.current_route, img)
        self._update_preview_info()

    def _apply_brush_size(self, value):  # 笔刷粗细变化。
        self.canvas.set_brush(size=int(value))

    def _on_eraser_toggled(self, checked):  # 橡皮模式切换。
        self.canvas.set_brush(eraser=checked)
        if checked:
            self.brush_status.setText("画笔: 橡皮(黑色/清除)")

    def _rebuild_palette(self, meta):  # 按 color_code 与上下色表重建调色板按钮。
        # 清旧按钮。
        for btn in self._swatch_buttons:
            btn.setParent(None)
            btn.deleteLater()
        self._swatch_buttons = []
        color_map = dict(meta.get('Color Code') or {})
        for k, v in (meta.get('Color Code Up Down') or {}).items():
            color_map.setdefault(k, v)
        first = True
        for rgb_str, command in color_map.items():
            rgb = parse_rgb(rgb_str)
            if rgb is None:
                continue
            btn = ColorSwatchButton(rgb)
            btn.picked = self._pick_color
            btn._cmd_text = str(command)  # 记录指令文本用于状态提示。
            self.palette_layout.insertWidget(self.palette_layout.count() - 1, btn)  # 插到尾部 stretch 前。
            self._swatch_buttons.append(btn)
            if first:
                self._pick_color(rgb, command, silent=True)  # 默认选中第一个色。
                btn.set_selected(True)
                first = False

    def _pick_color(self, rgb, command=None, silent=False):  # 选中某指令色为画笔。
        if command is None:  # 色块按钮回调不带 command，从按钮查。
            for btn in self._swatch_buttons:
                if btn.rgb == tuple(rgb):
                    command = getattr(btn, '_cmd_text', '')
                    break
        for btn in self._swatch_buttons:  # 更新高亮。
            btn.set_selected(btn.rgb == tuple(rgb))
        self.eraser_btn.setChecked(False)  # 选色退出橡皮。
        self.canvas.set_brush(rgb=rgb)
        self._brush_rgb = tuple(rgb)
        if not silent:
            self.brush_status.setText(f"画笔: RGB{tuple(rgb)} → {command}")

    # ------------------------------------------------------------------ 卡片 3.5：实时路线录制

    def _build_record_card(self):  # 录制控制卡：选目标路线 + 当前指令色实时描线 + 试定位校验。
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        self.record_route_check = SwitchButton()  # 开=录到当前选中路线（不新建），关=自动新建一条路线再录。
        row.addWidget(BodyLabel("录到当前路线:"))
        row.addWidget(self.record_route_check)
        row.addSpacing(12)
        self.try_locate_btn = PushButton(FluentIcon.SEARCH, "试定位当前帧")
        self.try_locate_btn.clicked.connect(self.on_try_locate)  # 单帧验证小地图/黄点几何。
        row.addWidget(self.try_locate_btn)
        self.record_btn = PrimaryPushButton(FluentIcon.PLAY, "开始录制")
        self.record_btn.clicked.connect(self.on_toggle_record)  # 开始/停止切换。
        row.addWidget(self.record_btn)
        row.addStretch(1)
        self.record_status = BodyLabel("未录制")  # 录制状态/统计文本（由轮询刷新）。
        row.addWidget(self.record_status, 2)
        self.add_card("实时路线录制", container)

    def _is_recording(self):  # 是否有运行中的录制线程。
        return self._recorder is not None and not self._recorder.finished  # 线程未结束即视为录制中。

    def on_toggle_record(self):  # 开始/停止录制切换按钮。
        if self._is_recording():  # 正在录制。
            self._stop_recorder()  # 请求停止（收尾落盘由线程完成，轮询回载）。
            return
        self._start_recorder()  # 否则开始录制。

    def _start_recorder(self):  # 校验前置条件并启动后台录制线程。
        if not self.current_map:  # 未选地图。
            self.record_status.setText("请先选择或新建地图。")  # 提示。
            return
        meta = load_meta(self.current_map)  # 读取定位参数快照。
        feature_name = str(meta.get('Minimap Feature') or '').strip()  # 小地图模板名。
        if not feature_name:  # 未配小地图模板。
            self.record_status.setText("请先在参数卡设置小地图模板名。")  # 提示。
            return
        if not self._feature_ready(feature_name):  # 模板未标注。
            self.record_status.setText(f"小地图模板未标注：{feature_name}，请去模板页标注。")  # 提示。
            return
        if self._capture_current_frame() is None:  # 取不到画面。
            self.record_status.setText("无画面：请先连接游戏窗口再录制。")  # 提示。
            return
        # 确定目标路线图：开关开则用当前选中路线（空则新建），关则总是新建一条。
        if self.record_route_check.isChecked() and self.current_route:  # 录到当前路线。
            target_route = self.current_route  # 用选中路线。
        else:  # 新建一条。
            target_route = add_route(self.current_map)  # 新建与 map.png 同尺寸的全黑路线。
            if not target_route:  # 无 map.png 无法定尺寸。
                self.record_status.setText("新建路线失败：请先导入或录制出底图。")  # 提示。
                return
            self._reload_route_list()  # 刷新路线下拉。
            idx = self.route_combo.findText(target_route)  # 定位新路线。
            if idx >= 0:
                self.route_combo.setCurrentIndex(idx)  # 切换过去。
        confirm = QMessageBox.question(
            self, "开始录制",
            f"录制将重建地图 '{self.current_map}' 的底图 map.png 与路线 {target_route}，"
            f"并边移动边按当前指令色描轨迹。确定开始？",
            QMessageBox.Yes | QMessageBox.No)  # 破坏性操作先确认。
        if confirm != QMessageBox.Yes:  # 取消。
            return
        self._recorder = MapRecorder(self.current_map, target_route, meta, lambda: self._brush_rgb)  # 创建线程，当前色通过回调实时取。
        self._recorder.start()  # 启动。
        self._rec_timer.start()  # 开始轮询。
        self.record_btn.setText("停止录制")  # 按钮切换为停止。
        self.record_route_check.setEnabled(False)  # 录制中锁定选项。
        self.record_status.setText(f"录制中→ {target_route}，请在游戏里控角色沿 intended 路线走一圈...")  # 提示。
        self.logger.info(f'start map recording: {self.current_map}/{target_route}')  # 日志。

    def _stop_recorder(self):  # 请求停止录制（不阻塞，落盘由线程收尾，轮询检测 finished 后回载）。
        if self._recorder is not None:  # 有线程。
            self._recorder.stop()  # 置停止标志。
        self.record_btn.setText("停止中...")  # 反馈。
        self.record_btn.setEnabled(False)  # 防重复点。

    def _poll_recorder(self):  # 定时器回调：刷新录制状态文本，线程结束后回载资产并停轮询。
        rec = self._recorder  # 局部引用。
        if rec is None:  # 无录制。
            self._rec_timer.stop()  # 停轮询。
            return
        self.record_status.setText(f"录制中：{rec.status}  贴图{rec.stats['pasted']} 落点{rec.stats['located']} 丢失{rec.stats['lost']}")  # 实时展示（仅读线程属性，跨线程安全）。
        if rec.finished:  # 线程已结束并完成落盘。
            self._rec_timer.stop()  # 停轮询。
            self.record_btn.setEnabled(True)  # 恢复按钮。
            self.record_btn.setText("开始录制")  # 恢复文案。
            self.record_route_check.setEnabled(True)  # 解锁。
            if rec.saved:  # 成功落盘。
                self.record_status.setText(f"已录制完成：{rec.status} 落点{rec.stats['located']}。")  # 提示。
            else:  # 未落盘。
                self.record_status.setText(f"录制未产生资产：{rec.error or '无数据'}。")  # 提示。
            self._reload_assets()  # 重新加载底图/路线，录制结果直接可见。
            self.logger.info(f'map recording finished saved={rec.saved} err={rec.error}')  # 日志。
            self._recorder = None  # 释放引用。

    def on_try_locate(self):  # 单帧试定位：验证小地图模板与黄点几何是否可正确提取（录制前置检查）。
        if not self.current_map:  # 未选地图。
            self.record_status.setText("请先选择地图。")  # 提示。
            return
        meta = load_meta(self.current_map)  # 读参数。
        feature_name = str(meta.get('Minimap Feature') or '').strip()  # 小地图模板名。
        frame = self._capture_current_frame()  # 取一帧。
        if frame is None:  # 无画面。
            self.record_status.setText("无画面：请先连接游戏窗口。")  # 提示。
            return
        box = self._match_minimap(frame, feature_name, float(meta.get('Minimap Threshold') or 0.8))  # 定位小地图框。
        if box is None:  # 未找到小地图。
            self.record_status.setText(f"试定位：未匹配到小地图（检查模板 '{feature_name}'/阈值）。")  # 提示。
            return
        rx, ry, rw, rh = compute_map_rect(box.x, box.y, box.width, box.height, str(meta.get('Map Rect') or ''))  # 实际地图区域。
        rx = max(0, rx); ry = max(0, ry)  # 夹到画面内。
        rw = min(rw, frame.shape[1] - rx); rh = min(rh, frame.shape[0] - ry)  # 防越界。
        dot = detect_yellow_dot(frame, (rx, ry, rw, rh), int(meta.get('Dot Hue Min') or 18),  # 检测黄点。
                                int(meta.get('Dot Hue Max') or 38), max(1, int(meta.get('Dot Min Pixels') or 4)))
        if dot is None:  # 未找到黄点。
            self.record_status.setText(f"试定位：小地图 {rw}x{rh} @({rx},{ry})，但未检测到黄点（调色相）。")  # 提示。
            return
        self.record_status.setText(f"试定位 OK：小地图 {rw}x{rh} @({rx},{ry})，黄点@({dot[0]},{dot[1]}) 面积{dot[2]}。可开始录制。")  # 成功。

    def _feature_ready(self, feature_name):  # 检查模板页是否已标注指定分类（不可用时返回 False）。
        executor = getattr(og, 'executor', None)  # 取执行器。
        feature_set = getattr(executor, 'feature_set', None) if executor is not None else None  # 取特征集。
        if feature_set is None:  # 不可用。
            return False  # 未就绪。
        try:  # 标注文件加载失败视为未就绪。
            return bool(feature_set.feature_exists(feature_name))  # 查询。
        except Exception:  # 异常。
            return False  # 未就绪。

    def _match_minimap(self, frame, feature_name, threshold):  # 全屏模板匹配小地图框，返回最高分框或 None。
        executor = getattr(og, 'executor', None)  # 取执行器。
        feature_set = getattr(executor, 'feature_set', None) if executor is not None else None  # 取特征集。
        if feature_set is None or not feature_name:  # 不可用或未配名。
            return None  # 无法匹配。
        try:  # 标注不存在会抛 ValueError。
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)  # 全屏灰度匹配，与录制线程同款。
        except (ValueError, Exception):  # 未标注或匹配异常。
            return None  # 未匹配。
        if not boxes:  # 无达标框。
            return None  # 未匹配。
        return max(boxes, key=lambda b: getattr(b, 'confidence', 0))  # 取最高分框。

    @staticmethod
    def _pair_widget(left, right):  # 把两个数框横向打包成一个表单行控件（如黄点色相下/上限）。
        holder = QWidget()  # 容器。
        hl = QHBoxLayout(holder)  # 横向布局。
        hl.setContentsMargins(0, 0, 0, 0)  # 去边距。
        hl.addWidget(left)  # 左数框。
        hl.addWidget(right)  # 右数框。
        return holder  # 返回打包控件。

    # ------------------------------------------------------------------ 卡片 4：指令表编辑

    def _build_table_card(self):  # 主表 + 上下表两张可编辑表格。
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.main_table = make_two_col_table()
        self.ud_table = make_two_col_table()
        grid.addWidget(BodyLabel("主指令色表 Color Code"), 0, 0)
        grid.addWidget(BodyLabel("上下专用色表 Up/Down"), 0, 1)
        grid.addWidget(self.main_table, 1, 0)
        grid.addWidget(self.ud_table, 1, 1)
        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        add_main = PushButton(FluentIcon.ADD, "加行(主表)")
        add_main.clicked.connect(lambda: self.main_table.insertRow(self.main_table.rowCount()))
        add_ud = PushButton(FluentIcon.ADD, "加行(上下)")
        add_ud.clicked.connect(lambda: self.ud_table.insertRow(self.ud_table.rowCount()))
        reset_btn = PushButton(FluentIcon.UPDATE, "恢复默认色表")
        reset_btn.clicked.connect(self.on_reset_color_tables)
        bl.addWidget(add_main)
        bl.addWidget(add_ud)
        bl.addStretch(1)
        bl.addWidget(reset_btn)
        grid.addWidget(btn_row, 2, 0, 1, 2)
        self.add_card("指令表编辑", widget)

    def _load_tables(self, meta):  # 用 meta 填充两张表。
        fill_color_table(self.main_table, meta.get('Color Code') or {})
        fill_color_table(self.ud_table, meta.get('Color Code Up Down') or {})

    def on_reset_color_tables(self):  # 恢复为内置默认色表。
        fill_color_table(self.main_table, dict(DEFAULT_COLOR_CODE))
        fill_color_table(self.ud_table, dict(DEFAULT_COLOR_CODE_UP_DOWN))
        self.logger.info('color tables reset to default')

    # ------------------------------------------------------------------ 卡片 5：参数与保存

    def _build_param_card(self):  # 导航/定位/打怪参数 + 保存按钮。
        widget = QWidget()
        outer = QHBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()  # 左列：导航 + 定位参数。
        self.search_range_spin = SpinBox()
        self.search_range_spin.setRange(1, 100)
        self.edge_color_edit = LineEdit()
        self.edge_color_edit.setPlaceholderText("255,127,127 留空禁用")
        self.teleport_walk_switch = SwitchButton()
        self.teleport_cd_spin = DoubleSpinBox()
        self.teleport_cd_spin.setRange(0.0, 30.0)
        self.teleport_cd_spin.setSingleStep(0.5)
        self.teleport_cd_spin.setDecimals(1)
        self.teleport_cd_spin.setSuffix(" s")
        self.minimap_feature_edit = LineEdit()
        self.minimap_feature_edit.setPlaceholderText("完整小地图")
        self.minimap_threshold_spin = DoubleSpinBox()
        self.minimap_threshold_spin.setRange(0.01, 1.0)
        self.minimap_threshold_spin.setSingleStep(0.05)
        self.minimap_threshold_spin.setDecimals(2)
        # —— 小地图几何与黄点检测（录制与运行时共用，与地图资产绑定）——
        self.map_rect_edit = LineEdit()
        self.map_rect_edit.setPlaceholderText("5,20,90,75 留空=整个小地图框")
        self.dot_hue_min_spin = SpinBox()
        self.dot_hue_min_spin.setRange(0, 179)
        self.dot_hue_max_spin = SpinBox()
        self.dot_hue_max_spin.setRange(0, 179)
        self.dot_min_spin = SpinBox()
        self.dot_min_spin.setRange(1, 500)
        # —— 录制参数（仅地图页签路线录制使用）——
        self.canvas_size_edit = LineEdit()
        self.canvas_size_edit.setPlaceholderText("1600,1200 拼接画布预分配尺寸 宽,高")
        self.trace_thickness_spin = SpinBox()
        self.trace_thickness_spin.setRange(1, 20)
        form.addRow("搜索半径(px)", self.search_range_spin)
        form.addRow("边缘保护色RGB", self.edge_color_edit)
        form.addRow("行走也用瞬移", self.teleport_walk_switch)
        form.addRow("瞬移冷却(s)", self.teleport_cd_spin)
        form.addRow("小地图模板名", self.minimap_feature_edit)
        form.addRow("小地图阈值", self.minimap_threshold_spin)
        form.addRow("地图区域(%)", self.map_rect_edit)
        form.addRow("黄点色相下/上", self._pair_widget(self.dot_hue_min_spin, self.dot_hue_max_spin))
        form.addRow("黄点最小面积", self.dot_min_spin)
        form.addRow("拼接画布尺寸", self.canvas_size_edit)
        form.addRow("描线粗细", self.trace_thickness_spin)

        right = QVBoxLayout()  # 右列：怪物特征 + 保存。
        self.monster_combo = EditableComboBox()  # 怪物特征（类别=怪物的标注，供特殊地图覆盖看板）。
        right.addWidget(BodyLabel("怪物特征(本地图覆盖,留空用看板)"))
        right.addWidget(self.monster_combo)
        right.addStretch(1)
        action = QWidget()
        al = QHBoxLayout(action)
        al.setContentsMargins(0, 0, 0, 0)
        self.save_button = PrimaryPushButton(FluentIcon.SAVE, "保存配置")
        self.save_button.clicked.connect(self.save)
        self.status_label = BodyLabel("")
        al.addWidget(self.save_button)
        al.addWidget(self.status_label, 1)
        right.addWidget(action)

        outer.addLayout(form, 1)
        outer.addLayout(right, 1)
        self.add_card("参数与保存", widget)

    def _load_params(self, meta):  # 用 meta 填充参数控件。
        self._loading = True
        self.search_range_spin.setValue(int(meta.get('Search Range', 10)))
        self.edge_color_edit.setText(str(meta.get('Edge Color', '')))
        self.teleport_walk_switch.setChecked(bool(meta.get('Use Teleport To Walk', False)))
        self.teleport_cd_spin.setValue(float(meta.get('Teleport Cooldown', 1.0)))
        self.minimap_feature_edit.setText(str(meta.get('Minimap Feature', '')))
        self.minimap_threshold_spin.setValue(float(meta.get('Minimap Threshold', 0.8)))
        self.map_rect_edit.setText(str(meta.get('Map Rect', '')))
        self.dot_hue_min_spin.setValue(int(meta.get('Dot Hue Min', 18)))
        self.dot_hue_max_spin.setValue(int(meta.get('Dot Hue Max', 38)))
        self.dot_min_spin.setValue(max(1, int(meta.get('Dot Min Pixels', 4))))
        self.canvas_size_edit.setText(str(meta.get('Record Canvas Size', '1600,1200')))
        self.trace_thickness_spin.setValue(max(1, int(meta.get('Record Trace Thickness', 2))))
        self.monster_combo.clear()
        try:
            monster_names = sorted((load_annotations_by_supercategory().get(SUPER_MONSTER) or {}).keys())
        except Exception:
            monster_names = []
        self.monster_combo.addItems([PLACEHOLDER] + monster_names)
        value = str(meta.get('Monster Features', '') or '')
        if value and self.monster_combo.findText(value) < 0:
            self.monster_combo.addItem(value)
        self.monster_combo.setCurrentText(value or PLACEHOLDER)
        self._loading = False

    def _combo_value(self, combo):  # 读下拉当前值，占位映射回空串。
        text = str(combo.currentText() or '').strip()
        return '' if text == PLACEHOLDER else text

    def save(self):  # 收集全部控件写回当前地图 meta.json。
        if not self.current_map:
            self.status_label.setText("请先选择或新建地图。")
            return
        meta = load_meta(self.current_map)  # 基于磁盘现有配置更新，避免丢未知键。
        meta['Search Range'] = int(self.search_range_spin.value())
        meta['Edge Color'] = self.edge_color_edit.text().strip()
        meta['Use Teleport To Walk'] = bool(self.teleport_walk_switch.isChecked())
        meta['Teleport Cooldown'] = round(float(self.teleport_cd_spin.value()), 1)
        meta['Minimap Feature'] = self.minimap_feature_edit.text().strip()
        meta['Minimap Threshold'] = round(float(self.minimap_threshold_spin.value()), 2)
        meta['Map Rect'] = self.map_rect_edit.text().strip()
        meta['Dot Hue Min'] = int(self.dot_hue_min_spin.value())
        meta['Dot Hue Max'] = int(self.dot_hue_max_spin.value())
        meta['Dot Min Pixels'] = int(self.dot_min_spin.value())
        meta['Record Canvas Size'] = self.canvas_size_edit.text().strip() or '1600,1200'
        meta['Record Trace Thickness'] = int(self.trace_thickness_spin.value())
        meta['Monster Features'] = self._combo_value(self.monster_combo)
        meta['Color Code'] = read_color_table(self.main_table)
        meta['Color Code Up Down'] = read_color_table(self.ud_table)
        save_meta(self.current_map, meta)
        self._rebuild_palette(meta)  # 色表可能变化，重建调色板。
        self._update_preview_info()
        self.status_label.setText(f"已保存到 maps/{self.current_map}/meta.json")
        self.logger.info(f'map meta saved for {self.current_map}')

    # ------------------------------------------------------------------ 概要信息

    def _update_preview_info(self):  # 刷新预览与校验卡片文本。
        if not self.current_map:
            self.preview_info.setText("未选择地图")
            return
        size = map_size(self.current_map)
        routes = list_routes(self.current_map)
        default = get_default_map()
        problems = [f"{r}: {err}" for r, err in validate_routes(self.current_map) if err]
        text = (f"地图 {self.current_map}  尺寸 {size[0] if size else '?'}x{size[1] if size else '?'}  "
                f"路线 {len(routes)} 条  默认地图: {default or '无'}")
        if self.current_route:
            text += f"  正在编辑: {self.current_route}"
        if problems:
            text += "\n⚠ 尺寸校验: " + "; ".join(problems)
        self.preview_info.setText(text)
