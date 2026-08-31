# 将框架“选择窗口”卡片由“整表列出全部窗口”改为“输入框搜索 + 下拉选择”，并缓存上次选择。
#
# 背景：去掉 windows.exe 锁定后，DeviceManager.update_pc_device() 会枚举系统全部可见窗口；
# 窗口数量较大时，框架原有 ListWidget 一次性创建并渲染所有条目，界面会明显卡顿。
# 这里改用 QComboBox(可编辑 + QCompleter 过滤)：下拉列表按需渲染不会卡，输入关键字即时过滤窗口列表。
#
# 框架的 configs/devices.json 只持久化临时窗口句柄（preferred/selected_hwnd），重启后窗口重开、
# 句柄必然变化，全枚举模式下无法恢复上次选择；因此本补丁额外用 exe 路径 + 窗口标题把选择缓存到
# configs/window_selection.json，启动后在设备列表里按标识回选并应用。
#
# 采用运行时猴子补丁（与 MyBaseTask.py 的框架补丁模式一致），不直接改 site-packages，
# 重装依赖后仍生效。只需在启动 GUI 之前 import 本模块。
import json  # 导入 json，用于读写窗口选择缓存文件。
import os  # 导入 os，用于定位缓存文件路径。

from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QCompleter

import ok.ui.qt.start.StartTab as _start_tab_module

_StartTab = _start_tab_module.StartTab

_original_init = _StartTab.__init__
_original_update_capture = _StartTab.update_capture
_original_update_selection = _StartTab.update_selection
_original_filter_devices = _StartTab.filter_devices

SELECTION_CACHE_FILE = os.path.join('configs', 'window_selection.json')  # 窗口选择缓存文件：重启后据此带回上次选中项。


def _exe_path_of(device):  # 取设备的 exe 完整路径（windows 设备的 exe 可能是列表，取第一个）。
    exe = device.get('exe')
    if isinstance(exe, (list, tuple)):  # 列表形式取首个非空项。
        exe = exe[0] if exe else ""
    return str(exe or "")


def _load_selection_cache():  # 读取上次窗口选择缓存，读失败按无缓存处理。
    try:
        with open(SELECTION_CACHE_FILE, encoding='utf-8') as f:  # 缓存为 JSON，存 exe 路径与窗口标题。
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):  # 文件不存在或内容损坏都视为首次使用。
        return None


def _save_selection_cache(exe_path, title):  # 把当前选中窗口的稳定标识写入缓存，供下次启动带回。
    try:
        with open(SELECTION_CACHE_FILE, 'w', encoding='utf-8') as f:  # 覆盖写入，只保留最后一次选择。
            json.dump({"exe_path": exe_path, "title": title}, f, ensure_ascii=False)
    except OSError:  # 缓存写失败不影响选择功能，只是下次启动无法自动带回。
        pass


def _device_display_text(self, device):  # 生成一条既可读又便于按标题/进程名搜索的描述文本。
    if device.get('device') == 'windows':
        kind = self.tr("PC")
    elif device.get('emulator'):
        kind = self.tr("Emulator")
    elif device.get('device') == 'browser':
        kind = self.tr("Browser")
    else:
        kind = self.tr("Android")
    connected = self.tr("Connected") if device.get('connected') else self.tr("Disconnected")
    nick = str(device.get('nick') or "")
    exe = _exe_path_of(device)  # 复用统一的 exe 路径提取。
    address = str(device.get('address') or "")
    resolution = str(device.get('resolution') or "")
    text = f"{kind} {connected}: {nick}"
    if exe:
        text += f" ({exe})"
    if address:
        text += f" {address}"
    if resolution:
        text += f" {resolution}"
    return text


def _install_searchable_combo(self):  # 用可搜索下拉框替换原窗口列表及其独立搜索框。
    old_list = self.device_list
    view_widget = old_list.parentWidget()
    layout = view_widget.layout() if view_widget is not None else None
    if self.device_search_box is not None:  # 搜索能力并入下拉框补全器，移除原搜索框避免重复。
        if layout is not None:
            layout.removeWidget(self.device_search_box)
        self.device_search_box.deleteLater()
        self.device_search_box = None
    if layout is not None:
        layout.removeWidget(old_list)
    old_list.deleteLater()

    combo = QComboBox()
    combo.setEditable(True)  # 允许键入关键字触发搜索。
    combo.setInsertPolicy(QComboBox.NoInsert)  # 输入仅用于搜索，不会新增条目。
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    combo.setPlaceholderText(self.tr("Search title or exe..."))  # 输入框占位提示，与框架原搜索框文案一致。
    completer = QCompleter(combo)
    completer.setCompletionMode(QCompleter.PopupCompletion)
    completer.setFilterMode(Qt.MatchContains)  # 关键字出现在任意位置即匹配。
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    model = QStringListModel(combo)
    completer.setModel(model)
    combo.setCompleter(completer)
    self._device_combo_model = model
    combo.currentIndexChanged.connect(lambda index: _on_combo_changed(self, index))
    completer.activated.connect(lambda text: _on_completer_activated(self, text))
    combo.lineEdit().textChanged.connect(lambda text: _on_search_text_changed(self, text))  # 输入即弹下拉展示过滤后的窗口。
    if layout is not None:
        layout.addWidget(combo)
    self.device_combo = combo
    self.device_list_row = -1


def _on_combo_changed(self, index):  # 下拉选中变化 -> 设为首选设备并刷新采集/交互列表，同时缓存选择。
    self.device_list_row = index
    if index == -1:
        return
    from ok import og
    devices = og.device_manager.get_devices()
    if 0 <= index < len(devices):  # 把选中窗口的稳定标识写入缓存，下次启动自动带回。
        device = devices[index]
        _save_selection_cache(_exe_path_of(device), str(device.get('nick') or ""))
    og.device_manager.set_preferred_device(index=index)
    self.capture_list.update_for_device()
    self.interaction_list.update_for_device()


def _on_completer_activated(self, text):  # 从搜索建议里点选某项 -> 定位到对应下标。
    index = self.device_combo.findText(text)
    if index < 0:
        return
    if self.device_combo.currentIndex() != index:
        self.device_combo.setCurrentIndex(index)  # 触发 currentIndexChanged -> _on_combo_changed。
    else:
        _on_combo_changed(self, index)  # 已是当前项时补全器不会触发信号，手动应用一次。
    self.device_combo.lineEdit().setText(self.device_combo.itemText(index))  # 选中后输入框展示完整窗口描述。
    self.device_combo.lineEdit().selectAll()  # 全选便于继续输入新关键字重新搜索。
    self.device_combo.hidePopup()  # 收起下拉，避免挡住画面。
    self.device_combo.clearFocus()  # 释放输入框焦点，避免后续键盘操作被输入框拦截。


def _on_search_text_changed(self, text):  # 输入框内容变化：保证下拉候选展示。
    combo = self.device_combo
    if not combo.lineEdit().hasFocus():
        return  # 程序化回显选中项也会触发本信号，只响应用户真实输入，避免破坏下拉状态。
    # 有关键字时补全器（PopupCompletion + MatchContains）会自动弹出过滤后的候选，无需手动干预；
    # 清空关键字且下拉未展开时，直接展示全部窗口供浏览挑选。
    if not text.strip() and not combo.view().isVisible():
        combo.showPopup()


def _restore_saved_selection(self, devices):  # 按缓存的 exe 路径 + 窗口标题在设备列表中找回上次选择。
    cache = _load_selection_cache()
    if not cache:  # 没有缓存（首次使用或文件损坏）：不做任何回选。
        return -1
    exe_path = str(cache.get('exe_path') or "")
    title = str(cache.get('title') or "")
    if not exe_path and not title:  # 缓存内容为空也没有可回选的目标。
        return -1
    exe_match = -1
    for row, device in enumerate(devices):  # 优先 exe 路径 + 标题都一致；其次只匹配 exe 路径（同程序标题可能变化）。
        if exe_path and _exe_path_of(device) == exe_path:
            if exe_match == -1:
                exe_match = row
            if str(device.get('nick') or "") == title:
                return row
    return exe_match


def _patched_init(self, config, exit_event):
    _original_init(self, config, exit_event)  # 原构造会先填充旧列表，随后替换控件。
    _install_searchable_combo(self)
    self.update_capture(True)  # 向新下拉框重新填充一次设备。


def _patched_update_capture(self, finished):  # 重写设备填充逻辑：下拉框按需渲染，支持大量窗口，并带回上次选择。
    if not hasattr(self, 'device_combo'):  # 构造过程中（下拉框尚未安装）走原逻辑。
        return _original_update_capture(self, finished)
    from ok import og
    devices = og.device_manager.get_devices()
    preferred = og.device_manager.config.get("preferred")
    combo = self.device_combo
    previous = combo.currentIndex()
    selected_index = -1
    combo.blockSignals(True)
    combo.clear()
    texts = []
    for row, device in enumerate(devices):
        if device.get('imei') == preferred:
            selected_index = row
        combo.addItem(_device_display_text(self, device))
    combo.blockSignals(False)
    # 补全器候选与下拉项保持同一顺序，保证下标即 get_devices() 的设备下标；
    # 输入搜索时补全器基于该全量模型自动过滤（MatchContains）。
    texts = [combo.itemText(i) for i in range(combo.count())]
    self._device_combo_model.setStringList(texts)
    saved_index = _restore_saved_selection(self, devices)  # 优先按缓存回选：框架存的句柄重启后必然失效。
    if saved_index >= 0:
        selected_index = saved_index
    if selected_index == -1 and 0 <= previous < combo.count():
        selected_index = previous  # 首选已不存在时尽量保持原选中。
    if selected_index >= 0:
        combo.blockSignals(True)
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)
        if saved_index >= 0 and saved_index != previous:  # 缓存命中且与框架默认不同：主动应用为首选设备。
            _on_combo_changed(self, selected_index)
        combo.lineEdit().setText(combo.itemText(selected_index))  # 输入框直接展示选中的窗口，一眼可见当前选择。
        combo.lineEdit().selectAll()
    self.device_list_row = selected_index
    if finished:
        self.start_card.refresh_button.setDisabled(False)
        self.start_card.refresh_button.setText(self.tr("Refresh"))
        self.capture_list.update_for_device()
        self.interaction_list.update_for_device()


def _patched_update_selection(self):
    if not hasattr(self, 'device_combo'):
        return _original_update_selection(self)
    from ok import og
    if og.executor.paused:  # 下拉框无“选择模式”概念，仅保留采集/交互列表的选择约束。
        self.capture_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.interaction_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.update_window_list()


def _patched_filter_devices(self, text=None):
    if not hasattr(self, 'device_combo'):
        return _original_filter_devices(self, text)
    return  # 搜索过滤已由下拉框补全器承担，无需再逐条隐藏。


_StartTab.__init__ = _patched_init
_StartTab.update_capture = _patched_update_capture
_StartTab.update_selection = _patched_update_selection
_StartTab.filter_devices = _patched_filter_devices
