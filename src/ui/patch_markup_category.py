# 模板标注窗口新增「类别」字段（位于分类与 X 行之间），落在 COCO 标准字段 categories[].supercategory。
#
# 背景：后续看板/任务需要按「类别」（测谎、测谎触发等）选择标注。COCO 格式的
# supercategory 字段本就存在（当前全为空字符串），直接复用，无需改动文件格式。
#
# 补丁点：
# 1. BBoxDialog.__init__          原构造后追加可编辑类别下拉框，插到表单第 1 行（分类与 X 之间）；
# 2. BBoxDialog._commit_supercategory 点确定时把类别缓存到模块级变量，供 _finish_drawing 取值
#    （画框流程是长方法，不整体重写，靠通道取值）；
# 3. AnnotationCanvas._finish_drawing  原逻辑后给新增标注补类别；
# 4. AnnotationCanvas._edit_annotation 整体重写（短方法），传入当前类别并在保存时回写；
# 5. MarkUpWindow.load_current_image   原逻辑后从 categories[].supercategory 回填画布标注类别；
# 6. MarkUpWindow.save_annotations     原逻辑后把画布类别写回 categories[].supercategory 并持久化。
#
# 采用运行时猴子补丁（与 patch_window_selector.py 模式一致），不直接改 site-packages，
# 重装依赖后仍生效。只需在启动 GUI 之前 import 本模块。
import os  # 导入 os，用于取当前图片文件名（与原逻辑一致）。

from PySide6.QtWidgets import QFormLayout
from qfluentwidgets import EditableComboBox

from ok.ui.qt.tasks.MarkUpWindow import (AnnotationCanvas, BBoxDialog, MarkUpWindow,
                                         save_coco)

# 类别下拉预设：首个空项表示未分类，下拉可编辑也支持手输新类别。
# 角色/怪物供看板角色栏与怪物栏按类别选标注，测谎/测谎触发供测谎栏使用。
PRESET_SUPERCATEGORIES = ['', '角色', '怪物', '测谎', '测谎触发', '服务区', '频道', '掉线']

_last_committed_supercategory = ''  # 最近一次在 BBoxDialog 点「确定」时的类别值（画框流程取值通道）。

_original_bbox_init = BBoxDialog.__init__
_original_finish_drawing = AnnotationCanvas._finish_drawing
_original_load_current_image = MarkUpWindow.load_current_image
_original_save_annotations = MarkUpWindow.save_annotations


def _collect_supercategory_options(dialog):  # 收集可选类别：预设值 + 当前 coco_data 已用过的类别。
    options = list(PRESET_SUPERCATEGORIES)
    widget = dialog.parent()
    while widget is not None:  # 对话框 parent 即标注窗口，向上找到持有 coco_data 的窗口。
        coco_data = getattr(widget, 'coco_data', None)
        if isinstance(coco_data, dict):
            for cat in coco_data.get('categories', []):
                super_name = str(cat.get('supercategory') or '').strip()
                if super_name and super_name not in options:
                    options.append(super_name)
            break
        widget = widget.parentWidget()
    return options


def _patched_bbox_init(self, parent, category="", x=0, y=0, w=0, h=0, existing_categories=None,
                       current_image_name="", editing_original_name="", supercategory=""):
    _original_bbox_init(self, parent, category, x, y, w, h,
                        existing_categories=existing_categories,
                        current_image_name=current_image_name,
                        editing_original_name=editing_original_name)
    self.super_input = EditableComboBox(self)  # 可编辑下拉：已有类别可选，也可输入新类别。
    self.super_input.addItems(_collect_supercategory_options(self))
    self.super_input.setCurrentText(supercategory)  # 无匹配项时保持空文本（未分类）。
    form_layout = None
    for i in range(self.viewLayout.count()):  # 找到原构造加入的 QFormLayout。
        item = self.viewLayout.itemAt(i)
        layout = item.layout() if item is not None else None
        if isinstance(layout, QFormLayout):
            form_layout = layout
            break
    if form_layout is not None:
        form_layout.insertRow(1, "类别:", self.super_input)  # 插在分类行（0）与 X 行之间。
    else:  # 框架结构变化时的兜底：放对话框末尾，保证功能不丢。
        self.viewLayout.addWidget(self.super_input)
    self.yesButton.clicked.connect(self._commit_supercategory)  # 确认时缓存类别，供画框流程读取。


def _commit_supercategory(self):  # 点「确定」时缓存类别值（先于 exec 返回，取值必为最新）。
    global _last_committed_supercategory
    _last_committed_supercategory = self.super_input.currentText().strip()


def _patched_finish_drawing(self, end_pos):  # 画框收尾原逻辑 + 给新增标注补类别。
    before = len(self.annotations)
    _original_finish_drawing(self, end_pos)
    if len(self.annotations) > before:  # 对话框确认后才会新增一条标注。
        self.annotations[-1]['supercategory'] = _last_committed_supercategory


def _patched_edit_annotation(self, idx):  # 重写标注编辑：传入当前类别并在保存时回写（原方法短，整体重写更清晰）。
    ann = self.annotations[idx]
    existing_cats = self._build_existing_categories_map()
    current_image = ""
    if self.markup_window and self.markup_window.image_list:
        current_image = os.path.basename(self.markup_window.image_list[self.markup_window.current_index])

    original_name = ann.get('category', '')

    dialog = BBoxDialog(self.window(),
                        original_name,
                        ann['x'], ann['y'], ann['w'], ann['h'],
                        existing_categories=existing_cats,
                        current_image_name=current_image,
                        editing_original_name=original_name,
                        supercategory=ann.get('supercategory', ''))
    if dialog.exec():
        cat, x, y, w, h = dialog.get_values()
        if cat:
            ann['category'] = cat
            ann['x'] = x
            ann['y'] = y
            ann['w'] = w
            ann['h'] = h
            ann['supercategory'] = dialog.super_input.currentText().strip()
            self.annotations_changed.emit()
            self.update()


def _patched_load_current_image(self):  # 图片加载原逻辑 + 从 categories[].supercategory 回填标注类别。
    _original_load_current_image(self)
    super_map = {cat['name']: str(cat.get('supercategory') or '')
                 for cat in self.coco_data.get('categories', [])}
    for can_ann in self.canvas.annotations:
        can_ann['supercategory'] = super_map.get(can_ann.get('category', ''), '')


def _patched_save_annotations(self):  # 保存原逻辑 + 把画布标注类别写回 categories[].supercategory。
    _original_save_annotations(self)
    super_map = {can_ann.get('category', ''): can_ann.get('supercategory', '')
                 for can_ann in self.canvas.annotations}
    changed = False
    for cat in self.coco_data.get('categories', []):
        new_value = super_map.get(cat['name'])
        if new_value is not None and cat.get('supercategory', '') != new_value:
            cat['supercategory'] = new_value
            changed = True
    if changed:  # 原方法已存过一次，仅类别有变化时补存，避免无谓写盘。
        save_coco(self.coco_data)


BBoxDialog.__init__ = _patched_bbox_init
BBoxDialog._commit_supercategory = _commit_supercategory
AnnotationCanvas._finish_drawing = _patched_finish_drawing
AnnotationCanvas._edit_annotation = _patched_edit_annotation
MarkUpWindow.load_current_image = _patched_load_current_image
MarkUpWindow.save_annotations = _patched_save_annotations
