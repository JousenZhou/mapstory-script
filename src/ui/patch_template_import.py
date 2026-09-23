# 模板页新增【导入图片】按钮：与框架【截图】按钮同流程，只是图片来源从实时采集帧换成本地图片文件。
#
# 背景：框架 TemplateTab 只能通过 take_screenshot 抓当前采集窗口画面做标注模板，
# 无法导入外部图片（如网上下载的游戏素材、别的机器截的图）。本补丁在工具栏
# 【截图】按钮右侧加一个【导入图片】按钮，选图后统一转存为 ok_templates/<序号>.png，
# 注册进 COCO 并插入模板网格首位，之后双击即可进入标注窗口，与截图流程完全一致。
#
# 补丁点：
# 1. TemplateTab.__init__                   原构造后创建导入按钮并插入工具栏（截图按钮右侧）；
# 2. TemplateTab.import_image               新增方法：文件对话框选图 → 转存 PNG → 注册 COCO → 刷新网格；
# 3. TemplateTab._update_selection_buttons  原逻辑后同步导入按钮可用状态（网格加载中/标注窗口打开时禁用）。
#
# 采用运行时猴子补丁（与 patch_markup_category.py 模式一致），不直接改 site-packages，
# 重装依赖后仍生效。只需在启动 GUI 之前 import 本模块。
import os  # 拼接模板目录下的目标文件路径。

from PySide6.QtGui import QImage  # 用 QImage 读写图片：原生支持中文路径，cv2.imread 在非 ASCII 路径下会失败。
from PySide6.QtWidgets import QFileDialog  # 系统文件选择对话框。
from qfluentwidgets import FluentIcon, PushButton  # 与框架工具栏按钮风格一致。

from ok.ui.qt.tasks.TemplateTab import (TemplateTab, ensure_template_folder,
                                        get_next_image_name)  # 复用框架的模板目录与序号命名逻辑。
from ok.util.logger import Logger  # 框架日志器，导入失败时留痕。

logger = Logger.get_logger(__name__)

IMAGE_FILE_FILTER = "Images 图片 (*.png *.jpg *.jpeg *.bmp *.webp)"  # 文件对话框过滤：常见图片格式。

_original_init = TemplateTab.__init__
_original_update_selection_buttons = TemplateTab._update_selection_buttons


def _patched_init(self, config):  # 原构造后追加【导入图片】按钮到工具栏。
    _original_init(self, config)
    self.import_btn = PushButton(FluentIcon.ADD, "Import Image 导入图片")
    self.import_btn.clicked.connect(self.import_image)
    toolbar = None
    main_layout = self.layout()
    if main_layout is not None and main_layout.count() > 0:
        first_item = main_layout.itemAt(0)
        toolbar = first_item.layout() if first_item is not None else None
    if toolbar is not None:  # 正常路径：插到工具栏【截图】按钮右侧。
        toolbar.insertWidget(toolbar.indexOf(self.screenshot_btn) + 1, self.import_btn)
    elif main_layout is not None:  # 框架布局变化时的兜底：按钮放主布局第二行，保证功能不丢。
        main_layout.insertWidget(1, self.import_btn)


def _import_image(self):  # 选择本地图片导入模板库，流程与 take_screenshot 一致。
    from ok.ui.qt.util.Alert import alert_error, alert_info
    source_path, _ = QFileDialog.getOpenFileName(
        self, "Import Image 导入图片", os.getcwd(), IMAGE_FILE_FILTER)
    if not source_path:  # 用户取消选择。
        return
    try:
        qimg = QImage(source_path)  # QImage 支持任意编码路径，读入后统一转存 PNG。
        if qimg.isNull():
            alert_error(f"Cannot read image: {source_path}. 无法读取图片：{source_path}")
            return
        folder = ensure_template_folder()  # ok_templates 目录，不存在时框架自动创建。
        name = get_next_image_name(folder, self.coco_data)  # 与截图一致的递增序号命名。
        file_path = os.path.join(folder, f"{name}.png")
        if not qimg.save(file_path, "PNG"):  # 统一转存 PNG，保证模板匹配与压缩保存流程兼容。
            raise RuntimeError(f"Could not write image to {file_path}")
        self._add_image_to_coco(file_path)  # 注册进 coco_annotations.json（含宽高）。
        alert_info(f"Image imported: {os.path.basename(file_path)}. 图片已导入：{os.path.basename(file_path)}")
        self._create_image_item(file_path)  # 生成缩略图卡片并插入网格首位。
        self.apply_filter()  # 刷新可见卡片集。
    except Exception as e:
        logger.error(f"Import image error: {e}")
        alert_error(f"Import failed: {e}. 导入失败：{e}")


def _patched_update_selection_buttons(self):  # 原逻辑后同步导入按钮可用状态。
    _original_update_selection_buttons(self)
    import_btn = getattr(self, 'import_btn', None)  # 构造早期可能还没建按钮，防御性取值。
    if import_btn is not None:
        import_btn.setEnabled(not self._grid_loading and self.markup_window is None)


TemplateTab.__init__ = _patched_init
TemplateTab.import_image = _import_image
TemplateTab._update_selection_buttons = _patched_update_selection_buttons
