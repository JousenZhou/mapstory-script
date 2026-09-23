# 模板页【导入图片】补丁（patch_template_import）无头冒烟测试：
# 验证补丁生效后工具栏出现导入按钮（位于截图按钮右侧）、导入流程把图片转存 PNG 进模板目录并插入网格卡片、
# 取消选择时不产生任何副作用。
# 模板目录重定向到临时目录、COCO 注册替换为本地桩，不触碰真实 ok_templates 与标注文件。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import shutil
import tempfile
import unittest
from unittest import mock

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


class TestTemplateImportPatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # 全部用例共享一个离屏应用实例。
        import src.ui.patch_template_import  # noqa: F401  应用补丁（模块级猴子补丁，重复导入无副作用）。

    def setUp(self):
        from ok.ui.qt.tasks.TemplateTab import TemplateTab
        self.tab = TemplateTab(None)  # 构造页签：只读既有配置，不触发网格加载（showEvent 才会）。
        self.temp_dir = tempfile.mkdtemp(prefix='template_import_test_')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)  # 清理临时目录及导入产物。

    def test_import_button_next_to_screenshot(self):
        toolbar = self.tab.layout().itemAt(0).layout()  # 主布局第一项即工具栏。
        self.assertIn('Import Image', self.tab.import_btn.text())  # 按钮文本为中英双语。
        self.assertGreater(toolbar.indexOf(self.tab.import_btn),
                           toolbar.indexOf(self.tab.screenshot_btn))  # 位于截图按钮右侧。

    def test_import_image_copies_png_and_inserts_card(self):
        source = os.path.join(self.temp_dir, '素材 图.png')  # 带中文与空格的源文件名，验证 QImage 路径兼容。
        image = QImage(64, 32, QImage.Format_RGB32)
        image.fill(0xFF336699)
        self.assertTrue(image.save(source, 'PNG'))
        import src.ui.patch_template_import as patch_module
        recorded = []
        self.tab._add_image_to_coco = lambda path: recorded.append(path)  # COCO 注册替换为本地桩，避免写真实标注文件。
        with mock.patch.object(patch_module.QFileDialog, 'getOpenFileName', return_value=(source, '')), \
                mock.patch.object(patch_module, 'ensure_template_folder', return_value=self.temp_dir):
            self.tab.import_image()
        self.assertEqual(1, len(recorded))  # 恰好注册一张图。
        imported = recorded[0]
        self.assertEqual(self.temp_dir, os.path.dirname(imported))  # 转存到模板目录。
        self.assertTrue(imported.endswith('.png'))  # 统一转存为 PNG。
        self.assertNotEqual(os.path.normcase(source), os.path.normcase(imported))  # 用序号新文件名，不覆盖源图。
        copied = QImage(imported)
        self.assertFalse(copied.isNull())  # 转存文件可正常读取。
        self.assertEqual((64, 32), (copied.width(), copied.height()))  # 尺寸保持原样。
        self.assertIn(imported, self.tab._items_by_path)  # 卡片已插入网格数据。
        self.assertIn(imported, self.tab._visible_image_paths)  # 刷新后可见。

    def test_import_cancelled_has_no_side_effect(self):
        import src.ui.patch_template_import as patch_module
        before_items = len(self.tab._image_items)
        with mock.patch.object(patch_module.QFileDialog, 'getOpenFileName', return_value=('', '')):
            self.tab.import_image()  # 取消选择应静默返回。
        self.assertEqual(before_items, len(self.tab._image_items))  # 网格数据无变化。
        self.assertEqual([], [p for p in os.listdir(self.temp_dir)])  # 临时目录无新文件。


if __name__ == '__main__':
    unittest.main()
