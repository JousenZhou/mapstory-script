# 路线绘制控件（src/ui/route_canvas.py）回归测试。
# 重点锁死三类问题：
#   1) 花屏：宽度非 4 倍数（如地图常见 1366）时，RGB888 每行字节数不满足 Qt 的 4 字节对齐行距，
#      缺省构造 QImage 会让每行错位若干字节，显示成整幅斜切条纹；断言逐像素颜色与源矩阵一致。
#   2) 叠色通道序：路线图按指令色 RGB 直写，底图是 cv2 的 BGR，叠加需换通道，否则选红笔显示成青色。
#   3) 合成缓存：落笔/撤销/清空后必须让 QPixmap 缓存失效，否则画面停在旧帧。
# 仅用 offscreen QApplication 驱动像素断言，不依赖真实设备与游戏画面。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 离屏平台设置，需在导入 PySide6 前生效。

import unittest

import numpy as np  # 图像矩阵。

from PySide6.QtWidgets import QApplication  # QImage/QPixmap 需要 QApplication 上下文。

from src.ui.route_canvas import RouteCanvas, _bgr_to_qimage  # 被测控件与转换函数。


def _odd_width_bgr(w=1366, h=60):  # 造一张宽度非 4 倍数、每行纯色互不相同的 BGR 图，便于暴露行错位。
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):  # 每行写不同的 BGR，行错位会立刻改变颜色。
        img[y, :] = (10 + y * 3 % 256, 200 - y % 200, 50 + (y * 7) % 206)
    return img


class TestBgrToQImage(unittest.TestCase):  # BGR -> QImage 的行距正确性。

    @classmethod
    def setUpClass(cls):  # QPixmap 转换需要 QApplication，offscreen 下不显示窗口。
        cls.app = QApplication.instance() or QApplication([])

    def test_odd_width_pixel_alignment(self):  # 宽度非 4 倍数时逐像素颜色须与源矩阵一致（花屏回归）。
        img = _odd_width_bgr()
        h, w, _ = img.shape
        self.assertNotEqual(0, (w * 3) % 4)  # 前置条件：RGB888 行距确实不 4 对齐，否则用例失去意义。
        qimg = _bgr_to_qimage(img)
        self.assertEqual(w, qimg.width())
        self.assertEqual(h, qimg.height())
        self.assertEqual(0, qimg.bytesPerLine() % 4)  # Qt 要求行距 4 字节对齐。
        for y in (0, 1, h // 2, h - 1):  # 抽查首行、末行与中间行。
            b, g, r = (int(v) for v in img[y, 0])
            color = qimg.pixelColor(0, y)
            self.assertEqual((r, g, b), (color.red(), color.green(), color.blue()), f"row {y} color shifted")  # 行错位会导致颜色偏移。

    def test_non_contiguous_input_still_correct(self):  # 传入非连续切片（ROI）也要正确转换（函数内部负责 contiguous 化）。
        img = _odd_width_bgr(w=40, h=20)
        roi = img[4:16, 6:32]  # 非连续视图。
        self.assertFalse(roi.flags['C_CONTIGUOUS'])
        qimg = _bgr_to_qimage(roi)
        self.assertEqual(roi.shape[1], qimg.width())
        color = qimg.pixelColor(0, 0)
        b, g, r = (int(v) for v in roi[0, 0])
        self.assertEqual((r, g, b), (color.red(), color.green(), color.blue()))


class TestRouteCanvasDisplay(unittest.TestCase):  # 控件合成与缓存失效。

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas_with_map(self):  # 载入 1366 宽底图的控件。
        canvas = RouteCanvas()
        canvas.set_images(_odd_width_bgr(), None)
        return canvas

    def test_composite_pixmap_matches_map(self):  # 无路线时缓存图尺寸等于底图且颜色不偏移。
        canvas = self._canvas_with_map()
        pm = canvas._ensure_pixmap()
        self.assertEqual(1366, pm.width())
        self.assertEqual(60, pm.height())
        img = pm.toImage()
        b, g, r = (int(v) for v in canvas._map[3, 0])
        color = img.pixelColor(0, 3)
        self.assertEqual((r, g, b), (color.red(), color.green(), color.blue()))

    def test_paint_invalidates_pixmap_cache(self):  # 落笔后 QPixmap 缓存须重建，显示叠加了指令色的合成图。
        canvas = self._canvas_with_map()
        before = canvas._ensure_pixmap()
        self.assertIsNotNone(before)
        self.assertIsNotNone(canvas._composite_pm)
        canvas.set_brush(rgb=(255, 0, 0), size=3)  # 红色指令画笔。
        canvas._paint_line((10, 10), (40, 10))  # 直接落一笔，绕开鼠标事件。
        self.assertIsNone(canvas._composite_pm)  # 缓存已失效。
        self.assertEqual((255, 0, 0), tuple(int(v) for v in canvas._route[10, 20]))  # 路线数组按指令色 RGB 直写。
        color = canvas._ensure_pixmap().toImage().pixelColor(20, 10)  # 显示端换通道后应仍是红色。
        self.assertEqual((255, 0, 0), (color.red(), color.green(), color.blue()))

    def test_brush_color_matches_palette(self):  # 非纯红绿蓝的指令色（如 (255,127,127)）显示也不能串位。
        canvas = self._canvas_with_map()
        canvas.set_brush(rgb=(255, 127, 127), size=2)
        canvas._paint_line((12, 30), (36, 30))
        color = canvas._ensure_pixmap().toImage().pixelColor(24, 30)
        self.assertEqual((255, 127, 127), (color.red(), color.green(), color.blue()))

    def test_map_background_not_swapped(self):  # 无路线像素处底图颜色须保持 cv2 原序显示，验证 BGR->RGB 只做一次。
        canvas = self._canvas_with_map()
        canvas.set_brush(rgb=(0, 255, 0), size=2)
        canvas._paint_line((100, 5), (120, 5))  # 在另一行画线，不影响采样点。
        img = canvas._ensure_pixmap().toImage()
        b, g, r = (int(v) for v in canvas._map[50, 0])
        color = img.pixelColor(0, 50)
        self.assertEqual((r, g, b), (color.red(), color.green(), color.blue()))

    def test_clear_and_undo_invalidate(self):  # 清空与撤销同样要让缓存失效。
        canvas = self._canvas_with_map()
        canvas.set_brush(rgb=(0, 255, 0), size=2)
        canvas._paint_line((5, 5), (30, 5))
        canvas.clear_route()
        self.assertIsNone(canvas._composite_pm)
        self.assertEqual(0, int(np.count_nonzero(canvas._route)))  # 已清空。
        canvas.undo()
        self.assertIsNone(canvas._composite_pm)
        self.assertGreater(int(np.count_nonzero(canvas._route)), 0)  # 撤销回退了笔迹。

    def test_grab_does_not_raise(self):  # 离屏真实走一遍 paintEvent，确认绘制路径不抛异常。
        canvas = self._canvas_with_map()
        canvas.resize(700, 400)
        canvas.set_brush(rgb=(0, 0, 255), size=4)
        canvas._paint_line((100, 20), (200, 20))
        pm = canvas.grab()  # 触发 paintEvent。
        self.assertFalse(pm.isNull())


if __name__ == '__main__':
    unittest.main()
