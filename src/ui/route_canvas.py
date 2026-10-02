# 路线图绘制控件：在地图底图上叠加绘制彩色指令路线，供地图页签（MapTab）内嵌使用。
#
# 交互模型：
#   - 左侧显示 map.png 底图，路线图（routeN.png）以纯指令色画在透明/黑色底上，非黑像素叠加显示；
#   - 按住鼠标拖动即以当前画笔色（或橡皮=黑色）在路线图上落笔，坐标按显示缩放比换算回图像像素；
#   - 落笔前自动压入撤销快照，Ctrl+Z / 调用 undo() 回退一步；
#   - 每次修改发出 route_changed 信号，由页签决定何时落盘保存。
#
# 说明：PNG 无损，指令色像素须与 meta 中 color_code 精确相等，因此画笔直接写入精确 RGB，不做任何抗锯齿/混合。
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import QWidget, QSizePolicy

import cv2  # 落笔绘制线段。
import numpy as np  # 图像矩阵。

MAX_UNDO = 24  # 撤销栈最大深度，限制内存占用。


def _bgr_to_qimage(img):  # BGR ndarray -> QImage（拷贝数据，避免底层缓冲区回收后失效）。
    # 必须转成 4 字节/像素并显式传行距：RGB888 每行 w*3 字节，宽度非 4 倍数（如 1366）时不满足 Qt 的 4 字节对齐行距，
    # 缺省构造会让每行错位若干字节，显示为整幅斜切条纹的花屏。
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra = np.ascontiguousarray(bgra)
    h, w, _ = bgra.shape
    return QImage(bgra.data, w, h, w * 4, QImage.Format_RGB32).copy()


class RouteCanvas(QWidget):
    """可绘制的路线图控件：显示 map 底图 + route 叠加，支持鼠标落笔与撤销。"""

    route_changed = Signal()  # 路线图内容发生变化（落笔/撤销/清空）时发出。

    def __init__(self, parent=None):
        super().__init__(parent)
        self._map = None      # 底图 BGR（np.uint8, HxWx3）。
        self._route = None    # 可编辑路线图 BGR，与底图同尺寸。
        self._composite = None  # 缓存的叠加显示图，route 变动时重建，避免每次 paint 全图混合。
        self._composite_pm = None  # 缓存的合成图 QPixmap，与 _composite 同生命周期，避免每次 paint 重复做全图颜色转换。
        self._scale = 1.0     # 图像到控件的等比缩放系数。
        self._offset = (0, 0)  # 居中留白偏移 (x, y)。
        self._brush_color = (255, 0, 0)  # 当前画笔色（BGR 之外的 RGB 指令色），由页签设置。
        self._brush_size = 2    # 画笔粗细（图像像素直径）。
        self._eraser = False    # True 时以黑色（无指令）落笔。
        self._undo_stack = []   # 撤销快照（route 副本）列表。
        self._drawing = False   # 是否处于按住拖动状态。
        self._last_pt = None    # 上一落笔图像坐标，用于画连续线段。
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(False)  # 仅在按键时接收 move 事件。

    # ------------------------------------------------------------------ 数据存取

    def set_images(self, map_img, route_img):  # 载入底图与路线图；尺寸不一致时把路线重置为与底图同尺寸的全黑。
        self._map = map_img.copy() if map_img is not None else None
        if self._map is None:
            self._route = None
        elif route_img is not None and route_img.shape[:2] == self._map.shape[:2]:
            self._route = route_img.copy()
        else:
            h, w = self._map.shape[:2]
            self._route = np.zeros((h, w, 3), dtype=np.uint8)  # 尺寸缺失/不符：给一条干净底路线供绘制。
        self._undo_stack.clear()
        self._invalidate()

    def route_image(self):  # 返回当前路线图副本（供保存）。
        return None if self._route is None else self._route.copy()

    def has_map(self):  # 是否已载入底图。
        return self._map is not None

    def set_brush(self, rgb=None, size=None, eraser=None):  # 更新画笔参数（rgb 为指令色三元组）。
        if rgb is not None:
            self._brush_color = tuple(int(c) for c in rgb)
            self._eraser = False  # 选色即退出橡皮模式。
        if size is not None:
            self._brush_size = max(1, int(size))
        if eraser is not None:
            self._eraser = bool(eraser)

    def undo(self):  # 回退一步。
        if not self._undo_stack:
            return
        self._route = self._undo_stack.pop()
        self._invalidate()
        self.route_changed.emit()

    def clear_route(self):  # 清空当前路线图为全黑（先压撤销快照）。
        if self._route is None:
            return
        self._push_undo()
        self._route[:] = 0
        self._invalidate()
        self.route_changed.emit()

    # ------------------------------------------------------------------ 渲染

    def _invalidate(self):  # 标记缓存失效并重绘。
        self._composite = None
        self._composite_pm = None
        self._update_scale()
        self.update()

    def _update_scale(self):  # 依据控件尺寸重算等比缩放与居中偏移（contain：整图可见）。
        if self._map is None:
            return
        h, w = self._map.shape[:2]
        cw, ch = max(1, self.width()), max(1, self.height())
        self._scale = max(0.1, min(cw / w, ch / h))
        disp_w, disp_h = w * self._scale, h * self._scale
        self._offset = (max(0, (cw - disp_w) / 2), max(0, (ch - disp_h) / 2))

    def _rebuild_composite(self):  # 把路线非黑像素叠加到底图上，缓存结果（合成为供显示的 BGR）。
        if self._map is None:
            return
        if self._route is None:
            self._composite = self._map.copy()
            return
        comp = self._map.copy()  # 底图来自 cv2 截图/imread，通道序是 BGR。
        mask = np.any(self._route != 0, axis=2)  # 非黑像素即有指令色。
        # 路线图通道序是指令色表同序的 RGB（录制端 trace 与画笔都直写 r,g,b），叠到 BGR 底图前必须交换通道，
        # 否则选红色笔会显示成青色，与调色板按钮颜色不一致。
        comp[mask] = self._route[mask][:, ::-1]  # RGB 转 BGR 后落到底图副本上。
        self._composite = comp

    def _ensure_pixmap(self):  # 取得合成图的 QPixmap，仅在 route/底图变动后才重建。
        if self._composite_pm is None:
            if self._composite is None:
                self._rebuild_composite()
            self._composite_pm = QPixmap.fromImage(_bgr_to_qimage(self._composite))
            self._composite = None  # ndarray 副本已转成 QPixmap，释放以省内存。
        return self._composite_pm

    def resizeEvent(self, event):  # 尺寸变化时重算缩放。
        super().resizeEvent(event)
        self._update_scale()

    def paintEvent(self, event):  # 绘制缓存合成图（无底图时留空由父级背景填充）。
        painter = QPainter(self)
        if self._map is None:
            painter.end()
            return
        pixmap = self._ensure_pixmap()
        target_w = int(pixmap.width() * self._scale)
        target_h = int(pixmap.height() * self._scale)
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(int(self._offset[0]), int(self._offset[1]), scaled)
        painter.end()

    # ------------------------------------------------------------------ 鼠标交互

    def _widget_to_image(self, pos):  # 控件坐标 -> 图像像素坐标（整数），越界裁剪。
        x = int((pos.x() - self._offset[0]) / self._scale)
        y = int((pos.y() - self._offset[1]) / self._scale)
        h, w = self._route.shape[:2]
        return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))

    def mousePressEvent(self, event):  # 按下开始一笔：先压撤销快照再落一个点。
        if event.button() != Qt.LeftButton or self._route is None:
            return
        self._push_undo()
        self._drawing = True
        self._last_pt = self._widget_to_image(event.position().toPoint())
        self._paint_at(self._last_pt)

    def mouseMoveEvent(self, event):  # 拖动：在上一与当前点之间画线段，避免快速拖动出现断点。
        if not self._drawing or self._route is None:
            return
        pt = self._widget_to_image(event.position().toPoint())
        if pt == self._last_pt:
            return
        self._paint_line(self._last_pt, pt)
        self._last_pt = pt

    def mouseReleaseEvent(self, event):  # 松开结束一笔并发出变更信号。
        if event.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        self._last_pt = None
        self.route_changed.emit()

    def _push_undo(self):  # 压入当前路线副本快照，超深度丢弃最早帧。
        if self._route is None:
            return
        self._undo_stack.append(self._route.copy())
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    def _paint_at(self, pt):  # 在单点落笔（画笔色或黑色）。
        color = (0, 0, 0) if self._eraser else tuple(self._brush_color)
        cv2.circle(self._route, pt, self._brush_size // 2 or 0, color, -1, cv2.LINE_8)
        self._invalidate()

    def _paint_line(self, p0, p1):  # 画两点间的粗线段。
        color = (0, 0, 0) if self._eraser else tuple(self._brush_color)
        cv2.line(self._route, p0, p1, color, self._brush_size, cv2.LINE_8)
        self._invalidate()

    def keyPressEvent(self, event):  # Ctrl+Z 撤销。
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()
        else:
            super().keyPressEvent(event)
