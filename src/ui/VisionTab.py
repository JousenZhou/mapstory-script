import cv2  # 导入 OpenCV，用于把 BGR 画面转成 RGB 显示。
from PySide6.QtCore import Qt, QTimer  # 导入 Qt 定时器与对齐常量。
from PySide6.QtGui import QImage, QPixmap  # 导入图像对象，用于把画面矩阵转成图片显示。
from PySide6.QtWidgets import QLabel, QSizePolicy  # 导入标签控件，用于显示实时画面。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于页签图标。

from ok import og  # 导入全局对象，用于读取任务线程推送的画面与截图设备。
from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。

CAPTURE_FPS = 30  # 截图采集固定帧率：实时画面刷新与无任务时的截图取帧都按该节拍。
CAPTURE_INTERVAL_MS = round(1000 / CAPTURE_FPS)  # 帧间隔毫秒数（33ms），QTimer 最小粒度即 1ms 足够。


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


class VisionTab(CustomTab):  # 定义实时识图画面页签，展示任务推送的带标注游戏画面。

    def __init__(self):  # 构造函数。
        super().__init__()  # 初始化父类。
        self.icon = FluentIcon.VIEW  # 页签图标。
        self.image_label = VisionLabel()  # 创建画面显示标签。
        self.add_card("Realtime Vision 实时识图画面", self.image_label, stretch=1)  # 把画面标签放进卡片并占满剩余空间。
        self.timer = QTimer(self)  # 创建刷新定时器。
        self.timer.timeout.connect(self.refresh)  # 定时刷新画面。
        self.timer.start(CAPTURE_INTERVAL_MS)  # 按固定 30FPS 节拍刷新，截图采集速率由此钉死。

    @property
    def name(self):  # 页签显示名称。
        return "Vision"  # 返回页签名。

    def refresh(self):  # 定时刷新：优先显示任务推送的带标注画面，任务未运行时显示原始截图。
        frame = None  # 待显示的画面。
        try:  # 读取共享画面失败时不应导致 UI 崩溃。
            if og.my_app is not None and hasattr(og.my_app, 'get_vision'):  # 全局对象提供画面共享接口时才读取。
                frame = og.my_app.get_vision(max_age=1.0)  # 读取 1 秒内的最新带标注画面。
            if frame is None:  # 任务未运行或画面过期时尝试直接截取游戏画面。
                frame = self.capture_raw()  # 从截图设备取一帧原始画面。
        except Exception as e:  # 任何读取异常都按无画面处理。
            self.logger.warning(f'VisionTab refresh failed: {e}')  # 记录异常日志。
            frame = None  # 按无画面处理。
        if frame is None:  # 仍无画面时显示占位提示。
            self.image_label.clear_frame("No frame, please connect a window and start the task. 暂无画面，请先连接游戏窗口并启动任务。")  # 提示用户操作步骤。
            return  # 结束本次刷新。
        self.image_label.set_frame(self.to_pixmap(frame))  # 把画面转成图片并显示。

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
