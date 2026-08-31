import os
import threading
import time

from PySide6.QtCore import QObject

from ok import Logger

logger = Logger.get_logger(__name__)


class Globals(QObject):

    def __init__(self, exit_event):
        super().__init__()
        self.yolo_model = None  # 存放懒加载的 YOLO 模型实例。
        self.yolo_load_failed = False  # 记录模型是否加载失败，避免重复加载报错。
        self.yolo_device = None  # YOLO 推理设备：优先 CUDA 显卡，缺失时回退 CPU。
        self._vision_lock = threading.Lock()  # 保护实时画面在任务线程与 UI 线程间的读写。
        self._vision_frame = None  # 最新一帧带标注的游戏画面（BGR 矩阵）。
        self._vision_time = 0.0  # 最新一帧的写入时间戳，用于判断画面是否过期。

    def update_vision(self, frame):  # 任务线程写入最新一帧带标注画面，供 UI 实时展示。
        with self._vision_lock:  # 加锁避免 UI 线程读到写一半的数据。
            self._vision_frame = frame  # 记录最新画面。
            self._vision_time = time.time()  # 记录写入时间。

    def get_vision(self, max_age=1.0):  # UI 线程读取最新画面，超过 max_age 秒未更新视为过期返回 None。
        with self._vision_lock:  # 加锁保证读取一致性。
            if self._vision_frame is None or time.time() - self._vision_time > max_age:  # 无画面或画面已过期。
                return None  # 返回 None 由 UI 显示占位提示。
            return self._vision_frame  # 返回最新带标注画面。

    def get_yolo(self, model_path='assets/yolo.pt'):  # 懒加载 YOLO 模型，首次调用时才导入 ultralytics。
        if self.yolo_model is None and not self.yolo_load_failed:  # 模型尚未加载且未失败时才尝试加载。
            if not os.path.exists(model_path):  # 模型文件不存在时不加载，由调用方给出提示。
                logger.warning(f'yolo model file not found: {model_path}')
                self.yolo_load_failed = True
                return None
            try:  # 尝试导入 ultralytics 并加载模型。
                from ultralytics import YOLO  # 导入 YOLO 类。
                self.yolo_model = YOLO(model_path)  # 加载用户训练的模型权重。
                self.yolo_device = self._pick_yolo_device()  # 选择推理设备：优先 CUDA 显卡。
                logger.info(f'yolo model loaded: {model_path} device: {self.yolo_device}')
            except Exception as e:  # 未安装 ultralytics 或权重损坏时记录并标记失败。
                logger.error(f'failed to load yolo model: {e}')
                self.yolo_load_failed = True
        return self.yolo_model  # 返回模型实例或 None。

    @staticmethod
    def _pick_yolo_device():  # 选择 YOLO 推理设备：有可用 CUDA 显卡返回 0，否则回退 'cpu'。
        try:
            import torch  # 导入 torch 探测 CUDA。
            if torch.cuda.is_available():  # 显卡驱动与 CUDA 运行时可用。
                return 0  # 使用第一块显卡做推理。
        except Exception as e:  # torch 未安装或探测异常。
            logger.warning(f'torch CUDA probe failed, fallback to cpu: {e}')
        return 'cpu'  # 回退 CPU 推理。

    def detect(self, frame, model_path='assets/yolo.pt'):  # 对一帧画面运行 YOLO，返回统一的检测结果。
        model = self.get_yolo(model_path)  # 获取已加载的模型。
        if model is None or frame is None:  # 模型或画面不可用时返回空结果。
            return []
        try:  # 运行推理，verbose=False 避免刷屏日志。
            results = model.predict(frame, verbose=False, device=self.yolo_device)  # 在选定设备（优先 CUDA）上推理。
        except Exception as e:  # 推理异常时记录并返回空结果，不让任务崩溃。
            logger.error(f'yolo predict failed: {e}')
            return []
        detections = []  # 存放统一格式的检测结果。
        for result in results:  # 逐张结果图解析。
            if result.boxes is None:  # 没有任何检测框时跳过。
                continue
            names = result.names  # 类别索引到类别名的映射。
            for box in result.boxes:  # 遍历每个检测框。
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 取检测框左上和右下角坐标。
                class_id = int(box.cls[0])  # 取预测的类别索引。
                detections.append({  # 追加一条统一格式的检测记录。
                    'name': names.get(class_id, str(class_id)),  # 类别名称。
                    'box': [x1, y1, x2, y2],  # 检测框坐标。
                    'conf': float(box.conf[0]),  # 置信度。
                })
        return detections  # 返回本帧全部检测结果。

