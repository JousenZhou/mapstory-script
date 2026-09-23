import threading
import time

from PySide6.QtCore import QObject

from src.liedetector.service import LieDetectorService  # 导入独立测谎监控服务，由 Globals 持有，随首个脚本任务启动（见任务 _ensure_lie_service）。


class Globals(QObject):

    def __init__(self, exit_event):
        super().__init__()
        self._vision_lock = threading.Lock()  # 保护实时画面在任务线程与 UI 线程间的读写。
        self._vision_frame = None  # 最新一帧带标注的游戏画面（BGR 矩阵）。
        self._vision_time = 0.0  # 最新一帧的写入时间戳，用于判断画面是否过期。
        # 检测发布通道：脚本任务用共享 GPU 匹配器每帧检测【测谎触发】【掉线2】，把命中框与分数发布到这里，
        # 后台测谎服务不再自己截图/CPU 匹配，改为消费任务发布的结果（仿 update_vision/get_vision）。
        self._detection_lock = threading.Lock()  # 保护检测结果在任务线程与服务线程间的读写。
        self._detection = None  # 最新一次发布的检测结果 dict（frame/trigger_box/trigger_score/disconnect_box/disconnect_score/seq/time）。
        self._detection_seq = 0  # 发布序号，每发布一帧自增，供服务判断结果是否为“新帧”去重。
        # 独立测谎监控服务：脱离脚本任务运行，值守看板【测谎触发】标注，命中即暂停当前任务并自动解测谎、结束后恢复。
        # 服务只创建不常驻启动：改为随首个脚本任务启动（见任务 _ensure_lie_service），无任务时不空转截图。
        self.lie_service = LieDetectorService(exit_event)  # 创建服务，与 app 共用退出事件。

    def update_vision(self, frame):  # 任务线程写入最新一帧带标注画面，供 UI 实时展示。
        with self._vision_lock:  # 加锁避免 UI 线程读到写一半的数据。
            self._vision_frame = frame  # 记录最新画面。
            self._vision_time = time.time()  # 记录写入时间。

    def get_vision(self, max_age=1.0):  # UI 线程读取最新画面，超过 max_age 秒未更新视为过期返回 None。
        with self._vision_lock:  # 加锁保证读取一致性。
            if self._vision_frame is None or time.time() - self._vision_time > max_age:  # 无画面或画面已过期。
                return None  # 返回 None 由 UI 显示占位提示。
            return self._vision_frame  # 返回最新带标注画面。

    def publish_detection(self, frame, trigger_box, trigger_score, disconnect_box, disconnect_score):  # 任务线程发布本帧测谎/掉线检测结果，供后台测谎服务消费。
        with self._detection_lock:  # 加锁避免服务线程读到写一半的数据。
            self._detection_seq += 1  # 递增发布序号，服务据此判断是否为新帧。
            self._detection = {  # 记录本次发布的完整检测结果（框可能为 None，分数低于阈值也保留供诊断）。
                'frame': frame,  # 本帧原始游戏画面（BGR 矩阵），求解阶段服务可复用。
                'trigger_box': trigger_box,  # 【测谎触发】最高分命中框，未命中为 None。
                'trigger_score': float(trigger_score or 0.0),  # 【测谎触发】最高分（无论是否达标）。
                'disconnect_box': disconnect_box,  # 【掉线2】最高分命中框，未命中为 None。
                'disconnect_score': float(disconnect_score or 0.0),  # 【掉线2】最高分（无论是否达标）。
                'seq': self._detection_seq,  # 本次发布对应的序号。
                'time': time.time(),  # 写入时间戳，用于判断检测结果是否过期。
            }

    def get_latest_detection(self, max_age=0.5):  # 服务线程读取最新检测结果，超过 max_age 秒未更新视为过期返回 None。
        with self._detection_lock:  # 加锁保证读取一致性。
            if self._detection is None or time.time() - self._detection['time'] > max_age:  # 无结果或结果已过期（无任务运行时）。
                return None  # 返回 None 让服务 park，不再独立截图值守。
            return self._detection  # 返回最新检测结果（同一时刻只有一个 maple 任务发布，最后写入者胜）。

