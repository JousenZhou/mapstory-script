# 测谎检验页签：用户上传谎言检测器录像，按固定 30FPS 节拍逐帧回放，
# 复刻参考项目 detect_transparent_shapes 流水线（模板定位→YOLO→ByteTrack→背景方向评分），
# 画面叠加检测框与运动轨迹，视频模式不发送鼠标，仅验证算法效果。
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2  # 导入 OpenCV，用于读视频与叠加绘制。
import numpy as np  # 导入 NumPy，用于画面矩阵。
from PySide6.QtCore import Qt, QThread, QTimer, Signal  # 导入 Qt 线程、定时器与信号。
from PySide6.QtGui import QImage, QPixmap  # 导入图像对象，用于把画面矩阵转成图片显示。
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QSizePolicy, QWidget  # 导入布局与控件。
from qfluentwidgets import BodyLabel, FluentIcon, PushButton, TextEdit  # 导入 Fluent 控件。

from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。
from src.ui.VisionTab import VisionLabel  # 复用实时识图页签的自适应画面标签。

TICK_FPS = 30  # 固定节拍帧率，与参考项目 FPS=30 一致。
TIMEOUT_TICKS = 545  # 超时兜底 545 tick（约 18 秒），与参考项目 solve_shape.rs 一致。
TRAIL_LENGTH = 30  # 每条轨迹保留最近 30 帧中心尾迹。

try:  # 推理依赖可选：缺失时页签降级为不可运行并提示安装。
    from src.liedetector.detector import (
        REGION_SIZE, TEMPLATE_THRESHOLD, LieDetectorRegion, MultiScaleTemplate,
        TransparentShapeDetector, _template_best_score)
    from src.liedetector.solver import TransparentShapeSolver
    import onnxruntime  # noqa: F401 仅探测安装情况。
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


class _EmitLogger:  # 把检测器内部日志转发成 Qt 信号的适配器。

    def __init__(self, emit):
        self._emit = emit

    def info(self, message):
        self._emit(str(message))


def _mid(rect):  # 矩形中心点（取整）。
    x, y, w, h = rect
    return int(x + w // 2), int(y + h // 2)


def draw_overlay(frame, region, detections, tracks, trails, target, cursor, bg_direction, tick, status):
    # 在全帧上叠加：区域框、全部检测框、轨迹尾迹与速度箭头、背景方向箭头、目标高亮、光标十字。
    rx, ry, rw, rh = region
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 图形区域边界。
    for (x, y, w, h), score in detections:  # 全部检测框（区域局部坐标转全帧坐标）。
        p1 = (rx + int(x), ry + int(y))
        p2 = (rx + int(x + w), ry + int(y + h))
        cv2.rectangle(frame, p1, p2, (0, 220, 0), 1)
        cv2.putText(frame, f"{score:.2f}", (p1[0], max(p1[1] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    alive_ids = set()
    for track in tracks:  # 每条轨迹：尾迹折线 + 卡尔曼速度箭头。
        tid = track.track_id
        alive_ids.add(tid)
        center = (rx + _mid(track.rect)[0], ry + _mid(track.rect)[1])
        trail = trails.setdefault(tid, deque(maxlen=TRAIL_LENGTH))
        trail.append(center)
        if len(trail) >= 2:
            points = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, (0, 200, 255), 1)
        vx, vy = track.kalman_velocity
        cv2.arrowedLine(frame, center, (center[0] + int(vx * 8), center[1] + int(vy * 8)),
                        (0, 128, 255), 1, tipLength=0.3)
    for tid in [t for t in trails if t not in alive_ids]:  # 清理已消亡轨迹的尾迹。
        del trails[tid]
    if bg_direction[0] != 0.0 or bg_direction[1] != 0.0:  # 背景方向大箭头画在区域左上角。
        origin = (rx + 30, ry + 30)
        end = (origin[0] + int(bg_direction[0] * 60), origin[1] + int(bg_direction[1] * 60))
        cv2.arrowedLine(frame, origin, end, (255, 80, 255), 2, tipLength=0.25)
    if target is not None:  # 当前目标高亮框。
        x, y, w, h = target.rect
        cv2.rectangle(frame, (rx + x, ry + y), (rx + x + w, ry + y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"target #{target.track_id}", (rx + x, max(ry + y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if cursor is not None:  # 预测光标十字。
        cx, cy = int(cursor[0]), int(cursor[1])
        cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 1)
    cv2.putText(frame, f"tick {tick}/{TIMEOUT_TICKS} tracks {len(tracks)} {status}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


class LieDetectorWorker(QThread):  # 工作线程：按 1/30s 节拍消费视频帧并运行完整流水线。

    log_message = Signal(str)
    provider_ready = Signal(str)
    finished_result = Signal(bool, str)  # (是否通过, 结果描述)

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=2)  # 只保留最新帧，避免 UI 积压。
        self._stopped = threading.Event()

    def stop(self):  # 请求停止并等待线程退出。
        self._stopped.set()
        self.wait(3000)

    def _push_frame(self, frame):  # 推送标注帧：队列满时丢弃旧帧。
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def run(self):  # 线程主循环：读帧→定位→检测→跟踪→求解→绘制。
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_message.emit(f"cannot open video 无法打开视频: {self.video_path}")
            self.finished_result.emit(False, "open failed")
            return
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频分辨率，诊断模板匹配不中时用。
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log_message.emit(f"video {frame_w}x{frame_h} opened 视频已打开")
        interval = 1.0 / TICK_FPS
        region_finder = LieDetectorRegion()
        # 启动探测：读前几帧多尺度匹配标题模板。裁剪的区域录像无标题（得分远低于阈值），
        # 全屏录像开头弹窗通常已在画面内；探测不中即锁定裁剪模式，
        # 避免全屏多尺度模板匹配拖垮 30FPS 节拍。
        crop_mode = False
        probe_best = -1.0
        probe_title = None
        for _ in range(5):
            ok, probe_frame = cap.read()
            if not ok:
                break
            probe_best = max(probe_best, _template_best_score(probe_frame, region_finder.title_template))
            if probe_best >= TEMPLATE_THRESHOLD:
                probe_title = region_finder.find_title(probe_frame)
                break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if probe_title is not None:
            self.log_message.emit(f"dialog title found in probe, fullscreen mode 探测命中标题，全屏模式")
        else:
            crop_mode = True
            self.log_message.emit(
                f"crop mode: no title in probe (best {probe_best:.3f} < {TEMPLATE_THRESHOLD}) "
                f"裁剪模式：探测未见弹窗标题（最高分 {probe_best:.3f} 低于阈值），整帧作为图形区域 {frame_w}x{frame_h}")
        prepare_matcher = MultiScaleTemplate(region_finder.prepare_template, TEMPLATE_THRESHOLD)
        detector = TransparentShapeDetector(logger=_EmitLogger(self.log_message.emit))
        solver = TransparentShapeSolver(fps=TICK_FPS)
        trails = {}
        found_dialog = False
        provider_emitted = False
        result_ok = False
        last_target_tick = 0  # 最近一次锁定目标的 tick，裁剪模式下视频播完时的通过判据。
        result_msg = f"timeout after {TIMEOUT_TICKS} ticks 超过 {TIMEOUT_TICKS} tick 未通过"
        tick = 0
        next_tick_time = time.perf_counter()
        try:
            while not self._stopped.is_set() and tick < TIMEOUT_TICKS:
                ok, frame = cap.read()
                if not ok:
                    if crop_mode and last_target_tick > 0 and tick - last_target_tick <= 10:
                        # 裁剪模式没有"弹窗消失"信号：视频播完时目标仍在被跟踪即视为验证通过。
                        result_ok = True
                        result_msg = (
                            f"crop mode: target tracked to tick {last_target_tick} "
                            f"裁剪模式：目标被持续跟踪至第 {last_target_tick} tick，验证通过")
                    else:
                        result_msg = "video ended before result 视频播完仍未出结果"
                    break
                tick += 1
                if crop_mode:  # 裁剪模式：无标题模板可查，整帧即区域，也不做准备中检测。
                    preparing = False
                    region = (0, 0, frame.shape[1], frame.shape[0])
                else:
                    # 弹窗已定位后才查“准备中”；未定位时只查标题，避免双倍模板匹配拖慢节拍。
                    preparing = prepare_matcher.match(frame)[0] if found_dialog else False
                    region = region_finder.find_region(frame)
                if preparing or region is None:  # 弹窗未出现或准备中：不推进求解。
                    if found_dialog and not preparing:  # 弹窗曾出现后消失，判定挑战完成通过。
                        result_ok = True
                        result_msg = f"dialog closed at tick {tick} 弹窗在第 {tick} tick 关闭，判定通过"
                        self.log_message.emit(result_msg)
                        cv2.putText(frame, "PASSED", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        self._push_frame(frame)
                        break
                    status = "preparing..." if preparing else "waiting for dialog..."
                    cv2.putText(frame, f"tick {tick}/{TIMEOUT_TICKS} {status}", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    if tick % 30 == 1 and not preparing:  # 诊断：每秒报告标题模板多尺度最高分，定位分辨率不匹配问题。
                        score = _template_best_score(frame, region_finder.title_template)
                        self.log_message.emit(
                            f"tick {tick}: title match score {score:.3f} (threshold {TEMPLATE_THRESHOLD}) "
                            f"标题模板最高分 {score:.3f}（阈值 {TEMPLATE_THRESHOLD}）")
                    self._push_frame(frame)
                else:
                    if not found_dialog:
                        found_dialog = True
                        if not crop_mode:
                            self.log_message.emit(f"dialog located at tick {tick}, region={region} 定位到弹窗区域")
                    rx, ry, rw, rh = region
                    crop = frame[ry:ry + rh, rx:rx + rw]
                    detections = detector.detect(crop)
                    if not provider_emitted:
                        provider_emitted = True
                        self.provider_ready.emit(detector.provider)
                    cursor, target = solver.solve(region, detections)
                    if target is not None:
                        last_target_tick = tick
                    if tick % 30 == 1:  # 诊断：每秒报告检测数量与轨迹数，定位检测为空问题。
                        self.log_message.emit(
                            f"tick {tick}: detections={len(detections)} tracks={len(solver.tracker.tracked)} "
                            f"检测数={len(detections)} 轨迹数={len(solver.tracker.tracked)}")
                    if target is not None and solver.current_track_id != getattr(self, "_last_target_id", None):
                        self.log_message.emit(f"target switched to #{solver.current_track_id} at tick {tick} 目标切换")
                    self._last_target_id = solver.current_track_id
                    draw_overlay(frame, region, detections, solver.tracker.tracked, trails,
                                 target, cursor, solver.bg_direction, tick, "")
                    self._push_frame(frame)
                next_tick_time += interval  # 固定节拍：处理完等待到下一个 1/30s 刻度。
                delay = next_tick_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick_time = time.perf_counter()
        except Exception as exc:  # 流水线异常不拖垮 UI，记录后按失败收尾。
            self.log_message.emit(f"pipeline error 流水线异常: {exc}")
            result_msg = f"pipeline error: {exc}"
        finally:
            cap.release()
        if self._stopped.is_set():
            result_msg = "stopped by user 用户手动停止"
        self.finished_result.emit(result_ok, result_msg)


class LieDetectorTab(CustomTab):  # 测谎检验页签：上传录像验证谎言检测器求解流水线。

    def __init__(self):
        super().__init__()
        self.icon = FluentIcon.LIBRARY
        self.worker = None
        self.video_path = ""

        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        self.pick_button = PushButton(FluentIcon.FOLDER, "Select Video 选择视频")
        self.pick_button.clicked.connect(self.pick_video)
        self.path_label = BodyLabel("No video selected 未选择视频")
        self.path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.start_button = PushButton(FluentIcon.PLAY, "Start 开始")
        self.start_button.clicked.connect(self.start)
        self.stop_button = PushButton(FluentIcon.PAUSE, "Stop 停止")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.fps_label = BodyLabel(f"FPS: {TICK_FPS} (fixed 固定)")
        self.provider_label = BodyLabel("Backend 推理后端: --")
        layout.addWidget(self.pick_button)
        layout.addWidget(self.path_label, 1)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.provider_label)
        self.add_card("Video & Run 视频与运行", control)

        if not INFERENCE_AVAILABLE:  # 依赖缺失时禁用运行并提示安装命令。
            self.start_button.setEnabled(False)
            self.path_label.setText("onnxruntime not installed, run: pip install -e \".[inference]\" 未安装推理依赖")

        self.image_label = VisionLabel()
        self.add_card("Process Vision 运算过程画面", self.image_label, stretch=1)

        self.log_edit = TextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(160)
        self.add_card("Run Log 运行日志", self.log_edit)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)  # 按 30FPS 节拍拉取标注帧显示。

    @property
    def name(self):
        return "LieDetector"

    def append_log(self, message):  # 追加一行运行日志。
        self.log_edit.append(message)

    def pick_video(self):  # 弹出文件对话框选择录像。
        path, _ = QFileDialog.getOpenFileName(
            self, "Select lie detector video 选择谎言检测器录像", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:
            self.video_path = path
            self.path_label.setText(os.path.basename(path))
            self.append_log(f"video selected 已选择视频: {path}")

    def start(self):  # 启动工作线程回放并验证。
        if not self.video_path or not Path(self.video_path).exists():
            self.append_log("please select a video first 请先选择视频")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.log_edit.clear()
        self.worker = LieDetectorWorker(self.video_path, self)
        self.worker.log_message.connect(self.append_log)
        self.worker.provider_ready.connect(self.on_provider_ready)
        self.worker.finished_result.connect(self.on_finished)
        self.start_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.append_log(f"start verifying at fixed {TICK_FPS} FPS 开始按固定 {TICK_FPS}FPS 验证")
        self.worker.start()

    def stop(self):  # 请求停止工作线程。
        if self.worker is not None and self.worker.isRunning():
            self.append_log("stopping... 正在停止")
            self.worker.stop()

    def on_provider_ready(self, provider):  # 推理后端确定后更新徽标。
        short = "CUDA" if "CUDA" in provider else ("DirectML" if "Dml" in provider else "CPU")
        self.provider_label.setText(f"Backend 推理后端: {short}")
        self.append_log(f"inference backend 推理后端: {provider}")

    def on_finished(self, ok, message):  # 工作线程结束：恢复按钮并输出结论。
        self.start_button.setEnabled(INFERENCE_AVAILABLE)
        self.pick_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        verdict = "PASSED 通过" if ok else "NOT PASSED 未通过"
        self.append_log(f"result 结果: {verdict} - {message}")

    def refresh(self):  # 定时从工作线程队列取最新标注帧显示。
        if self.worker is None:
            return
        frame = None
        try:
            while True:  # 丢弃旧帧只保留最新一帧。
                frame = self.worker.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)
        self.image_label.set_frame(QPixmap.fromImage(image.copy()))

    def closeEvent(self, event):  # 页签关闭时确保工作线程退出。
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)
