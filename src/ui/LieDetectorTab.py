# -*- coding: utf-8 -*-
"""测谎检验页签：d/lie Qt 交互原样移植（PySide6）。

交互与 D:\\lie ui/main_window.py 一致：
- 工具栏：导入视频 / 开始 / 暂停(继续) / 单帧步进 / 停止 + 彩色状态标签●；
- 每次点「开始」都是全新分析会话（设备/置信度此时锁定）；
- 进度条拖动 = 暂停并预览原始帧（未分析），下次「开始」从该帧重开会话；
- 选项：显示轨迹 / 显示检测框 / 播放模式（逐帧最高精度·原速测实时）/ 计算设备（CPU·GPU TRT INT8）/ 置信度；
- 底部调试面板：状态/帧/检测/目标ID/角速度/丢帧/实际fps。

本项目扩展：历史录像下拉（lie_records 边车）+ 刷新/删除；复算结束给出锁定判定。
视频一律按【测谎区域】裁剪录像处理（recorder 只录区域），整帧即管线输入。
"""
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QWidget
from qfluentwidgets import (BodyLabel, CheckBox, ComboBox, DoubleSpinBox, FluentIcon,
                            PushButton, TextEdit)

from ok.gui.widget.CustomTab import CustomTab
from src.liedetector.recorder import LIE_OUTCOME_LABELS, LIE_RECORD_DIR, delete_record, list_records
from src.liedetector.vision import config as vision_config
from src.liedetector.vision.frame_rate_controller import FrameRateController, SourceClock
from src.liedetector.vision.overlay_renderer import render
from src.liedetector.vision.flow_session import create_session
from src.liedetector.vision.video_source import VideoFileSource, analyze_video


def _to_qimage(frame) -> QImage:
    """BGR ndarray -> QImage（拷贝缓冲，数组离开作用域后仍有效）。"""
    h, w, ch = frame.shape
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class DetectThread(QThread):
    """管线线程：读视频 → d/lie Pipeline → 叠加渲染 → 发帧与统计（d/lie DetectThread 原样移植）。"""

    frame_ready = Signal(QImage, dict)
    status = Signal(str)
    finished_run = Signal()

    def __init__(self, path: str, start_frame: int = 0, device: str = "cpu", conf: float = None,
                 parent=None):
        super().__init__(parent)
        self.path = path
        self.start_frame = max(0, start_frame)
        self.device = device
        self.conf = conf
        self.detector = None
        self.paused = False
        self.step_once = False
        self.show_boxes = True
        self.show_trails = True
        # False=逐帧（每帧都进算法，最高精度）；True=原速（跟随源时钟，处理不过来就丢帧）。
        self.realtime = False
        self.processed = 0
        self.dropped = 0
        self.max_behind = 0
        self._first_at = None
        self._pause_at = None
        self._paused_sec = 0.0
        self._stop = False

    def run(self):
        from src.liedetector.vision.detector import Detector
        from src.liedetector.vision import tensorrt_backend

        source = VideoFileSource(self.path)
        fps = source.fps or 30.0
        try:
            if self.device == "gpu" and not tensorrt_backend.engine_cache_path(
                    vision_config.YOLO_MODEL_PATH, vision_config.YOLO_INPUT_SIZE[0]).exists():
                self.status.emit("首次使用 GPU：正在构建 TensorRT INT8 引擎（约 1-5 分钟，之后直接加载缓存）…")
            detector = Detector(device=self.device, conf=self.conf, calib_video=self.path)
            self.detector = detector
            session = create_session(detector, fps)  # FLOW_PRIMARY 时走光流主会话。
            # 分析总是从倒计时开始：seek 进入等于新会话，状态机时钟锚定到该帧。
            if self.start_frame:
                source.seek(self.start_frame)
            session.start(source.frame_idx)
            controller = FrameRateController(fps)
            clock = None

            while not self._stop:
                if self.paused and not self.step_once:
                    if self._pause_at is None:
                        self._pause_at = __import__("time").perf_counter()
                    if clock is not None:
                        clock.halt()
                    self.msleep(30)
                    continue
                if self._pause_at is not None:
                    # 暂停时间不算吞吐证据。
                    self._paused_sec += __import__("time").perf_counter() - self._pause_at
                    self._pause_at = None
                stepping = self.step_once
                self.step_once = False

                # 模式切换后重锚：调度表锚到播放实际所在帧。
                if self.realtime and clock is None:
                    clock = SourceClock(fps, source.frame_idx)
                elif not self.realtime:
                    clock = None

                if clock is not None:
                    if stepping:
                        # 单步即播放一帧：既不能当逾期丢掉，也不能等下一个间隔。
                        clock.anchor(source.frame_idx)
                    else:
                        clock.resume()
                        behind = clock.behind_by(source.frame_idx)
                        if behind > 0:
                            self.max_behind = max(self.max_behind, behind)
                            for _ in range(behind):
                                if not source.read()[0]:
                                    break
                                self.dropped += 1
                        elif clock.ahead(source.frame_idx):
                            self.msleep(5)
                            continue

                ret, frame = source.read()
                if not ret:
                    break

                res = session.process(frame, source.frame_idx)
                out = render(frame, res.tracks, res.target_id, res.state,
                             show_boxes=self.show_boxes, show_trails=self.show_trails)
                target = next((t for t in res.tracks if t.track_id == res.target_id), None)
                self._count_frame()
                self.frame_ready.emit(_to_qimage(out), self._stats(source, fps, res, target))

                # 只有管线快过源帧率时才等待。
                if clock is None:
                    controller.wait()
        except Exception as exc:  # 管线异常显式上报，不静默死线程。
            self.frame_ready.emit(QImage(), {"error": str(exc)})
        finally:
            source.release()
            self.finished_run.emit()

    def _count_frame(self) -> None:
        import time
        if self._first_at is None:
            self._first_at = time.perf_counter()
        self.processed += 1

    def _stats(self, source, fps: float, res, target) -> dict:
        import time
        consumed = self.processed + self.dropped
        wall = max(time.perf_counter() - (self._first_at or time.perf_counter()) - self._paused_sec, 1.0)
        return {
            "frame": source.frame_idx,
            "total": source.total_frames,
            "fps": source.fps,
            "state": res.state,
            "dets": res.detect_count,
            "tracks": len(res.tracks),
            "target_id": res.target_id,
            "omega": res.omega,
            "visible": bool(target and target.lost_frames == 0),
            "mode": "原速" if self.realtime else "逐帧",
            "device": self.detector.backend_label if self.detector else "?",
            "conf": self.detector.conf if self.detector else self.conf or 0.0,
            "start": self.start_frame,
            "processed": self.processed,
            "dropped": self.dropped,
            "drop_pct": 100.0 * self.dropped / consumed if consumed else 0.0,
            "eff_fps": self.processed / wall,
            "max_behind": self.max_behind,
        }

    def stop(self):
        self._stop = True
        self.paused = False


class LieDetectorTab(CustomTab):
    """测谎检验页签：历史/本地录像回放复算，验证 d/lie 管线锁定效果。"""

    name = "LieDetector"
    STATE_LABEL_COLORS = {
        "IDLE": "gray",
        "COUNTDOWN": "blue",
        "TRANSITION": "yellow",
        "TRACKING": "green",
        "TARGET_LOST": "red",
        "GAME_OVER": "purple",
    }

    def __init__(self):
        super().__init__()
        self.icon = FluentIcon.LIBRARY
        self.worker = None  # 当前 DetectThread。
        self.video_path = None  # 当前导入/选中的视频路径。
        self.start_frame = 0  # 分析起点（进度条拖动设定）。
        self.ever_locked = False  # 本会话是否曾锁定目标（复算判定用）。
        self._scrubbing = False
        self._scrub_target = None
        self._scrub_cap = None  # 预览专用独立解码句柄（回退 seek 需重开）。
        self._scrub_cap_at = 0
        self._history_by_video = {}  # mp4 文件名 -> 边车 dict。
        self._build_ui()

    # ------------------------------------------------------------------ UI 构建

    def _build_ui(self):
        # 工具栏（d/lie 交互）：导入/历史 | 开始/暂停/单帧/停止 | 状态标签。
        control = QWidget()
        bar = QHBoxLayout(control)
        bar.setContentsMargins(0, 0, 0, 0)
        self.btn_import = PushButton(FluentIcon.FOLDER, "导入视频")
        self.history_combo = ComboBox()  # 历史录像下拉（lie_records 边车摘要）。
        self.history_combo.setMinimumWidth(240)
        self.history_combo.activated.connect(self._on_history_selected)
        self.btn_refresh = PushButton(FluentIcon.SYNC, "刷新")
        self.btn_refresh.clicked.connect(self._reload_history)
        self.btn_delete = PushButton(FluentIcon.DELETE, "删除")
        self.btn_delete.clicked.connect(self._delete_selected_history)
        self.btn_start = PushButton(FluentIcon.PLAY, "开始")
        self.btn_pause = PushButton(FluentIcon.PAUSE, "暂停")
        self.btn_step = PushButton(FluentIcon.SKIP_FORWARD, "单帧步进")
        self.btn_stop = PushButton(FluentIcon.CANCEL, "停止")
        for b in (self.btn_start, self.btn_pause, self.btn_step, self.btn_stop):
            b.setEnabled(False)
        self.lbl_state = BodyLabel("● IDLE")
        self.lbl_state.setStyleSheet(f"color: {self.STATE_LABEL_COLORS['IDLE']}; font-weight: bold")
        bar.addWidget(self.btn_import)
        bar.addWidget(self.history_combo)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(self.btn_delete)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_step)
        bar.addWidget(self.btn_stop)
        bar.addStretch(1)
        bar.addWidget(self.lbl_state)
        self.add_card("视频与运行", control)

        # 视频区。
        self.video_label = QLabel("未导入视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 420)
        self.video_label.setStyleSheet("background:#111;color:#bbb;")
        self.add_card("画面", self.video_label)

        # 进度条 + 位置。
        seek_host = QWidget()
        seek = QHBoxLayout(seek_host)
        seek.setContentsMargins(0, 0, 0, 0)
        from qfluentwidgets import Slider
        self.slider = Slider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.lbl_pos = BodyLabel("0 / 0")
        seek.addWidget(self.lbl_pos)
        seek.addWidget(self.slider, 1)
        self.add_card("进度", seek_host)

        # 选项行：轨迹/检测框/播放模式/计算设备/置信度。
        opts_host = QWidget()
        opts = QHBoxLayout(opts_host)
        opts.setContentsMargins(0, 0, 0, 0)
        self.chk_trail = CheckBox("显示轨迹")
        self.chk_trail.setChecked(True)
        self.chk_boxes = CheckBox("显示检测框")
        self.chk_boxes.setChecked(True)
        self.cmb_mode = ComboBox()
        self.cmb_mode.addItem("逐帧播放（最高精度：每帧都进算法）")
        self.cmb_mode.addItem("原视频速度（测实时：落后就丢帧）")
        self._build_device_combo()
        self.spn_conf = DoubleSpinBox()
        self.spn_conf.setRange(0.05, 0.95)
        self.spn_conf.setSingleStep(0.05)
        self.spn_conf.setDecimals(2)
        self.spn_conf.setValue(vision_config.YOLO_CONF_THRESHOLD)
        self.spn_conf.setToolTip("检测置信度阈值。与计算设备一样只在点「开始」的新分析会话生效。")
        opts.addWidget(self.chk_trail)
        opts.addWidget(self.chk_boxes)
        opts.addWidget(BodyLabel("播放模式:"))
        opts.addWidget(self.cmb_mode)
        opts.addWidget(BodyLabel("计算设备:"))
        opts.addWidget(self.cmb_device)
        opts.addWidget(BodyLabel("置信度:"))
        opts.addWidget(self.spn_conf)
        opts.addStretch(1)
        self.add_card("选项", opts_host)

        # 调试面板。
        self.debug = TextEdit()
        self.debug.setReadOnly(True)
        self.debug.setFixedHeight(110)
        self.add_card("调试", self.debug)

        # 信号接线。
        self.btn_import.clicked.connect(self.on_import)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_step.clicked.connect(self.on_step)
        self.btn_stop.clicked.connect(lambda: self.on_stop())
        self.chk_trail.toggled.connect(self._sync_options)
        self.chk_boxes.toggled.connect(self._sync_options)
        self.cmb_mode.currentIndexChanged.connect(self._sync_options)
        self.slider.sliderMoved.connect(self._on_scrub)
        self.slider.sliderPressed.connect(lambda: self._on_scrub(self.slider.value()))
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(60)
        self._scrub_timer.timeout.connect(self._flush_scrub)

        self._reload_history()

    def _build_device_combo(self):
        """计算设备下拉：CPU / GPU·TensorRT INT8（显卡三探针门控；不可用时不出此项）。"""
        self.cmb_device = ComboBox()
        self.cmb_device.addItem("CPU")
        from src.liedetector.vision.device import gpu_option_state
        self._gpu_option = gpu_option_state()
        if self._gpu_option.enabled:
            self.cmb_device.addItem("GPU · TensorRT INT8")
            self.cmb_device.setCurrentIndex(1)  # GPU 是首选：可用时默认选中。
        else:
            self.cmb_device.setToolTip(f"GPU 不可用：{self._gpu_option.reason}")

    # ------------------------------------------------------------------ 历史录像

    def _reload_history(self):
        """重建历史下拉：lie_records 边车按时间倒序，摘要含结果中文标签。"""
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self._history_by_video = {}
        self.history_combo.addItem("历史录像…", userData=None)
        for record in list_records():
            video = record.get("video")
            if not video:
                continue
            outcome = record.get("outcome") or ""
            label = LIE_OUTCOME_LABELS.get(outcome, outcome)
            stamp = str(record.get("timestamp") or "")[:16].replace("T", " ")
            self.history_combo.addItem(f"{stamp}  {label}", userData=video)
            self._history_by_video[video] = record
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, index: int):
        video = self.history_combo.itemData(index)
        if not video:
            return
        path = os.path.join(LIE_RECORD_DIR, video)
        if not os.path.exists(path):
            self.debug.setPlainText(f"录像不存在：{path}")
            return
        self.load(path)
        record = self._history_by_video.get(video) or {}
        meta = record.get("meta") or {}
        conf = meta.get("conf")
        if conf:
            self.spn_conf.setValue(float(conf))  # 回填实战置信度，复算与线上同参。

    def _delete_selected_history(self):
        """删除当前选中的历史录像（mp4+json 成对）。"""
        video = self.history_combo.currentData()
        if not video:
            self.debug.setPlainText("请先在下拉中选中一条历史录像再删除。")
            return
        from PySide6.QtWidgets import QMessageBox
        if QMessageBox.question(self, "删除录像", f"确定删除 {video}（mp4+json）？") != QMessageBox.Yes:
            return
        if delete_record(video):
            self.debug.setPlainText(f"已删除：{video}")
            self._reload_history()

    # ------------------------------------------------------------------ 导入与控制

    def load(self, path: str):
        self.video_path = path
        info = analyze_video(path)
        self.video_label.setText(os.path.basename(path))
        self.start_frame = 0
        self._scrubbing = False
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, info["total_frames"] - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)
        self.lbl_pos.setText(f"0 / {info['total_frames']}")
        self.debug.setPlainText(
            f"导入: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, "
            f"{info['total_frames']} 帧 / {info['duration_sec']:.2f}s"
        )
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入视频", "", "Video (*.mp4 *.avi *.mkv)")
        if path:
            self.load(path)

    def on_start(self):
        if not self.video_path:
            return
        self.on_stop(reset_position=False)  # 全新线程=全新状态机。
        self._scrubbing = False
        self.ever_locked = False
        device = "gpu" if self.cmb_device.currentIndex() == 1 else "cpu"
        conf = float(self.spn_conf.value())
        self.worker = DetectThread(self.video_path, start_frame=self.start_frame,
                                   device=device, conf=conf)
        self.worker.show_boxes = self.chk_boxes.isChecked()
        self.worker.show_trails = self.chk_trail.isChecked()
        self.worker.realtime = self.cmb_mode.currentIndex() == 1
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.debug.setPlainText)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()
        self._set_state("COUNTDOWN")
        self.btn_pause.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.cmb_device.setEnabled(False)  # 设备/置信度只在会话开始时生效，运行中固定。
        self.spn_conf.setEnabled(False)

    def on_pause(self):
        if not self.worker:
            return
        self.worker.paused = not self.worker.paused
        self.btn_pause.setText("继续" if self.worker.paused else "暂停")

    def on_step(self):
        if self.worker:
            self.worker.paused = True
            self.btn_pause.setText("继续")
            self.worker.step_once = True

    def on_stop(self, reset_position: bool = True):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self._set_state("IDLE")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_step.setEnabled(False)
        self.cmb_device.setEnabled(True)
        self.spn_conf.setEnabled(True)
        if reset_position:
            # 停止即回到开头；只有拖动进度条才移动分析起点。
            self.start_frame = 0
            self._scrubbing = False
            self.slider.blockSignals(True)
            self.slider.setValue(0)
            self.slider.blockSignals(False)

    def on_frame(self, image: QImage, stats: dict):
        if "error" in stats:
            self.debug.setPlainText(f"[ERROR] {stats['error']}")
            return
        if not image.isNull():
            scaled = QPixmap.fromImage(image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)
        if stats.get("target_id") is not None:
            self.ever_locked = True
        self._set_state(stats["state"])
        self._follow_slider(stats)
        self.debug.setPlainText(
            "状态: {state}  帧: {frame}/{total}  源fps: {fps:.1f}  设备: {device}  置信度: {conf:.2f}\n"
            "检测: {dets}  轨迹: {tracks}  目标ID: {target_id}  "
            "角速度: {omega:+.4f} rad/帧  {vis}\n"
            "模式: {mode}  分析起点: 第{start}帧  已处理: {processed}  丢帧: {dropped} ({drop_pct:.1f}%)  "
            "落后峰值: {max_behind} 帧  实际速度: {eff_fps:.1f} / {fps:.1f} fps".format(
                vis="可见" if stats["visible"] else "预测中", **stats
            )
        )

    def _on_finished(self):
        """复算结束：给出锁定判定（曾锁定目标或到 GAME_OVER）。"""
        verdict = "锁定成功（曾锁定目标）" if self.ever_locked else "未锁定（全程无目标）"
        self._set_state("GAME_OVER" if self.ever_locked else "IDLE")
        current = self.debug.toPlainText()
        self.debug.setPlainText(f"{current}\n复算判定: {verdict}")
        self.btn_pause.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.cmb_device.setEnabled(True)
        self.spn_conf.setEnabled(True)

    # ------------------------------------------------------------------ 进度条（拖动=预览未分析帧）

    def _on_scrub(self, pos: int):
        """拖动仅预览：分析无法从任意帧重新推导，只能重开会话。"""
        self._scrubbing = True
        if self.worker and not self.worker.paused:
            self.on_pause()
        self._scrub_target = pos
        self._scrub_timer.start()

    def _flush_scrub(self):
        if self._scrub_target is None or not self.video_path:
            return
        pos = self._scrub_target
        self._scrub_target = None
        frame = self._raw_frame(pos)
        if frame is None:
            return
        self.start_frame = pos
        self._show_raw(frame)
        self.lbl_pos.setText(f"{pos + 1} / {self.slider.maximum() + 1}  预览（未分析）")

    def _raw_frame(self, pos: int):
        """独立句柄读原始帧：回退 seek 不可靠，重开句柄。"""
        if self._scrub_cap is None or pos < self._scrub_cap_at:
            if self._scrub_cap is not None:
                self._scrub_cap.release()
            self._scrub_cap = cv2.VideoCapture(self.video_path)
        if not self._scrub_cap.isOpened():
            return None
        self._scrub_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = self._scrub_cap.read()
        if not ret:
            return None
        self._scrub_cap_at = pos + 1
        return frame

    def _show_raw(self, frame):
        scaled = QPixmap.fromImage(_to_qimage(frame)).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _follow_slider(self, stats: dict):
        if self._scrubbing:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(max(0, stats["frame"] - 1))
        self.slider.blockSignals(False)
        self.lbl_pos.setText(f"{stats['frame']} / {stats['total']}")

    def _sync_options(self):
        if self.worker:
            self.worker.show_boxes = self.chk_boxes.isChecked()
            self.worker.show_trails = self.chk_trail.isChecked()
            self.worker.realtime = self.cmb_mode.currentIndex() == 1

    def _set_state(self, state: str):
        self.lbl_state.setText(f"● {state}")
        self.lbl_state.setStyleSheet(
            f"color: {self.STATE_LABEL_COLORS.get(state, 'gray')}; font-weight: bold")

    def deleteLater(self):
        if self._scrub_cap is not None:
            self._scrub_cap.release()
        self.on_stop()
        super().deleteLater()
