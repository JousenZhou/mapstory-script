# -*- coding: utf-8 -*-
"""光流为主的解题会话编排器（宿主新增，D:\\lie 无此模块）。

三相状态机：
- phase-1 倒计时/褪色：复用现有 Pipeline（YOLO 全帧率），照常锁定白目标；
- phase-2 跟踪：FlowTracker 逐帧像素携带身份，follow_bbox 持续输出；每
  FLOW_VERIFY_INTERVAL 帧一次 YOLO 全帧校验——与光流框 IoU 达标的检测直接吸附
  （重播种，杀漂移），连续 FLOW_VERIFY_MISS 次无匹配判滑移转 phase-3；
- phase-3 重获：全帧率 YOLO + 全新 Tracker 簿记，try_recover 四路证据评分，
  光流末位框/速度作为锚与预测（比 d/lie 丢失瞬间的静态快照更强）；恢复成功
  重播种回 phase-2，RECOVERY_TIMEOUT_FRAMES 超时 → GAME_OVER。

对外接口与 Pipeline 兼容（process -> FrameResult），solver 除工厂替换外零改动。
FLOW_PRIMARY=False 时 create_session 返回原 Pipeline，行为与 D:\\lie 完全一致。
"""
from typing import List, Optional

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.flow_tracker import FlowTracker
from src.liedetector.vision.pipeline import FrameResult, Pipeline
from src.liedetector.vision.recovery import try_recover
from src.liedetector.vision.state_machine import GameState
from src.liedetector.vision.tracks import Track, TrackState
from src.liedetector.vision.tracker import Tracker
from src.liedetector.vision import kinematics
from src.liedetector.vision.logger import get_logger

log = get_logger("flow_session")

_TRAIL_LEN = 50  # 合成轨迹的历史长度（渲染用）。


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    a2 = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


class FlowSession:
    """光流为主的三相解题会话。process(frame, frame_idx) -> FrameResult。"""

    def __init__(self, detector, fps: float):
        self.detector = detector
        self.fps = fps
        self._pipeline = Pipeline(detector, fps)  # phase-1 复用 d/lie 全套（含其自身恢复）。
        self._pipeline.sm.start(0)
        self._flow = FlowTracker()
        self._phase = "pipeline"  # pipeline / flow / reacquire / game_over
        self._handoff_stable = 0
        # 交接身份证书（phase-1 结束时捕获）。
        self._target_id = None
        self._target_class_id = 0
        self._target_class_name = ""
        self._target_omega = None
        self._target_modulation = 0.0
        self._target_confidence = 0.0
        # phase-2 状态。
        self._follow_hist = []  # (bbox) 中值平滑窗。
        self._disp_hist = []  # 最近光流位移（重获速度锚）。
        self._verify_misses = 0
        self._last_verify_idx = -10**9
        self._last_dets = 0
        # phase-3 状态。
        self._tracker = None  # 重获期新建的 OC-SORT 簿记。
        self._loss_bbox = None
        self._loss_frame = None
        self._coast_bbox = None
        self._coast_vel = None
        # 渲染轨迹。
        self._track = None

    # ------------------------------------------------------------------ 主入口

    def start(self, frame_idx: int = 0) -> None:
        """（重）锚定 phase-1 状态机时钟：seek 进入等于新会话，与 Pipeline.sm.start 对齐。"""
        self._phase = "pipeline"
        self._pipeline.sm.start(frame_idx)

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        if self._phase == "pipeline":
            return self._process_pipeline(frame, frame_idx)
        if self._phase == "flow":
            return self._process_flow(frame, frame_idx)
        if self._phase == "reacquire":
            return self._process_reacquire(frame, frame_idx)
        return self._emit(frame, frame_idx, GameState.GAME_OVER.value, None)

    # ------------------------------------------------------------------ phase-1

    def _process_pipeline(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        res = self._pipeline.process(frame, frame_idx)
        if res.state == GameState.GAME_OVER.value:
            self._phase = "game_over"
            return res
        if res.state != GameState.TRACKING.value or res.follow_bbox is None:
            self._handoff_stable = 0
            return res
        self._handoff_stable += 1
        if self._handoff_stable < config.FLOW_HANDOFF_STABLE:
            return res
        self._handoff(res, frame, frame_idx)
        return self._process_flow(frame, frame_idx)  # 交接帧立即以光流接管。

    def _handoff(self, res: FrameResult, frame: np.ndarray, frame_idx: int) -> None:
        """褪色完成、身份确定：捕获证书并播种光流。"""
        self._target_id = res.target_id
        locked = next((t for t in res.tracks if t.track_id == res.target_id), None)
        if locked is not None:
            self._target_class_id = locked.class_id
            self._target_class_name = locked.class_name
        self._target_omega = self._pipeline.sm.target_omega
        self._target_modulation = self._pipeline.sm.target_modulation
        self._target_confidence = res.confidence
        self._track = Track(track_id=int(res.target_id), class_id=self._target_class_id,
                            class_name=self._target_class_name, created_frame=frame_idx,
                            is_confirmed=True, is_target=True, state=TrackState.CONFIRMED,
                            last_seen_frame=frame_idx)
        self._follow_hist = [res.follow_bbox]
        self._disp_hist = []
        self._verify_misses = 0
        self._last_verify_idx = frame_idx
        self._flow.start(self._to_gray(frame), res.follow_bbox)
        self._phase = "flow"
        log.info("flow session handoff at frame %s (target=%s omega=%s)",
                 frame_idx, self._target_id, self._target_omega)

    # ------------------------------------------------------------------ phase-2

    def _process_flow(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        gray = self._to_gray(frame)
        res = self._flow.update(gray)
        if res is None:
            self._enter_reacquire(frame_idx)
            return self._process_reacquire(frame, frame_idx)
        bbox, disp = res
        self._disp_hist.append(disp)
        if len(self._disp_hist) > 15:
            del self._disp_hist[0]
        self._follow_hist.append(bbox)
        if len(self._follow_hist) > config.FLOW_SMOOTH_FRAMES:
            del self._follow_hist[0]
        self._update_track_hist(bbox, frame_idx)
        follow = self._smoothed()
        detect_count = 0
        # 周期性 YOLO 校验：吸附杀漂移，连续无匹配判滑移。
        if frame_idx - self._last_verify_idx >= config.FLOW_VERIFY_INTERVAL:
            self._last_verify_idx = frame_idx
            snap, dets = self._verify(frame, gray, bbox)
            detect_count = dets
            if snap is not None:
                self._follow_hist = [snap]
                follow = snap
                self._update_track_hist(snap, frame_idx)
            else:
                self._verify_misses += 1
                if self._verify_misses >= config.FLOW_VERIFY_MISS:
                    log.warning("flow verify missed %s times at frame %s, reacquire",
                                self._verify_misses, frame_idx)
                    self._enter_reacquire(frame_idx)
                    return self._process_reacquire(frame, frame_idx)
        return self._emit(frame, frame_idx, GameState.TRACKING.value, follow,
                          detect_count=detect_count)

    def _verify(self, frame: np.ndarray, gray: np.ndarray, bbox):
        """YOLO 全帧校验：返回 (吸附框 or None, 检测数)。"""
        try:
            dets = self.detector.track(frame)
        except Exception as e:
            log.warning("flow verify detector failed: %s", e)
            return None, 0
        self._last_dets = len(dets)
        best, best_iou = None, 0.0
        for d in dets:
            iou = _iou(bbox, d.bbox)
            if iou > best_iou:
                best, best_iou = d, iou
        if best is None or best_iou < config.FLOW_VERIFY_IOU:
            return None, len(dets)
        snap = tuple(float(v) for v in best.bbox)
        self._flow.start(gray, snap)  # 吸附重播种：漂移清零、载波年龄清零。
        self._verify_misses = 0
        return snap, len(dets)

    # ------------------------------------------------------------------ phase-3

    def _enter_reacquire(self, frame_idx: int) -> None:
        self._phase = "reacquire"
        self._loss_bbox = self._smoothed() or self._flow.bbox or self._follow_hist[-1]
        self._loss_frame = frame_idx
        self._tracker = Tracker(self.detector)  # 全新簿记：id 与旧局/旧轨无关。
        vel = self._median_disp()
        self._coast_vel = vel
        self._coast_bbox = self._loss_bbox
        log.warning("flow lost at frame %s (bbox=%s vel=%s), reacquire", frame_idx,
                    self._loss_bbox, vel)

    def _process_reacquire(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        try:
            tracks = self._tracker.update(frame, frame_idx)
        except Exception as e:
            log.warning("reacquire tracker failed: %s", e)
            tracks = []
        self._last_dets = len(self._tracker.last_detections)
        elapsed = max(frame_idx - self._loss_frame, 1)
        # 光流惯性外推 follow：鼠标沿原轨迹继续走，不在丢失点冻结。
        follow = None
        if elapsed <= config.FLOW_COAST_ON_LOST and self._coast_vel is not None:
            dx, dy = self._coast_vel
            b = self._coast_bbox
            self._coast_bbox = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
            follow = self._coast_bbox
            self._update_track_hist(follow, frame_idx)
        recovered = try_recover(
            tracks,
            target_id=self._target_id,
            anchor_bbox=self._loss_bbox,
            elapsed_frames=elapsed,
            lost_velocity=self._median_disp(),
            target_area=max((self._loss_bbox[2] - self._loss_bbox[0]) *
                            (self._loss_bbox[3] - self._loss_bbox[1]), 1.0),
            omega_observable=self._target_modulation >= config.MIN_ROTATION_MODULATION,
            target_omega=self._target_omega,
            group_vel=kinematics.group_velocity(tracks),
            last_measured_frame=self._loss_frame,
            loss_center=_center(self._loss_bbox),
            near_positions={},  # 光流模式无丢失快照：晚出生轨迹按 near 通过（有意偏离）。
            exclude_ids={self._target_id},
        )
        if recovered is not None:
            log.info("flow reacquired id=%s at frame %s", recovered.track_id, frame_idx)
            gray = self._to_gray(frame)
            self._target_id = recovered.track_id
            self._follow_hist = [recovered.bbox]
            self._flow.start(gray, recovered.bbox)
            self._phase = "flow"
            return self._emit(frame, frame_idx, GameState.TRACKING.value, recovered.bbox,
                              detect_count=self._last_dets)
        if elapsed >= config.RECOVERY_TIMEOUT_FRAMES:
            log.warning("flow reacquire timed out at frame %s", frame_idx)
            self._phase = "game_over"
            return self._emit(frame, frame_idx, GameState.GAME_OVER.value, None)
        return self._emit(frame, frame_idx, GameState.TARGET_LOST.value, follow,
                          detect_count=self._last_dets)

    # ------------------------------------------------------------------ 工具

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _median_disp(self) -> Optional[tuple]:
        if not self._disp_hist:
            return None
        arr = np.asarray(self._disp_hist, dtype=np.float64)
        return tuple(float(v) for v in np.median(arr, axis=0))

    def _smoothed(self):
        if not self._follow_hist:
            return None
        if len(self._follow_hist) < config.FLOW_SMOOTH_FRAMES:
            return self._follow_hist[-1]
        arr = np.asarray(self._follow_hist, dtype=np.float64)
        return tuple(float(v) for v in np.median(arr, axis=0))

    def _update_track_hist(self, bbox, frame_idx: int) -> None:
        t = self._track
        if t is None or bbox is None:
            return
        t.bbox_history.append(tuple(float(v) for v in bbox))
        t.centroid_history.append(_center(bbox))
        t.last_seen_frame = frame_idx
        if len(t.bbox_history) > _TRAIL_LEN:
            del t.bbox_history[0]
            del t.centroid_history[0]

    def _emit(self, frame, frame_idx: int, state: str, follow_bbox,
              detect_count: int = 0) -> FrameResult:
        if self._track is not None and follow_bbox is not None:
            self._update_track_hist(follow_bbox, frame_idx)
        return FrameResult(
            frame=frame,
            tracks=[self._track] if self._track is not None else [],
            state=state,
            target_id=self._target_id,
            omega=self._target_omega or 0.0,
            confidence=self._target_confidence,
            frame_idx=frame_idx,
            detect_count=detect_count,
            follow_bbox=follow_bbox,
        )


def create_session(detector, fps: float):
    """solver 工厂：FLOW_PRIMARY=True 走光流主会话，False 走原 d/lie Pipeline。"""
    if config.FLOW_PRIMARY:
        return FlowSession(detector, fps)
    pipeline = Pipeline(detector, fps)
    pipeline.sm.start(0)
    return pipeline
