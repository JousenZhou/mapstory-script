"""Game state machine (doc 3) and target confirmation (doc 6.1)."""

from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.logger import get_logger

log = get_logger("state")


class GameState(Enum):
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    TRANSITION = "TRANSITION"
    TRACKING = "TRACKING"
    TARGET_LOST = "TARGET_LOST"
    GAME_OVER = "GAME_OVER"


ALLOWED = {
    GameState.IDLE: {GameState.COUNTDOWN, GameState.GAME_OVER},
    GameState.COUNTDOWN: {GameState.TRANSITION, GameState.GAME_OVER},
    # TARGET_LOST is reachable mid-fade: an anchor that dies in TRANSITION goes straight
    # to the recovery machinery instead of coasting a stale box to the timeout
    GameState.TRANSITION: {GameState.TRACKING, GameState.TARGET_LOST, GameState.GAME_OVER},
    GameState.TRACKING: {GameState.TARGET_LOST, GameState.GAME_OVER},
    GameState.TARGET_LOST: {GameState.TRACKING, GameState.GAME_OVER},
    GameState.GAME_OVER: set(),
}


def white_fill_ratio(frame: np.ndarray, bbox) -> float:
    """
    Share of the box covered by white pixels. A filled shape only covers its own
    silhouette - a 5-point star fills ~0.35 of its bbox - so this is bounded well below 1
    even for the true target, and the threshold must be calibrated to that, not to 0.5.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0

    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    lo, hi = config.WHITE_HSV_RANGE
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return float(mask.mean()) / 255.0


def is_white_filled(frame: np.ndarray, bbox) -> bool:
    return white_fill_ratio(frame, bbox) >= config.WHITE_FILL_RATIO


def confirm_initial_target(detections: List[Detection], frame: np.ndarray) -> Optional[Detection]:
    """Whitest filled candidate wins; area and confidence do not distinguish target from glint."""
    scored = [(d, white_fill_ratio(frame, d.bbox)) for d in detections]
    candidates = [(d, r) for d, r in scored if r >= config.WHITE_FILL_RATIO]
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning("%d white-filled candidates, taking the whitest", len(candidates))
    return max(candidates, key=lambda dr: dr[1])[0]


class StateMachine:
    def __init__(self, fps: float):
        self.fps = fps
        self.state = GameState.IDLE
        self.state_started_frame = 0
        self.target_id: Optional[int] = None
        self.target_class: Optional[str] = None
        self.target_shape: Optional[str] = None
        self.target_class_id: Optional[int] = None
        self.target_area: float = 0.0
        self.target_modulation: float = 0.0
        self.target_omega: Optional[float] = None
        self.target_omegas = deque(maxlen=config.TARGET_OMEGA_HISTORY)
        self.lost_velocity = None
        self.lost_area: float = 0.0
        self.allowed_classes: List[int] = []
        self.lost_frame_idx: Optional[int] = None
        self.target_confidence: float = 0.0
        self.fade_white_ratio: float = 1.0
        self.fade_low_frames: int = 0
        self.target_bbox: Optional[Tuple[float, float, float, float]] = None
        # frame at which the currently locked track was last measured; a candidate that
        # already existed and was far away then is a different object, not a re-label
        self.target_last_measured_frame: int = -1

    @property
    def countdown_end_frame(self) -> int:
        return int(self.state_started_frame + config.COUNTDOWN_DURATION_SEC * self.fps)

    def transition(self, new_state: GameState, frame_idx: int) -> bool:
        if new_state == self.state:
            return False
        if new_state not in ALLOWED[self.state]:
            log.warning("ignored transition %s -> %s", self.state.value, new_state.value)
            return False
        log.info("state %s -> %s (frame %s)", self.state.value, new_state.value, frame_idx)
        self.state = new_state
        self.state_started_frame = frame_idx
        return True

    def start(self, frame_idx: int = 0) -> None:
        self.state = GameState.COUNTDOWN
        self.state_started_frame = frame_idx

    def set_target(self, det: Detection, track_id: int, allowed_classes: List[int]) -> None:
        self.target_id = track_id
        self.target_class = det.class_name
        self.target_shape = det.shape or det.class_name
        self.target_class_id = det.class_id
        self.target_area = det.area
        self.target_confidence = det.confidence
        self.target_bbox = det.bbox
        self.target_last_measured_frame = 0
        self.allowed_classes = list(allowed_classes)
        log.info("target locked: id=%s class=%s classes=%s", track_id, det.class_name, allowed_classes)

    def hand_off_target(self, track_id: int, bbox, iou: float) -> None:
        """
        OC-SORT re-identified the target (overlaps and box jitter make ids volatile).
        Following the geometry keeps one continuous target identity instead of a false loss.
        """
        log.info("target handoff %s -> %s (iou %.2f)", self.target_id, track_id, iou)
        self.target_id = track_id
        self.target_bbox = bbox

    def record_target_omega(self, omega: float) -> None:
        """
        Identity certificate used by the handoff and recovery gates. One fit can be dragged
        off by a contaminated contour while stars overlap, and a sign flip then authenticates
        the wrong track for the rest of the run, so this keeps the median of the recent fits.
        """
        self.target_omegas.append(omega)
        self.target_omega = float(np.median(list(self.target_omegas)))

    def reset_target_omegas(self) -> None:
        """After a corrected silent swap (or a recovery from a mis-lock) the deque's tail can
        belong to the distractor; keeping it would make the certificate reject the true
        carrier. Clear it and let the new carrier's fits rebuild the evidence."""
        self.target_omegas.clear()
        self.target_omega = None

    def is_target_visible(self, tracks) -> bool:
        if self.target_id is None:
            return False
        for t in tracks:
            if t.track_id == self.target_id:
                # doc 15.2: a dropped detection is bridged by the tracker's EKF, not a loss
                return t.lost_frames <= config.TARGET_LOST_TOLERANCE_FRAMES
        return False

    def lost_elapsed_frames(self, frame_idx: int) -> int:
        return 0 if self.lost_frame_idx is None else frame_idx - self.lost_frame_idx

    def recovery_timeout(self, frame_idx: int) -> bool:
        return self.lost_elapsed_frames(frame_idx) > config.RECOVERY_TIMEOUT_FRAMES
