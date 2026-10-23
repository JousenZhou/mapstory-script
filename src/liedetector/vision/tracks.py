"""Track data structures and lifecycle rules (doc 5.2, 5.3, 15.1)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.logger import get_logger

log = get_logger("tracks")


class TrackState(Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    shape: str = ""
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # measured (frame, x, y) samples for the kinematic profile; occluded frames carry no
    # entry so diffs across a gap average out instead of producing a spurious jump
    motion_history: List[Tuple[int, float, float]] = field(default_factory=list)
    angle_history: List[float] = field(default_factory=list)
    angle_measured: List[bool] = field(default_factory=list)
    omega: float = 0.0
    omega_valid: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    signature: Optional[np.ndarray] = None
    modulation: float = 0.0
    last_shift_score: float = 0.0
    is_confirmed: bool = False
    is_target: bool = False
    color: Tuple[int, int, int] = config.DISTRACTOR_COLOR
    created_frame: int = 0
    last_seen_frame: int = 0
    last_measured_frame: int = -1
    lost_frames: int = 0
    hits: int = 0
    state: TrackState = TrackState.CANDIDATE

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        return self.centroid_history[-1] if self.centroid_history else None

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def area(self) -> float:
        b = self.bbox
        if b is None:
            return 0.0
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    @property
    def omega_samples(self) -> int:
        """Frames that actually contributed an angle; a fresh track has no rotation evidence yet."""
        return sum(1 for m in self.angle_measured if m)


def update_track(track: Track, det: Detection, frame_idx: int) -> None:
    track.centroid_history.append(det.centroid)
    track.bbox_history.append(det.bbox)
    track.class_id = det.class_id
    track.class_name = det.class_name
    track.shape = det.shape or track.shape
    track.last_seen_frame = frame_idx
    track.last_measured_frame = frame_idx
    track.lost_frames = 0
    track.hits += 1
    track.velocity = _velocity(track)
    if not track.is_confirmed and track.hits >= config.OC_SORT_MIN_HITS:
        track.state = TrackState.CONFIRMED
        track.is_confirmed = True
    else:
        track.state = TrackState.CONFIRMED if track.is_confirmed else TrackState.CANDIDATE
    _trim(track)


def mark_missing(track: Track, frame_idx: int) -> None:
    """Occluded: keep the track alive and extrapolate. Deletion is handled separately."""
    track.lost_frames += 1
    if track.centroid_history and track.bbox_history:
        cx, cy = track.centroid
        vx, vy = track.velocity
        nx, ny = cx + vx, cy + vy
        track.centroid_history.append((nx, ny))
        x1, y1, x2, y2 = track.bbox
        track.bbox_history.append((x1 + vx, y1 + vy, x2 + vx, y2 + vy))
        track.angle_history.append(track.angle_history[-1] if track.angle_history else 0.0)
        track.angle_measured.append(False)
        track.state = TrackState.OCCLUDED
    _trim(track)


def prune_tracks(tracks: Dict[int, Track], frame_size: Tuple[int, int], frame_idx: int) -> List[int]:
    """
    Doc 15.1 rule 1: leaving the frame is deletion, being occluded is EKF extrapolation.
    They must not be conflated, so the off-screen check runs on the extrapolated centroid too.
    """
    w, h = frame_size
    removed: List[int] = []
    for tid, t in list(tracks.items()):
        c = t.centroid
        if c is None:
            removed.append(tid)
            del tracks[tid]
            continue

        cx, cy = c
        off_screen = not (
            -config.EDGE_MARGIN <= cx <= w + config.EDGE_MARGIN
            and -config.EDGE_MARGIN <= cy <= h + config.EDGE_MARGIN
        )
        if off_screen:
            log.info("track %s removed: centroid left frame at (%.1f,%.1f)", tid, cx, cy)
            removed.append(tid)
            del tracks[tid]
            continue

        if t.lost_frames > config.OC_SORT_MAX_AGE:
            log.info("track %s removed: lost for %s frames", tid, t.lost_frames)
            removed.append(tid)
            del tracks[tid]
    return removed


def _velocity(track: Track) -> Tuple[float, float]:
    h = track.centroid_history
    if len(h) < 2:
        return track.velocity
    (x0, y0), (x1, y1) = h[-2], h[-1]
    return (x1 - x0, y1 - y0)


def _trim(track: Track) -> None:
    n = config.MAX_TRAIL_LENGTH
    if len(track.centroid_history) > n:
        track.centroid_history = track.centroid_history[-n:]
        track.bbox_history = track.bbox_history[-n:]
    if len(track.angle_history) > n:
        track.angle_history = track.angle_history[-n:]
        track.angle_measured = track.angle_measured[-n:]
