"""Per-track kinematic profiles and the flock estimate that identity scoring reads.

Measured on the validation set: the distractors share one motion model and move as a
coherent flock (cos of their velocity with the group median ~ +0.97), while the target
wanders on its own (cos ~ +0.1, sometimes negative). That separation is far wider than
any other identity signal available after the fade, so it anchors the handoff/recovery
score; path inertia (constant-velocity prediction) and omega agreement support it.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.tracks import Track
from src.liedetector.vision.logger import get_logger

log = get_logger("kinematics")


def record_measurement(track: Track, frame_idx: int) -> None:
    """Measured samples only, with their frames: diffs across an occlusion gap then give
    an averaged velocity instead of a huge spurious jump."""
    track.motion_history.append((frame_idx, track.centroid[0], track.centroid[1]))
    overflow = len(track.motion_history) - (config.KIN_SMOOTH_WINDOW + 2)
    if overflow > 0:
        del track.motion_history[:overflow]


def smoothed_velocity(track: Track, window: int = None) -> Optional[Tuple[float, float]]:
    """Component-wise median of per-interval velocities. A median over ~15 measured
    intervals shrugs off a couple of jittered boxes; the raw last-step velocity cannot."""
    h = track.motion_history[-(window or config.KIN_SMOOTH_WINDOW) - 1:]
    if len(h) < config.KIN_MIN_SAMPLES:
        return None
    arr = np.asarray(h, dtype=np.float64)
    dt = np.diff(arr[:, 0])
    ok = dt > 0
    if not ok.any():
        return None
    return (float(np.median(np.diff(arr[:, 1])[ok] / dt[ok])),
            float(np.median(np.diff(arr[:, 2])[ok] / dt[ok])))


def group_velocity(tracks: List[Track]) -> Optional[Tuple[float, float]]:
    """
    Median velocity across the live measured tracks: with 13/14 of the objects being
    distractors, the median IS the flock velocity and stays correct even though the
    (unknown) target is inside the sample set.
    """
    vs = []
    for t in tracks:
        if t.lost_frames != 0 or not t.is_confirmed:
            continue
        v = smoothed_velocity(t)
        if v is not None:
            vs.append(v)
    if len(vs) < config.KIN_GROUP_MIN_TRACKS:
        return None
    arr = np.asarray(vs)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def flock_cosine(track_vel, group_vel) -> Optional[float]:
    """cos(track velocity, flock velocity); None when either side is too slow for a
    direction to mean anything."""
    if track_vel is None or group_vel is None:
        return None
    nt = math.hypot(*track_vel)
    ng = math.hypot(*group_vel)
    if nt < config.KIN_MIN_SPEED or ng < config.KIN_MIN_SPEED:
        return None
    return (track_vel[0] * group_vel[0] + track_vel[1] * group_vel[1]) / (nt * ng)


def independence_score(cos: Optional[float]) -> float:
    """Evidence that a track is NOT a flock member: 1 = clearly independent (target-like),
    0 = indistinguishable from the flock. None (no measurable direction) = no evidence."""
    if cos is None:
        return 0.0
    hi, lo = config.KIN_FLOCK_COS_HI, config.KIN_FLOCK_COS_LO
    if cos >= hi:
        return 0.0
    if cos <= lo:
        return 1.0
    return (hi - cos) / (hi - lo)


def path_score(predicted_xy, cand_xy, sigma: float) -> float:
    """Gaussian closeness to the constant-velocity prediction of where the target should
    have emerged from the overlap."""
    if predicted_xy is None or sigma <= 0:
        return 0.0
    d = math.hypot(predicted_xy[0] - cand_xy[0], predicted_xy[1] - cand_xy[1])
    return math.exp(-(d * d) / (2.0 * sigma * sigma))


def was_near_at(track: Track, frame_idx: int, center, radius: float) -> bool:
    """
    Was `track` within `radius` of `center` around `frame_idx`? A track born after that
    moment could not have been anywhere else than "not yet separated", so it counts as
    near: the merge-split case where the tracker re-labels one half of a blob.
    """
    if track.created_frame > frame_idx:
        return True
    best = None
    for entry in track.motion_history:
        if best is None or abs(entry[0] - frame_idx) < abs(best[0] - frame_idx):
            best = entry
    if best is None or abs(best[0] - frame_idx) > config.KIN_NEAR_FRAME_TOLERANCE:
        return False
    return math.hypot(best[1] - center[0], best[2] - center[1]) <= radius


def body_length(bbox) -> float:
    if bbox is None:
        return 0.0
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1])
