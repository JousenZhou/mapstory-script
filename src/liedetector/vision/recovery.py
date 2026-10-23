"""Target loss recovery ranked by the measured identity signals.

The flock is the primary evidence: every distractor shares one motion model, so a
candidate whose velocity agrees with the group median is a distractor by definition
(measured cos ~ +0.97 vs the target's ~ +0.1). Path inertia (constant-velocity
prediction of where the target should emerge from the overlap) and omega agreement
with the certificate support the decision; area and IoU are tie-breakers.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision import kinematics
from src.liedetector.vision.tracks import Track
from src.liedetector.vision.logger import get_logger

log = get_logger("recovery")


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    a2 = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _reach(anchor_bbox, elapsed_frames: int) -> float:
    """One body length even for an instant recovery, then the distance budget grows with time."""
    if anchor_bbox is None:
        return 0.0
    size = max(anchor_bbox[2] - anchor_bbox[0], anchor_bbox[3] - anchor_bbox[1])
    return max(size, config.RECOVERY_MAX_SPEED * max(elapsed_frames, 1))


def _within_reach(track: Track, center, reach: float) -> bool:
    if track.centroid is None or center is None:
        return False
    return float(math.hypot(track.centroid[0] - center[0], track.centroid[1] - center[1])) <= reach


def score_candidate(
    cand: Track,
    group_vel=None,
    predicted_xy=None,
    sigma: float = 0.0,
    target_area: float = 0.0,
    cert_omega: Optional[float] = None,
    anchor_bbox=None,
    appearance_sim: Optional[float] = None,
) -> Tuple[float, float]:
    """(score, available): the second is the evidence mass this run can actually produce,
    so the threshold stays one number whether or not the flock/certificate exist."""
    w = config.RECOVERY_SCORE_WEIGHTS
    score = available = 0.0

    if group_vel is not None:
        cos = kinematics.flock_cosine(kinematics.smoothed_velocity(cand), group_vel)
        score += w["group"] * kinematics.independence_score(cos)
        available += w["group"]

    if predicted_xy is not None and sigma > 0 and cand.centroid is not None:
        score += w["path"] * kinematics.path_score(predicted_xy, cand.centroid, sigma)
        available += w["path"]

    if cert_omega is not None:
        if cand.omega_valid:
            agree = max(0.0, 1.0 - abs(cand.omega - cert_omega) / config.RECOVERY_OMEGA_TOLERANCE)
        else:
            agree = 0.0
        score += w["omega"] * agree
        available += w["omega"]

    # shape evidence from the fade-complete certificate: same-class distractors are
    # mostly dissimilar shapes (measured pairwise Hu correlation ~0), sim in [-1, 1]
    # maps to [0, 1] evidence
    if appearance_sim is not None:
        score += w["appearance"] * max(0.0, (appearance_sim + 1.0) / 2.0)
        available += w["appearance"]

    if target_area > 0 and cand.area > 0:
        score += w["area"] * (min(cand.area, target_area) / max(cand.area, target_area))
        available += w["area"]

    if anchor_bbox is not None:
        score += w["iou"] * _iou(cand.bbox, anchor_bbox)
        available += w["iou"]

    return score, available


def rank_candidates(
    candidates: List[Track],
    group_vel=None,
    lost_center=None,
    lost_velocity=None,
    elapsed_frames: int = 1,
    target_area: float = 0.0,
    omega_observable: bool = True,
    target_omega: Optional[float] = None,
    anchor_bbox=None,
) -> List[Tuple[Track, float]]:
    cert = target_omega if (omega_observable and target_omega is not None
                            and abs(target_omega) >= config.TARGET_OMEGA_CERT_FLOOR) else None
    predicted = sigma = None
    if lost_center is not None and lost_velocity is not None:
        predicted = (lost_center[0] + lost_velocity[0] * elapsed_frames,
                     lost_center[1] + lost_velocity[1] * elapsed_frames)
        sigma = _reach(anchor_bbox, elapsed_frames) / 2.0 or 30.0
    scored = [
        (t, score_candidate(t, group_vel, predicted, sigma or 0.0, target_area, cert, anchor_bbox)[0])
        for t in candidates
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def near_at_loss(track: Track, last_measured_frame: int, center, radius: float,
                 near_positions: dict = None) -> bool:
    """
    Was `track` near the loss point when the target was last measured? The snapshot taken
    at the loss frame is authoritative: motion_history is trimmed to the smoothing window
    and forgets the loss frame ~17 frames later, which starved this gate exactly when a
    long recovery needed it (score1.00_medium.mp4 f592+).
    """
    if track.created_frame > last_measured_frame:
        return True  # born after the loss: a re-label or a re-entry from off-screen
    if near_positions and track.track_id in near_positions:
        px, py = near_positions[track.track_id]
        return math.hypot(px - center[0], py - center[1]) <= radius
    return kinematics.was_near_at(track, last_measured_frame, center, radius)


def try_recover(
    tracks: List[Track],
    target_id: int,
    anchor_bbox=None,
    elapsed_frames: int = 1,
    lost_velocity=None,
    target_area: float = 0.0,
    omega_observable: bool = True,
    target_omega: Optional[float] = None,
    group_vel=None,
    last_measured_frame: int = -1,
    loss_center=None,
    near_positions: dict = None,
    exclude_ids: set = None,
    drop_anchor_score: bool = False,
    appearance_scores: dict = None,
) -> Optional[Track]:
    # a candidate must be measured this frame; its rotation fit state is NOT a gate: the
    # carrier of a re-appearing target is often a young track whose omega has not
    # converged yet, and an unconverged fit simply scores zero instead of disqualifying
    # (ids flagged as impostors - OC-SORT id theft caught at resumption - stay excluded)
    candidates = [t for t in tracks
                  if t.lost_frames == 0 and t.track_id != target_id
                  and t.track_id not in (exclude_ids or set())]
    reach = _reach(anchor_bbox, elapsed_frames)
    center = loss_center if loss_center is not None else (_center(anchor_bbox) if anchor_bbox else None)
    if center is not None:
        candidates = [t for t in candidates if _within_reach(t, center, reach)]
    if not candidates:
        return None

    near_radius = config.TARGET_NEAR_RADIUS_FACTOR * kinematics.body_length(anchor_bbox)
    kept = []
    for t in candidates:
        # an old track that was nowhere near the loss point when the target was last
        # measured is a different object drifting in, not the target re-appearing
        if not near_at_loss(t, last_measured_frame, center, near_radius, near_positions):
            continue
        # moving with the flock identifies a distractor regardless of every other score
        cos = kinematics.flock_cosine(kinematics.smoothed_velocity(t), group_vel)
        if cos is not None and cos >= config.KIN_FLOCK_COS_HI:
            continue
        kept.append(t)
    if not kept:
        return None

    cert = target_omega if (omega_observable and target_omega is not None
                            and abs(target_omega) >= config.TARGET_OMEGA_CERT_FLOOR) else None
    predicted = sigma = None
    if center is not None and lost_velocity is not None:
        predicted = (center[0] + lost_velocity[0] * elapsed_frames,
                     center[1] + lost_velocity[1] * elapsed_frames)
        sigma = max(reach / 2.0, 30.0)

    best, best_score, best_avail = None, -1.0, 0.0
    for t in kept:
        # at a suspected-mis-lock loss the anchor box belongs to the distractor: scoring
        # IoU against it rewards whatever sits near the WRONG box and masks the true
        # carrier in the best-then-ratio selection (measured: scene_703824074)
        s, a = score_candidate(t, group_vel, predicted, sigma or 0.0, target_area, cert,
                               None if drop_anchor_score else anchor_bbox,
                               (appearance_scores or {}).get(t.track_id))
        if s > best_score:
            best, best_score, best_avail = t, s, a
    if best is None:
        return None

    ratio = best_score / best_avail if best_avail > 0 else 0.0
    # at a poisoned loss the path/anchor/omega score terms are gone on purpose, so the
    # evidence mass shrinks to group+area - and in edge scenes the true carrier flies
    # WITH the flock (measured cos 0.64-1.0), capping the achievable ratio near 0.4.
    # The four hard gates (reach, near-at-loss, flock-cos ceiling, confirmed+area) still
    # apply in full; this only re-balances the soft threshold to the surviving evidence.
    min_ratio = (config.RECOVERY_MIN_RATIO_POISONED if drop_anchor_score
                 else config.RECOVERY_MIN_RATIO)
    # a direction-less candidate (too slow or too young to have a velocity) that only
    # enters the loss area late carries no independence evidence at all; without a higher
    # bar it is exactly how a slow distractor drifting past gets adopted (scene_2078906197).
    # Only applies when a flock estimate exists: without one, "no direction" is not the
    # candidate's fault and every candidate is in the same boat.
    if (ratio < config.RECOVERY_LATE_MIN_RATIO and elapsed_frames > config.RECOVERY_LATE_FRAMES
            and group_vel is not None):
        best_cos = kinematics.flock_cosine(kinematics.smoothed_velocity(best), group_vel)
        if best_cos is None:
            log.debug("late direction-less candidate id=%s ratio %.2f < %.2f",
                      best.track_id, ratio, config.RECOVERY_LATE_MIN_RATIO)
            return None
    if ratio < min_ratio:
        log.debug("best candidate id=%s ratio %.2f < %.2f", best.track_id, ratio, min_ratio)
        return None

    log.info("recovered target id=%s ratio %.2f (group-independence, path, omega agree)",
             best.track_id, ratio)
    return best
