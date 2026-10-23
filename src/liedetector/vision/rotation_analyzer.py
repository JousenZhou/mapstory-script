"""
Rotation angle / angular velocity estimation.

Doc 6.4 plan A is not usable: cv2.minAreaRect of an axis-aligned bbox is pure 0/90 noise,
and even fed a real contour its angle jumps within [-90,0) while width/height swap, so no
mod-period branch selection recovers the true orientation (measured, see README).

Instead each frame's contour is turned into a radial signature whose cyclic shift equals
the rotation, and the *per-frame increment* is measured by correlation restricted to
MAX_SHIFT_DEG. Symmetry aliases sit far outside that window, so they can never be picked
and the accumulated angle is continuous by construction - no unwrap ambiguity.
"""

from typing import Optional

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.tracks import Track
from src.liedetector.vision.geometry import estimate_shift, radial_signature
from src.liedetector.vision.logger import get_logger

log = get_logger("rotation")


def segment_contour(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Densest boundary of the shape inside bbox: bright filled pixels, else edges."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    area = float((x2 - x1) * (y2 - y1))

    mask = cv2.threshold(gray, config.WHITE_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    contour = _largest_contour(mask, area)
    if contour is not None:
        return contour

    # transparent-outline distractors: closed edges rather than filled pixels
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return _largest_contour(edges, area)


def _largest_contour(mask: np.ndarray, roi_area: float) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < config.MIN_CONTOUR_AREA_RATIO * roi_area:
        return None
    return best.reshape(-1, 2).astype(np.float64)


def update_angle(frame: np.ndarray, track: Track) -> bool:
    """
    Append this frame's accumulated angle to the track. Returns False when the angle
    could not be measured (no contour, or the shape carries no rotational signal),
    in which case the previous angle is repeated and excluded from the omega fit.
    """
    if track.bbox is None:
        return _hold(track)

    contour = segment_contour(frame, track.bbox)
    sig = None if contour is None else radial_signature(contour, config.SIGNATURE_BINS, config.SIGNATURE_HARMONICS)
    if sig is None:
        return _hold(track)

    track.modulation = sig.modulation

    if track.signature is None:
        # first usable frame: establishes the orientation reference, angle 0
        track.signature = sig
        track.angle_history.append(0.0)
        track.angle_measured.append(True)
        return True

    observable = sig.modulation >= config.MIN_ROTATION_MODULATION
    shift = estimate_shift(
        track.signature.signal, sig.signal, config.MAX_SHIFT_DEG, sig.modulation
    )
    track.signature = sig
    if shift is None or not observable:
        return _hold(track)

    track.angle_history.append(track.angle_history[-1] + shift.delta_rad)
    track.angle_measured.append(True)
    track.last_shift_score = shift.score
    return True


def hold_angle(track: Track) -> bool:
    """Public hold: repeat the previous angle, excluded from the omega fit. Used when the
    box is contaminated (deep overlap) and measuring would poison the accumulated angle."""
    return _hold(track)


def _hold(track: Track) -> bool:
    track.angle_history.append(track.angle_history[-1] if track.angle_history else 0.0)
    track.angle_measured.append(False)
    return False


def estimate_omega(track: Track, window: int = None) -> Optional[float]:
    """
    Angular velocity in rad/frame, least squares over the last `window` measured angles.
    Returns None when there is not enough data: 0.0 is a statement that the shape is not
    rotating, so an unfinished fit must never be reported as one. The fit also uses each
    sample's real frame position, otherwise occluded frames compact the time base and the
    slope comes out too steep.
    """
    window = window or config.OMEGA_WINDOW
    tail = list(zip(track.angle_history, track.angle_measured))[-window * 2:]
    points = [(j, angle) for j, (angle, ok) in enumerate(tail) if ok]
    if len(points) < window:
        return None
    points = points[-window:]
    frames = np.asarray([p[0] for p in points], dtype=np.float64)
    angles = np.unwrap(np.asarray([p[1] for p in points], dtype=np.float64))
    span = frames[-1] - frames[0]
    if span <= 0:
        return None
    return float(np.polyfit(frames - frames[0], angles, 1)[0])


def is_rotating(track: Track) -> bool:
    """A shape with no angular modulation (circle) cannot be certified as rotating."""
    if track.modulation < config.MIN_ROTATION_MODULATION:
        return False
    return track.omega_valid and abs(track.omega) > config.OMEGA_THRESHOLD


def omega_observable(track: Track) -> bool:
    return track.modulation >= config.MIN_ROTATION_MODULATION
