"""Geometry helpers: centroid, rotated box, contour rotation signature."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.liedetector.vision.logger import get_logger

log = get_logger("geometry")


def contour_centroid(contour: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def bbox_center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def rotated_box(bbox, angle_rad: float) -> np.ndarray:
    """Axis-aligned bbox -> rotated rectangle vertices for the given orientation."""
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    size = (x2 - x1, y2 - y1)
    box = cv2.boxPoints((center, size, np.degrees(angle_rad)))
    return np.int32(box).reshape(-1, 1, 2)


@dataclass
class Signature:
    signal: np.ndarray  # mean-subtracted, L2-normalised: correlate to find the shift
    modulation: float   # relative radius variation; ~0 for rotation-invariant shapes


def _lowpass(sig: np.ndarray, harmonics: int) -> np.ndarray:
    """
    Keep only the first `harmonics` circular harmonics. A polygon's angular structure
    lives in low harmonics (triangle -> 3rd, square -> 4th, star -> 5th), while the
    per-bin ripple from rasterising a small circle is spread across high ones, so this
    is what keeps modulation from being an artefact of boundary pixelation.
    """
    n = len(sig)
    harmonics = max(1, min(harmonics, n // 2))
    spectrum = np.fft.rfft(sig)
    mask = np.zeros(len(spectrum))
    mask[: harmonics + 1] = 1.0
    return np.fft.irfft(spectrum * mask, n)


@dataclass
class ShiftEstimate:
    delta_rad: float
    score: float
    modulation: float


def radial_signature(
    contour: np.ndarray,
    bins: int = 96,
    harmonics: int = 8,
) -> Optional[Signature]:
    """
    Mean boundary radius per angle bin. Rotating the shape by phi cyclically shifts this
    signal, so phi is recoverable by correlation with no assumption about symmetry.
    Modulation is measured before normalisation: a circle's radius is flat no matter how
    it spins, and normalising would amplify its pixelation ripple into fake structure.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return None
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    d = pts - np.array([cx, cy])
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)
    idx = np.minimum((th / (2 * np.pi) * bins).astype(int), bins - 1)

    sums = np.zeros(bins)
    counts = np.zeros(bins)
    np.add.at(sums, idx, r)
    np.add.at(counts, idx, 1.0)
    raw = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    if counts.min() == 0:
        present = counts > 0
        if present.sum() < 8:
            return None
        xs = np.arange(bins)
        raw = np.interp(xs, xs[present], raw[present], period=bins)

    mean_r = float(raw.mean())
    if mean_r <= 0:
        return None
    shape = raw - mean_r
    modulation = float(np.ptp(_lowpass(shape, harmonics)) / mean_r)
    sig = _lowpass(shape, harmonics)
    norm = float(np.linalg.norm(sig))
    if norm == 0:
        return None
    return Signature(sig / norm, modulation)


def estimate_shift(
    prev: np.ndarray,
    curr: np.ndarray,
    max_shift_deg: float = 25.0,
    modulation: float = 0.0,
) -> Optional[ShiftEstimate]:
    """
    Per-frame rotation increment, searched only within max_shift_deg so the shape's
    symmetry aliases (which sit far away) can never be chosen. Sub-bin refinement by
    parabola through the three best bins.
    """
    bins = len(prev)
    if curr is None or prev is None:
        return None
    kmax = max(1, int(round(np.radians(max_shift_deg) / (2 * np.pi) * bins)))
    scores = np.array([float(np.dot(np.roll(prev, k), curr)) for k in range(-kmax, kmax + 1)])
    j = int(np.argmax(scores))
    if 0 < j < len(scores) - 1:
        y0, y1, y2 = scores[j - 1], scores[j], scores[j + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        frac = float(np.clip(frac, -0.5, 0.5))
    else:
        frac = 0.0
    step = 2 * np.pi / bins
    return ShiftEstimate((j - kmax + frac) * step, float(scores[j]), modulation)


def direction_similarity(v1, v2) -> float:
    """Cosine similarity of two 2D vectors; 0.0 when either is degenerate."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
