"""Appearance identity evidence: rotation-invariant Hu-moment signatures.

Distractors share the target's CLASS (measured: every object in a scene is the same
class), but compound targets are built from primitive combinations and the pairwise Hu
correlation between same-class objects is near zero (measured med -0.13..0.08, i.e.
mostly dissimilar) - so a shape signature extracted when the identity is certain gives
recovery/correction an evidence channel that does not depend on motion at all.

The signature MUST be taken in the outline (post-fade) form: the countdown shows the
target white-filled while tracking shows it as an outline, and the two have different
moments. It is captured once at the fade-complete transition.
"""

from typing import Optional

import cv2
import numpy as np

from src.liedetector.vision.logger import get_logger

log = get_logger("appearance")


def hu_signature(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """7-dim rotation/scale-invariant Hu signature of the shape inside `bbox`.

    The shape is an outline over a textured background, so the signature comes from the
    gradient magnitude (edges), thresholded by the ROI's own statistics - robust to the
    global illumination drift the simulator applies. None when the box has no structure.
    """
    if bbox is None:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
    x2, y2 = min(w, int(round(bbox[2]))), min(h, int(round(bbox[3])))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    thresh = max(40.0, float(np.percentile(mag, 85)))
    mask = (mag >= thresh).astype(np.uint8)
    if mask.sum() < 40:
        return None
    m = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def signature_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Pearson correlation of two Hu signatures in [-1, 1]; None if either is missing.
    H7 is numerically unstable (|h| ~ 1e-18) and pure noise - excluded from the match."""
    if a is None or b is None:
        return None
    c = np.corrcoef(a[:6], b[:6])[0, 1]
    return None if np.isnan(c) else float(c)
