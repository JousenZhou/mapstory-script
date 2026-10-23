"""Overlay rendering: rotated boxes, trails, centroids, vectors, labels (doc 7)."""

from typing import List, Optional

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.tracks import Track
from src.liedetector.vision.geometry import rotated_box


def render(
    frame: np.ndarray,
    tracks: List[Track],
    target_id: Optional[int] = None,
    state: str = "IDLE",
    show_boxes: bool = True,
    show_trails: bool = True,
) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        is_target = t.track_id == target_id
        color = config.TARGET_COLOR if is_target else config.DISTRACTOR_COLOR
        if show_boxes and t.bbox is not None:
            angle = t.angle_history[-1] if t.angle_history else 0.0
            cv2.polylines(out, [rotated_box(t.bbox, angle)], True, color, config.BOX_THICKNESS)
        if show_trails and len(t.centroid_history) > 1:
            trail_color = config.TARGET_COLOR if is_target else (80, 80, 80)
            pts = np.array(t.centroid_history[-config.MAX_TRAIL_LENGTH:], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, trail_color, config.TRAIL_THICKNESS)
        if t.centroid is None:
            continue
        cx, cy = int(t.centroid[0]), int(t.centroid[1])
        cv2.circle(out, (cx, cy), config.CENTROID_RADIUS, color, -1)

        vx, vy = t.velocity
        if abs(vx) + abs(vy) > 0.5:
            cv2.arrowedLine(out, (cx, cy), (int(cx + vx * 10), int(cy + vy * 10)),
                            config.VELOCITY_COLOR, 1, tipLength=0.3)

        if t.angle_history:
            a = t.angle_history[-1]
            length = max(6.0, (t.bbox[2] - t.bbox[0]) / 2.0) if t.bbox else 20.0
            cv2.line(out, (cx, cy), (int(cx + length * np.cos(a)), int(cy + length * np.sin(a))),
                     config.ANGLE_COLOR, 1)

        label = f"ID:{t.track_id} w:{t.omega:+.3f}"
        cv2.putText(out, label, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(out, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                _state_color(state), 2)
    return out


def _state_color(state: str):
    return config.STATE_LIGHT_COLORS.get(state, config.DISTRACTOR_COLOR)
