"""v8/v9 identity behaviours: presentation-layer follow smoothing and the snapshot near gate.

The follow-bbox median is presentation-only: it must absorb a 1-2 frame detection drag
without touching identity state, and it must never mix two objects across an id change.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.recovery import near_at_loss
from src.liedetector.vision.state_machine import GameState
from src.liedetector.vision.tracks import Track


def det(cx, cy, tid, half=20, conf=0.9):
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=conf,
        class_id=3,
        class_name="pentagram",
        centroid=(cx, cy),
        track_id=tid,
        shape="star",
    )


class StubDetector:
    def __init__(self, frames):
        self.frames = frames
        self.idx = -1

    def detect(self, frame):
        return self._current()

    def track(self, frame):
        self.idx += 1
        return self._current()

    def _current(self):
        return list(self.frames[self.idx]) if 0 <= self.idx < len(self.frames) else []

    def classes_for_shape(self, shape):
        return [3]


def white_frame(box):
    frame = np.zeros((300, 300, 3), np.uint8)
    x1, y1, x2, y2 = map(int, box)
    frame[y1:y2, x1:x2] = 255
    return frame


def play(monkeypatch, frames, boxes):
    monkeypatch.setattr(config, "COUNTDOWN_DURATION_SEC", 1 / 30.0)
    pipeline = Pipeline(StubDetector(frames), fps=30.0)
    pipeline.sm.state = GameState.COUNTDOWN
    pipeline.sm.state_started_frame = 0
    results = []
    for idx in range(len(frames)):
        frame = white_frame(boxes[min(idx, len(boxes) - 1)])
        results.append(pipeline.process(frame, idx))
    return pipeline, results


class StubTrack:
    """Minimal track stand-in for driving _smoothed_follow_bbox deterministically."""

    def __init__(self, track_id, bbox, lost_frames=0):
        self.track_id = track_id
        self.bbox = bbox
        self.lost_frames = lost_frames


def _pipeline():
    return Pipeline(StubDetector([]), fps=30.0)


def test_follow_bbox_absorbs_single_frame_drag():
    """A distractor grazing the target drags the detection for one frame; the 3-frame
    median output stays on the target position (scene_2078906183's final-frame miss)."""
    p = _pipeline()
    stable = (100.0, 100.0, 140.0, 140.0)
    dragged = (100.0, 115.0, 140.0, 155.0)
    out = [
        p._smoothed_follow_bbox(StubTrack(1, stable), i)
        for i in range(5)
    ]
    out.append(p._smoothed_follow_bbox(StubTrack(1, dragged), 5))
    out.append(p._smoothed_follow_bbox(StubTrack(1, stable), 6))
    assert out[0] == stable and out[1] == stable  # passthrough until the window fills
    assert out[3] == stable and out[4] == stable  # full window: stable median
    assert out[5] == stable  # the dragged frame's median is the stable box
    assert out[6] == stable  # and the window washes the drag out immediately


def test_follow_bbox_resets_on_id_change():
    """The median window must never mix two objects: after an id switch the output is a
    passthrough of the new carrier's box, not a median of old and new."""
    p = _pipeline()
    old_box = (100.0, 100.0, 140.0, 140.0)
    new_box = (200.0, 200.0, 240.0, 240.0)
    for i in range(4):
        p._smoothed_follow_bbox(StubTrack(1, old_box), i)
    assert p._smoothed_follow_bbox(StubTrack(7, new_box), 4) == new_box
    # window refills with the new id only
    assert p._smoothed_follow_bbox(StubTrack(7, new_box), 5) == new_box
    assert p._smoothed_follow_bbox(StubTrack(7, new_box), 6) == new_box


def test_near_at_loss_prefers_the_position_snapshot():
    """The snapshot answers 'where was this track at the reference frame' even when
    motion_history has already forgotten it (17-entry trim, 5-frame lookup tolerance)."""
    t = Track(track_id=11, class_id=3, class_name="pentagram")
    t.created_frame = 0
    # history covers frames 100-116; the reference frame 90 is far outside the window
    for i in range(100, 117):
        t.motion_history.append((i, 200.0, 200.0))
    assert near_at_loss(t, 90, (205.0, 200.0), 50.0, {11: (205.0, 200.0)}) is True
    assert near_at_loss(t, 90, (400.0, 400.0), 50.0, {11: (205.0, 200.0)}) is False
    # without a snapshot, motion_history cannot reach frame 90 (best entry 100, off by 10
    # frames > KIN_NEAR_FRAME_TOLERANCE)
    assert near_at_loss(t, 90, (205.0, 200.0), 50.0, None) is False
