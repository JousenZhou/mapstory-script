"""Pipeline-level identity and state behaviour with a scripted detector.

These pin down what only shows up on the real video: volatile OC-SORT ids, brief missed
detections, and a fade that must be measured on the target rather than assumed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.state_machine import GameState
from src.liedetector.vision.tracks import Track

BOX = (100.0, 150.0, 140.0, 190.0)


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
    """Replays one fixed detection list per frame; the pipeline drives it through track()."""

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


def white_frame():
    frame = np.zeros((300, 300, 3), np.uint8)
    x1, y1, x2, y2 = map(int, BOX)
    frame[y1:y2, x1:x2] = 255
    return frame


def play(monkeypatch, frames, white_frames=1, countdown_frames=1):
    monkeypatch.setattr(config, "COUNTDOWN_DURATION_SEC", countdown_frames / 30.0)
    pipeline = Pipeline(StubDetector(frames), fps=30.0)
    pipeline.sm.state = GameState.COUNTDOWN
    pipeline.sm.state_started_frame = 0
    results = []
    for idx in range(len(frames)):
        frame = white_frame() if idx < white_frames else np.zeros((300, 300, 3), np.uint8)
        results.append(pipeline.process(frame, idx))
    return pipeline, results


def test_target_survives_an_id_switch(monkeypatch):
    """A star that OC-SORT hands a new id is the same star, not a lost target. The adoption
    may wait out the grace window (OC-SORT's own re-association chance) but must happen."""
    frames = [
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [det(121, 171, 7)],
        [det(122, 172, 7)],
        [det(123, 173, 7)],
        [det(124, 174, 7)],
        [det(125, 175, 7)],
        [det(126, 176, 7)],
    ]
    pipeline, results = play(monkeypatch, frames)
    assert [r.target_id for r in results[:2]] == [1, 1]
    settle = config.TARGET_GRACE_FRAMES + 2
    assert all(r.target_id == 7 for r in results[settle:])
    assert all(r.state != GameState.TARGET_LOST.value for r in results)


def test_brief_missed_detection_is_bridged(monkeypatch):
    """Doc 15.2: one frame without a detection is covered by the tracker's prediction."""
    frames = [
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [],
        [det(120, 170, 1)],
        [det(120, 170, 1)],
    ]
    pipeline, results = play(monkeypatch, frames)
    assert results[-1].state == GameState.TRACKING.value
    assert pipeline.sm.target_id == 1


def test_prolonged_absence_is_a_loss(monkeypatch):
    monkeypatch.setattr(config, "TRANSITION_TIMEOUT_FRAMES", 3)
    frames = [[det(120, 170, 1)] for _ in range(3)] + [[] for _ in range(20)]
    pipeline, _ = play(monkeypatch, frames)
    assert pipeline.sm.state in (GameState.TARGET_LOST, GameState.GAME_OVER)


def test_dead_anchor_mid_transition_goes_to_recovery(monkeypatch):
    """An anchor that dies while the fade is unresolved must not coast to the timeout and
    then guess blind: it enters TARGET_LOST immediately with fresh loss context
    (score1.00_medium f240-397: the coast ended in a distractor adoption)."""
    frames = [[det(120, 170, 1)] for _ in range(2)] + [[] for _ in range(8)]
    pipeline, results = play(monkeypatch, frames)  # TRANSITION_TIMEOUT stays at 120
    assert results[-1].state == GameState.TARGET_LOST.value
    assert pipeline.sm.lost_frame_idx is not None
    assert pipeline.sm.lost_area > 0


def test_transition_holds_while_the_target_is_still_white(monkeypatch):
    """The fade is measured: a target that has not faded must keep the state machine in TRANSITION."""
    frames = [[det(120, 170, 1)] for _ in range(8)]
    pipeline, results = play(monkeypatch, frames, white_frames=len(frames))
    assert results[-1].state == GameState.TRANSITION.value


def test_transition_does_not_finish_on_a_missing_target(monkeypatch):
    """A box we only extrapolated is not evidence that the fade completed."""
    frames = [
        [det(120, 170, 1)],
        [det(120, 170, 1)],
        [],
        [],
        [],
        [det(120, 170, 1)],
    ]
    pipeline, results = play(monkeypatch, frames)
    assert results[2].state == results[4].state == GameState.TRANSITION.value
    assert results[5].state == GameState.TRANSITION.value
    assert results[5].target_id == 1


def faded_pipeline():
    """A pipeline parked in TRACKING after the fade, with a rotation certificate."""
    pipeline = Pipeline(StubDetector([[]]), fps=30.0)
    sm = pipeline.sm
    sm.state = GameState.TRACKING
    sm.target_id = 1
    sm.target_bbox = BOX
    sm.target_omega = 0.12
    sm.target_modulation = 0.9
    sm.allowed_classes = [3]
    return pipeline


def track_at(tid, box, omega=None):
    t = Track(track_id=tid, class_id=3, class_name="pentagram")
    t.bbox_history.append(box)
    t.centroid_history.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    if omega is not None:
        t.omega, t.omega_valid = omega, True
    return t


def test_a_faded_handoff_needs_rotation_evidence():
    """
    Once the fill is gone the spin rate is the only identity evidence, so an overlapping
    neighbour must not take the target's id just by sitting near the anchor box.
    """
    pipeline = faded_pipeline()
    black = np.zeros((300, 300, 3), np.uint8)
    neighbour = track_at(2, (108.0, 158.0, 148.0, 198.0), omega=0.55)

    pipeline._reanchor_target(black, [neighbour], 5)
    assert pipeline.sm.target_id == 1

    same_rate = track_at(2, (108.0, 158.0, 148.0, 198.0), omega=0.13)
    pipeline._reanchor_target(black, [same_rate], 6)
    assert pipeline.sm.target_id == 2


def test_a_track_without_a_converged_fit_only_inherits_an_identical_box():
    """An unconverged omega is absence of evidence, so it must not authorize a loose handoff."""
    pipeline = faded_pipeline()
    black = np.zeros((300, 300, 3), np.uint8)
    fresh = track_at(3, (108.0, 158.0, 148.0, 198.0))
    pipeline._reanchor_target(black, [fresh], 5)
    assert pipeline.sm.target_id == 1

    relabelled = track_at(4, (101.0, 151.0, 141.0, 191.0))
    pipeline._reanchor_target(black, [relabelled], 6)
    assert pipeline.sm.target_id == 4


def test_target_leaving_the_frame_ends_the_game(monkeypatch):
    """Doc 15.1: exiting the picture deletes the track, so there is nothing to recover."""
    monkeypatch.setattr(config, "RECOVERY_TIMEOUT_FRAMES", 2)
    monkeypatch.setattr(config, "TRANSITION_TIMEOUT_FRAMES", 3)
    inside = [[det(120, 170, 1)] for _ in range(4)]
    gone = [[det(400, 400, 1)] for _ in range(6)]
    pipeline, results = play(monkeypatch, inside + gone, white_frames=1)
    assert pipeline.sm.state == GameState.GAME_OVER
    assert pipeline.sm.target_id is not None  # it was tracked before it left


def test_countdown_without_a_white_target_does_not_lock(monkeypatch):
    """Doc 15.2: no white fill in COUNTDOWN means no lock, and the machine must not invent one."""
    outline = [[det(120, 170, tid)] for tid in range(1, 6)]
    pipeline, results = play(monkeypatch, outline, white_frames=0, countdown_frames=5)
    assert pipeline.sm.target_id is None
    assert all(r.state != GameState.TRACKING.value for r in results)


def test_empty_track_set_is_not_a_crash(monkeypatch):
    pipeline, results = play(monkeypatch, [[] for _ in range(6)], white_frames=0)
    assert results[-1].state in (GameState.TRANSITION.value, GameState.TRACKING.value)
    assert results[-1].target_id is None
