"""State machine transitions (doc 3.2) and white-filled target confirmation (doc 6.1)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.state_machine import (
    GameState,
    StateMachine,
    confirm_initial_target,
    is_white_filled,
    white_fill_ratio,
)


def box_det(cx, cy, half=25, cls=0, name="square"):
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.9, class_id=cls, class_name=name, centroid=(cx, cy), track_id=1,
    )


def frame_with(filled_center=None, outline_centers=(), size=200):
    """Dark background; one filled white blob and/or hollow outline blobs, as the game draws them."""
    f = np.zeros((size, size, 3), np.uint8)
    for cx, cy in outline_centers:
        cv2.rectangle(f, (cx - 20, cy - 20), (cx + 20, cy + 20), (250, 250, 250), 1)
    if filled_center:
        cv2.rectangle(f, (filled_center[0] - 20, filled_center[1] - 20),
                      (filled_center[0] + 20, filled_center[1] + 20), (255, 255, 255), -1)
    return f


def test_white_filled_vs_outline():
    f = frame_with(filled_center=(60, 60), outline_centers=[(150, 150)])
    filled = white_fill_ratio(f, box_det(60, 60, half=22).bbox)
    outline = white_fill_ratio(f, box_det(150, 150, half=22).bbox)
    # the threshold must sit between the two regimes, with room on both sides
    assert filled > config.WHITE_FILL_RATIO > outline * 2, (filled, outline)
    assert is_white_filled(f, box_det(60, 60, half=22).bbox)
    assert not is_white_filled(f, box_det(150, 150, half=22).bbox)


def test_confirm_single_white_target():
    f = frame_with(filled_center=(60, 60), outline_centers=[(150, 150), (60, 150)])
    dets = [box_det(60, 60), box_det(150, 150), box_det(60, 150)]
    picked = confirm_initial_target(dets, f)
    assert picked is not None and picked.centroid == (60, 60)


def test_confirm_no_white_target_returns_none():
    f = frame_with(outline_centers=[(150, 150)])
    assert confirm_initial_target([box_det(150, 150)], f) is None


def test_happy_path_transitions():
    sm = StateMachine(fps=30.0)
    assert sm.state == GameState.IDLE
    sm.start(0)
    assert sm.state == GameState.COUNTDOWN
    assert sm.transition(GameState.TRANSITION, 90)
    assert sm.transition(GameState.TRACKING, 100)
    assert sm.transition(GameState.TARGET_LOST, 200)
    sm.lost_frame_idx = 200
    assert sm.transition(GameState.TRACKING, 210)
    assert sm.transition(GameState.GAME_OVER, 400)


def test_illegal_transition_is_rejected():
    sm = StateMachine(fps=30.0)
    sm.start(0)
    assert not sm.transition(GameState.TRACKING, 10)   # must pass through TRANSITION
    assert sm.state == GameState.COUNTDOWN
    assert not sm.transition(GameState.IDLE, 11)
    assert sm.state == GameState.COUNTDOWN


def test_lost_to_game_over_after_timeout():
    sm = StateMachine(fps=30.0)
    sm.start(0)
    sm.transition(GameState.TRANSITION, 90)
    sm.transition(GameState.TRACKING, 100)
    sm.transition(GameState.TARGET_LOST, 150)
    sm.lost_frame_idx = 150
    assert not sm.recovery_timeout(150 + config.RECOVERY_TIMEOUT_FRAMES - 1)
    assert sm.recovery_timeout(150 + config.RECOVERY_TIMEOUT_FRAMES + 1)
    assert sm.transition(GameState.GAME_OVER, 300)


def test_countdown_end_frame_uses_fps():
    sm = StateMachine(fps=30.0)
    sm.start(10)
    assert sm.countdown_end_frame == 10 + int(config.COUNTDOWN_DURATION_SEC * 30)


def test_target_omega_certificate_survives_one_bad_fit():
    """One contaminated frame must not flip the sign the handoff and recovery gates trust."""
    sm = StateMachine(fps=30.0)
    for _ in range(config.TARGET_OMEGA_HISTORY):
        sm.record_target_omega(0.12)
    sm.record_target_omega(-0.10)
    assert sm.target_omega == pytest.approx(0.12)


def test_target_omega_certificate_follows_a_sustained_change():
    """The rate really does change when the target starts to spin freely after the fade."""
    sm = StateMachine(fps=30.0)
    for _ in range(config.TARGET_OMEGA_HISTORY):
        sm.record_target_omega(0.06)
    for _ in range(config.TARGET_OMEGA_HISTORY):
        sm.record_target_omega(0.13)
    assert sm.target_omega == pytest.approx(0.13)
