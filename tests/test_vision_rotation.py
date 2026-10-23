"""Rotation estimation on synthetic shapes: omega sign and magnitude, and unobservable cases."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.liedetector.vision.rotation_analyzer import (
    estimate_omega,
    is_rotating,
    omega_observable,
    segment_contour,
    update_angle,
)
from src.liedetector.vision.tracks import Track

SIZE = 240


def polygon(kind: str, angle_rad: float, cx: float = 120.0, cy: float = 120.0, r: float = 60.0):
    if kind == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            a = angle_rad + i * np.pi / 5 - np.pi / 2
            pts.append([cx + rr * np.cos(a), cy + rr * np.sin(a)])
        return np.array(pts, np.float32)
    n = {"triangle": 3, "square": 4, "hexagon": 6}[kind]
    off = np.pi / 4 if kind == "square" else 0.0
    pts = []
    for i in range(n):
        a = angle_rad + off + i * 2 * np.pi / n - np.pi / 2
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(pts, np.float32)


def render(kind: str, angle_rad: float, filled: bool = True, cx: float = 120.0) -> np.ndarray:
    """White shape on a dark background, as the game presents it."""
    frame = np.full((SIZE, SIZE, 3), 30, np.uint8)
    pts = polygon(kind, angle_rad, cx=cx).astype(np.int32)
    if filled:
        cv2.fillPoly(frame, [pts], (255, 255, 255))
    else:
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    return frame


def fit(track: Track) -> Track:
    """Mirror what the pipeline does with the fit: an unconverged one is not a zero rotation."""
    omega = estimate_omega(track)
    track.omega_valid = omega is not None
    track.omega = omega if omega is not None else 0.0
    return track


def build(kind: str, omega: float, frames: int, filled: bool = True, drift: float = 0.0) -> Track:
    track = Track(track_id=1, class_id=0, class_name=kind, shape=kind)
    for i in range(frames):
        frame = render(kind, omega * i, filled=filled, cx=120 + drift * i)
        x1, y1, x2, y2 = bbox_of(frame)
        track.bbox_history.append((x1, y1, x2, y2))
        track.centroid_history.append(((x1 + x2) / 2, (y1 + y2) / 2))
        update_angle(frame, track)
        fit(track)
    return track


def bbox_of(frame: np.ndarray):
    mask = cv2.inRange(frame, (240, 240, 240), (255, 255, 255))
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@pytest.mark.parametrize("kind", ["triangle", "square", "star", "hexagon"])
def test_recovers_positive_omega(kind):
    """CW rotation in image coords (y down) is positive atan2 growth; sign must survive."""
    true_omega = 0.11
    track = build(kind, true_omega, 60)
    assert abs(track.omega) == pytest.approx(true_omega, rel=0.12), track.omega
    assert is_rotating(track)


@pytest.mark.parametrize("kind", ["triangle", "square", "star"])
def test_recovers_negative_omega(kind):
    track = build(kind, -0.09, 60)
    assert track.omega < 0
    assert abs(track.omega) == pytest.approx(0.09, rel=0.15)


def test_direction_is_signed_not_aliased():
    """Symmetry aliases would make CW and CCW indistinguishable; increments must not."""
    cw = build("triangle", 0.11, 60).omega
    ccw = build("triangle", -0.11, 60).omega
    assert cw > 0 > ccw
    assert abs(abs(cw) - abs(ccw)) < 0.02


@pytest.mark.parametrize("kind", ["triangle", "square", "star"])
def test_stationary_shape_has_zero_omega(kind):
    track = build(kind, 0.0, 60)
    assert abs(track.omega) < 0.005
    assert not is_rotating(track)


def test_translation_alone_is_not_rotation():
    track = build("triangle", 0.0, 60, drift=1.5)
    assert abs(track.omega) < 0.005


def test_circle_rotation_is_unobservable():
    """A spinning circle looks identical: contrast must flag it so omega is never trusted."""
    track = Track(track_id=1, class_id=0, class_name="circle", shape="circle")
    for i in range(40):
        frame = np.full((SIZE, SIZE, 3), 30, np.uint8)
        cv2.circle(frame, (120, 120), 60, (255, 255, 255), -1)
        track.bbox_history.append((60.0, 60.0, 180.0, 180.0))
        track.centroid_history.append((120.0, 120.0))
        update_angle(frame, track)
        fit(track)
    assert not omega_observable(track)
    assert not is_rotating(track)


def test_outline_shape_still_measurable():
    """Distractors are transparent outlines; the signature must still track their edges."""
    track = build("star", 0.10, 60, filled=False)
    assert abs(track.omega) == pytest.approx(0.10, rel=0.2)


def test_undetected_frame_holds_angle_without_breaking_fit():
    true_omega = 0.11
    track = Track(track_id=1, class_id=0, class_name="triangle", shape="triangle")
    for i in range(60):
        frame = render("triangle", true_omega * i)
        x1, y1, x2, y2 = bbox_of(frame)
        track.bbox_history.append((x1, y1, x2, y2))
        track.centroid_history.append(((x1 + x2) / 2, (y1 + y2) / 2))
        if i == 30:
            blank = np.full((SIZE, SIZE, 3), 30, np.uint8)
            assert update_angle(blank, track) is False
        else:
            update_angle(frame, track)
        fit(track)
    assert abs(track.omega) == pytest.approx(true_omega, rel=0.15)


def test_occlusion_does_not_compress_the_time_base():
    """
    Held frames are unmeasured, so a fit that renumbered the remaining samples as consecutive
    frames would read twice the rate while the star was being missed every other frame.
    """
    true_omega = 0.11
    track = Track(track_id=1, class_id=0, class_name="triangle", shape="triangle")
    for i in range(60):
        if i >= 30 and i % 2 == 0:
            # 0.11 rad over two frames still sits inside MAX_SHIFT_DEG, so the angle is real
            blank = np.full((SIZE, SIZE, 3), 30, np.uint8)
            track.bbox_history.append((60.0, 60.0, 180.0, 180.0))
            track.centroid_history.append((120.0, 120.0))
            assert update_angle(blank, track) is False
        else:
            frame = render("triangle", true_omega * i)
            x1, y1, x2, y2 = bbox_of(frame)
            track.bbox_history.append((x1, y1, x2, y2))
            track.centroid_history.append(((x1 + x2) / 2, (y1 + y2) / 2))
            update_angle(frame, track)
        fit(track)
    assert track.omega_valid
    assert track.omega == pytest.approx(true_omega, rel=0.15)


def test_an_unconverged_fit_is_not_read_as_no_rotation():
    """Below a full window of measured angles there is no rate to compare against."""
    track = build("star", 0.11, 5)
    assert estimate_omega(track) is None
    assert not track.omega_valid


def test_segment_contour_rejects_tiny_roi():
    frame = np.full((SIZE, SIZE, 3), 30, np.uint8)
    assert segment_contour(frame, (0, 0, 2, 2)) is None
