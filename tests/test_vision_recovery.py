"""Loss recovery ranking (doc 6.3) - omega continuity is what separates the target from a wobble."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.liedetector.vision.recovery import rank_candidates, try_recover
from src.liedetector.vision.tracks import Track


def track(tid, omega, velocity=(0.0, 0.0), size=80.0, samples=10, lost=0, converged=True):
    t = Track(track_id=tid, class_id=3, class_name="pentagram", lost_frames=lost)
    t.bbox_history.append((0.0, 0.0, size, size))
    t.centroid_history.append((size / 2, size / 2))
    t.velocity = velocity
    t.omega = omega if converged else 0.0
    t.omega_valid = converged
    t.angle_measured = [True] * samples
    return t


def test_prefers_the_candidate_rotating_at_the_targets_rate():
    steady = track(11, omega=0.12)
    wobble = track(12, omega=0.03)  # smaller, so a plain |omega| ranking would pick this one wrong
    best = try_recover([wobble, steady], target_id=6, lost_velocity=(3.0, 0.0),
                       target_area=80 * 80, target_omega=0.12)
    assert best is not None and best.track_id == 11


def test_rejects_a_candidate_spinning_at_a_different_rate():
    wrong_way = track(13, omega=-0.30)
    assert try_recover([wrong_way], target_id=6, lost_velocity=(3.0, 0.0),
                       target_area=80 * 80, target_omega=0.12) is None


def test_rejects_a_non_rotating_candidate():
    still = track(14, omega=0.0, velocity=(3.0, 0.0))
    assert try_recover([still], target_id=6, target_area=80 * 80, target_omega=0.12) is None


def test_stale_tracks_are_not_candidates():
    stale = track(15, omega=0.12, lost=2)
    assert try_recover([stale], target_id=6, target_area=80 * 80, target_omega=0.12) is None


def test_a_track_without_a_converged_fit_scores_no_omega_not_zero_caution():
    """An unconverged omega is absence of evidence: the track stays a candidate (young
    re-appearances carry no converged fit) but cannot win against a matching one."""
    fresh = track(18, omega=0.12, velocity=(0.5, 0.0), converged=False)
    steady = track(19, omega=0.12, velocity=(3.0, 0.0))
    best = try_recover([fresh, steady], target_id=6, lost_velocity=(3.0, 0.0),
                       target_area=80 * 80, target_omega=0.12)
    assert best is not None and best.track_id == 19


def test_a_lone_unconverged_track_is_not_enough_evidence():
    fresh = track(23, omega=0.12, velocity=(3.0, 0.0), converged=False)
    assert try_recover([fresh], target_id=6, lost_velocity=(3.0, 0.0),
                       target_area=80 * 80, target_omega=0.12) is None


def moved(tid, cx, cy, omega=0.06):
    t = track(tid, omega=omega, velocity=(3.0, 0.0))
    t.bbox_history = [(cx - 40.0, cy - 40.0, cx + 40.0, cy + 40.0)]
    t.centroid_history = [(cx, cy)]
    return t


def test_a_distant_look_alike_is_not_the_target_coming_back():
    """Stars do not teleport, so recovery is bounded by the last predicted position."""
    anchor = (30.0, 30.0, 110.0, 110.0)  # centre (70,70), one body 80px
    near, far = moved(20, 100.0, 100.0), moved(21, 640.0, 640.0)
    best = try_recover([far, near], target_id=6, anchor_bbox=anchor, elapsed_frames=1,
                       lost_velocity=(3.0, 0.0), target_area=80 * 80, target_omega=0.06)
    assert best is not None and best.track_id == 20


def test_the_reach_grows_with_the_time_since_the_loss():
    """The longer the target has been hidden, the further it could legitimately have travelled."""
    anchor, candidate = (30.0, 30.0, 110.0, 110.0), lambda: moved(22, 340.0, 340.0)
    kwargs = dict(target_id=6, anchor_bbox=anchor, lost_velocity=(3.0, 0.0),
                  target_area=80 * 80, target_omega=0.06)
    assert try_recover([candidate()], elapsed_frames=2, **kwargs) is None
    assert try_recover([candidate()], elapsed_frames=70, **kwargs) is not None


def test_circle_target_falls_back_to_motion_and_size():
    """Doc 6.3's omega term is meaningless for a rotation-invariant shape."""
    near = track(16, omega=0.0, velocity=(3.0, 0.0), size=80.0)
    far = track(17, omega=0.0, velocity=(-3.0, 0.0), size=20.0)
    scored = rank_candidates([far, near], lost_velocity=(3.0, 0.0), target_area=80 * 80,
                             omega_observable=False)
    assert scored[0][0].track_id == 16


def test_matched_tracks_are_not_candidates():
    same = track(6, omega=0.12)
    assert try_recover([same], target_id=6, target_omega=0.12) is None
