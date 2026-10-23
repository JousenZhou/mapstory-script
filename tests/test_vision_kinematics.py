"""Kinematic profiles and the overlap identity gates they feed (validation-set measured)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.liedetector.vision import config
from src.liedetector.vision import kinematics
from src.liedetector.vision.recovery import try_recover
from src.liedetector.vision.tracks import Track


def mk(tid, cx=400.0, cy=300.0, created=0):
    t = Track(track_id=tid, class_id=3, class_name="pentagram", created_frame=created)
    t.bbox_history.append((cx - 47, cy - 47, cx + 47, cy + 47))
    t.centroid_history.append((cx, cy))
    t.is_confirmed = True
    return t


def fly(t, vx, vy, frames=20, start=0):
    """Append a measured motion history: constant velocity plus per-frame samples."""
    x, y = t.centroid
    for f in range(start, start + frames):
        x, y = x + vx, y + vy
        t.motion_history.append((f, x, y))
        t.centroid_history.append((x, y))
    t.velocity = (vx, vy)


# --- kinematics --------------------------------------------------------

def test_smoothed_velocity_recovers_constant_motion():
    t = mk(1)
    fly(t, 3.0, -1.5, frames=20)
    vx, vy = kinematics.smoothed_velocity(t)
    assert vx == pytest.approx(3.0, abs=0.01) and vy == pytest.approx(-1.5, abs=0.01)


def test_smoothed_velocity_is_robust_to_box_jitter():
    t = mk(1)
    fly(t, 2.0, 2.0, frames=20)
    # a couple of wildly displaced samples (merging boxes) must not drag the median
    t.motion_history[-2] = (t.motion_history[-2][0], t.motion_history[-2][1] + 40, t.motion_history[-2][2])
    vx, vy = kinematics.smoothed_velocity(t)
    assert vx == pytest.approx(2.0, abs=0.1) and vy == pytest.approx(2.0, abs=0.1)


def test_group_velocity_median_ignores_the_one_independent_mover():
    flock = []
    for i in range(7):
        t = mk(i)
        fly(t, 4.0, 0.0, frames=20)
        flock.append(t)
    odd = mk(99)
    fly(odd, -5.0, 5.0, frames=20)  # the target: different direction entirely
    gv = kinematics.group_velocity(flock + [odd])
    assert gv is not None
    assert gv[0] == pytest.approx(4.0, abs=0.1) and gv[1] == pytest.approx(0.0, abs=0.1)


def test_group_velocity_needs_a_quorum():
    t = mk(1)
    fly(t, 4.0, 0.0)
    assert kinematics.group_velocity([t]) is None


def test_flock_cosine_and_independence_boundaries():
    assert kinematics.independence_score(None) == 0.0
    assert kinematics.independence_score(0.97) == 0.0     # measured distractor
    assert kinematics.independence_score(0.11) == 1.0     # measured target
    assert kinematics.independence_score(-0.28) == 1.0
    assert kinematics.independence_score(0.5) == pytest.approx(0.5, abs=0.01)
    assert kinematics.flock_cosine((0.1, 0.0), (4.0, 0.0)) is None  # too slow to have a direction


def test_path_score_decays_with_distance():
    assert kinematics.path_score((0, 0), (0, 0), 50.0) == pytest.approx(1.0)
    assert kinematics.path_score((0, 0), (150.0, 0.0), 50.0) == pytest.approx(0.011, abs=0.002)


def test_was_near_at_birth_or_proximity():
    old_far = mk(5, cx=700.0, cy=100.0, created=10)
    fly(old_far, 1.0, 0.0, frames=30, start=10)
    assert not kinematics.was_near_at(old_far, 25, (400, 300), radius=141.0)

    old_near = mk(6, cx=450.0, cy=320.0, created=10)  # was in a blob with the target at f25
    fly(old_near, 1.0, 0.0, frames=30, start=10)
    assert kinematics.was_near_at(old_near, 25, (400, 300), radius=141.0)

    born_after = mk(7, cx=700.0, cy=100.0, created=40)  # re-label: did not exist to be near
    fly(born_after, 0.0, 0.0, frames=5, start=40)
    assert kinematics.was_near_at(born_after, 25, (400, 300), radius=141.0)


# --- the gates they feed ----------------------------------------------

def _t(tid, cx, cy, vel, omega=0.03, converged=True):
    t = mk(tid, cx=cx, cy=cy)
    fly(t, *vel, frames=20, start=25)  # measured through frames 25..44
    t.omega, t.omega_valid = omega, converged
    return t


def test_recovery_rejects_a_flock_mover_even_with_perfect_omega():
    """The flock gate outranks every other score: cos ~ +0.97 is a distractor by definition."""
    flock_vel = (4.0, 0.0)
    imposter = _t(11, 420.0, 300.0, flock_vel, omega=0.03)
    assert try_recover(
        [imposter], target_id=1, anchor_bbox=(353.0, 253.0, 447.0, 347.0),
        elapsed_frames=1, lost_velocity=(3.0, 0.0), target_area=8836,
        target_omega=0.03, group_vel=(4.0, 0.0), last_measured_frame=39,
        loss_center=(400.0, 300.0),
    ) is None


def test_recovery_accepts_the_independent_mover_near_the_predicted_path():
    wanderer = _t(12, 391.0, 304.0, (-2.0, 1.0), omega=0.03)  # moves against the flock
    best = try_recover(
        [wanderer], target_id=1, anchor_bbox=(353.0, 253.0, 447.0, 347.0),
        elapsed_frames=3, lost_velocity=(-2.0, 1.0), target_area=8836,
        target_omega=0.03, group_vel=(4.0, 0.0), last_measured_frame=37,
        loss_center=(400.0, 300.0),
    )
    assert best is not None and best.track_id == 12


def test_recovery_rejects_an_old_far_track_drifting_into_the_anchor():
    """The f232 failure mode: a long-lived track that was far away when the target was last
    measured drifts into reach; the birth gate must reject it even with matching omega."""
    drifter = _t(13, 430.0, 310.0, (-4.0, 0.0), omega=0.03)
    drifter.created_frame = 5
    # was at (600,300) around the target's last measurement (frame 10), 200px away
    drifter.motion_history = [(f, 600.0 - 4.0 * (f - 10), 300.0) for f in range(5, 41)]
    assert try_recover(
        [drifter], target_id=1, anchor_bbox=(353.0, 253.0, 447.0, 347.0),
        elapsed_frames=30, lost_velocity=(-2.0, 1.0), target_area=8836,
        target_omega=0.03, group_vel=None, last_measured_frame=10,
        loss_center=(400.0, 300.0),
    ) is None


def test_near_gate_survives_history_trimming_via_snapshot():
    """score1.00_medium.mp4: the carrier is an old track; ~17 frames after the loss its
    motion_history no longer contains the loss frame, so the history lookup starves. The
    loss-frame snapshot must keep the gate alive for the whole recovery window."""
    carrier = _t(14, 470.0, 300.0, (-3.0, -1.0), omega=0.03)
    carrier.created_frame = 528
    carrier.motion_history = [(f, 520.0 - 1.0 * (f - 580), 300.0) for f in range(575, 601)]
    # history starts at f575: the loss frame (566) is long gone from it
    snapshot = {14: (520.0, 300.0)}  # where it stood at the loss frame
    best = try_recover(
        [carrier], target_id=5, anchor_bbox=(453.0, 253.0, 547.0, 347.0),
        elapsed_frames=34, lost_velocity=(-1.0, 0.5), target_area=8836,
        target_omega=0.03, group_vel=None, last_measured_frame=566,
        loss_center=(500.0, 300.0), near_positions=snapshot,
    )
    assert best is not None and best.track_id == 14

    # without the snapshot the trimmed history cannot answer and the gate starves
    starved = try_recover(
        [carrier], target_id=5, anchor_bbox=(453.0, 253.0, 547.0, 347.0),
        elapsed_frames=34, lost_velocity=(-1.0, 0.5), target_area=8836,
        target_omega=0.03, group_vel=None, last_measured_frame=566,
        loss_center=(500.0, 300.0),
    )
    assert starved is None


def test_handoff_grace_blocks_a_one_frame_miss():
    """A 1-3 frame miss is OC-SORT re-associating; the anchor must not walk to a neighbour."""
    import numpy as np
    from src.liedetector.vision.pipeline import Pipeline

    class Stub:
        def classes_for_shape(self, s):
            return [3]

    pipeline = Pipeline(Stub(), fps=30.0)
    sm = pipeline.sm
    sm.state = _tracking_state()
    sm.target_id = 1
    sm.target_bbox = (100.0, 100.0, 140.0, 140.0)
    sm.allowed_classes = [3]

    missing = mk(1, cx=120.0, cy=120.0, created=0)
    fly(missing, 1.0, 0.0, frames=30)
    missing.lost_frames = 1  # missed exactly one frame
    neighbour = mk(9, cx=138.0, cy=120.0, created=0)
    fly(neighbour, 4.0, 0.0, frames=30)  # flock mover
    pipeline._reanchor_target(np.zeros((300, 300, 3), np.uint8), [missing, neighbour], 31)
    assert sm.target_id == 1  # grace held: no adoption during the re-association window


# --- silent swap sanity gates (score1.00_high.mp4 f659 failure) --------

def _swap_stage(pipeline, frame_idx=659):
    """Pipeline parked mid-swap-suspicion: locked track flock-aligned past the threshold."""
    from src.liedetector.vision.state_machine import GameState

    sm = pipeline.sm
    sm.state = GameState.TRACKING
    sm.target_id = 58
    sm.target_bbox = (380.0, 350.0, 510.0, 480.0)  # ~130px star, area ~16900
    sm.allowed_classes = [3]
    sm.target_last_measured_frame = frame_idx
    pipeline._swap_suspect = config.SWAP_SUSPECT_FRAMES
    pipeline._healthy_vel = (0.0, -2.4)
    pipeline._healthy_pos = (445.0, 415.0)
    pipeline._healthy_frame = frame_idx - 20
    return sm


def _swap_tracks():
    """Locked flock-flying target + flock + one independent mover candidate slot."""
    target = mk(58, cx=445.0, cy=415.0, created=400)
    fly(target, 3.0, 0.0, frames=20, start=640)
    target.is_confirmed = True

    flock = []
    for i, (cx, cy) in enumerate([(100.0, 100.0), (600.0, 120.0), (150.0, 400.0), (650.0, 420.0)]):
        t = mk(90 + i, cx=cx, cy=cy, created=0)
        fly(t, 3.0, 0.0, frames=20, start=640)
        t.is_confirmed = True
        flock.append(t)
    return target, flock


def test_silent_swap_rejects_a_micro_detection_fragment():
    """The f659 failure: a 173px fragment jitters 'independently' near the predicted path
    while the real target flies with the flock; it must not steal the anchor."""
    import numpy as np
    from src.liedetector.vision.pipeline import Pipeline

    class Stub:
        def classes_for_shape(self, s):
            return [3]

    pipeline = Pipeline(Stub(), fps=30.0)
    sm = _swap_stage(pipeline)
    target, flock = _swap_tracks()
    fragment = mk(95, cx=447.0, cy=409.0, created=654)
    fly(fragment, -0.5, 2.0, frames=6, start=654)  # young, jittery, tiny
    fragment.is_confirmed = False
    fragment.bbox_history[-1] = (441.0, 403.0, 454.0, 416.0)  # ~13x13 px

    pipeline._check_silent_swap([target] + flock + [fragment], target, 659)
    assert sm.target_id == 58  # no swap: the only "independent mover" is noise


def test_silent_swap_takes_a_confirmed_same_scale_independent_mover():
    import numpy as np
    from src.liedetector.vision.pipeline import Pipeline

    class Stub:
        def classes_for_shape(self, s):
            return [3]

    pipeline = Pipeline(Stub(), fps=30.0)
    sm = _swap_stage(pipeline)
    target, flock = _swap_tracks()
    mover = mk(93, cx=460.0, cy=400.0, created=620)
    fly(mover, -3.0, 1.0, frames=20, start=640)  # independent, real-star scale
    mover.is_confirmed = True

    pipeline._check_silent_swap([target] + flock + [mover], target, 659)
    assert sm.target_id == 93


def test_silent_swap_rejects_an_old_far_track():
    import numpy as np
    from src.liedetector.vision.pipeline import Pipeline

    class Stub:
        def classes_for_shape(self, s):
            return [3]

    pipeline = Pipeline(Stub(), fps=30.0)
    sm = _swap_stage(pipeline)
    target, flock = _swap_tracks()
    drifter = mk(97, cx=462.0, cy=398.0, created=500)
    fly(drifter, -3.0, 1.0, frames=20, start=640)
    drifter.is_confirmed = True
    drifter.motion_history = [(f, 300.0, 100.0) for f in range(500, 660)]  # was far away

    pipeline._check_silent_swap([target] + flock + [drifter], target, 659)
    assert sm.target_id == 58


def _tracking_state():
    from src.liedetector.vision.state_machine import GameState
    return GameState.TRACKING
