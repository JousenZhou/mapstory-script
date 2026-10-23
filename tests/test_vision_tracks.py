"""Track lifecycle: leaving the frame deletes, occlusion extrapolates, max_age deletes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.tracks import Track, TrackState, mark_missing, prune_tracks, update_track

FRAME = (1000, 800)


def det(cx, cy, size=40, tid=1, cls=0, name="square"):
    return Detection(
        bbox=(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
        confidence=0.9,
        class_id=cls,
        class_name=name,
        centroid=(cx, cy),
        track_id=tid,
    )


def _new(tid=1, cx=500.0, cy=400.0):
    t = Track(track_id=tid, class_id=0, class_name="square", created_frame=0)
    update_track(t, det(cx, cy, tid=tid), 1)
    return t


def test_min_hits_confirms_track():
    t = _new()
    for i in range(2, 2 + config.OC_SORT_MIN_HITS):
        update_track(t, det(500 + i, 400, tid=t.track_id), i)
    assert t.is_confirmed and t.state == TrackState.CONFIRMED


def test_centroid_off_frame_deletes_immediately():
    tracks = {1: _new(cx=500, cy=400)}
    removed = prune_tracks(tracks, FRAME, 5)
    assert removed == []

    # walk it out of the frame in one step
    update_track(tracks[1], det(1200, 400, tid=1), 6)
    removed = prune_tracks(tracks, FRAME, 6)
    assert removed == [1]
    assert 1 not in tracks


def test_offscreen_deletes_even_while_unmatched():
    """Extrapolation must not rescue a track that drifted outside the frame."""
    t = _new(cx=990, cy=400)
    t.velocity = (60.0, 0.0)
    tracks = {t.track_id: t}
    mark_missing(tracks[t.track_id], 2)  # appends a predicted centroid at ~1050
    removed = prune_tracks(tracks, FRAME, 2)
    assert removed == [t.track_id]


def test_occlusion_keeps_track_and_extrapolates():
    t = _new(cx=500, cy=400)
    t.velocity = (10.0, 0.0)
    tracks = {t.track_id: t}
    for f in range(2, 6):
        mark_missing(tracks[t.track_id], f)
    assert t.track_id in tracks
    assert t.state == TrackState.OCCLUDED
    assert t.lost_frames == 4
    assert tracks[t.track_id].centroid[0] == pytest.approx(540.0)


def test_max_age_deletes():
    t = _new(cx=500, cy=400)
    tracks = {t.track_id: t}
    for f in range(2, 2 + config.OC_SORT_MAX_AGE + 2):
        mark_missing(tracks[t.track_id], f)
    prune_tracks(tracks, FRAME, f)
    assert t.track_id not in tracks


def test_reappearance_resets_lost_frames():
    t = _new(cx=500, cy=400)
    tracks = {t.track_id: t}
    mark_missing(tracks[t.track_id], 2)
    mark_missing(tracks[t.track_id], 3)
    assert tracks[t.track_id].lost_frames == 2
    update_track(tracks[t.track_id], det(520, 400, tid=1), 4)
    assert tracks[t.track_id].lost_frames == 0
    assert tracks[t.track_id].state == TrackState.CANDIDATE or tracks[t.track_id].is_confirmed
