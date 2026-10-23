"""Playback schedule for original-speed mode: what the drop counter is measured against."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.liedetector.vision.frame_rate_controller import SourceClock


def clock_at(fps, frame_idx=0, age_sec=0.0):
    c = SourceClock(fps, frame_idx)
    c.start -= age_sec  # pretend the run already went this long, without sleeping for it
    return c


def test_due_frame_advances_one_frame_per_interval():
    c = clock_at(50.0, age_sec=0.20)
    assert c.due_frame() == 10


def test_ahead_of_the_schedule_waits_instead_of_consuming():
    c = clock_at(30.0, frame_idx=0, age_sec=0.0)
    assert c.ahead(0) is True
    assert c.behind_by(0) == 0


def test_behind_the_schedule_reports_how_many_frames_to_drop():
    c = clock_at(30.0, frame_idx=0, age_sec=1.0)
    # 30 frames were due, frame 0 is next, so 29 of them can never be shown again
    assert c.behind_by(0) == 29
    assert c.ahead(0) is False


def test_paused_time_is_not_billed_to_the_video():
    c = SourceClock(30.0, frame_idx=0)
    c.start -= 1.0
    c.halt()
    time.sleep(0.2)
    c.resume()
    assert c.due_frame() == 30


def test_a_source_without_a_rate_has_no_schedule_to_miss():
    c = clock_at(0.0, frame_idx=5)
    assert c.ahead(5) is False
    assert c.behind_by(5) == 0
