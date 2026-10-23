"""Paces processing to the video's original frame rate (doc 9.3)."""

import time


class FrameRateController:
    def __init__(self, target_fps: float):
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self.last_time = time.perf_counter()

    def wait(self) -> None:
        elapsed = time.perf_counter() - self.last_time
        sleep_time = self.frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.perf_counter()

    def reset(self) -> None:
        self.last_time = time.perf_counter()


class SourceClock:
    """
    Wall-clock schedule for original-speed playback: which frame index should be consumed by now.

    FrameRateController alone cannot express real speed, because it only ever sleeps: a pipeline
    slower than the source simply finishes late, and nothing says how much of the video it fell
    behind. Playback then cannot tell a dropped frame from a slow run, which is the one thing
    real-speed mode exists to measure.
    """

    def __init__(self, fps: float, frame_idx: int = 0):
        self.frame_interval = 1.0 / fps if fps > 0 else 0.0
        self.base_frame = frame_idx
        self.start = time.perf_counter()
        self._halted_at = None

    def due_frame(self) -> int:
        """Frames the schedule wants consumed already; unbounded when there is no rate to hold."""
        if self.frame_interval <= 0:
            return 1 << 62
        return self.base_frame + int((time.perf_counter() - self.start) / self.frame_interval)

    def ahead(self, next_frame_idx: int) -> bool:
        return next_frame_idx >= self.due_frame()

    def behind_by(self, next_frame_idx: int) -> int:
        if self.frame_interval <= 0:
            return 0
        return max(0, self.due_frame() - next_frame_idx - 1)

    def halt(self) -> None:
        if self._halted_at is None:
            self._halted_at = time.perf_counter()

    def resume(self) -> None:
        # a pause must not bill its own waiting time to the video, or resuming drops a burst
        if self._halted_at is not None:
            self.start += time.perf_counter() - self._halted_at
            self._halted_at = None

    def anchor(self, frame_idx: int) -> None:
        """Re-tie the schedule to where playback actually stands."""
        self.base_frame = frame_idx
        self.start = time.perf_counter()
        self._halted_at = None
