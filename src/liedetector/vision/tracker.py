"""OC-SORT wrapper: ultralytics built-in tracker + outer track bookkeeping."""

from typing import Dict, List, Tuple

import numpy as np

from src.liedetector.vision.detector import Detection, Detector
from src.liedetector.vision import kinematics
from src.liedetector.vision.tracks import Track, TrackState, mark_missing, prune_tracks, update_track
from src.liedetector.vision.logger import get_logger

log = get_logger("tracker")


class Tracker:
    """
    The Kalman/association state lives inside ultralytics' OC-SORT; this layer keeps the
    per-track history the state machine, rotation analyzer and renderer need.
    """

    def __init__(self, detector: Detector):
        self.detector = detector
        self.tracks: Dict[int, Track] = {}
        self.last_detections: List[Detection] = []
        self._last_frame_idx = -1

    def update(self, frame: np.ndarray, frame_idx: int, allowed_classes: List[int] = None) -> List[Track]:
        raw = self.detector.track(frame)
        # ultralytics only returns detections attached to a track; reused as this frame's detections
        self.last_detections = [d for d in raw if d.track_id is not None]
        dets = self.last_detections
        if allowed_classes is not None:
            dets = [d for d in dets if d.class_id in allowed_classes]

        seen = set()
        for d in dets:
            tid = int(d.track_id)
            seen.add(tid)
            track = self.tracks.get(tid)
            if track is None:
                track = Track(
                    track_id=tid,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    created_frame=frame_idx,
                )
                self.tracks[tid] = track
            update_track(track, d, frame_idx)
            kinematics.record_measurement(track, frame_idx)

        for tid, track in self.tracks.items():
            if tid not in seen and track.state != TrackState.LOST:
                mark_missing(track, frame_idx)

        removed = prune_tracks(self.tracks, _frame_size(frame), frame_idx)
        if removed:
            log.debug("pruned tracks: %s", removed)
        self._last_frame_idx = frame_idx
        return list(self.tracks.values())

    def get_target_track(self, target_id: int) -> Track | None:
        return self.tracks.get(target_id)

    def new_tracks_since(self, frame_idx: int) -> List[Track]:
        return [t for t in self.tracks.values() if t.created_frame > frame_idx]


def _frame_size(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    return w, h
