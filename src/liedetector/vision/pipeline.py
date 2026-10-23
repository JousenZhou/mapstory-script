"""Per-frame pipeline shared by the offline runner and the UI thread (doc 10.1)."""

from dataclasses import dataclass
from typing import List, Optional

import math

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision import appearance, kinematics
from src.liedetector.vision.detector import Detection, Detector
from src.liedetector.vision.recovery import near_at_loss, try_recover
from src.liedetector.vision.rotation_analyzer import estimate_omega, hold_angle, update_angle
from src.liedetector.vision.state_machine import GameState, StateMachine, confirm_initial_target, white_fill_ratio
from src.liedetector.vision.tracks import Track
from src.liedetector.vision.tracker import Tracker
from src.liedetector.vision.logger import get_logger

log = get_logger("pipeline")


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks: List[Track]
    state: str
    target_id: Optional[int]
    omega: float
    confidence: float
    frame_idx: int
    detect_count: int
    error: Optional[str] = None
    # presentation-only: 3-frame component-wise median of the locked track's box. A star
    # grazing the target drags the detection box for 1-2 frames (measured: a 99.8% scene
    # lost the final frame to a 22px drag); the median absorbs that without touching any
    # internal state. None until the window fills, and the window resets on id change.
    follow_bbox: Optional[tuple] = None


class Pipeline:
    def __init__(self, detector: Detector, fps: float):
        self.detector = detector
        self.tracker = Tracker(detector)
        self.sm = StateMachine(fps)
        self.fps = fps
        self._overlapping: set = set()
        self._swap_suspect = 0
        self._lock_with_flock = False
        self._healthy_vel = None
        self._healthy_pos = None
        self._healthy_bbox = None
        self._healthy_frame = None
        self._last_known_vel = None
        self._loss_snapshot = {}
        self._loss_origin = None
        self._frame_wh = None
        self._follow_hist = []
        self._moving_candidate_frames = 0
        self._impostor_ids = set()
        self._loss_context_poisoned = False
        self._theft_watch = None
        self._target_hu = None
        self._healthy_near = {}
        self._ccos_hist = {}
        # theft-watch context, populated by _begin_loss; None until the first loss so
        # _arm_theft_watch from a neural M1 correction (no prior loss) degrades to no-op
        self._loss_vel = None
        self._loss_bbox = None
        self._loss_near = {}
        self._loss_frame = None
        # neural evidence channel state: False = not tried yet, None = init failed,
        # NeuralEvidence = active. See NEURAL_FUSION_PLAN.md; consumed only under
        # distress (_neural_m1) or target loss (_neural_m2), never on healthy frames.
        self._neural = False
        self._neural_center = None   # last video-px regression center
        self._neural_frame = None
        self._neural_div = 0         # M1: consecutive distress frames diverging from neural
        self._neural_pick = None     # M1/M2: (track_id, consecutive frames chosen)
        # 会话复位适配：warm Detector 跨会话复用时，构造新 Pipeline 即清掉上一会话遗留的
        # OC-SORT 轨迹簿；d/lie 的桩检测器没有 reset_tracking，自动跳过（保持原行为）。
        reset = getattr(self.detector, "reset_tracking", None)
        if callable(reset):
            reset()


    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        sm = self.sm
        if sm.state == GameState.IDLE:
            return self._result(frame, [], frame_idx, [])

        # one inference per frame: the tracker's detections double as this frame's detection stage
        tracks = self.tracker.update(frame, frame_idx)
        detections = self.tracker.last_detections
        self._frame_wh = (frame.shape[1], frame.shape[0])

        # two contours inside one ROI poison the radial signature; those angles are held
        # (repeated, excluded from the fit) instead of measured
        live = [t for t in tracks if t.lost_frames == 0]
        self._overlapping = _deep_overlap_ids(live)
        for t in tracks:
            if t.lost_frames == 0:
                if t.track_id in self._overlapping:
                    hold_angle(t)
                else:
                    update_angle(frame, t)
            fitted = estimate_omega(t)
            t.omega_valid = fitted is not None
            t.omega = fitted if fitted is not None else 0.0

        # the omega certificate and the last measured kinematics are maintained in every
        # state: the countdown spin is the only rotation evidence that exists before the
        # fade, and its former empty window (cert written only in TRACKING) cost whole
        # videos when the anchor died mid-TRANSITION. While a silent swap is suspected the
        # locked track is presumed to be a distractor: its omega and velocity would pollute
        # the certificate and the loss context (measured: a polluted certificate was the
        # only gate that rejected the true carrier at the sole correctable moment)
        target = self._target(tracks)
        if target is not None and target.lost_frames == 0:
            v = kinematics.smoothed_velocity(target)
            if v is not None and not self._lock_with_flock:
                self._last_known_vel = v
            if target.omega_valid and target.track_id not in self._overlapping \
                    and not self._lock_with_flock:
                sm.record_target_omega(target.omega)

        self._reanchor_target(frame, tracks, frame_idx)

        if sm.state == GameState.COUNTDOWN:
            self._on_countdown(detections, frame, frame_idx, tracks)
        elif sm.state == GameState.TRANSITION:
            self._on_transition(frame, tracks, frame_idx)
        elif sm.state in (GameState.TRACKING, GameState.TARGET_LOST):
            self._on_tracking_or_lost(frame, tracks, frame_idx)

        target = self._target(tracks)
        if config.NEURAL_EVIDENCE_ENABLED:
            self._neural_tick(frame, tracks, frame_idx)
        impostor = target is not None and target.track_id in self._impostor_ids
        follow_bbox = None if impostor else self._smoothed_follow_bbox(target, frame_idx)
        return FrameResult(
            frame=frame,
            tracks=tracks,
            state=sm.state.value,
            target_id=None if impostor else sm.target_id,
            omega=target.omega if target else 0.0,
            confidence=sm.target_confidence,
            frame_idx=frame_idx,
            detect_count=len(detections),
            follow_bbox=follow_bbox,
        )

    def _smoothed_follow_bbox(self, target: Optional[Track], frame_idx: int):
        """Component-wise median of the last FOLLOW_SMOOTH_FRAMES measured boxes of the
        locked track. Passthrough until the window fills; resets on id change so the
        median never mixes two objects."""
        if target is None or target.bbox is None or target.lost_frames != 0:
            self._follow_hist = []
            return None
        if not self._follow_hist or self._follow_hist[-1][0] != target.track_id:
            self._follow_hist = []
        self._follow_hist.append((target.track_id, target.bbox))
        if len(self._follow_hist) > config.FOLLOW_SMOOTH_FRAMES:
            del self._follow_hist[0]
        if len(self._follow_hist) < config.FOLLOW_SMOOTH_FRAMES:
            return target.bbox
        arr = np.asarray([b for _, b in self._follow_hist], dtype=np.float64)
        return tuple(float(v) for v in np.median(arr, axis=0))

    # --- state handlers -------------------------------------------------

    def _on_countdown(self, detections: List[Detection], frame: np.ndarray, frame_idx: int,
                      tracks: List[Track] = None) -> None:
        """
        Until the fade the white fill is the ground truth for which star is the target, so it
        outranks the tracker's id: re-confirm it every frame and follow whichever track carries it.

        Two measured confounders are filtered before the whitest-wins confirmation:
        - the countdown digit: white (fill up to 0.52, sometimes whiter than a compound
          target), always top-centre, and its track barely moves while alive (~30 frames
          per digit) - a real target covers >=72px in 30 frames at its slowest;
        - double detections of a compound target: the detector emits BOTH the compound box
          and a brighter primitive inside it; the primitive is a fragment of the same white
          region, so a candidate whose centre lies inside another candidate's box is dropped.
        A lock found to sit on a digit-like track is a digit lock: white fill wins, and a
        sustained moving candidate re-locks it.
        """
        sm = self.sm
        tracks = tracks or []

        scored = [(d, white_fill_ratio(frame, d.bbox)) for d in detections]
        white = [d for d, r in scored if r >= config.WHITE_FILL_RATIO]
        white = _drop_fragments(white)
        white = [d for d in white if not self._digit_like(d, tracks, frame_idx)]

        det = confirm_initial_target(white, frame)
        if det is not None and det.track_id is not None:
            tid = int(det.track_id)
            if sm.target_id is None:
                sm.set_target(det, tid, self.detector.classes_for_shape(det.shape))
            elif tid != sm.target_id and sm.target_bbox is not None:
                value = _iou(sm.target_bbox, det.bbox)
                if value >= config.TARGET_HANDOFF_IOU:
                    sm.hand_off_target(tid, det.bbox, value)
                else:
                    # the lock sits on a digit-like track while a real white candidate
                    # persists: the lock is wrong, white fill wins
                    locked = next((t for t in tracks if t.track_id == sm.target_id), None)
                    if self._digit_like_track(locked, frame_idx):
                        self._moving_candidate_frames += 1
                        if self._moving_candidate_frames >= config.COUNTDOWN_DIGIT_ESCAPE_FRAMES:
                            log.warning("countdown lock was digit-like, re-locking to the "
                                        "white candidate id %s at frame %s", tid, frame_idx)
                            sm.hand_off_target(tid, det.bbox, value)
                            self._moving_candidate_frames = 0
                    else:
                        self._moving_candidate_frames = 0
        else:
            self._moving_candidate_frames = 0
        # the countdown must expire even while no target is confirmed, or the machine hangs
        if frame_idx >= self.sm.countdown_end_frame:
            self.sm.transition(GameState.TRANSITION, frame_idx)

    def _on_transition(self, frame: np.ndarray, tracks: List[Track], frame_idx: int) -> None:
        """Fade is done once the locked target's box stops being white-filled."""
        target = self._target(tracks)
        if target is not None and target.lost_frames == 0 and target.bbox:
            ratio = white_fill_ratio(frame, target.bbox)
            self.sm.fade_white_ratio = ratio
            self.sm.fade_low_frames = self.sm.fade_low_frames + 1 if ratio < config.FADE_DONE_RATIO else 0
            if self.sm.fade_low_frames >= config.FADE_CONFIRM_FRAMES:
                log.info("fade complete at frame %s (white ratio %.2f)", frame_idx, ratio)
                # identity is certain here and the target just settled into its outline
                # form: capture the appearance certificate for recovery/correction
                self._target_hu = appearance.hu_signature(frame, target.bbox)
                self.sm.transition(GameState.TRACKING, frame_idx)
                return
        # the anchor died mid-fade: coasting a stale box to the timeout only postpones a
        # blind guess. Hand the identity problem to the recovery machinery while the loss
        # is fresh (small reach, live snapshot, omega certificate) instead. A run that
        # never locked anything has nothing to recover and still rides to the timeout.
        if (self.sm.target_id is not None
                and (target is None or target.lost_frames > config.TARGET_GRACE_FRAMES)):
            log.warning("anchor died mid-TRANSITION at frame %s: entering recovery", frame_idx)
            self._begin_loss(tracks, target, frame_idx)
            self.sm.transition(GameState.TARGET_LOST, frame_idx)
            return
        # a box we only extrapolated measures nothing; waiting is cheaper than a false TRACKING
        if frame_idx - self.sm.state_started_frame >= config.TRANSITION_TIMEOUT_FRAMES:
            self._force_track(frame_idx, "fade never resolved")

    def _force_track(self, frame_idx: int, why: str) -> None:
        log.warning("leaving TRANSITION at frame %s: %s", frame_idx, why)
        self.sm.fade_low_frames = 0
        self.sm.transition(GameState.TRACKING, frame_idx)

    def _begin_loss(self, tracks: List[Track], target: Optional[Track], frame_idx: int) -> None:
        """Bookkeeping shared by every path into TARGET_LOST: fresh velocity/area context
        and the snapshot of where every track stood at the loss moment."""
        sm = self.sm
        self._theft_watch = None  # a new loss invalidates any pending watch
        sm.lost_frame_idx = frame_idx
        sm.lost_velocity = ((kinematics.smoothed_velocity(target)
                             if target is not None else None)
                            or (target.velocity if target else None)
                            or self._last_known_vel)
        # pre-loss context for the theft watch: where the lock stood and how it moved when
        # it was last trustworthy (captured before the poisoning below)
        self._loss_bbox = sm.target_bbox
        self._loss_vel = sm.lost_velocity
        # a lock that was already swap-suspected when it died was following a distractor:
        # its velocity poisons the constant-velocity prediction, and its box poisons the
        # IoU score term (both soft scores; every hard gate stays). Dropping them lets
        # recovery be judged on independence/area instead of on proximity to the wrong
        # box (measured: scene_703824074, ratio 0.44 -> adoption).
        self._loss_context_poisoned = self._swap_suspect >= config.SWAP_SUSPECT_FRAMES
        if self._loss_context_poisoned:
            sm.lost_velocity = None
        sm.lost_area = target.area if target else sm.target_area
        # the near-at-loss lookup cannot use motion_history, which is trimmed to the
        # smoothing window and forgets the loss frame ~17 frames later
        self._loss_snapshot = {t.track_id: t.centroid for t in tracks if t.centroid}
        self._loss_near = dict(self._loss_snapshot)
        self._loss_frame = frame_idx

    def _resumption_is_theft(self, target: Track, frame_idx: int) -> bool:
        """True when the locked track re-measures after a real coasting gap and its
        rotation contradicts the certificate: OC-SORT handed the id to a different
        object while it coasted. Only decided on real evidence: a short gap is
        re-association noise, and no certificate (circle) or an unconverged fit
        means the check has nothing to say."""
        gap = frame_idx - target.last_measured_frame if target.last_measured_frame >= 0 else 0
        if gap < config.RESUME_THEFT_MIN_GAP:
            return False
        sm = self.sm
        if sm.target_omega is None or abs(sm.target_omega) < config.TARGET_OMEGA_CERT_FLOOR:
            return False
        if sm.target_modulation < config.MIN_ROTATION_MODULATION:
            return False
        if not target.omega_valid:
            return False
        return abs(target.omega - sm.target_omega) > config.RECOVERY_OMEGA_TOLERANCE

    def _on_tracking_or_lost(self, frame: np.ndarray, tracks: List[Track], frame_idx: int) -> None:
        sm = self.sm
        target = self._target(tracks)
        if sm.state == GameState.TRACKING:
            if target is not None:
                # keep the best rotation signal seen while locked; a circle never yields one
                sm.target_modulation = max(sm.target_modulation, target.modulation)
                if config.SWAP_DETECTOR_ENABLED and target.lost_frames == 0:
                    self._check_silent_swap(tracks, target, frame_idx)
                if self._theft_watch is not None:
                    self._check_theft_watch(frame, tracks, target, frame_idx)
            if not sm.is_target_visible(tracks):
                self._begin_loss(tracks, target, frame_idx)
                sm.transition(GameState.TARGET_LOST, frame_idx)
            return

        # the anchor already follows the box, so a live track under it needs no re-scoring.
        # EXCEPT right after a coasting gap: OC-SORT can re-associate the id to a different
        # object while the track coasted (measured: 15/16 final-frame failures in the edge
        # categories had the locked id survive through the failure onset - "id theft"), and
        # accepting it back rubber-stamps the theft. The rotation certificate is the only
        # identity evidence that survives the gap: a resumed carrier that rotates at odds
        # with the certificate is an impostor - the lock stays lost and recovery searches.
        if target is not None and target.lost_frames == 0:
            if self._resumption_is_theft(target, frame_idx):
                if target.track_id not in self._impostor_ids:
                    log.warning("resumption rejected at frame %s: id %s re-measured with "
                                "omega %.4f vs certificate %.4f - treating as id theft",
                                frame_idx, target.track_id, target.omega,
                                self.sm.target_omega)
                    self._impostor_ids.add(target.track_id)
                # do not resume: fall through to recovery with the impostor excluded
            else:
                self._impostor_ids.discard(target.track_id)
                sm.lost_frame_idx = None
                self._loss_snapshot = {}
                sm.transition(GameState.TRACKING, frame_idx)
                # a single-point theft check is impossible (measured: 3/13 stolen
                # resumptions look genuine at the resumption frame), so instead of
                # trusting the resumption blindly, watch it
                self._arm_theft_watch(frame_idx)
                return

        # shape evidence: compare each candidate's outline with the certificate captured
        # at fade completion; None everywhere when no certificate exists (the score term
        # then drops out of the evidence mass entirely)
        appearance_scores = self._appearance_scores(frame, tracks)

        recovered = try_recover(
            tracks,
            target_id=sm.target_id,
            anchor_bbox=(target.bbox or sm.target_bbox) if target is not None else sm.target_bbox,
            elapsed_frames=max(sm.lost_elapsed_frames(frame_idx), 1),
            lost_velocity=sm.lost_velocity,
            target_area=sm.lost_area,
            omega_observable=sm.target_modulation >= config.MIN_ROTATION_MODULATION,
            target_omega=sm.target_omega,
            group_vel=kinematics.group_velocity(tracks),
            last_measured_frame=sm.target_last_measured_frame,
            loss_center=_bbox_center(sm.target_bbox),
            near_positions=self._loss_snapshot,
            exclude_ids=self._impostor_ids,
            drop_anchor_score=self._loss_context_poisoned,
            appearance_scores=appearance_scores,
        )
        if recovered is not None:
            sm.target_id = recovered.track_id
            sm.target_bbox = recovered.bbox
            sm.target_last_measured_frame = frame_idx
            sm.lost_frame_idx = None
            self._swap_suspect = 0
            self._loss_snapshot = {}
            self._loss_origin = None
            self._healthy_vel, self._healthy_pos, self._healthy_frame = None, None, None
            self._healthy_bbox = None
            # the certificate predates the loss: rebuild it from the recovered carrier
            sm.reset_target_omegas()
            sm.transition(GameState.TRACKING, frame_idx)
            self._arm_theft_watch(frame_idx)
        elif sm.recovery_timeout(frame_idx):
            log.warning("recovery timed out at frame %s", frame_idx)
            sm.transition(GameState.GAME_OVER, frame_idx)

    def _appearance_scores(self, frame: np.ndarray, tracks: List[Track]) -> Optional[dict]:
        """Per-candidate Hu-signature similarity to the fade-complete certificate; None
        when no certificate exists (the score term then vanishes from the evidence mass)."""
        if not config.APPEARANCE_EVIDENCE_ENABLED or self._target_hu is None:
            return None
        scores = {}
        for t in tracks:
            if t.lost_frames == 0 and t.bbox is not None:
                sim = appearance.signature_similarity(
                    self._target_hu, appearance.hu_signature(frame, t.bbox))
                if sim is not None:
                    scores[t.track_id] = sim
        return scores or None

    def _arm_theft_watch(self, frame_idx: int) -> None:
        """After a gap-bridged return to TRACKING, the resumed carrier is watched: a
        stolen id reveals itself by jumping into the flock's direction (the pre-loss
        lock was the independent target, the impostor is a flock member)."""
        if not config.THEFT_WATCH_ENABLED or self._loss_vel is None or self._loss_bbox is None:
            self._theft_watch = None
            return
        self._theft_watch = {
            "until": frame_idx + config.THEFT_WATCH_FRAMES,
            "bbox": self._loss_bbox,
            "vel": self._loss_vel,
            "near": self._loss_near,
            "frame": self._loss_frame,
            "aligned": 0,
        }

    def _check_theft_watch(self, frame: np.ndarray, tracks: List[Track], target: Track, frame_idx: int) -> None:
        w = self._theft_watch
        if frame_idx > w["until"]:
            self._theft_watch = None
            return
        sm = self.sm
        group_vel = kinematics.group_velocity(tracks)
        v_now = kinematics.smoothed_velocity(target)
        if group_vel is None or v_now is None:
            return
        cos_now = kinematics.flock_cosine(v_now, group_vel)
        if cos_now is None:
            return
        if cos_now >= config.KIN_FLOCK_COS_HI:
            w["aligned"] = w.get("aligned", 0) + 1
        else:
            w["aligned"] = 0
        pre_cos = kinematics.flock_cosine(w["vel"], group_vel)
        jump = None if pre_cos is None else cos_now - pre_cos
        v_rev = kinematics.flock_cosine(w["vel"], v_now)
        suspicious = ((jump is not None and jump >= config.THEFT_WATCH_JUMP)
                      or (v_rev is not None and v_rev < 0.0))
        if not suspicious or w["aligned"] < config.THEFT_WATCH_ALIGNED_FRAMES:
            return
        # re-capture anchored at the pre-loss snapshot: every hard gate of the normal
        # recovery applies, the impostor id is excluded; finding nothing costs nothing
        recovered = try_recover(
            tracks,
            target_id=sm.target_id,
            anchor_bbox=w["bbox"],
            elapsed_frames=max(frame_idx - w["frame"], 1),
            lost_velocity=w["vel"],
            target_area=sm.lost_area,
            omega_observable=sm.target_modulation >= config.MIN_ROTATION_MODULATION,
            target_omega=sm.target_omega,
            group_vel=group_vel,
            last_measured_frame=sm.target_last_measured_frame,
            loss_center=_bbox_center(w["bbox"]),
            near_positions=w["near"],
            exclude_ids={target.track_id} | self._impostor_ids,
            appearance_scores=self._appearance_scores(frame, tracks),
        )
        if recovered is not None and recovered.track_id != target.track_id:
            log.warning("theft watch re-captured at frame %s: id %s -> %s (resumed carrier "
                        "jumped into the flock)", frame_idx, target.track_id, recovered.track_id)
            self._impostor_ids.add(target.track_id)
            sm.hand_off_target(recovered.track_id, recovered.bbox,
                               _iou(sm.target_bbox or w["bbox"], recovered.bbox))
            sm.target_last_measured_frame = frame_idx
            sm.reset_target_omegas()
            self._swap_suspect = 0
            self._theft_watch = None

    def _check_silent_swap(self, tracks: List[Track], target: Track, frame_idx: int) -> None:
        """
        OC-SORT can hand the target's id to a distractor inside an overlap; afterwards the
        anchor follows the flock and never looks back. A locked track that keeps flying
        with the flock while an independent mover sits near the target's predicted path is
        that failure, and the anchor moves back to the independent mover.
        """
        group_vel = kinematics.group_velocity(tracks)
        cos = kinematics.flock_cosine(kinematics.smoothed_velocity(target), group_vel)
        self._lock_with_flock = cos is not None and cos >= config.KIN_FLOCK_COS_HI
        self._record_ccos(tracks, group_vel, frame_idx)
        if cos is None:
            return
        if cos <= config.KIN_FLOCK_COS_LO:
            self._swap_suspect = 0
            self._healthy_vel = kinematics.smoothed_velocity(target)
            self._healthy_pos = target.centroid
            self._healthy_bbox = target.bbox
            self._healthy_frame = frame_idx
            # position snapshot of the whole scene at this moment: motion_history trims to
            # ~17 entries and forgets this frame soon after, which would starve the
            # snapshot near gate exactly when a long correction needed it
            self._healthy_near = {t.track_id: t.centroid for t in tracks
                                  if t.centroid and t.lost_frames == 0}
            return
        self._swap_suspect += 1
        if self._swap_suspect < config.SWAP_SUSPECT_FRAMES or self._healthy_vel is None:
            return

        elapsed = max(frame_idx - self._healthy_frame, 1)
        predicted = (self._healthy_pos[0] + self._healthy_vel[0] * elapsed,
                     self._healthy_pos[1] + self._healthy_vel[1] * elapsed)
        reach = config.RECOVERY_MAX_SPEED * elapsed + kinematics.body_length(target.bbox)
        cert_valid = (self.sm.target_omega is not None
                      and abs(self.sm.target_omega) >= config.TARGET_OMEGA_CERT_FLOOR)
        near_center = _bbox_center(self.sm.target_bbox) or target.centroid
        near_radius = config.TARGET_NEAR_RADIUS_FACTOR * kinematics.body_length(target.bbox)
        near_frame = self.sm.target_last_measured_frame
        best, best_d = None, None
        for cand in tracks:
            if cand.track_id == target.track_id or cand.lost_frames > 0 or cand.centroid is None:
                continue
            ccos = kinematics.flock_cosine(kinematics.smoothed_velocity(cand), group_vel)
            if not self._independent_enough(cand, ccos, frame_idx):
                continue  # only an independent mover can be the target; this gate is what
                # keeps the corrector inert while a TRUE target legitimately flies with
                # the flock (then every nearby candidate is a flock member and is excluded)
            # the swap takes over the anchor, so its candidate must pass the same sanity
            # gates as a handoff: a spurious micro-detection also jitters "independently",
            # and without these it can steal the target from a healthy locked track
            if not cand.is_confirmed:
                continue  # a noise fragment lives a handful of frames
            if cand.class_id not in self.sm.allowed_classes:
                continue
            if target.area > 0 and min(cand.area, target.area) / max(cand.area, target.area) \
                    < config.SWAP_MIN_AREA_RATIO:
                continue  # must be the same object scale as the locked track
            if not kinematics.was_near_at(cand, near_frame, near_center, near_radius):
                continue
            if cert_valid and cand.omega_valid:
                if abs(cand.omega - self.sm.target_omega) > config.RECOVERY_OMEGA_TOLERANCE:
                    continue
            d = float(np.hypot(cand.centroid[0] - predicted[0], cand.centroid[1] - predicted[1]))
            if d > reach:
                continue
            if best_d is None or d < best_d:
                best, best_d = cand, d
        if best is not None:
            log.warning("silent swap corrected at frame %s: id %s -> %s (locked track flew with "
                        "the flock %s frames)", frame_idx, target.track_id, best.track_id,
                        self._swap_suspect)
            self.sm.hand_off_target(best.track_id, best.bbox, _iou(target.bbox, best.bbox))
            self.sm.target_last_measured_frame = frame_idx
            # the deque's tail belongs to the distractor; the certificate must rebuild from
            # the corrected carrier or it keeps rejecting the true target
            self.sm.reset_target_omegas()
            self._swap_suspect = 0
            self._healthy_vel, self._healthy_pos, self._healthy_frame = None, None, None
            self._healthy_bbox = None
            return

    def _record_ccos(self, tracks: List[Track], group_vel, frame_idx: int) -> None:
        """Per-candidate independence history. In edge scenes the target zigzags along the
        border and its flock-cosine oscillates through the full range with a ~10 frame
        period, so the single-frame ccos gate excludes the true carrier half the time."""
        if group_vel is None:
            return
        cutoff = frame_idx - config.SWAP_INDEP_HIST_WINDOW
        for t in tracks:
            if t.lost_frames != 0 or not t.is_confirmed or t.centroid is None:
                continue
            c = kinematics.flock_cosine(kinematics.smoothed_velocity(t), group_vel)
            if c is None:
                continue
            h = self._ccos_hist.setdefault(t.track_id, [])
            h.append((frame_idx, c))
            while h and h[0][0] < cutoff:
                del h[0]

    def _independent_enough(self, cand: Track, ccos, frame_idx: int) -> bool:
        if ccos is not None and ccos <= config.KIN_FLOCK_COS_LO:
            return True
        h = self._ccos_hist.get(cand.track_id, [])
        hits = sum(1 for _, c in h if c <= config.SWAP_INDEP_HIST_COS)
        return hits >= config.SWAP_INDEP_HIST_MIN

    # --- neural evidence channel (see NEURAL_FUSION_PLAN.md) -------------

    def _neural_tick(self, frame: np.ndarray, tracks: List[Track], frame_idx: int) -> None:
        """Keep the motion-regression servo warm and run the two gated consumers.

        The servo must feed continuously (stride 2, the V7 runtime's native
        source_stride): a mid-video cold start cannot acquire a mover far from the
        panel prior (measured: the regression wanders near (320,212)). After any stale
        gap it reseeds with the tracker's OWN position as prior, so the regression
        restarts from our belief instead of the panel center. M1/M2 consume the center
        only under distress / TARGET_LOST; healthy frames pay only the feed cost."""
        if self._neural is False:
            try:
                from src.liedetector.vision.neural_evidence import NeuralEvidence
                self._neural = NeuralEvidence((frame.shape[1], frame.shape[0]))
            except Exception as e:
                log.warning("neural evidence disabled: %s", e)
                self._neural = None
                return
        if self._neural is None:
            return
        sm = self.sm
        if sm.state not in (GameState.TRACKING, GameState.TARGET_LOST):
            self._neural_div = 0
            self._neural_pick = None
            return
        gap = (frame_idx - self._neural_frame) if self._neural_frame is not None else None
        if gap != 0 and (gap is None or gap >= config.NEURAL_EVIDENCE_STRIDE):
            if gap is not None and gap > config.NEURAL_EVIDENCE_STRIDE:
                target = self._target(tracks)
                belief = None
                if target is not None and target.lost_frames == 0 \
                        and target.centroid is not None:
                    belief = (float(target.centroid[0]), float(target.centroid[1]))
                elif sm.target_bbox is not None:
                    c = _bbox_center(sm.target_bbox)
                    belief = (float(c[0]), float(c[1]))
                self._neural.reset(belief)
            center = self._neural.compute(frame, frame_idx)
            self._neural_frame = frame_idx
            if center is not None:
                self._neural_center = center
        if sm.state == GameState.TRACKING:
            self._neural_m1(tracks, frame_idx)
        else:
            self._neural_m2(tracks, frame_idx)

    def _neural_valid(self, center, frame_idx: int) -> bool:
        """Reject the channel's two known failure signatures before its evidence is
        consumed: the (320,212) prior fallback and a faster-than-motion re-lock jump."""
        if np.hypot(center[0] - self._neural.prior_video[0],
                    center[1] - self._neural.prior_video[1]) < config.NEURAL_PRIOR_MIN_DIST:
            return False
        for f, c in reversed(self._neural.recent):
            if f < frame_idx:
                dt = max(1, frame_idx - f)
                if np.hypot(center[0] - c[0], center[1] - c[1]) / dt > config.NEURAL_MAX_SPEED:
                    return False
                break
        return True

    def _neural_candidate(self, tracks, center, body, exclude, area_ref,
                          near_frame, near_center, near_radius):
        """Confirmed, allowed-class track nearest the neural center; area-compatible
        with the reference object and, when a healthy snapshot exists, born near it."""
        best, best_d = None, None
        for cand in tracks:
            if (cand.lost_frames != 0 or not cand.is_confirmed
                    or cand.centroid is None or cand.track_id in exclude
                    or cand.class_id not in self.sm.allowed_classes):
                continue
            if area_ref and min(cand.area, area_ref) / max(cand.area, area_ref) \
                    < config.SWAP_MIN_AREA_RATIO:
                continue
            if near_center is not None and near_frame is not None and not kinematics.was_near_at(
                    cand, near_frame, near_center, near_radius):
                continue
            d = float(np.hypot(cand.centroid[0] - center[0], cand.centroid[1] - center[1]))
            if d <= config.NEURAL_NEAR_FACTOR * max(1.0, body) \
                    and (best_d is None or d < best_d):
                best, best_d = cand, d
        return best, best_d

    def _neural_m1(self, tracks: List[Track], frame_idx: int) -> None:
        """Distress-gated divergence arbitration. While the lock is swap-suspected
        (or watched after a gap-bridged resume) and the motion-only channel persistently
        disagrees with the locked track, hand the lock to the confirmed track sitting on
        the neural center. No independence / omega-certificate requirement - the edge
        failures happen exactly where those gates have no evidence - but area, birth
        (when a healthy snapshot exists) and a 2-frame confirmation stay."""
        target = self._target(tracks)
        if target is None or target.lost_frames != 0 or target.centroid is None:
            self._neural_div = 0
            return
        distress = self._swap_suspect >= config.SWAP_SUSPECT_FRAMES
        if not distress:
            # lock_with_flock and the post-recovery theft watch are NOT enough: measured
            # on the v19 full run they fire M1 inside healthy free-motion locks (24
            # regressions, concentrated in 四组合体/三角形 自由位移). Sustained flock
            # alignment (the same suspicion state the silent-swap corrector acts on)
            # is the trigger; see NEURAL_FUSION_PLAN.md.
            self._neural_div = 0
            self._neural_pick = None
            return
        center = self._neural_center
        if center is None or not self._neural_valid(center, frame_idx):
            self._neural_div = 0
            return
        body = kinematics.body_length(target.bbox)
        d = float(np.hypot(center[0] - target.centroid[0], center[1] - target.centroid[1]))
        if d <= config.NEURAL_DIVERGENCE_TAU * max(1.0, body):
            self._neural_div = 0
            self._neural_pick = None
            return
        self._neural_div += 1
        if self._neural_div < config.NEURAL_DIVERGENCE_FRAMES:
            return
        best, _ = self._neural_candidate(
            tracks, center, body,
            exclude={target.track_id} | self._impostor_ids,
            area_ref=target.area,
            near_frame=self._healthy_frame, near_center=self._healthy_pos,
            near_radius=config.TARGET_NEAR_RADIUS_FACTOR * body)
        if best is None:
            return
        if self._neural_pick and self._neural_pick[0] == best.track_id:
            self._neural_pick = (best.track_id, self._neural_pick[1] + 1)
        else:
            self._neural_pick = (best.track_id, 1)
        if self._neural_pick[1] < config.NEURAL_CONFIRM_FRAMES:
            return
        log.warning("neural M1 correction at frame %s: id %s -> %s (divergence %.0f px, "
                    "distress %s frames)", frame_idx, target.track_id, best.track_id,
                    d, self._swap_suspect)
        self.sm.hand_off_target(best.track_id, best.bbox, _iou(target.bbox, best.bbox))
        self.sm.target_last_measured_frame = frame_idx
        self.sm.reset_target_omegas()
        self._swap_suspect = 0
        self._neural_div = 0
        self._neural_pick = None
        self._healthy_vel = self._healthy_pos = self._healthy_frame = None
        self._healthy_bbox = None
        self._arm_theft_watch(frame_idx)

    def _neural_m2(self, tracks: List[Track], frame_idx: int) -> None:
        """Loss re-capture. When normal recovery finds nothing, a confirmed track that
        sits on a *valid* neural center for NEURAL_CONFIRM_FRAMES consecutive frames is
        adopted as the original identity: the neural+area double-gate adoption the plan
        commits to. The near-at-loss gate is dropped by design - the lost-type failures
        are far from the loss area - but no other live track may be hijacked and the
        normal post-adoption bookkeeping (certificate rebuild, theft watch) applies."""
        sm = self.sm
        if sm.lost_elapsed_frames(frame_idx) < config.NEURAL_M2_MIN_LOST_FRAMES:
            return  # the real recovery machinery has priority inside a young loss
        center = self._neural_center
        if center is None or not self._neural_valid(center, frame_idx):
            self._neural_pick = None
            return
        body_ref = 2.0 * (sm.lost_area ** 0.5) if sm.lost_area else 60.0
        best, _ = self._neural_candidate(
            tracks, center, body_ref,
            exclude=self._impostor_ids,
            area_ref=sm.lost_area,
            near_frame=None, near_center=None, near_radius=None)
        if best is None:
            self._neural_pick = None
            return
        if self._neural_pick and self._neural_pick[0] == best.track_id:
            self._neural_pick = (best.track_id, self._neural_pick[1] + 1)
        else:
            self._neural_pick = (best.track_id, 1)
        if self._neural_pick[1] < config.NEURAL_CONFIRM_FRAMES:
            return
        log.warning("neural M2 re-capture at frame %s: id %s adopted after %s lost frames",
                    frame_idx, best.track_id, sm.lost_elapsed_frames(frame_idx))
        sm.hand_off_target(best.track_id, best.bbox, 0.0)
        sm.target_last_measured_frame = frame_idx
        sm.reset_target_omegas()
        sm.lost_frame_idx = None
        self._swap_suspect = 0
        self._loss_snapshot = {}
        self._loss_origin = None
        self._healthy_vel, self._healthy_pos, self._healthy_frame = None, None, None
        self._healthy_bbox = None
        self._neural_pick = None
        sm.transition(GameState.TRACKING, frame_idx)
        self._arm_theft_watch(frame_idx)

    def _digit_like_track(self, track: Optional[Track], frame_idx: int) -> bool:
        """The countdown digit sits top-centre of the frame. Measured across all 280
        validation scenes: the GT target NEVER enters this zone during the countdown
        (first entry anywhere is frame 162, countdown ends at 151) - the simulator keeps
        the spawn and the digit apart - so within the countdown the zone is a clean
        digit discriminator."""
        if track is None or track.centroid is None or self._frame_wh is None:
            return False
        w, h = self._frame_wh
        cx, cy = track.centroid
        return (config.COUNTDOWN_DIGIT_ZONE_X[0] * w <= cx <= config.COUNTDOWN_DIGIT_ZONE_X[1] * w
                and cy <= config.COUNTDOWN_DIGIT_ZONE_Y * h)

    def _digit_like(self, det: Detection, tracks: List[Track], frame_idx: int) -> bool:
        t = next((x for x in tracks if x.track_id == det.track_id), None)
        return self._digit_like_track(t, frame_idx)

    # --- helpers -------------------------------------------------------

    def _reanchor_target(self, frame: np.ndarray, tracks: List[Track], frame_idx: int) -> None:
        """
        Keep the target identity continuous even when OC-SORT gives it a new id: if the locked
        id is absent or unmatched this frame, adopt the track that now covers its box - but
        only on real evidence. Four gates measured on the validation set:
        grace (a 1-3 frame miss is re-association, not a loss), birth-or-merged-near (an old
        track that was far away at the last measurement is a different object), flock (a
        candidate moving with the group median is a distractor), and the rotation
        certificate with a floor: |cert| below it is no evidence, so only a box that is
        essentially the anchor itself may be adopted.
        """
        sm = self.sm
        if sm.target_id is None:
            return
        current = self._target(tracks)
        if current is not None and current.lost_frames == 0:
            sm.target_bbox = current.bbox
            sm.target_last_measured_frame = frame_idx
            return
        # grace: the extrapolated anchor walks toward whatever is nearby; handing it off
        # during OC-SORT's own re-association window is how a wrong star gets adopted
        if current is not None and current.lost_frames <= config.TARGET_GRACE_FRAMES:
            return

        anchor = current.bbox if current is not None and current.bbox else sm.target_bbox
        if anchor is None:
            return
        anchor_ratio = white_fill_ratio(frame, anchor)
        group_vel = kinematics.group_velocity(tracks)
        cert_valid = (sm.target_omega is not None
                      and abs(sm.target_omega) >= config.TARGET_OMEGA_CERT_FLOOR
                      and sm.target_modulation >= config.MIN_ROTATION_MODULATION)
        last_measured_center = _bbox_center(sm.target_bbox) or _bbox_center(anchor)
        near_radius = config.TARGET_NEAR_RADIUS_FACTOR * kinematics.body_length(anchor)

        best, best_iou = None, 0.0
        for cand in tracks:
            if cand.track_id == sm.target_id or cand.lost_frames > 0:
                continue
            if cand.class_id not in sm.allowed_classes:
                continue
            value = _iou(anchor, cand.bbox)
            if value <= best_iou or _area_drift(anchor, cand.bbox) > config.TARGET_HANDOFF_MAX_AREA_DRIFT:
                continue
            if not kinematics.was_near_at(
                cand, sm.target_last_measured_frame, last_measured_center, near_radius,
            ):
                continue
            cos = kinematics.flock_cosine(kinematics.smoothed_velocity(cand), group_vel)
            if cos is not None and cos >= config.KIN_FLOCK_COS_HI:
                continue
            if anchor_ratio >= config.WHITE_FILL_RATIO:
                # while the target is still visibly filled, a handoff may not change that
                if white_fill_ratio(frame, cand.bbox) < anchor_ratio - config.TARGET_HANDOFF_MAX_RATIO_GAP:
                    continue
            if cert_valid:
                if cand.omega_valid:
                    if abs(cand.omega - sm.target_omega) > config.RECOVERY_OMEGA_TOLERANCE:
                        continue
                elif value < config.TARGET_HANDOFF_BLIND_IOU:
                    # an unconverged fit is not evidence of anything, so only a box that is
                    # essentially the anchor itself may be adopted as the same star re-labelled
                    continue
            elif value < config.TARGET_HANDOFF_BLIND_IOU:
                continue
            best, best_iou = cand, value
        if best is not None and best_iou >= config.TARGET_HANDOFF_IOU:
            sm.hand_off_target(best.track_id, best.bbox, best_iou)
            sm.target_last_measured_frame = frame_idx

    def _target(self, tracks: List[Track]) -> Optional[Track]:
        if self.sm.target_id is None:
            return None
        for t in tracks:
            if t.track_id == self.sm.target_id:
                return t
        return None

    def _result(self, frame, tracks, frame_idx, detections) -> FrameResult:
        return FrameResult(
            frame=frame,
            tracks=tracks,
            state=self.sm.state.value,
            target_id=self.sm.target_id,
            omega=0.0,
            confidence=self.sm.target_confidence,
            frame_idx=frame_idx,
            detect_count=len(detections),
        )


def _drop_fragments(white_dets: List[Detection]) -> List[Detection]:
    """A compound target is often detected twice: the compound box plus a brighter
    primitive inside it. The primitive is a fragment of the same white region, so a
    candidate almost fully contained in a MUCH larger white candidate (>=2x area,
    >=80% of its own area covered) is dropped and the container stays. The area and
    containment floors keep a dim oversized glitch box from swallowing a bright target
    that merely overlaps it (measured: scene_476768057)."""
    if len(white_dets) < 2:
        return white_dets

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    kept = []
    for d in white_dets:
        fragment = False
        for o in white_dets:
            if o is d or d.bbox is None or o.bbox is None:
                continue
            if area(o.bbox) < config.COUNTDOWN_FRAGMENT_MIN_AREA_RATIO * area(d.bbox):
                continue
            x1, y1 = max(d.bbox[0], o.bbox[0]), max(d.bbox[1], o.bbox[1])
            x2, y2 = min(d.bbox[2], o.bbox[2]), min(d.bbox[3], o.bbox[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area(d.bbox) > 0 and inter / area(d.bbox) >= 0.8:
                fragment = True
                break
        if not fragment:
            kept.append(d)
    return kept


def _deep_overlap_ids(live_tracks: List[Track]) -> set:
    """Ids whose box overlaps another live box enough that the rotation signature would
    see two contours. Overlap is the norm, not an event: ~80% of tracking frames have one."""
    ids = set()
    for i, a in enumerate(live_tracks):
        if a.bbox is None:
            continue
        for b in live_tracks[i + 1:]:
            if b.bbox is not None and _iou(a.bbox, b.bbox) >= config.OVERLAP_OMEGA_IOU:
                ids.add(a.track_id)
                ids.add(b.track_id)
    return ids


def _bbox_center(bbox):
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _area(bbox) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _either_clipped(b1, b2, frame_wh) -> bool:
    """A box clipped by the frame border loses most of its area while still being the same
    object, so scale comparisons are meaningless for it."""
    if b1 is None or b2 is None or frame_wh is None:
        return False
    w, h = frame_wh
    m = config.SWAP_EDGE_MARGIN

    def touches(b):
        return b[0] <= m or b[1] <= m or b[2] >= w - m or b[3] >= h - m

    return touches(b1) or touches(b2)


def _area_drift(b1, b2) -> float:
    """Relative bbox area change; a rotating star varies ~20%, a different object does not."""
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return abs(a2 - a1) / a1


def _iou(b1, b2) -> float:
    if b1 is None or b2 is None:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0
