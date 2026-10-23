"""Neural motion-evidence channel (V7 trajectory regression, vendor port).

Faithful port of ShapeTrackStudio's onnx_trajectory_features / onnx_shape_runtime /
preprocess (source: 大肥鱼RoundV4/ShapeTrackStudio, itself a port of the V7 runtime).
Single-purpose second-opinion channel for the lie pipeline:

    full frame -> clean panel 640x424 (green-cursor inpaint is a no-op here)
    -> DIS optical flow two-frame features [current | residual | curl]
    -> appearance channel replaced by motion energy (appearance_removed)
    -> ONNX regresses the salient mover's center, seeded by the previous center

The channel is *motion-only*: no detection, no association, so its failure modes
(multi-object regression confusion, drift toward the panel prior under fast tail
motion) are disjoint from the pipeline's own. It is only consumed under distress
(swap-suspect / theft-watch / TARGET_LOST); it never overwrites tracker state
directly - see pipeline._neural_m1 / _neural_m2 for the gated consumers.

All magic numbers are recovered from the V7 runtime (see source config.py);
changing them breaks the network contract.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.logger import get_logger

log = get_logger("neural")

# --- V7 network contract (do not tune) ---------------------------------------
PANEL_WH = (640, 424)
NET_INPUT_WH = (320, 224)
PRIOR_CENTER_PANEL = (320.0, 212.0)
FLOW_PRESET = "DISOPTICAL_FLOW_PRESET_MEDIUM"
FLOW_FINEST_SCALE = 1
FLOW_GRAD_DESCENT_ITERS = 12
FLOW_VAR_REFINE_ITERS = 3
FLOW_SCALE = 0.5
SMOOTH_SIGMA = 9
DELTA_BLUR_KSIZE = (3, 3)
DELTA_BLUR_SIGMA = 0.7
FLOW_CURL_BLUR_SIGMA = 2.5
CURL_DIVISOR = 8.0
RESIDUAL_OFFSET = 1.5
RESIDUAL_GAIN = 9.0
ROTATION_GAIN = 90.0
ROTATION_BIAS = 128
ENERGY_GAIN = 4.0
GREEN_HSV_PANEL = ((35, 75, 55), (95, 255, 255))
CURSOR_DILATE_KSIZE = (7, 7)
INPAINT_RADIUS = 3


def _remove_green_cursor(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = GREEN_HSV_PANEL
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if not mask.any():
        return bgr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CURSOR_DILATE_KSIZE)
    mask = cv2.dilate(mask, kernel)
    return cv2.inpaint(bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


class _TwoFrameFeatures:
    """Causal two-image features; channel 1 residual, channel 2 flow curl."""

    def __init__(self):
        preset = getattr(cv2, FLOW_PRESET)
        self.flow = cv2.DISOpticalFlow_create(preset)
        self.flow.setFinestScale(FLOW_FINEST_SCALE)
        self.flow.setGradientDescentIterations(FLOW_GRAD_DESCENT_ITERS)
        self.flow.setVariationalRefinementIterations(FLOW_VAR_REFINE_ITERS)
        w, h = PANEL_WH
        self.grid = np.indices((h, w), dtype=np.float32)

    def encode_gray(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        w, h = PANEL_WH
        small = (round(w * FLOW_SCALE), round(h * FLOW_SCALE))
        p = cv2.resize(previous, small, interpolation=cv2.INTER_AREA)
        c = cv2.resize(current, small, interpolation=cv2.INTER_AREA)
        vf = self.flow.calc(c, p, None)
        ratio = np.array([small[0] / w, small[1] / h], np.float32)
        v = cv2.resize(vf, (w, h), interpolation=cv2.INTER_LINEAR) / ratio

        smooth = cv2.GaussianBlur(v, (0, 0), SMOOTH_SIGMA)
        yy, xx = self.grid
        aligned = cv2.remap(previous, xx + smooth[..., 0], yy + smooth[..., 1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        delta = cv2.GaussianBlur(cv2.absdiff(current, aligned).astype(np.float32),
                                 DELTA_BLUR_KSIZE, DELTA_BLUR_SIGMA)
        v_blur = cv2.GaussianBlur(v, (0, 0), FLOW_CURL_BLUR_SIGMA)
        dvx = cv2.Sobel(v_blur[..., 0], cv2.CV_32F, 0, 1, ksize=3)
        dvy = cv2.Sobel(v_blur[..., 1], cv2.CV_32F, 1, 0, ksize=3)
        curl = (dvy - dvx) / CURL_DIVISOR

        residual = np.clip((delta - RESIDUAL_OFFSET) * RESIDUAL_GAIN, 0, 255).astype(np.uint8)
        rotation = np.clip(curl * ROTATION_GAIN + ROTATION_BIAS, 0, 255).astype(np.uint8)
        return np.stack([current, residual, rotation], axis=-1)


class NeuralEvidence:
    """Lazy ONNX session + per-call motion regression in video pixels."""

    def __init__(self, frame_wh: Tuple[int, int]):
        self.frame_wh = frame_wh
        self.panel_to_video = np.array([frame_wh[0] / PANEL_WH[0],
                                        frame_wh[1] / PANEL_WH[1]], np.float32)
        self.prior_video = tuple(np.array(PRIOR_CENTER_PANEL, np.float32) * self.panel_to_video)
        self._session = None
        self._features = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_center: Optional[np.ndarray] = None
        self.recent: deque = deque(maxlen=8)  # (frame_idx, center_video)

    # -- session ---------------------------------------------------------
    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        path = Path(config.NEURAL_ONNX_PATH)
        if not path.exists():
            log.warning("neural evidence: onnx missing at %s", path)
            return False
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = []
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                log.warning("neural evidence: no GPU provider, CPU fallback is ~80ms/frame")
            self._session = ort.InferenceSession(str(path), sess_options=options,
                                                 providers=providers)
            self._features = _TwoFrameFeatures()
            log.info("neural evidence ready (%s, active=%s)", path.name,
                     self._session.get_providers()[0])
            return True
        except Exception as e:  # missing dep / device error: channel stays off
            log.warning("neural evidence init failed: %s", e)
            self._session = None
            return False

    def reset(self, belief_video=None) -> None:
        """Reseed the servo. belief_video (pipeline's own position) beats the panel
        prior: a mid-video cold start cannot acquire a far mover on its own (measured:
        the regression wanders near the prior), so after any stale gap the network
        restarts from where the tracker already believes the target is."""
        self._prev_gray = None
        if belief_video is not None:
            self._prev_center = (np.asarray(belief_video, np.float32)
                                 / self.panel_to_video)
        else:
            self._prev_center = np.array(PRIOR_CENTER_PANEL, np.float32)
        self.recent.clear()

    # -- inference -------------------------------------------------------
    def compute(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        """Regression center in video pixels, or None when unavailable."""
        if not self._ensure_session():
            return None
        panel = cv2.resize(_remove_green_cursor(frame_bgr), PANEL_WH,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            if self._prev_center is None:
                self._prev_center = np.array(PRIOR_CENTER_PANEL, np.float32)
            return None  # no motion pair yet: first frame only seeds the prior
        try:
            encoded = self._features.encode_gray(self._prev_gray, gray)
            evidence = encoded.copy()
            energy = np.abs(encoded[:, :, 2].astype(np.float32) - ROTATION_BIAS)
            energy = np.clip(cv2.GaussianBlur(energy, (0, 0), 3) * ENERGY_GAIN,
                             0, 255).astype(np.uint8)
            evidence[:, :, 0] = energy
            evidence = cv2.resize(evidence, NET_INPUT_WH, interpolation=cv2.INTER_AREA)
            motion = np.ascontiguousarray(
                evidence[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
            prior = (self._prev_center.reshape(1, 2)
                     / np.array([[PANEL_WH[0], PANEL_WH[1]]], np.float32)).astype(np.float32)
            out = self._session.run(None, {"motion": motion[None], "prior": prior})[0][0]
            center_panel = np.clip(out * np.array(PANEL_WH, np.float32),
                                   [0, 0], [PANEL_WH[0] - 1, PANEL_WH[1] - 1])
        except Exception as e:
            log.warning("neural evidence inference error: %s", e)
            return None
        finally:
            self._prev_gray = gray

        self._prev_center = center_panel
        center_video = center_panel * self.panel_to_video
        self.recent.append((frame_idx, (float(center_video[0]), float(center_video[1]))))
        return center_video
