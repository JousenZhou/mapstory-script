# -*- coding: utf-8 -*-
"""重构等价性对拍：把 _chain 抽取 + CUDA Graph 改造之前的 _compute 原样抄一份当参照，
与新实现在**同一门面、同一光流引擎**下逐帧对拍，要求逐位相等（CPU）或极小差（显卡）。

同时验证 CUDA Graph 路径与 eager 路径逐位一致——图内外跑的是同一份 _chain，
唯一区别是输入换成了固定地址的静态缓冲，因此结果必须完全相同。

跑法：.venv\\Scripts\\python.exe _diag_chain_equiv.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import gpu_shape_backend as gsb  # noqa: E402
from src.liedetector.tensor_evidence import TensorTemporalAligner, gaussian3x3  # noqa: E402
from src.liedetector.torch_flow import Cv2DisFlow, Cv2FarnebackFlow, TorchFarnebackFlow  # noqa: E402

LAGS = (1, 2, 4)
FLOW_PARAMS = dict(levels=3, iterations=3, winsize=15, poly_n=5, poly_sigma=1.2, pyr_scale=0.5)


def original_compute(aligner, frame):
    """改造前 _compute 的逐行抄本（唯一实现，作为等价性参照）。"""

    xp = aligner.xp
    with xp.inference():
        color = xp.clone(xp.asarray(frame))
        gray = xp.to_gray(color)
        aligner._history.append((color, gray))
        available = [lag for lag in aligner.lags if len(aligner._history) > lag]
        if not available:
            return None

        height = int(color.shape[0])
        width = int(color.shape[1])
        play_height = int(height * aligner.play_height_ratio)
        play_mask, band_mask = aligner._masks(height, width, play_height)

        stacked_gray = xp.stack([aligner._history[-1 - lag][1] for lag in available], axis=0)
        flows = aligner.flow_engine.calc_batch(gray, stacked_gray)

        current_color = xp.astype(color, xp.float32)
        normalized_maps = []
        raw_means = []

        for index in range(len(available)):
            previous_color = aligner._history[-1 - available[index]][0]
            aligned = xp.astype(xp.warp(previous_color, flows[index]), xp.float32)
            delta = xp.max(xp.abs(current_color - aligned), axis=2)
            valid_delta = xp.narrow(delta, 0, 0, play_height)
            median = xp.astype(xp.median(valid_delta), xp.float32)
            mad = xp.astype(xp.median(xp.abs(valid_delta - median)), xp.float32)
            sigma = xp.clip(mad * 1.4826, 1.0, None)
            normalized = xp.clip((delta - median) / sigma, 0.0, 12.0) * play_mask
            normalized_maps.append(normalized)
            raw_means.append(xp.mean(valid_delta))

        combined = xp.max(xp.stack(normalized_maps, axis=0), axis=0)
        combined = gaussian3x3(xp, combined) * band_mask
        raw_mean = float(xp.to_numpy(xp.min(xp.stack(raw_means, axis=0), axis=0)))
        return combined, raw_mean


def make_sequence(count, height=214, width=320, seed=7):
    """带平移方块的合成运动序列（uint8 BGR）。"""

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        frame = base.copy()
        top, left = 60 + index, 40 + index * 2
        frame[top:top + 46, left:left + 46] = 235
        frames.append(frame)
    return frames


def as_numpy(value):
    detach = getattr(value, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(value)


def compare(label, left, right, frames, exact):
    """left.update 走新实现，right 用 original_compute 驱动同一套内部状态。"""

    worst_map = 0.0
    worst_raw = 0.0
    pairs = 0
    for frame in frames:
        a = left.update(frame)
        b = original_compute(right, frame)
        if a is None or b is None:
            assert (a is None) == (b is None), f"{label}: 可用 lag 判定不一致"
            continue
        pairs += 1
        x = as_numpy(a.normalized).astype(np.float64)
        y = as_numpy(b[0]).astype(np.float64)
        worst_map = max(worst_map, float(np.abs(x - y).max()))
        worst_raw = max(worst_raw, abs(a.raw_mean - b[1]))
    passed = (worst_map == 0.0 and worst_raw == 0.0) if exact else (worst_map <= 1e-4)
    verdict = "OK" if passed else "FAIL"
    print(f"[{label}] frames={pairs} max|Δmap|={worst_map:.9f} max|Δraw|={worst_raw:.9f} -> {verdict}")
    return passed


def compare_graph(label, frames):
    """同一份帧序列：一个对齐器禁用图（纯 eager），一个启用图，两者输出必须逐位一致。"""

    backend = gsb.require_gpu_backend()
    if not backend.is_gpu:
        print(f"[{label}] skip: 后端非显卡")
        return True
    eager = TensorTemporalAligner(LAGS, 1.0, TorchFarnebackFlow(backend.xp, **FLOW_PARAMS), backend)
    eager._graph_disabled = True  # 强制纯 eager。
    graphed = TensorTemporalAligner(LAGS, 1.0, TorchFarnebackFlow(backend.xp, **FLOW_PARAMS), backend)
    worst_map = 0.0
    worst_raw = 0.0
    captured_at = None
    for index, frame in enumerate(frames):
        a = eager.update(frame)
        b = graphed.update(frame)
        if graphed._graph is not None and captured_at is None:
            captured_at = index
        if a is None or b is None:
            assert (a is None) == (b is None), f"{label}: 可用 lag 判定不一致"
            continue
        # 图的输出是静态缓冲，必须当场拷走，否则下一帧 replay 会覆盖。
        x = as_numpy(a.normalized).astype(np.float64).copy()
        y = as_numpy(b.normalized).astype(np.float64).copy()
        worst_map = max(worst_map, float(np.abs(x - y).max()))
        worst_raw = max(worst_raw, abs(a.raw_mean - b.raw_mean))
    ok = worst_map == 0.0 and worst_raw == 0.0 and captured_at is not None
    print(
        f"[{label}] captured_at_frame={captured_at} graph_active={graphed._graph is not None} "
        f"max|Δmap|={worst_map:.9f} max|Δraw|={worst_raw:.9f} -> {'OK' if ok else 'FAIL'}"
    )
    # reset() 之后图必须被丢弃，并在再次进入稳态时重新捕获。
    graphed.reset()
    dropped = graphed._graph is None and graphed._static_current_color is None
    for frame in frames:
        graphed.update(frame)
    recaptured = graphed._graph is not None
    print(f"[{label}] reset 释放图={dropped} 重新捕获={recaptured} -> {'OK' if dropped and recaptured else 'FAIL'}")
    return ok and dropped and recaptured


def main():
    frames = make_sequence(14)
    numpy_backend = gsb.numpy_backend()
    ok = True

    # 检查 1：CPU 门面 + DIS，重构前后必须逐位相等（打包版行为零变化的硬要求）。
    ok &= compare(
        "1-cpu-dis 重构等价",
        TensorTemporalAligner(LAGS, 1.0, Cv2DisFlow(2), numpy_backend),
        TensorTemporalAligner(LAGS, 1.0, Cv2DisFlow(2), numpy_backend),
        frames,
        exact=True,
    )
    # 检查 2：CPU 门面 + cv2 Farneback，同样要求逐位相等（与引擎无关，验证的是 _chain 抽取本身）。
    ok &= compare(
        "2-cpu-farneback 重构等价",
        TensorTemporalAligner(LAGS, 1.0, Cv2FarnebackFlow(**FLOW_PARAMS), numpy_backend),
        TensorTemporalAligner(LAGS, 1.0, Cv2FarnebackFlow(**FLOW_PARAMS), numpy_backend),
        frames,
        exact=True,
    )
    # 检查 3/4：显卡门面，eager 重构等价 + CUDA Graph 与 eager 逐位一致。
    if gsb.gpu_backend_available():
        backend = gsb.require_gpu_backend()
        ok &= compare(
            "3-gpu eager 重构等价",
            TensorTemporalAligner(LAGS, 1.0, TorchFarnebackFlow(backend.xp, **FLOW_PARAMS), backend),
            TensorTemporalAligner(LAGS, 1.0, TorchFarnebackFlow(backend.xp, **FLOW_PARAMS), backend),
            frames,
            exact=True,
        )
        ok &= compare_graph("4-gpu CUDA Graph 等价", frames)
    else:
        print("[3/4] skip: 无 torch CUDA")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
