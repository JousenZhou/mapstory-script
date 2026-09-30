# -*- coding: utf-8 -*-
"""解测谎单帧耗时分段剖析：定位「显卡化之后反而没变快」的时间去哪了。

在真实录像上跑 high/ultra/extreme 三档，把 session.update 拆成
缩放 / 证据链 / 白块检测 / 粒子推进 / 估计 / 观测 / 重定位 若干段，每段前后都做一次
``torch.cuda.synchronize()``（显卡后端）以拿到真实设备耗时，而不是被异步排队掩盖。

注意：同步点本身会打断异步流水，因此这里量到的 total 会比未插桩时偏大，
各段之和也会偏大。用途是看**占比**，不是看绝对值；绝对值以 ``_diag_out/p3torch_*.json``
里未插桩的 timing_ms 为准（脚本会一并读出来对照）。

跑法：
    .venv\\Scripts\\python.exe _diag_perf_probe.py            # 全部素材 × high/ultra/extreme
    .venv\\Scripts\\python.exe _diag_perf_probe.py 2231 ultra # 只跑名字含 2231 的、ultra 档
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _diag_lie_replay import collect_videos  # noqa: E402
from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.torch_array import torch_gpu_available  # noqa: E402

MAX_FRAMES = 300
WARMUP_FRAMES = 20
DEFAULT_TIERS = ("high", "ultra", "extreme")
# 剖析的阶段名；顺序即 session.update 里的调用顺序，rest 由减法得出。
STAGES = ("resize", "evidence", "white", "propagate", "estimate", "observe", "relocate")


def sync():
    """显卡后端下强制同步，让分段计时拿到真实设备耗时。"""

    if torch_gpu_available():
        from src.liedetector.torch_array import torch_module

        torch_module().torch.cuda.synchronize()


class Recorder:
    """把 (tick, 毫秒) 攒进桶，按 tick >= WARMUP_FRAMES 过滤后折算成「每帧毫秒」。"""

    def __init__(self):
        self.buckets = defaultdict(list)
        self.frames = 0  # 参与统计的帧数（去掉预热帧）。

    def add(self, stage, tick, ms):
        if tick >= WARMUP_FRAMES:
            self.buckets[stage].append((tick, ms))

    def per_frame(self, stage):
        values = self.buckets.get(stage)
        if not values:
            return 0.0, 0
        return sum(ms for _, ms in values) / self.frames, len(values)


def instrument(session, recorder, ticks, undo):
    """给 session 的各阶段包一层计时。undo 收集 (对象, 属性名, 原值) 供复原。"""

    def wrap(obj, name, stage):
        raw = getattr(obj, name)

        def timed(*args, **kwargs):
            sync()
            started = time.perf_counter()
            out = raw(*args, **kwargs)
            sync()
            recorder.add(stage, ticks[0], (time.perf_counter() - started) * 1000.0)
            return out

        undo.append((obj, name, raw))
        setattr(obj, name, timed)

    wrap(session.aligner, "update", "evidence")  # 证据链（光流 + 残差归一化）。
    # 这两个是模块级函数，session.update 里按模块属性查找，patch 模块即可。
    wrap(ss, "detect_white_shapes", "white")
    wrap(ss, "resize_for_processing", "resize")


def wrap_tracker(tracker, recorder, ticks, undo):
    """跟踪器是首帧之后才建出来的，建出来后单独包一次。"""

    def wrap(name, stage):
        raw = getattr(tracker, name)

        def timed(*args, **kwargs):
            sync()
            started = time.perf_counter()
            out = raw(*args, **kwargs)
            sync()
            recorder.add(stage, ticks[0], (time.perf_counter() - started) * 1000.0)
            return out

        undo.append((tracker, name, raw))
        setattr(tracker, name, timed)

    wrap("propagate", "propagate")
    wrap("estimate", "estimate")
    wrap("observe_color", "observe")
    wrap("observe_border", "observe")
    wrap("_relocate", "relocate")


def reference_ms(tier, name):
    """读未插桩的回放基准耗时（同一份含 CUDA Graph 的代码），用于判断插桩本身带来了多少失真。"""

    path = os.path.join("_diag_out", f"p4graph_{tier}_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["timing_ms"]["mean"]


def run(path, region, tier):
    """插桩回放一段录像，返回分段耗时。"""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recorder = Recorder()
    ticks = [0]  # 用单元素列表让闭包能读到当前帧号。
    undo = []
    session = None
    wrapped = None
    totals = []
    try:
        while ticks[0] < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame
            if region is not None:
                x, y, w, h = region
                crop = frame[y:y + h, x:x + w]
                if crop.size == 0:
                    break
            if session is None:
                session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=tier), logger=None)
                session.reset(crop.shape[1], crop.shape[0])
                instrument(session, recorder, ticks, undo)
            if session.tracker is not None and wrapped is not session.tracker:
                wrapped = session.tracker
                wrap_tracker(wrapped, recorder, ticks, undo)
            sync()
            started = time.perf_counter()
            session.update(crop, float(src_fps))
            sync()
            if ticks[0] >= WARMUP_FRAMES:
                totals.append((time.perf_counter() - started) * 1000.0)
                recorder.frames += 1
            ticks[0] += 1
    finally:
        cap.release()
        for obj, name, raw in undo:  # 复原所有 patch，避免影响下一段。
            setattr(obj, name, raw)
    if session is None or not totals:
        return None
    name = os.path.basename(path)
    total_mean = float(np.mean(totals))
    stages = {}
    for stage in STAGES:
        ms, count = recorder.per_frame(stage)
        stages[stage] = {"ms": round(ms, 3), "calls": count}
    accounted = sum(info["ms"] for info in stages.values())
    stages["rest"] = {"ms": round(total_mean - accounted, 3), "calls": recorder.frames}
    return {
        "tier": tier,
        "video": name,
        "frames": ticks[0],
        "backend": session.backend.name,
        "engine": getattr(session.aligner, "engine_name", ""),
        "region": list(region) if region else None,
        "scale": round(float(session.effective_scale), 4),
        "total_mean": round(total_mean, 2),
        "total_p95": round(float(np.percentile(totals, 95)), 2),
        "reference_mean": reference_ms(tier, name),  # 未插桩的基准，None 表示没跑过。
        "stages": stages,
        "sources": dict(session.source_counts),
    }


def show(data):
    print(
        f"\n=== {data['tier']} {data['video']} frames={data['frames']} scale={data['scale']} "
        f"backend={data['backend']} engine={data['engine']} region={data['region']} ==="
    )
    ref = data["reference_mean"]
    skew = f"  未插桩={ref}ms 插桩放大={data['total_mean'] / ref:.2f}x" if ref else ""
    print(f"  total mean={data['total_mean']}ms p95={data['total_p95']}ms{skew}")
    print(f"  sources={data['sources']}")
    for stage in STAGES + ("rest",):
        info = data["stages"][stage]
        share = (info["ms"] / data["total_mean"] * 100.0) if data["total_mean"] else 0.0
        print(f"  {stage:10s} {info['ms']:8.3f}ms/frame {share:5.1f}%  calls={info['calls']}")


def main():
    name_filter = sys.argv[1] if len(sys.argv) > 1 else ""
    tier_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    tiers = (tier_filter,) if tier_filter in DEFAULT_TIERS else DEFAULT_TIERS
    videos = [item for item in collect_videos() if name_filter in item[0]]
    if not videos:
        print(f"no video matched {name_filter!r}")
        return 1
    results = []
    for name, path, region in videos:
        for tier in tiers:
            data = run(path, region, tier)
            if data is None:
                print(f"SKIP {name} {tier}")
                continue
            results.append(data)
            show(data)
    if len(results) > 1:  # 多段时按帧数加权汇总，看整体时间去向。
        weights = np.asarray([row["frames"] for row in results], dtype=np.float64)
        print("\n===== 加权汇总（按帧数）=====")
        print(f"{'tier':6s}{'stage':12s}{'ms/frame':>10s}{'share':>8s}")
        for tier in dict.fromkeys(row["tier"] for row in results):
            subset = [row for row in results if row["tier"] == tier]
            sub_weights = np.asarray([row["frames"] for row in subset], dtype=np.float64)
            total = float(np.average([row["total_mean"] for row in subset], weights=sub_weights))
            for stage in STAGES + ("rest",):
                ms = float(np.average([row["stages"][stage]["ms"] for row in subset], weights=sub_weights))
                print(f"{tier:6s}{stage:12s}{ms:10.3f}{ms / total * 100.0:7.1f}%")
            print(f"{tier:6s}{'TOTAL':12s}{total:10.3f}")
        del weights
    os.makedirs("_diag_out", exist_ok=True)
    with open(os.path.join("_diag_out", "perf_probe.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
