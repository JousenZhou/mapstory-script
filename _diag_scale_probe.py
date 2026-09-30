# -*- coding: utf-8 -*-
"""对比处理分辨率对解测谎「耗时」与「跟踪质量」的影响：extreme 默认尺度 vs 原分辨率(1.0x)。

只隔离「处理分辨率」这一个变量：粒子/对照/光流参数全部沿用 extreme 档，
仅把 process_scale 抬到 1.0、解除 max_process_side 上限，让算法直接吃原始帧。
每段素材统计 total 的 mean/p95 与 source 分布——prediction 占比是「盲跟外推」的代理指标，
越低说明真观测（color/border）越充分、跟踪质量越好。

跑法：
    .venv\\Scripts\\python.exe _diag_scale_probe.py                 # 三个代表分辨率素材
    .venv\\Scripts\\python.exe _diag_scale_probe.py 081653 001619   # 指定素材名子串
"""
import os
import sys
import time
import glob

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.liedetector.shape_session import ShapeTrackSession, ShapeTrackParams

WARMUP = 20  # 预热帧（含模板学习/CUDA Graph 捕获），不计入统计。
MAX_FRAMES = 220  # 每段最多处理帧数，够看清稳定态耗时与 source 分布。


def run(path, full_scale):
    """跑一段录像：full_scale=True 时用原分辨率(1.0x)，否则用 extreme 默认尺度。"""

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h, w = frame.shape[:2]
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)
    if full_scale:  # 只改分辨率相关两参，其余（粒子/对照/光流）仍是 extreme，隔离变量。
        session.params.process_scale = 1.0
        session.params.max_process_side = 1e9
    session.reset(w, h)  # reset 里按 params 算 effective_scale。
    times = []
    n = 0
    while ok and n < MAX_FRAMES:
        t0 = time.perf_counter()
        session.update(frame, 30.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if n >= WARMUP:
            times.append(dt)
        n += 1
        ok, frame = cap.read()
    cap.release()
    if not times:
        return None
    src = session.source_counts
    total_src = sum(src.values()) or 1
    scale = session.effective_scale
    return {
        "scale": round(scale, 3),
        "proc": f"{int(round(w * scale))}x{int(round(h * scale))}",
        "mean": round(float(np.mean(times)), 2),
        "p95": round(float(np.percentile(times, 95)), 2),
        "pred": round(src.get("prediction", 0) / total_src * 100, 1),
        "border": round(src.get("border", 0) / total_src * 100, 1),
        "color": round(src.get("color", 0) / total_src * 100, 1),
    }


def main():
    tags = sys.argv[1:] or ["081653", "001619", "001738"]  # 728x486 / 768x508 / 1366x768 三种代表分辨率。
    for tag in tags:
        matches = [p for p in sorted(glob.glob("lie_records/*.mp4")) if tag in p]
        if not matches:
            print(f"no video for {tag!r}")
            continue
        path = matches[0]
        cap = cv2.VideoCapture(path)
        res = f"{int(cap.get(3))}x{int(cap.get(4))}"
        cap.release()
        print(f"\n=== {os.path.basename(path)}  原分辨率={res} ===")
        for full in (False, True):
            r = run(path, full)
            label = "原图 1.0x " if full else "extreme默认"
            if r is None:
                print(f"  {label}: SKIP")
                continue
            fps = 1000.0 / r["mean"] if r["mean"] else 0.0
            print(f"  {label}  scale={r['scale']:<5} 处理={r['proc']:<9} "
                  f"mean={r['mean']:>7}ms p95={r['p95']:>7}ms (~{fps:>4.1f}fps)  "
                  f"color={r['color']:>4}% border={r['border']:>4}% prediction={r['pred']:>4}%")


if __name__ == "__main__":
    main()
