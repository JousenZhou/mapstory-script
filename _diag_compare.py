# -*- coding: utf-8 -*-
"""对比 test_case 三个参考视频：分辨率/帧数 + 全流程分段耗时，解释 fps 差异。"""

import glob
import time

import cv2
import numpy as np

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession

CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"
videos = sorted(glob.glob(CASE_DIR + r"\*.mp4"))

for path in videos:
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n===== {path.split(chr(92))[-1]} : {w}x{h} frames={n} src_fps={fps:.1f}")

    # 内容复杂度采样：取中间 5 帧看白点候选与边缘密度。
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    white_counts = []
    edge_means = []
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_counts.append(len([c for c in cnts if cv2.contourArea(c) >= 40]))
        edge_means.append(float(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 180).mean()))
    print(f"  sample whites/frame={white_counts} canny_mean={edge_means}")

    # 全流程基准：前 120 帧（跳过前 5 帧光流预热不计入均值）。
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    params = ShapeTrackParams()
    session = ShapeTrackSession(params=params)
    session.reset(w, h)
    decode_times, track_times = [], []
    sources = {}
    whites_per_frame = []
    frame = None
    for i in range(120):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_dec = time.perf_counter()
        if not ok:
            print(f"  video ended at frame {i}")
            break
        result = session.update(frame, 30)
        t_trk = time.perf_counter()
        if i >= 5:
            decode_times.append(t_dec - t0)
            track_times.append(t_trk - t_dec)
        sources[result.source] = sources.get(result.source, 0) + 1
        whites_per_frame.append(result.white_candidates)
    total = np.mean(decode_times) + np.mean(track_times)
    print(f"  effective_scale={session.effective_scale:.3f}")
    print(f"  decode={np.mean(decode_times) * 1000:.1f}ms track={np.mean(track_times) * 1000:.1f}ms "
          f"total={total * 1000:.1f}ms -> {1000 / (total * 1000):.1f} fps")
    print(f"  sources={sources} whites_avg={np.mean(whites_per_frame):.1f} max={max(whites_per_frame)}")
    cap.release()
