# -*- coding: utf-8 -*-
"""实测候选性能配置：不同 max_process_side/control_count/particle_count 下的 track 耗时与 source 分布。"""
import glob
import sys
import time

sys.path.insert(0, '.')
import cv2
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession

CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"

CONFIGS = {
    "baseline(400/C260/P280)": dict(),
    "320/C260/P280": dict(max_process_side=320.0),
    "320/C130/P280": dict(max_process_side=320.0, control_count=130),
    "320/C130/P200": dict(max_process_side=320.0, control_count=130, particle_count=200),
}

videos = sorted(glob.glob(CASE_DIR + r"\*.mp4"))
for path in videos:
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for _ in range(90):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"\n===== {path.split(chr(92))[-1]} {w}x{h}")
    for name, overrides in CONFIGS.items():
        sess = ShapeTrackSession(params=ShapeTrackParams(**overrides))
        sess.reset(w, h)
        t_all = 0.0
        n = 0
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            sess.update(frame, 30.0)
            if i >= 5:
                t_all += time.perf_counter() - t0
                n += 1
        print(f"  {name:26s} scale={sess.effective_scale:.3f} track={t_all / n * 1000:5.1f}ms sources={sess.source_counts}")
