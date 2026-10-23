# -*- coding: utf-8 -*-
"""逐帧诊断 20260930_001619：重点看 155 帧附近为何目标处凭空多出大量目标框。

逐帧跑 detector.track + Pipeline，导出：
1. 每帧全部检测（bbox/conf/cls/track_id）与轨迹表（id/state/lost_frames/is_confirmed）
2. 渲染叠加图 PNG（140~170 帧逐帧）供目检
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

VIDEO = r"C:\Users\Administrator\Desktop\lie_records\20260930_001619_score1.00_ultra.mp4"
OUT_DIR = os.path.join("cache", "diag_001619")
LO, HI = 130, 175  # 关注帧区间。

from src.liedetector.vision.detector import Detector
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.overlay_renderer import render
from src.liedetector.vision.state_machine import GameState

os.makedirs(OUT_DIR, exist_ok=True)

det = Detector(device="gpu")
print(f"backend={det.backend_label} conf={det.conf}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video {int(cap.get(3))}x{int(cap.get(4))} @{fps:.2f} total={total}")

pipeline = Pipeline(det, fps)
pipeline.sm.start(0)

frame_idx = -1
while True:
    ok, frame = cap.read()
    frame_idx += 1
    if not ok or frame_idx > HI:
        break
    res = pipeline.process(frame, frame_idx)
    if frame_idx < LO:
        continue
    dets = det.last_track if False else pipeline.tracker.last_detections
    lines = [f"f{frame_idx:03d} state={res.state} target={res.target_id} omega={res.omega:+.3f} "
             f"conf={res.confidence:.2f} dets={res.detect_count} tracks={len(res.tracks)}"]
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        lines.append(f"   det id={d.track_id} cls={d.class_name} c={d.confidence:.2f} "
                     f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) cen=({d.centroid[0]:.0f},{d.centroid[1]:.0f})")
    for t in res.tracks:
        if t.lost_frames == 0:
            lines.append(f"   trk id={t.track_id} {t.state.value} confirmed={t.is_confirmed} "
                         f"lost={t.lost_frames} omega={t.omega if t.omega is None else round(t.omega,3)} "
                         f"created={t.created_frame} age={frame_idx - t.created_frame}")
    print("\n".join(lines))
    out = render(frame, res.tracks, res.target_id, res.state, show_boxes=True, show_trails=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"f{frame_idx:03d}.png"), out)

cap.release()
print("saved overlays to", OUT_DIR)
