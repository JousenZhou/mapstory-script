# -*- coding: utf-8 -*-
"""全长回放精度探针：不设帧数上限地跑完一局录像，逐帧落盘轨迹，并对关键帧保存叠加画面。

用途：_diag_lie_replay 只回前 400 帧（白相），看不到边界段的精度衰减；这里跑满全程，
每 ~15 帧打印一次 source/conf/snr/center，并在「白转边」「conf 跌破 0.3」「末帧」处存 PNG，
人工核对绿色中心点是否压在透明图形上。

用法：python _diag_star_full_replay.py [录像名过滤，默认 1908]
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

name_filter = sys.argv[1] if len(sys.argv) > 1 else "1908"  # 素材过滤。
mp4 = next(iter(glob.glob(f"lie_records/*{name_filter}*.mp4")), None)  # 找到录像。
if mp4 is None:
    print("no video")
    sys.exit(1)
side_path = mp4[:-4] + ".json"  # 边车路径。
side = json.load(open(side_path, encoding="utf-8-sig")) if os.path.exists(side_path) else {}  # 边车内容。
fps = float(side.get("fps") or 30.0)  # 喂帧口径：边车记录的录像名义帧率。
tier = "extreme"  # 精度档：上限档。
print(f"video={os.path.basename(mp4)} fps={fps:.2f} tier={tier} sidecar_outcome={side.get('outcome')}")

cap = cv2.VideoCapture(mp4)
session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=tier), logger=None)
rows = []  # 逐帧 (tick, source, cx, cy, conf, snr)。
tick = 0
prev_source = None
saved = set()  # 已存 PNG 的 tick，避免重复。


def save_overlay(t, frame, result, tag):  # 把中心点画到全尺帧上存盘。
    if t in saved:
        return
    saved.add(t)
    canvas = frame.copy()
    c = result.center
    if c is not None:
        x, y = int(c[0] * session.effective_scale), int(c[1] * session.effective_scale)  # 处理尺度->全尺度。
        cv2.drawMarker(canvas, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(canvas, (x, y), int(max(6.0, 18.0 * session.effective_scale)), (0, 255, 0), 1)
    out = f"_diag_out/prec_{tag}_{t:04d}.png"
    cv2.imwrite(out, canvas)
    print(f"  saved {out} source={result.source} conf={result.confidence:.2f}")


while True:
    ok, frame = cap.read()
    if not ok:
        break
    if tick == 0:
        session.reset(frame.shape[1], frame.shape[0])
    result = session.update(frame, fps)
    c = result.center
    rows.append((
        tick, result.source,
        None if c is None else round(float(c[0]), 2),
        None if c is None else round(float(c[1]), 2),
        round(float(result.confidence), 3), round(float(result.border_snr), 3),
    ))
    if result.source != prev_source:  # 来源切换点：切到 border 的最后一帧白相与首帧边界都值得存图。
        print(f"f{tick:4d} SOURCE {prev_source} -> {result.source} conf={result.confidence:.3f}")
        if result.source == "border":
            save_overlay(tick - 1, frame, result, "lastcolor")
        if result.source == "prediction":
            save_overlay(tick, frame, result, "pred")
        prev_source = result.source
    elif result.source == "border" and result.confidence < 0.3 and tick not in saved:
        save_overlay(tick, frame, result, "lowconf")
    tick += 1
cap.release()
n = len(rows)
print(f"total frames={n} region={session.region_width}x{session.region_height} scale={session.effective_scale:.4f}")
src_count = {}
for r in rows:
    src_count[r[1]] = src_count.get(r[1], 0) + 1
print(f"sources={src_count}")
# 每 15 帧打一行摘要（前 120 行），看 conf/snr/中心的趋势。
for r in rows[::15]:
    print(f"f{r[0]:4d} {r[1]:>10} conf={r[4]:.3f} snr={r[5]:.3f} center=({r[2]},{r[3]})")
# 末帧存图：最终中心落在哪个角落。
cap = cv2.VideoCapture(mp4)
cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
ok, last = cap.read()
cap.release()
if ok:
    class _R:  # 用末帧 rows 构造一个假 result 供 save_overlay 复用。
        center = None if rows[-1][2] is None else np.array([rows[-1][2], rows[-1][3]], np.float32)
        source = rows[-1][1]
        confidence = rows[-1][4]
    cv2.imwrite("_diag_out/prec_final_frame.png", last)  # 原始末帧。
    canvas = last.copy()
    if _R.center is not None:
        x, y = int(_R.center[0] * session.effective_scale), int(_R.center[1] * session.effective_scale)
        cv2.drawMarker(canvas, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.imwrite("_diag_out/prec_final_overlay.png", canvas)
    print(f"final center(process)={None if _R.center is None else tuple(np.round(_R.center,1))} final full px={None if _R.center is None else (int(_R.center[0]*session.effective_scale), int(_R.center[1]*session.effective_scale))}")
json.dump(rows, open(f"_diag_out/prec_rows_{name_filter}.json", "w"), ensure_ascii=False)
print(f"rows saved -> _diag_out/prec_rows_{name_filter}.json")
