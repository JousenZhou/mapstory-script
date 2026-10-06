# -*- coding: utf-8 -*-
"""9s 段逐帧对账：把回放落盘的跟踪输出与该帧真实画面并排取出来，用于定位「丢失并往右上平滑」的第一现场。

9s 对应帧号：素材 fps=50.207 → 9×50.207≈452（17s 硬验收帧 853 同源换算）。

输出：
    _diag_out/nine_s_trace.log   每帧 tick/source/center/conf/border_snr/见证候选/是否重锚
    _diag_out/nine_s/t{tick}.png 对应原始帧（不裁剪、不缩放，直接可肉眼判读目标在哪）

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_probe.py [tick_from] [tick_to] [存图步长]
"""

import json  # 读回放产物。
import glob  # 找文件。
import os  # 路径。
import sys  # 参数。

import cv2  # 读视频帧。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 目标素材（已是区域裁剪结果）。
VIDEO_KEY = "20261001_190829"  # 回放产物过滤。
OUT_DIR = "_diag_out"  # 落盘目录。
TIER = "extreme"  # 生产默认档。
TAG = os.environ.get("LIE_PROBE_TAG", "anchor2")  # 用哪次回放的结果做对账。


def load_rows():  # 取回放的逐帧行：[tick, source, cx, cy, conf, border_snr, witness, cum, applied]。
    hits = [p for p in glob.glob(os.path.join(OUT_DIR, f"{TAG}_{TIER}_{VIDEO_KEY}_*.json")) if f"_{TIER}_" in p]
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        return json.load(handle)["frames"]


def main():
    tick_from = int(sys.argv[1]) if len(sys.argv) > 1 else 420  # 起始帧。
    tick_to = int(sys.argv[2]) if len(sys.argv) > 2 else 540  # 结束帧。
    save_step = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # >0 时按该步长存图。
    frames = load_rows()
    if frames is None:
        print(f"MISS 无 {TAG}/{TIER} 回放产物")
        return 1
    rows = {r[0]: r for r in frames}
    os.makedirs(os.path.join(OUT_DIR, "nine_s"), exist_ok=True)
    prev = None  # 上一有效中心，用于算逐帧位移（「平滑」的量化口径）。
    for tick in range(tick_from, tick_to + 1):
        row = rows.get(tick)
        if row is None:
            continue
        source, cx, cy, conf, snr = row[1], row[2], row[3], row[4], row[5]
        witness = row[6] if len(row) > 6 else None
        cum = row[7] if len(row) > 7 else 0.0
        applied = row[8] if len(row) > 8 else False
        move = "" if (prev is None or cx is None) else f"mv={np.hypot(cx - prev[0], cy - prev[1]):5.1f}"
        if cx is not None:
            prev = (cx, cy)
        wtext = "none" if not witness else f"({witness[0]:.0f},{witness[1]:.0f})"
        print(f"t{tick:4d} {source:12s} c=({cx:6.1f},{cy:6.1f}) conf={conf:4.2f} snr={snr:5.2f} "
              f"wit={wtext:14s} cum={cum:6.1f} reanchor={int(bool(applied))} {move}")
    if save_step:  # 单独一遍顺序解码存图（VideoCapture 逐帧读必须顺序，上面那遍不做解码）。
        cap = cv2.VideoCapture(VIDEO)
        tick = 0
        wanted = set(range(tick_from, tick_to + 1, save_step))
        while tick <= tick_to:
            ok, frame = cap.read()
            if not ok:
                break
            if tick in wanted:
                row = rows.get(tick)
                path = os.path.join(OUT_DIR, "nine_s", f"t{tick}.png")
                canvas = frame.copy()
                if row and row[2] is not None:  # 把跟踪中心画上去，肉眼直接判「跟的是谁」。
                    scale = 1.0
                    cv2.drawMarker(canvas, (int(row[2] * scale), int(row[3] * scale)), (0, 255, 0),
                                   cv2.MARKER_CROSS, 22, 2)
                    if row[6]:
                        cv2.circle(canvas, (int(row[6][0]), int(row[6][1])), 10, (0, 128, 255), 2)
                cv2.imwrite(path, canvas)
            tick += 1
        cap.release()
        print(f"已存图 {sorted(wanted)} -> {os.path.join(OUT_DIR, 'nine_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
