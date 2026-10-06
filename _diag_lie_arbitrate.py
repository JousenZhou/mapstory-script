# -*- coding: utf-8 -*-
"""仲裁「谁压在星形上」：把回放输出的跟踪中心、探针逐帧真值质心、生产证据簇质心画到同一张放大裁剪图上。

背景：判据的 median≈27px 已被 _diag_lie_bias.py 分解成「常量偏置 28px（几乎纯 -y）+ 去偏残差 7px」，
说明跟踪点与探针质心锁的是同一个物理目标，只差一个固定矢量。到底是哪一个是星形的真实中心，只能看画面。

用法：
    .venv\\Scripts\\python.exe _diag_lie_arbitrate.py 702 720 744 776
产物：_diag_out/arbitrate_t{tick}.png（放大裁剪）、_diag_out/arbitrate_t{tick}_full.png（全景缩略）
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 图像。
import numpy as np  # 数值。

from _diag_lie_target_score import load_truth, truth_at  # 共用真值口径（逐帧内插）。

OUT_DIR = "_diag_out"  # 产物目录。
VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
TRACKS = os.path.join(OUT_DIR, "witness_tracks_20261001_190829_score1.00_extreme_extreme.json")  # 生产簇快照。
ZOOM = 4  # 放大倍数。
HALF = 110  # 裁剪半宽（全尺度像素）。
SNAP_TAGS = ("cluster3", "cluster", "cadence")  # 参与对比的回放 tag（绿/黄/紫）。
TAG_COLORS = {"cluster3": (0, 255, 0), "cluster": (0, 200, 255), "cadence": (255, 0, 255)}  # tag -> BGR。


def cluster_snapshot(tick, tolerance=3):  # 取距 tick 最近（tolerance 帧内）的检查点簇快照，返回 (快照帧号, 簇列表)。
    if not os.path.exists(TRACKS):
        return None, []
    with open(TRACKS, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    best = None  # 最近快照。
    for snap in data["snapshots"]:
        distance = abs(int(snap["tick"]) - tick)  # 帧距。
        if distance > tolerance:
            continue
        if best is None or distance < abs(int(best["tick"]) - tick):
            best = snap
    if best is None:
        return None, []
    return int(best["tick"]), best.get("clusters", [])[:4]


def replay_center(tag, tick):  # 某 tag 回放输出的跟踪中心（精确帧）。
    for path in glob.glob(os.path.join(OUT_DIR, f"{tag}_extreme_*.json")):
        if "190829" not in path:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload["frames"]:
            if row[0] == tick:
                return row[1], ((row[2], row[3]) if row[2] is not None else None)
    return None, None


def mark(img, pos, color):  # 画十字+小圈（文字在放大后补，避免拉伸模糊）。
    if pos is None:
        return
    x, y = int(round(pos[0])), int(round(pos[1]))
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (x, y), 4, color, 1)


def main():
    ticks = [int(a) for a in sys.argv[1:]] or [776]
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    points, _ = load_truth()  # 真值折线。
    for tick in ticks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"t{tick} 读帧失败")
            continue
        canvas = frame.copy()  # 标注画布。
        notes = []  # (全尺度坐标, 颜色, 文字)。
        truth = truth_at(points, tick)  # 真值（可能为 None：path 断轨）。
        snap_tick, clusters = cluster_snapshot(tick)  # 生产簇（检查点口径）。
        for tag in SNAP_TAGS:
            source, center = replay_center(tag, tick)  # 该 tag 输出。
            if center is None:
                continue
            color = TAG_COLORS[tag]
            mark(canvas, center, color)
            distance = None if truth is None else round(float(np.hypot(center[0] - truth[0], center[1] - truth[1])), 1)  # 到真值距离。
            notes.append((center, color, f"{tag[:6]} {source[:3]} d={distance}"))
        if truth is not None:
            mark(canvas, truth, (0, 0, 255))
            notes.append((truth, (0, 0, 255), "probe#311"))
        for index, cand in enumerate(clusters):
            mark(canvas, cand["center"], (255, 255, 0))
            notes.append((cand["center"], (255, 255, 0),
                          f"clu{index} s={cand['samples']} w={cand['windowed']} rep={int(bool(cand['reported']))}"))
        anchor = truth if truth is not None else clusters[0]["center"] if clusters else (frame.shape[1] // 2, frame.shape[0] // 2)
        cx, cy = int(round(anchor[0])), int(round(anchor[1]))  # 裁剪中心。
        x0, x1 = max(0, cx - HALF), min(frame.shape[1], cx + HALF)  # 列范围。
        y0, y1 = max(0, cy - HALF), min(frame.shape[0], cy + HALF)  # 行范围。
        crop = canvas[y0:y1, x0:x1]
        crop = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
        for (px, py), color, text in notes:  # 放大后写字。
            lx = int((px - x0) * ZOOM) + 16
            ly = int((py - y0) * ZOOM) - 12
            if 0 <= lx < crop.shape[1] - 30 and 14 <= ly < crop.shape[0] - 6:
                cv2.putText(crop, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        thumb = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))  # 全景。
        out_png = os.path.join(OUT_DIR, f"arbitrate_t{tick}.png")
        cv2.imwrite(out_png, crop)
        cv2.imwrite(os.path.join(OUT_DIR, f"arbitrate_t{tick}_full.png"), thumb)
        print(f"t{tick} truth={None if truth is None else (round(truth[0],1), round(truth[1],1))} snap@t{snap_tick}")
        for _, _, label in notes:  # 逐条打印标注。
            print(f"   {label}")
        print(f"   -> {out_png}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
