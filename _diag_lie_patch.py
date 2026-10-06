# -*- coding: utf-8 -*-
"""补丁对比取证：把若干帧的同一区域裁出来横排（可叠加跟踪中心/真值标记），用肉眼判断目标图形到底移动到哪。

用途：本素材相机在持续平移，固定坐标判据会在早期帧误判；旋转结束后（帧 776+）答案图形不再转，
残差证据消失，只能靠图像对比确认它相对地形停在哪，从而给「17s 应在右下角」建立可信真值。

用法：
    .venv\\Scripts\\python.exe _diag_lie_patch.py                       # 默认：旋转结束前后右下区域
    .venv\\Scripts\\python.exe _diag_lie_patch.py 190829 620 400 140 776 800 820 853

输出：_diag_out/truth/patch_<tag>.png（每格左上角标帧号，红叉=该帧生产跟踪中心，绿圈=探针真值内插）。
"""

import glob  # 素材过滤。
import json  # 读回放逐帧 trace。
import os  # 路径。
import sys  # 参数。

import cv2  # 图像。
import numpy as np  # 拼接。

RECORD_DIR = "lie_records"  # 录像目录。
OUT_DIR = os.path.join("_diag_out", "truth")  # 输出目录。
TIER = "extreme"  # 读哪档的 trace。
# 探针 winner #311：birth=702 start=(610.6,333.7) → death=776 end=(594.4,412.6)；776 后不再旋转，按端点延申。
TRUTH_BIRTH, TRUTH_START = 702.0, (610.6, 333.7)
TRUTH_DEATH, TRUTH_END = 776.0, (594.4, 412.6)
DEFAULT_TAG = "cadence"  # 默认读哪个回放 tag 的 trace。


def truth_at(tick):  # 真值轨迹内插（旋转期线性，之后停在端点，之前不判返回 None）。
    if tick < TRUTH_BIRTH:
        return None  # 旋转前无残差证据，位置不可知。
    if tick >= TRUTH_DEATH:
        return TRUTH_END  # 停止旋转后按端点近似。
    ratio = (tick - TRUTH_BIRTH) / (TRUTH_DEATH - TRUTH_BIRTH)  # 内插比例。
    return (TRUTH_START[0] + (TRUTH_END[0] - TRUTH_START[0]) * ratio,
            TRUTH_START[1] + (TRUTH_END[1] - TRUTH_START[1]) * ratio)


def load_trace(tag, name_filter):  # 读回放 trace，返回 tick -> (cx, cy, source, conf)。
    paths = [p for p in sorted(glob.glob(os.path.join("_diag_out", f"{tag}_{TIER}_*.json"))) if name_filter in os.path.basename(p)]
    if not paths:
        return {}  # 无产物。
    with open(paths[-1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {row[0]: (row[2], row[3], row[1], row[4]) for row in payload["frames"] if row[2] is not None}


def main():  # 入口。
    name_filter = sys.argv[1] if len(sys.argv) > 1 else "190829"  # 素材。
    cx = float(sys.argv[2]) if len(sys.argv) > 2 else 620.0  # 裁剪中心 x。
    cy = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0  # 裁剪中心 y。
    half = int(sys.argv[4]) if len(sys.argv) > 4 else 140  # 半宽。
    ticks = [int(v) for v in sys.argv[5:]] or [702, 740, 776, 800, 820, 840, 853, 880]  # 帧列表。
    tag = os.environ.get("LIE_PATCH_TRACE", DEFAULT_TAG)  # trace 来源 tag。
    paths = [p for p in sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4"))) if name_filter in os.path.basename(p)]
    if not paths:
        print(f"no video matched {name_filter}")
        return 1
    trace = load_trace(tag, name_filter)  # 生产跟踪中心。
    cap = cv2.VideoCapture(paths[-1])  # 打开录像。
    tiles = []  # 横排块。
    want = set(ticks)  # 目标帧集合。
    frames = {}  # 帧缓存。
    tick = 0
    while tick <= max(want):  # 顺序读。
        ok, frame = cap.read()
        if not ok:
            break
        if tick in want:
            frames[tick] = frame.copy()  # 缓存。
        tick += 1
    cap.release()  # 释放。
    height, width = next(iter(frames.values())).shape[:2]  # 帧尺寸。
    x0, x1 = int(max(0, cx - half)), int(min(width, cx + half))  # 裁剪框。
    y0, y1 = int(max(0, cy - half)), int(min(height, cy + half))  # 裁剪框。
    zoom = 2  # 放大倍数，便于看细节。
    for frame_tick in ticks:  # 按序出图。
        frame = frames.get(frame_tick)
        if frame is None:
            continue
        tile = cv2.resize(frame[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)  # 裁剪放大。
        marker = lambda p: (int((p[0] - x0) * zoom), int((p[1] - y0) * zoom))  # 全尺度→tile 坐标。
        tracked = trace.get(frame_tick)  # 该帧生产中心。
        if tracked is not None:  # 画红叉 + 距离。
            px, py = marker((tracked[0], tracked[1]))
            cv2.drawMarker(tile, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)  # 红叉。
            cv2.putText(tile, f"trk {tracked[2][:5]} {tracked[3]:.2f}", (4, tile.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)  # 来源与置信。
        truth = truth_at(frame_tick)  # 真值。
        if truth is not None:  # 画绿圈。
            tx, ty = marker(truth)
            cv2.circle(tile, (tx, ty), int(45 * zoom), (0, 255, 0), 1)  # 命中半径。
            cv2.drawMarker(tile, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)  # 十字。
        label = f"t{frame_tick}"  # 帧号标签。
        if truth is not None and tracked is not None:  # 标两者距离。
            label += f"  d_trk_truth={((tracked[0] - truth[0]) ** 2 + (tracked[1] - truth[1]) ** 2) ** 0.5:.0f}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), -1)  # 标签底。
        cv2.putText(tile, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)  # 标签。
        tiles.append(tile)  # 入列。
    if not tiles:
        print("no frames read")
        return 1
    strip = cv2.hconcat([t[:, :tiles[0].shape[1]] for t in tiles])  # 等宽横排（裁剪框一致，理论同宽）。
    os.makedirs(OUT_DIR, exist_ok=True)  # 建目录。
    out = os.path.join(OUT_DIR, f"patch_{name_filter}_{int(cx)}x{int(cy)}_{tag}.png")
    cv2.imwrite(out, strip)  # 落盘。
    print(f"patch -> {out} ({strip.shape[1]}x{strip.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
