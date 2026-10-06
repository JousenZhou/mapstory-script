# -*- coding: utf-8 -*-
"""逐帧定位画面里的绿色靶心（高饱和绿标记）中心，与回放输出、探针真值对齐比较。

用途：判据真值用的是「证据团块质心」，但画面上还有一个游戏/录制自带的绿色标记，它才是人眼所指的
「目标」。三者谁在星形中心、谁随时间怎么走，必须逐帧量化，不能靠看图猜。

用法：
    .venv\\Scripts\\python.exe _diag_lie_green_marker.py 700 720 744 760 776
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 图像。
import numpy as np  # 数值。

from _diag_lie_target_score import load_truth, truth_at  # 真值口径。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
OUT_DIR = "_diag_out"  # 产物目录。


def green_center(frame):  # 找画面里最「纯绿」的团块中心（BGR：G 明显高于 R、B）。
    f = frame.astype(np.int16)  # 有符号便于做差。
    score = f[:, :, 1] - np.maximum(f[:, :, 0], f[:, :, 2])  # 绿超其他通道。
    mask = (score >= 60).astype(np.uint8)  # 二值。
    if mask.sum() < 20:
        return None, 0, None
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)  # 连通域。
    if count <= 1:
        return None, int(mask.sum()), None
    best = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])  # 最大块。
    area = int(stats[best, cv2.CC_STAT_AREA])  # 面积。
    box = stats[best, :4].tolist()  # x,y,w,h。
    return (float(centroids[best][0]), float(centroids[best][1])), area, box


def replay_centers(tick):  # 各 tag 在该帧的输出中心。
    out = {}  # tag -> (source, center)。
    for path in glob.glob(os.path.join(OUT_DIR, "*_extreme_20261001_190829_*.mp4.json")):
        tag = os.path.basename(path).split("_extreme_")[0]  # tag 名。
        if tag in ("rotation_diag", "witness_tracks"):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload["frames"]:
            if row[0] == tick and row[2] is not None:
                out[tag] = (row[1], (row[2], row[3]))
                break
    return out


def main():
    ticks = [int(a) for a in sys.argv[1:]] or list(range(700, 781, 10))
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 区域宽。
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 区域高。
    print(f"region {w}x{h}")
    points, _ = load_truth()  # 真值。
    for tick in ticks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
        ok, frame = cap.read()
        if not ok:
            continue
        pos, area, box = green_center(frame)  # 绿色标记。
        truth = truth_at(points, tick)  # 探针真值。
        outs = replay_centers(tick)  # 各回放输出。
        # 帧间绿色标记的连续性用灰度均值辅助判断（是否还在画面里）。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度。
        print(f"t{tick:4d} green={None if pos is None else (round(pos[0],1), round(pos[1],1))} area={area} box={None if box is None else [round(v) for v in box]}"
              f" truth={None if truth is None else (round(truth[0],1), round(truth[1],1))}"
              f" out={ {k: (v[0][:3], (round(v[1][0]), round(v[1][1]))) for k, v in outs.items() } }"
              f" mean={gray.mean():.1f}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
