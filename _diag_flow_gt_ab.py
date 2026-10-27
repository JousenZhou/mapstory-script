# -*- coding: utf-8 -*-
"""光流主会话 vs d/lie Pipeline 的「真值质量」A/B：跟随框中心到模拟器真值点的距离。

数据源：D:\\lie/failed_videos/<类别>/<子类>/scene_*/（preview.mp4 + trajectory.json 真值）。
指标（仅统计 GT valid 且会话处于 TRACKING/TARGET_LOST 的帧）：
- on-target%：dist <= 45px（星体半径量级）帧占比
- med_dist / p90_dist：距离中位数与 90 分位
- follow%：follow_bbox 非空帧占比（含重获期惯性外推——质量以 on-target 为准）
- game_over：是否提前终局
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from src.liedetector.vision.detector import Detector
from src.liedetector.vision.flow_session import FlowSession
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.state_machine import GameState

ON_TARGET_PX = 45.0


def load_gt(video_path):
    traj = json.load(open(video_path + ".trajectory.json", encoding="utf-8"))
    pts = {}
    for f in traj["frames"]:
        pts[int(f["frame_index"])] = (f.get("point"), bool(f.get("valid")))
    return pts


def run_video(path, make_session, gt):
    det = Detector(device="gpu")
    session = make_session(det)
    cap = cv2.VideoCapture(path)
    i = -1
    dists = []
    follow_n = 0
    total = 0
    scored = 0
    game_over_at = None
    states = {}
    while True:
        ok, frame = cap.read()
        i += 1
        if not ok:
            break
        res = session.process(frame, i)
        total += 1
        states[res.state] = states.get(res.state, 0) + 1
        if res.state == GameState.GAME_OVER.value and game_over_at is None:
            game_over_at = i
        if res.follow_bbox is not None:
            follow_n += 1
        gt_point, gt_valid = gt.get(i, (None, False))
        if not gt_valid or res.state not in ("TRACKING", "TARGET_LOST"):
            continue
        scored += 1
        if res.follow_bbox is not None:
            c = ((res.follow_bbox[0] + res.follow_bbox[2]) / 2, (res.follow_bbox[1] + res.follow_bbox[3]) / 2)
            dists.append(float(np.hypot(c[0] - gt_point[0], c[1] - gt_point[1])))
    cap.release()
    d = np.array(dists)
    return {
        "total": total,
        "scored": scored,
        "follow_pct": 100.0 * follow_n / max(total, 1),
        "on_target_pct": 100.0 * float((d <= ON_TARGET_PX).sum()) / max(scored, 1),
        "med_dist": float(np.median(d)) if len(d) else -1,
        "p90_dist": float(np.percentile(d, 90)) if len(d) else -1,
        "game_over_at": game_over_at,
    }


def make_pipe(det):
    p = Pipeline(det, 30.0)
    p.sm.start(0)
    return p


def main():
    root = r"D:\lie\failed_videos"
    scenes = []
    for dirpath, _dirs, files in os.walk(root):
        if "preview.mp4" in files and os.path.exists(os.path.join(dirpath, "preview.mp4.trajectory.json")):
            scenes.append(os.path.join(dirpath, "preview.mp4"))
    scenes.sort()
    print(f"scenes: {len(scenes)}")
    rows = []
    det_label = {0: "d/lie", 1: "flow"}
    for idx, v in enumerate(scenes):
        gt = load_gt(v)
        try:
            r0 = run_video(v, make_pipe, gt)
            r1 = run_video(v, lambda d: FlowSession(d, 30.0), gt)
        except Exception as e:
            print(f"[{idx}] {os.path.basename(os.path.dirname(v))}: ERROR {e}")
            continue
        rel = os.path.relpath(os.path.dirname(v), root)
        rows.append((rel, r0, r1))
        print(f"[{idx:02d}] {rel}")
        print(f"     d/lie: on={r0['on_target_pct']:5.1f}% med={r0['med_dist']:6.1f} p90={r0['p90_dist']:6.1f} "
              f"follow={r0['follow_pct']:5.1f}% over@{r0['game_over_at']}")
        print(f"     flow : on={r1['on_target_pct']:5.1f}% med={r1['med_dist']:6.1f} p90={r1['p90_dist']:6.1f} "
              f"follow={r1['follow_pct']:5.1f}% over@{r1['game_over_at']}")
    if rows:
        agg = [(np.mean([r0["on_target_pct"] for _, r0, r1 in rows]),
                np.mean([r1["on_target_pct"] for _, r0, r1 in rows])),
               (np.mean([r0["follow_pct"] for _, r0, r1 in rows]),
                np.mean([r1["follow_pct"] for _, r0, r1 in rows]))]
        wins = sum(1 for _, r0, r1 in rows if r1["on_target_pct"] >= r0["on_target_pct"])
        print(f"\n== 平均 on-target: d/lie {agg[0][0]:.1f}% vs flow {agg[0][1]:.1f}% | "
              f"平均 follow: {agg[1][0]:.1f}% vs {agg[1][1]:.1f}% | flow 不劣于: {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
