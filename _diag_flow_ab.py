# -*- coding: utf-8 -*-
"""光流主会话 vs 原 d/lie Pipeline 的录像 A/B 对比。

对每段录像分别以两种会话逐帧复算，输出：
- follow 覆盖率（follow_bbox 非空帧占比）
- TARGET_LOST 帧数、GAME_OVER 到达帧
- 光流相态分布（pipeline/flow/reacquire/game_over）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detector
from src.liedetector.vision.flow_session import FlowSession
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.state_machine import GameState


def run_video(path, make_session):
    det = Detector(device="gpu")
    session = make_session(det)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    follow = lost = total = 0
    game_over_at = None
    first_follow = None
    t0 = time.perf_counter()
    i = -1
    while True:
        ok, frame = cap.read()
        i += 1
        if not ok:
            break
        res = session.process(frame, i)
        total += 1
        if res.follow_bbox is not None:
            follow += 1
            if first_follow is None:
                first_follow = i
        if res.state == GameState.TARGET_LOST.value:
            lost += 1
        if res.state == GameState.GAME_OVER.value and game_over_at is None:
            game_over_at = i
    cap.release()
    wall = time.perf_counter() - t0
    return {
        "total": total,
        "follow_pct": 100.0 * follow / max(total, 1),
        "lost_frames": lost,
        "game_over_at": game_over_at,
        "first_follow": first_follow,
        "fps": total / wall if wall > 0 else 0,
    }


def ab(path):
    print(f"== {os.path.basename(path)}")

    def flow_session(det):
        s = FlowSession(det, 30.0)
        return s

    def pipe_session(det):
        p = Pipeline(det, 30.0)
        p.sm.start(0)
        return p

    base = run_video(path, pipe_session)
    flow = run_video(path, flow_session)
    print(f"  d/lie  : follow={base['follow_pct']:.1f}% lost={base['lost_frames']} "
          f"game_over@{base['game_over_at']} first_follow@{base['first_follow']} ({base['fps']:.0f}fps 复算)")
    print(f"  光流主 : follow={flow['follow_pct']:.1f}% lost={flow['lost_frames']} "
          f"game_over@{flow['game_over_at']} first_follow@{flow['first_follow']} ({flow['fps']:.0f}fps 复算)")
    return base, flow


if __name__ == "__main__":
    videos = sys.argv[1:] or [
        os.path.join("lie_records", "20261025_232340.mp4"),
    ]
    for v in videos:
        if os.path.exists(v):
            ab(v)
