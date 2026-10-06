# -*- coding: utf-8 -*-
"""9s 段真值轨迹：用 YOLO 透明图形检测器逐帧检出所有形状，再把 17s 那枚答案星形的检测框串成链。

为什么不用外观模板：t853 的锐利星形模板回溯到 t820 之前匹配分只有 0.08~0.35（=背景水平），
因为答案在透明相位外观完全不同；而 YOLO 检测器正是为「看见透明形状」训练的，位置不依赖外观。
为什么不用相机平移累计：那条曲线（_diag_out/truth/traj_190829.json）幅度 ±230px、周期 ~440 帧，
是水波往复而非刚体平移，上轮已否掉。

做法：t853 取离 (606.4,401.6) 最近的检测为锚，逐帧（步长 1）向两侧最近邻接龙，门限 40px
（全场形状间距 ~150px，漂移 ~2.5px/帧，40px 足够且不串台）；断帧时按常速外推补一次预测。

输出：
    _diag_out/nine_s_truth.json   tick -> [x, y, score, 是否补帧]
    _diag_out/nine_s_truth.log    逐 10 帧真值 + 同期跟踪中心 + 误差

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_truth.py
"""

import glob  # 找回放产物。
import json  # 落盘。
import os  # 路径。
import sys  # 导入 src。

import cv2  # 读帧。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 允许脚本目录直接导入包。

from src.liedetector.detector import TransparentShapeDetector  # YOLO 透明图形检测器。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 目标素材。
ANCHOR_TICK = 853  # 17s 真值锚点。
ANCHOR_POS = (606.4, 401.6)  # 上轮定案的答案星形中心。
LO, HI = 380, 900  # 建链区间。
GATE = 40.0  # 最近邻接入门限（像素）。
OUT_DIR = "_diag_out"  # 产物目录。
REPLAY_TAG = "anchor2"  # 对比用的回放批次。
REPLAY_KEY = "20261001_190829"  # 回放产物过滤。
REPLAY_TIER = "extreme"  # 回放档。


def detect_all():  # 逐帧检测，返回 tick -> [(cx, cy, score)]。
    cap = cv2.VideoCapture(VIDEO)
    detector = TransparentShapeDetector()  # 建一次会话，循环复用。
    frames = {}  # 结果容器。
    raw = {}  # tick -> 原帧（只留建链需要的帧，避免占内存：这里全留灰度即可）。
    tick = 0
    while tick <= HI:
        ok, frame = cap.read()
        if not ok:
            break
        if tick >= LO:
            dets = []  # 本帧检测列表。
            for box, score in detector.detect(frame):  # 只关心中心与置信。
                dets.append((float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0), float(score)))
            frames[tick] = dets
            raw[tick] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 存灰度供后续画验证图。
        tick += 1
    cap.release()
    return frames, raw


def walk(frames, start_tick, start_pos, step):  # 单向接龙（step=-1 回溯 / +1 前推）：最近邻入门，超门限则按常速外推并标记补帧。
    pos = np.asarray(start_pos, dtype=np.float64)  # 当前位置。
    vel = np.zeros(2)  # 常速。
    out = {}  # 结果。
    t = start_tick
    while LO <= t <= HI:
        dets = frames.get(t) or []
        score, filled = -1.0, True
        if dets:
            arr = np.array([[d[0], d[1]] for d in dets], dtype=np.float64)
            dist = np.linalg.norm(arr - pos, axis=1)
            k = int(np.argmin(dist))
            if dist[k] <= GATE:
                vel = 0.6 * vel + 0.4 * (arr[k] - pos)
                pos = arr[k]
                score, filled = dets[k][2], False
            else:
                pos = pos + vel
        else:
            pos = pos + vel
        out[t] = (pos.copy(), score, filled)
        t += step
    return out


def load_replay():  # 取回放逐帧中心：tick -> (cx, cy, source)。
    hits = [p for p in glob.glob(os.path.join(OUT_DIR, f"{REPLAY_TAG}_{REPLAY_TIER}_{REPLAY_KEY}_*.json")) if f"_{REPLAY_TIER}_" in p]
    if not hits:
        return {}
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        rows = json.load(handle)["frames"]
    return {r[0]: (r[2], r[3], r[1]) for r in rows if r[2] is not None}


def main():
    frames, _raw = detect_all()  # 全量检测。
    print(f"检测帧数 {len(frames)}，平均每帧 {np.mean([len(v) for v in frames.values()]):.1f} 个形状")
    anchor_dets = frames[ANCHOR_TICK]
    arr = np.array([[d[0], d[1]] for d in anchor_dets])  # 锚帧中心。
    k = int(np.argmin(np.linalg.norm(arr - np.array(ANCHOR_POS), axis=1)))  # 最近检测。
    start = arr[k]
    print(f"锚点 t{ANCHOR_TICK} 最近检测 ({start[0]:.1f},{start[1]:.1f}) 距 ANCHOR_POS "
          f"{np.linalg.norm(start - np.array(ANCHOR_POS)):.1f}px score={anchor_dets[k][2]:.2f}")
    back = walk(frames, int(ANCHOR_TICK), start, -1)  # 回溯
    fwd = walk(frames, int(ANCHOR_TICK) + 1, start, +1)  # 前推
    traj = {}
    traj.update(back)
    traj.update({t: v for t, v in fwd.items() if t not in traj})
    json.dump({str(int(t)): [float(v[0][0]), float(v[0][1]), round(float(v[1]), 3), bool(v[2])] for t, v in sorted(traj.items())},
              open(os.path.join(OUT_DIR, "nine_s_truth.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    replay = load_replay()
    filled = sum(1 for v in traj.values() if v[2])
    print(f"建链 {min(traj)}..{max(traj)} 补帧 {filled} 次")
    print("tick  真值(x,y)  score  |  回放中心  误差px  source")
    for t in sorted(traj):
        if t % 10 or t < LO + 20:
            continue
        pos, score, gap = traj[t]
        row = replay.get(t)
        if row is None:
            print(f"t{t:4d} ({pos[0]:6.1f},{pos[1]:6.1f}) {score:4.2f}{' 补' if gap else '   '} | 无回放")
            continue
        err = float(np.hypot(row[0] - pos[0], row[1] - pos[1]))
        print(f"t{t:4d} ({pos[0]:6.1f},{pos[1]:6.1f}) {score:4.2f}{' 补' if gap else '   '} | "
              f"({row[0]:6.1f},{row[1]:6.1f}) err={err:6.1f} {row[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
