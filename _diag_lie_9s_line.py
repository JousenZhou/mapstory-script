# -*- coding: utf-8 -*-
"""检验「答案沿直线匀速往右下」假设：在 (t452,(395,55)) → (t702,(610.6,333.7)) 的连线各帧位置上，
量到最近 YOLO 检测的距离。若沿途每帧都有形状（距离 < 20px），假设成立，这条线就能当 9s 段真值。

对照组：同样量「回放跟踪中心」到最近检测的距离，以及「跟踪中心连线的法向偏移线」上的距离，
用来排除「反正场上星形密，随便哪里都能凑到一个」这种假阳性。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_line.py
"""

import glob  # 找回放产物。
import json  # 读缓存。

import numpy as np  # 数值。

CACHE = "_diag_out/yolo_dets_190829.json"  # 逐帧检测中心。
A_TICK, A_POS = 452, np.array([395.0, 55.0])  # 假设起点（白相掉头位置）。
B_TICK, B_POS = 702, np.array([610.6, 333.7])  # 假设终点（旋转窗实测）。
REPLAY = sorted(glob.glob("_diag_out/anchor2_extreme_20261001_190829_*.json"))[-1]  # 回放。


def nearest(dets, pos):  # 最近检测距离与坐标。
    if len(dets) == 0:
        return 999.0, None
    arr = np.array(dets, dtype=np.float64)
    d = np.linalg.norm(arr - pos, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), arr[k]


def main():
    frames = {int(k): v for k, v in json.load(open(CACHE, encoding="utf-8")).items()}  # 检测。
    rows = {r[0]: r for r in json.load(open(REPLAY, encoding="utf-8"))["frames"]}  # 回放行。
    vel = (B_POS - A_POS) / (B_TICK - A_TICK)  # 匀速假设下的速度。
    print(f"匀速假设速度 = ({vel[0]:+.3f},{vel[1]:+.3f}) px/帧，即 {np.hypot(*vel)*50.207:.0f} px/s")
    print("tick   线上预测      最近检测距离   |  跟踪中心      到检测距离  跟踪到线距离")
    hits_line, hits_trk = 0, 0
    n = 0
    for t in range(int(A_TICK), int(B_TICK) + 1, 10):
        pred = A_POS + vel * (t - A_TICK)  # 线上预测点。
        dets = frames.get(t, [])  # 本帧检测。
        dl, _ = nearest(dets, pred)  # 线到检测。
        row = rows.get(t)
        trk = np.array([row[2], row[3]]) if row and row[2] is not None else None  # 跟踪中心。
        dt = nearest(dets, trk)[0] if trk is not None else -1  # 跟踪中心到检测。
        # 点到直线距离
        if trk is not None:
            rel = trk - A_POS
            vv = vel / np.linalg.norm(vel)
            along = float(rel @ vv)
            perp = float(np.linalg.norm(rel - along * vv))
        else:
            perp = -1
        n += 1
        hits_line += dl < 20.0
        hits_trk += (dt >= 0 and dt < 20.0)
        print(f"t{t:4d} ({pred[0]:6.1f},{pred[1]:6.1f}) {dl:6.1f}      |  "
              f"({trk[0]:6.1f},{trk[1]:6.1f}) {dt:6.1f}   {perp:6.1f}")
    print(f"命中率(<20px)：线上 {hits_line}/{n}，跟踪中心 {hits_trk}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
