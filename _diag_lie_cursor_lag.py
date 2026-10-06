# -*- coding: utf-8 -*-
"""判定「跟踪中心被绿色准星残差捕获」还是「准星本来就在瞄答案」——用互相关求最佳滞后。

因果性是唯一能区分二者的观测事实：
  · 若跟踪器被准星这个 UI 叠加层的时序残差吸引，跟踪中心必须**滞后**准星（滤波时间常数决定），
    滞后量为正帧数；
  · 若玩家准星一直瞄着答案（白相 319 帧中位差 2.0px 支持这一点），而跟踪器也压在答案上，
    两者应近似同相（lag≈0），或跟踪中心因粒子平滑而滞后；
  · 若跟踪中心**领先**准星（负 lag），则二者跟的不是同一个东西，准星路径与跟踪路径只是恰好平行。

同时给出透明相早期（border 起点之后）与 9s 窗的分别估计，以及准星自身的逐帧速度
（用来把 lag 换算成像素）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_cursor_lag.py
"""

import glob  # 找回放产物。
import json  # 读。

import numpy as np  # 数值。

REPLAY = sorted(glob.glob("_diag_out/anchor2_extreme_20261001_190829_*.json"))[-1]  # 回放。
CURSOR = "_diag_out/cursor_traj_190829.json"  # 准星轨迹缓存。
MAX_LAG = 40  # 搜索 ±40 帧。


def best_lag(rows, cur, t0, t1):
    """在 [t0,t1) 上搜使 Σ|跟踪(t) - 准星(t+lag)| 最小的 lag；正 lag = 跟踪滞后准星。"""
    out = []
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        d = []
        for t in range(t0, t1):
            r = rows.get(t)
            if r is None or r[2] is None:
                continue
            c = cur.get(t + lag)
            if c is None:
                continue
            d.append(np.hypot(r[2] - c[0], r[3] - c[1]))
        if len(d) >= (t1 - t0) * 0.6:
            out.append((float(np.mean(d)), lag, len(d)))
    if not out:
        return None
    out.sort()
    return out[0]


def main():
    rows = {r[0]: r for r in json.load(open(REPLAY, encoding="utf-8"))["frames"]}  # 回放行。
    cur = {int(k): v for k, v in json.load(open(CURSOR, encoding="utf-8")).items()}  # 准星。
    windows = [
        ("白相（权威，准星=答案）", 40, 330),
        ("透明相早期 t340-430", 340, 430),
        ("9s 窗 t430-520", 430, 520),
        ("丢失后 t520-620", 520, 620),
        ("后段 t620-760", 620, 760),
        ("旋转窗 t790-870", 790, 870),
    ]
    print(f"{'窗口':24s} {'最佳lag(帧)':>10s} {'平均距离':>9s} {'同相距离':>9s} 解读")
    for name, t0, t1 in windows:
        got = best_lag(rows, cur, t0, t1)
        if got is None:
            print(f"{name:24s} 样本不足")
            continue
        mean_d, lag, n = got
        same = [np.hypot(r[2] - c[0], r[3] - c[1]) for t, r in rows.items() if t0 <= t < t1 and r[2] is not None
                for c in [cur.get(t)] if c]
        same_d = float(np.mean(same)) if same else float("nan")
        speed = float(np.median([np.hypot(*(np.array(cur[t]) - np.array(cur[t - 1]))) for t in range(t0 + 1, t1)
                                 if t in cur and t - 1 in cur])) if t1 - t0 > 2 else float("nan")
        if lag > 3:
            read = f"跟踪滞后准星（UI 残差捕获特征），准星速度 {speed:.2f}px/帧 → 滞后折 {lag*speed:.0f}px"
        elif lag < -3:
            read = "跟踪领先准星 → 二者非同一对象（准星在别处）"
        else:
            read = "近似同相 → 准星即答案（或捕获极轻）"
        print(f"{name:24s} {lag:10d} {mean_d:9.1f} {same_d:9.1f} {read} (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
