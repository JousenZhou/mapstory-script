# -*- coding: utf-8 -*-
"""偏差分解：把「到真值轨迹的距离」拆成 常量偏移 + 跟随残差 + 时间超前滞后，用来判断 20~35px 到底是
「跟错目标」还是「同一目标的定位口径差」。

用法：
    .venv\\Scripts\\python.exe _diag_lie_bias.py cluster3 cluster cadence
"""

import glob  # 找产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import numpy as np  # 统计。

from _diag_lie_target_score import WINDOW, load_truth, truth_at  # 共用真值口径。

OUT_DIR = "_diag_out"  # 产物目录。
TIER = "extreme"  # 判据档。


def load_tag(tag, video_filter="190829"):  # 取某 tag 的 extreme 回放 payload。
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_*.json"))):
        if video_filter and video_filter not in os.path.basename(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            return os.path.basename(path), json.load(handle)
    return None, None


def analyse(tag):  # 对一个 tag 做偏差分解。
    name, payload = load_tag(tag)
    if payload is None:
        print(f"[NODATA] {tag}")
        return
    points, _ = load_truth()  # 真值逐帧。
    by_tick = {row[0]: row for row in payload["frames"]}  # 帧索引。
    ticks, dx, dy = [], [], []  # 样本。
    for tick in range(WINDOW[0], WINDOW[1] + 1):
        truth = truth_at(points, tick)  # 真值。
        row = by_tick.get(tick)  # 输出。
        if truth is None or row is None or row[2] is None:
            continue
        ticks.append(tick)  # 记录。
        dx.append(row[2] - truth[0])  # x 偏移。
        dy.append(row[3] - truth[1])  # y 偏移。
    if len(ticks) < 10:
        print(f"[SHORT] {tag} 只有 {len(ticks)} 帧")
        return
    dx, dy = np.array(dx), np.array(dy)  # 转数组。
    raw = np.hypot(dx, dy)  # 原始距离。
    mx, my = float(dx.mean()), float(dy.mean())  # 常量偏移。
    ex, ey = dx - mx, dy - my  # 去偏后的残差。
    resid = np.hypot(ex, ey)  # 残差距离。
    # 时间滞后：把真值整体平移 k 帧，看哪一刻去偏残差最小（跟踪点落后还是超前）。
    tx = np.array([truth_at(points, t)[0] for t in ticks])  # 真值 x 序列。
    ty = np.array([truth_at(points, t)[1] for t in ticks])  # 真值 y 序列。
    best = None  # 最优滞后。
    for k in range(-12, 13):  # 搜索 ±12 帧。
        shifted = []  # 平移后的真值。
        for index, tick in enumerate(ticks):
            t2 = truth_at(points, tick - k)  # 真值在 tick-k 的值 = 若跟踪滞后 k 帧则应对齐它。
            shifted.append(t2 if t2 is not None else (tx[index], ty[index]))
        sx, sy = np.array([s[0] for s in shifted]), np.array([s[1] for s in shifted])
        d = np.hypot((tx + mx) - sx, (ty + my) - sy)  # 用「真值+常量偏移到哪都跟着」近似。
        score = float(np.median(d))  # 中位残差。
        if best is None or score < best[1]:
            best = (k, score)
    print(f"[{tag}] {name} n={len(ticks)}")
    print(f"   raw     median={np.median(raw):5.1f} p95={np.percentile(raw,95):5.1f} max={raw.max():5.1f}")
    print(f"   bias    ({mx:+6.1f}, {my:+6.1f}) |b|={np.hypot(mx,my):5.1f}  方向=atan2({my:+.0f},{mx:+.0f})={np.degrees(np.arctan2(my,mx)):+6.1f}°")
    print(f"   去偏残差 median={np.median(resid):5.1f} p95={np.percentile(resid,95):5.1f} max={resid.max():5.1f}")
    print(f"   最佳时间滞后 k={best[0]:+3d} 帧 -> median={best[1]:5.1f}")
    print(f"   偏移漂移 std=({dx.std():5.1f}, {dy.std():5.1f})  （小=稳定常量偏置，大=真的在飘）")


def main():
    for tag in (sys.argv[1:] or ["cluster"]):
        analyse(tag)


if __name__ == "__main__":
    main()
