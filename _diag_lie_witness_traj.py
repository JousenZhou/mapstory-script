# -*- coding: utf-8 -*-
"""把回放里逐帧记录的「旋转见证器候选」单独拉出来看轨迹。

见证器是独立于 border 证据的第二双眼睛（从证据图+光流里找唯一在旋转的团块），
它的候选序列如果连续往右下走，就同时提供了：
  a) 9s 段的第三方真值参照（与白相末端、旋转窗实测点互验）；
  b) 修复方向 —— 让见证候选在更低累计分下就能否决 border 中心。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_witness_traj.py [t_from] [t_to]
"""

import glob  # 找回放产物。
import json  # 读。
import sys  # 参数。

import numpy as np  # 数值。

REPLAY = sorted(glob.glob("_diag_out/anchor2_extreme_20261001_190829_*.json"))[-1]  # 回放。
ANCHORS = {336: (401.7, 134.4), 452: None, 702: (610.6, 333.7), 776: (594.4, 412.6), 853: (606.4, 401.6)}  # 已知锚点。


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 337  # 起始帧。
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 934  # 结束帧。
    rows = {r[0]: r for r in json.load(open(REPLAY, encoding="utf-8"))["frames"]}  # 回放行。
    prev = None  # 上一候选，算步进。
    seen = 0  # 有效候选帧数。
    seg_start, seg_last = None, None  # 当前连续段。
    for t in range(t_from, t_to + 1):
        row = rows.get(t)
        if row is None:
            continue
        source, cx, cy, conf, snr, wit, cum, applied = row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
        if wit:
            seen += 1
            step = "" if prev is None else f"步进={np.hypot(wit[0]-prev[0], wit[1]-prev[1]):5.1f}"
            # 跟踪中心可能为空（waiting/scene-ended），空值不参与偏离统计。
            ctr = "  --  " if cx is None else f"({cx:6.1f},{cy:6.1f})"
            dv = "" if cx is None else f"Δ跟踪={np.hypot(wit[0]-cx, wit[1]-cy):6.1f}"
            da = ""
            if t in ANCHORS and ANCHORS[t]:
                da = f"Δ锚={np.hypot(wit[0]-ANCHORS[t][0], wit[1]-ANCHORS[t][1]):6.1f}"
            print(f"t{t:4d} {source:11s} 候选=({wit[0]:6.1f},{wit[1]:6.1f}) cum={cum:6.1f} "
                  f"applied={int(bool(applied))} 跟踪={ctr} {dv} {da} {step}")
            prev = list(wit)
            seg_last = t
            if seg_start is None:
                seg_start = t
        else:
            if seg_start is not None and t - seg_last > 30:  # 段结束，汇总。
                print(f"  → 连续段 t{seg_start}..t{seg_last} 共 {seen} 帧有候选（累计自段首重算）")
                seen, seg_start, seg_last, prev = 0, None, None, None
    if seg_start is not None:
        print(f"  → 连续段 t{seg_start}..t{seg_last}")
    # 汇总：每 50 帧取候选中位数，看总体走向
    print("\n== 候选按 50 帧分桶（中位位置 / 出现帧数 / 跟踪中心偏离中位）==")
    bucket = {}
    for t in range(t_from, t_to + 1):
        row = rows.get(t)
        if row is None or not row[6]:
            continue
        b = t // 50 * 50
        bucket.setdefault(b, {"w": [], "d": []})
        bucket[b]["w"].append(row[6])
        if row[2] is not None:
            bucket[b]["d"].append(np.hypot(row[6][0] - row[2], row[6][1] - row[3]))
    for b in sorted(bucket):
        w = np.array(bucket[b]["w"])
        d = np.array(bucket[b]["d"]) if bucket[b]["d"] else np.array([np.nan])
        print(f"t{b:4d}-{b+49:4d} n={len(w):3d} 中位=({np.median(w[:,0]):6.1f},{np.median(w[:,1]):6.1f}) "
              f"span=({w[:,0].max()-w[:,0].min():6.1f},{w[:,1].max()-w[:,1].min():6.1f}) Δ跟踪中位={np.median(d):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
