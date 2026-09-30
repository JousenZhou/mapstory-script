# -*- coding: utf-8 -*-
"""给回归门控选口径：把「逐帧 Δcenter」拆成几个判别力不同的统计量，看哪个能区分
「数值混沌噪声」与「真实的逻辑回归」。

五对比较：
    A baseline(CuPy+DIS)  ↔ n3facade(numpy+cv2 Farneback)   纯 CPU、完全不碰 torch，只换了光流算法
    B n3facade            ↔ p3torch(torch+torch Farneback)  同一份代码，只换数组库与设备
    C baseline            ↔ p3torch                         端到端总差异（Phase 3 收尾口径）
    D baseline            ↔ p4graph(+CUDA Graph)            端到端总差异（Phase 4 最终口径）
    E p3torch             ↔ p4graph                         只差 CUDA Graph：应逐帧 Δ=0、仅耗时下降

若 C/D 的每个统计量都落在 A/B 的范围内，就说明总差异全部由数值混沌解释，没有逻辑回归。
E 是硬不变量：CUDA Graph 与 eager 跑同一份 _chain，证据图逐位相等（见 _diag_chain_equiv.py），
下游粒子滤波消费相同输入 → 跟踪结果必须完全一致，任何非零 Δ 都说明图捕获引入了副作用。
"""

import glob
import json
import os
import sys

import numpy as np

OUT_DIR = "_diag_out"


def load(tag):
    result = {}
    for path in glob.glob(os.path.join(OUT_DIR, f"{tag}_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result[(payload["tier"], payload["video"])] = payload
    return result


def stats(left, right):
    index = {row[0]: row for row in right["frames"]}
    pairs = [(row, index[row[0]]) for row in left["frames"] if row[0] in index]
    if not pairs:
        return None
    same_source = 0
    all_delta = []
    same_delta = []
    tracked_delta = []  # 两侧都是 color/border（真正在跟目标，而不是外推预测）的帧。
    tracked = ("color", "border")
    for a, b in pairs:
        if a[1] == b[1]:
            same_source += 1
        if a[2] is None or b[2] is None:
            continue
        delta = float(np.hypot(a[2] - b[2], a[3] - b[3]))
        all_delta.append(delta)
        if a[1] == b[1]:
            same_delta.append(delta)
        if a[1] in tracked and b[1] in tracked:
            tracked_delta.append(delta)
    total_left = sum(left["sources"].values())
    total_right = sum(right["sources"].values())
    names = set(left["sources"]) | set(right["sources"])
    tvd = 0.5 * sum(
        abs(left["sources"].get(n, 0) / total_left - right["sources"].get(n, 0) / total_right) for n in names
    )
    return {
        "frames": len(pairs),
        "source_agreement": same_source / len(pairs),
        "tvd": tvd,
        "all_mean": float(np.mean(all_delta)) if all_delta else float("nan"),
        "all_p95": float(np.percentile(all_delta, 95)) if all_delta else float("nan"),
        "all_median": float(np.median(all_delta)) if all_delta else float("nan"),
        "same_mean": float(np.mean(same_delta)) if same_delta else float("nan"),
        "same_p95": float(np.percentile(same_delta, 95)) if same_delta else float("nan"),
        "same_median": float(np.median(same_delta)) if same_delta else float("nan"),
        "trk_mean": float(np.mean(tracked_delta)) if tracked_delta else float("nan"),
        "trk_p95": float(np.percentile(tracked_delta, 95)) if tracked_delta else float("nan"),
        "trk_median": float(np.median(tracked_delta)) if tracked_delta else float("nan"),
        "trk_n": len(tracked_delta),
        "base_ms": left["timing_ms"]["mean"],
        "new_ms": right["timing_ms"]["mean"],
        "new_p95_ms": right["timing_ms"]["p95"],
    }


def report(label, base_tag, new_tag):
    base, new = load(base_tag), load(new_tag)
    rows = []
    for key in sorted(base):
        if key in new:
            row = stats(base[key], new[key])
            if row:
                rows.append((key, row))
    print(f"\n===== {label}: {base_tag} -> {new_tag}  ({len(rows)} 段) =====")
    header = (
        f"{'tier':5s} {'video':40s} {'src_agr':>7s} {'tvd':>6s} "
        f"{'all_mdn':>8s} {'all_mean':>9s} {'all_p95':>8s} "
        f"{'trk_mdn':>8s} {'trk_mean':>9s} {'trk_p95':>8s} {'trk_n':>6s} {'ms':>14s}"
    )
    print(header)
    for (tier, video), r in rows:
        print(
            f"{tier:5s} {video[:40]:40s} {r['source_agreement']:7.3f} {r['tvd']:6.3f} "
            f"{r['all_median']:8.3f} {r['all_mean']:9.3f} {r['all_p95']:8.2f} "
            f"{r['trk_median']:8.3f} {r['trk_mean']:9.3f} {r['trk_p95']:8.2f} {r['trk_n']:6d} "
            f"{r['base_ms']:6.1f}->{r['new_ms']:6.1f}"
        )
    aggregate = {}
    for field in ("source_agreement", "tvd", "all_median", "all_mean", "all_p95", "trk_median", "trk_mean", "trk_p95"):
        values = np.asarray([r[field] for _, r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            aggregate[field] = (float(np.min(values)), float(np.median(values)), float(np.max(values)))
    print("  -- 分布 (min / median / max) --")
    for field, (lo, mid, hi) in aggregate.items():
        print(f"     {field:16s} {lo:9.3f} {mid:9.3f} {hi:9.3f}")
    speedup = [r["base_ms"] / r["new_ms"] for _, r in rows if r["new_ms"]]
    if speedup:
        print(f"     {'speedup':16s} {min(speedup):9.3f} {float(np.median(speedup)):9.3f} {max(speedup):9.3f}")
    return rows


def main():
    report("A 纯CPU换光流算法（对照噪声下界）", "baseline", "n3facade")
    report("B 同代码换数组库+设备（对照噪声）", "n3facade", "p3torch")
    report("C 端到端总差异（Phase 3 被检验对象）", "baseline", "p3torch")
    report("D 端到端总差异（含 CUDA Graph，Phase 4 最终口径）", "baseline", "p4graph")
    report("E 只差 CUDA Graph（硬不变量：应 Δ=0，仅提速）", "p3torch", "p4graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
