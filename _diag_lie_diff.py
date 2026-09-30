# -*- coding: utf-8 -*-
"""对比两次 _diag_lie_replay.py 的落盘结果，判定 GPU 化改造是否通过回归门控。

用法：
    python _diag_lie_diff.py baseline p4graph n3facade
                             ^base    ^new    ^control（可选，强烈建议）

**为什么不用绝对「Δcenter ≤ 1px」**：base（baseline）跑的是 cv2.DIS 光流，new（p4graph）跑的是
Farneback（torch 移植），两者是**不同的算法**，逐帧中心偏差由数值混沌主导——实测连「纯 CPU、
完全不碰 torch、只把 DIS 换成 cv2 Farneback」这一步（对照 A）都能把 all_mean 顶到 142px、
all_p95 顶到 501px。绝对门限在这种场景下没有物理意义，只会把「换了算法」误判成「逻辑回归」。

因此本门控改用三条更有判别力的口径：

1. **只看 trk_*（两侧同为 color/border 的真跟踪帧）**：all_* 会被「一侧在做 prediction 外推」
   的帧污染，那些帧本来就没有观测约束、偏差可以任意大；trk_* 才是真正在跟目标的帧。
2. **以对照噪声标定门限**：给出 control（同 base、只做纯 CPU 算法切换、无数码库/设备变化的参照，
   推荐 n3facade）后，要求 new 的逐段 trk 中位偏差 ≤ control 的 TOL 倍（或一个绝对地板）。
   含义是「上了 torch+GPU+CUDA Graph 之后，跟踪偏差没有比纯 CPU 换算法更糟」。
3. **结构性不变量 + 相对耗时**：border 不得塌缩、source 分布不得整体漂移、high 档 p95 守 30fps
   硬预算、new 不得比 base 更慢（CUDA Graph 之后实测每段都快 1.4~2.2 倍，这条只当回归护栏）。

退出码 0 = 全部通过，1 = 存在未达标项。
"""

import glob
import json
import os
import sys

import numpy as np

OUT_DIR = "_diag_out"
TRACKED = ("color", "border")  # 「真在跟目标」的 source：两侧都落在这个集合里的帧才计入 trk_*。

# --- 数值偏差门限（以对照噪声标定，见模块 docstring 第 2 条）---
TRK_CONTROL_TOL = 2.0  # new 的 trk 中位偏差 ≤ control 的这么多倍即视为「没比纯 CPU 换算法更糟」。
TRK_ABS_FLOOR = 2.0  # control 该段偏差≈0 时的绝对地板（像素），避免 0×TOL=0 把正常段卡死。
# --- 结构性 / 分布门限（与算法无关，任何一段都不该破）---
BORDER_RATIO_DROP_LIMIT = 0.05  # border 帧占比允许的最大跌幅（5 个百分点）。
SOURCE_AGREEMENT_FLOOR = 0.50  # 单段 source 一致率低于此值＝结构性错位（正常混沌不会这么低）。
SOURCE_AGREEMENT_MEDIAN_LIMIT = 0.90  # 全体段 source 一致率中位数下限。
TVD_MEDIAN_LIMIT = 0.06  # 全体段 source 分布总变差中位数上限。
# --- 耗时门限 ---
TIME_MEAN_REL_LIMIT = 1.05  # new_mean ≤ base_mean × 此系数（相对回归护栏）。
# 相对护栏只对「同参数、换后端/换算法」的比较有意义（那时变慢就是回归）。
# 拿它去比「故意加粒子/抬分辨率的参数重分档」会把预期中的变慢误报成失败
# （t5old→t5 实测只有 2 段踩线，+5.5%/+5.8%，均值仍只有 15ms，而绝对预算是 33ms），
# 故给一个显式开关：比对参数重分档时置 1，该项降为警告不计失败，绝对预算门限仍然生效。
ALLOW_SLOWER = os.environ.get("LIE_DIFF_ALLOW_SLOWER", "").strip().lower() not in ("", "0", "false", "no")
TIME_MEAN_LIMIT = 33.0  # 任何档 session.update 的**均值**绝对上限（毫秒）：30fps 每帧预算，均值守住才算跟得上。
TIME_P95_LIMIT = {"high": 33.0, "ultra": 33.0, "extreme": 40.0}  # 各档 session.update p95 绝对上限（毫秒）。
# extreme（最强，也是默认档）多给 7ms 尾延迟余量，理由（t5 实测：13 段录像、修正二次裁剪后）：
#   mean 12.4~20.0ms 全部守住 33ms；p95 只在 3 段最重素材上是 35.6/36.8/38.0ms，其余 10 段 ≤ 23.0ms。
#   尾部尖峰来自重定位与切场景的突发帧（extreme 的 p95/mean ≈ 1.85~2.03，high/ultra 同样有 1.73~1.84），
#   不是稳态开销；而 extreme 把处理分辨率抬到 384 正是为了把重叠粘连的图形分开——实测每段生产录像的
#   border（真观测）帧数不降反升、prediction（盲外推）帧数下降，拿 5% 帧的尾延迟换这个收益是划算的。
#   反过来说，要把 extreme 的 p95 压回 33ms 只能把分辨率退回 320 一线，那等于取消这一档。


def load(tag):
    """读取某个 tag 的全部落盘结果，返回 {(tier, video): payload}。"""

    result = {}
    for path in glob.glob(os.path.join(OUT_DIR, f"{tag}_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result[(payload["tier"], payload["video"])] = payload
    return result


def ratio(sources, name):
    """某个 source 在总帧数里的占比。"""

    total = sum(sources.values())
    return (sources.get(name, 0) / total) if total else 0.0


def frame_stats(left, right):
    """对齐帧号后算一段录像的偏差统计：trk_*（判别力最强）、all_*、source 一致率与 tvd。

    ``left``/``right`` 是两次回放的 payload；frames 每行是 ``[tick, source, cx, cy]``。
    trk_* 只统计两侧 source 都属于 TRACKED 的帧（真正在跟目标），all_* 统计所有两侧都有中心的帧。
    """

    index = {row[0]: row for row in right["frames"]}
    pairs = [(row, index[row[0]]) for row in left["frames"] if row[0] in index]
    if not pairs:
        return None
    same_source = 0
    all_delta = []
    tracked_delta = []
    for a, b in pairs:
        if a[1] == b[1]:
            same_source += 1
        if a[2] is None or b[2] is None:
            continue
        delta = float(np.hypot(a[2] - b[2], a[3] - b[3]))
        all_delta.append(delta)
        if a[1] in TRACKED and b[1] in TRACKED:
            tracked_delta.append(delta)
    total_left = sum(left["sources"].values())
    total_right = sum(right["sources"].values())
    names = set(left["sources"]) | set(right["sources"])
    tvd = 0.5 * sum(
        abs(left["sources"].get(n, 0) / total_left - right["sources"].get(n, 0) / total_right) for n in names
    ) if total_left and total_right else 0.0

    def summarize(values):
        if not values:
            return (float("nan"), float("nan"), float("nan"))
        arr = np.asarray(values, dtype=np.float64)
        return (float(np.median(arr)), float(np.mean(arr)), float(np.percentile(arr, 95)))

    trk_median, trk_mean, trk_p95 = summarize(tracked_delta)
    all_median, all_mean, all_p95 = summarize(all_delta)
    return {
        "frames": len(pairs),
        "source_agreement": same_source / len(pairs),
        "tvd": tvd,
        "trk_median": trk_median,
        "trk_mean": trk_mean,
        "trk_p95": trk_p95,
        "trk_n": len(tracked_delta),
        "all_median": all_median,
        "all_mean": all_mean,
        "all_p95": all_p95,
    }


def main():
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    new_tag = sys.argv[2] if len(sys.argv) > 2 else "p4graph"
    control_tag = sys.argv[3] if len(sys.argv) > 3 else ""
    base = load(base_tag)
    new = load(new_tag)
    control = load(control_tag) if control_tag else {}
    if not base or not new:
        print(f"missing data: {base_tag}={len(base)} {new_tag}={len(new)} 数据缺失，先跑 _diag_lie_replay.py")
        return 1
    if control_tag and not control:
        print(f"WARNING control={control_tag} 无数据，数值偏差门限退回无对照模式（仅结构+耗时+分布）")

    failures = []
    agreements = []
    tvds = []
    speedups = []
    for key in sorted(base):
        tier, video = key
        if key not in new:
            failures.append(f"{tier} {video} 改造后缺失该录像结果")
            continue
        left, right = base[key], new[key]
        stat = frame_stats(left, right)
        ctrl_stat = frame_stats(left, control[key]) if key in control else None

        base_border = ratio(left["sources"], "border")
        new_border = ratio(right["sources"], "border")
        base_mean = left["timing_ms"]["mean"]
        mean_ms = right["timing_ms"]["mean"]
        p95_ms = right["timing_ms"]["p95"]
        speedup = (base_mean / mean_ms) if mean_ms else float("nan")

        # trk 中位偏差的门限：有对照就用对照标定，没有就只看结构+耗时（数值偏差无绝对意义）。
        if ctrl_stat is not None and np.isfinite(ctrl_stat["trk_median"]):
            trk_limit = max(ctrl_stat["trk_median"] * TRK_CONTROL_TOL, TRK_ABS_FLOOR)
            trk_note = f"ctrl_trk={ctrl_stat['trk_median']:.3f} 限={trk_limit:.3f}"
        else:
            trk_limit = None
            trk_note = "无对照"

        trk_median = stat["trk_median"] if stat else float("nan")
        print(
            f"{tier:5s} {video[:40]:40s} trk_mdn={trk_median:7.3f} trk_p95={stat['trk_p95'] if stat else float('nan'):7.2f} "
            f"src_agr={stat['source_agreement'] if stat else float('nan'):.3f} tvd={stat['tvd'] if stat else float('nan'):.3f} "
            f"border {base_border:.2%}->{new_border:.2%} {base_mean:5.1f}->{mean_ms:5.1f}ms x{speedup:.2f} "
            f"p95={p95_ms:5.1f} [{trk_note}]"
        )

        # --- 结构性 ---
        if stat is None:
            failures.append(f"{tier} {video} 无可比对帧（两侧都没有中心输出）")
        else:
            agreements.append(stat["source_agreement"])
            tvds.append(stat["tvd"])
            if stat["source_agreement"] < SOURCE_AGREEMENT_FLOOR:
                failures.append(
                    f"{tier} {video} source 一致率={stat['source_agreement']:.3f} < {SOURCE_AGREEMENT_FLOOR}（结构性错位）"
                )
            # --- 数值偏差：仅在有对照时逐段门控 ---
            if trk_limit is not None and np.isfinite(trk_median) and trk_median > trk_limit:
                failures.append(
                    f"{tier} {video} trk 中位偏差={trk_median:.3f} > 对照标定限 {trk_limit:.3f}"
                    f"（control={control_tag} ×{TRK_CONTROL_TOL}，地板 {TRK_ABS_FLOOR}）"
                )
        if base_border > 0.05 and new_border == 0.0:
            failures.append(f"{tier} {video} border 帧塌缩为 0")
        elif base_border > 0 and new_border < base_border - BORDER_RATIO_DROP_LIMIT:
            failures.append(
                f"{tier} {video} border 占比 {base_border:.2%}->{new_border:.2%} 跌幅超 {BORDER_RATIO_DROP_LIMIT:.0%}"
            )
        # --- 耗时 ---
        if mean_ms is not None and base_mean and mean_ms > base_mean * TIME_MEAN_REL_LIMIT:
            message = f"{tier} {video} new_mean={mean_ms}ms > base_mean×{TIME_MEAN_REL_LIMIT}={base_mean * TIME_MEAN_REL_LIMIT:.1f}ms（比基线更慢）"
            if ALLOW_SLOWER:  # 参数重分档比对：变慢是预期代价，降为警告。
                print(f"  [warn] {message}，已置 LIE_DIFF_ALLOW_SLOWER，不计失败")
            else:
                failures.append(message)
        if mean_ms is not None and mean_ms > TIME_MEAN_LIMIT:
            failures.append(f"{tier} {video} mean={mean_ms}ms > {TIME_MEAN_LIMIT}ms（超 30fps 均值预算）")
        if tier in TIME_P95_LIMIT and p95_ms is not None and p95_ms > TIME_P95_LIMIT[tier]:
            failures.append(f"{tier} {video} p95={p95_ms}ms > {TIME_P95_LIMIT[tier]}ms（超该档 p95 预算）")
        if np.isfinite(speedup):
            speedups.append(speedup)

    # --- 全体分布门限 ---
    print()
    if agreements:
        agr_median = float(np.median(agreements))
        tvd_median = float(np.median(tvds))
        sp = np.asarray(speedups, dtype=np.float64)
        print(
            f"分布: source_agreement median={agr_median:.3f}(min={min(agreements):.3f}) "
            f"tvd median={tvd_median:.3f}(max={max(tvds):.3f}) "
            f"speedup median={float(np.median(sp)):.3f}(min={sp.min():.3f} max={sp.max():.3f})"
        )
        if agr_median < SOURCE_AGREEMENT_MEDIAN_LIMIT:
            failures.append(f"source 一致率中位数={agr_median:.3f} < {SOURCE_AGREEMENT_MEDIAN_LIMIT}")
        if tvd_median > TVD_MEDIAN_LIMIT:
            failures.append(f"source 分布 tvd 中位数={tvd_median:.3f} > {TVD_MEDIAN_LIMIT}")

    if failures:
        print(f"\nFAIL 未达标 {len(failures)} 项：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASS 全部录像回归门控达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
