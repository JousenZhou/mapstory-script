# -*- coding: utf-8 -*-
"""「跟没跟到旋转真目标」自动判据：给若干 _diag_lie_replay.py 的落盘 tag 打分，秒级判定一次改动是否有效。

真值 = 探针 _diag_lie_rotation.py 唯一 winner #311 的**逐帧质心轨迹**（lag12_floor2.0，29 点，帧 702–776），
不是固定坐标。原因：本素材相机/图形自身在移动，#311 的质心从 (610.6,333.7) 走到 (594.4,412.6)，
写死单点会在大部分帧上误判（也曾在相位相关外推上栽过：周期性地形让配准锁错周期，自检误差 57px，弃用）。

时间口径：素材容器标注 50.207fps 是写入侧固定值，真实采集率约 45fps（录制固定帧率写变速采集的已知坑），
所以用户口中的「17 秒」≈ 帧 17×45 = **765**，正落在旋转窗末端、真值 (594,413) 即画面右下。
判据因此以旋转窗为主，并同时报告 45fps 与 30fps 两种秒口径下的距离。

用法：
    .venv\\Scripts\\python.exe _diag_lie_target_score.py cadence rollback minsamples
    .venv\\Scripts\\python.exe _diag_lie_target_score.py newtag --radius 45

退出码 0=达标，1=未达标。达标定义（全部满足）：
    ① 旋转窗（702–776）内到真值距离的中位数 ≤ 20px、p95 ≤ 45px；
    ② 17s（45fps 口径 = 帧 765）距离 ≤ 45px 且该帧 source 不是 prediction/scene-ended；
    ③ 旋转窗内命中率（≤45px）≥ 90%。
"""

import argparse  # 参数解析。
import glob  # 找产物。
import json  # 读 JSON。
import os  # 路径。

import numpy as np  # 统计。

OUT_DIR = "_diag_out"  # 回放产物目录。
TIER = "extreme"  # 判据只看失败复现档。
PROBE = os.path.join(OUT_DIR, "rotation_diag_extreme_20261001_190829_score1.00_extreme.mp4.json")  # 真值来源。
PROBE_COMBO = "lag12_floor2.0"  # 探针参数组合（生产 witness 的 span 与 lag12 同口径）。
PROBE_TRACK_ID = 311  # 唯一 winner 轨迹号。
FPS_TRUE = 45.0  # 估算的真实采集帧率（17s → 帧 765）。
FPS_LIVE = 30.0  # 生产供帧节拍（保守口径）。
WINDOW = (702, 776)  # 旋转窗（真值轨迹覆盖范围）。
MEDIAN_LIMIT = 20.0  # 窗内中位距离上限（px）。
P95_LIMIT = 45.0  # 窗内 p95 距离上限（px）。
HIT_RATIO_LIMIT = 0.9  # 窗内命中率下限。
BAD_SOURCES = ("prediction", "scene-ended")  # 这些来源视为未真正跟住。


def load_truth():  # 读探针 winner 的逐帧真值轨迹。
    with open(PROBE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for row in data["combos"][PROBE_COMBO]:  # 找目标轨迹。
        if row["id"] == PROBE_TRACK_ID and row.get("path"):
            points = [(float(t), float(x), float(y)) for t, x, y in row["path"]]  # 逐帧点。
            return points, row
    raise RuntimeError(f"track #{PROBE_TRACK_ID} path missing in {PROBE_COMBO}")


def truth_at(points, tick):  # 线性内插真值；窗外返回 None。
    if tick < points[0][0] or tick > points[-1][0]:
        return None  # 无证据。
    for index in range(len(points) - 1):  # 逐段找。
        t0, x0, y0 = points[index]  # 段起点。
        t1, x1, y1 = points[index + 1]  # 段终点。
        if t0 <= tick <= t1:  # 落在段内。
            ratio = 0.0 if t1 == t0 else (tick - t0) / (t1 - t0)  # 内插比例。
            return x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio  # 真值。
    return None


def load_tag(tag, video_filter):  # 取某 tag 下匹配素材的 extreme 回放 payload。
    payloads = []  # 结果。
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_*.json"))):
        if video_filter and video_filter not in os.path.basename(path):  # 素材过滤。
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payloads.append((os.path.basename(path), json.load(handle)))
    return payloads


def score(payload, points, radius):  # 对一个回放结果按真值轨迹打分。
    by_tick = {row[0]: row for row in payload["frames"]}  # 帧号索引。
    distances = []  # 窗内逐帧 (tick, dist, source, conf)。
    for tick in range(WINDOW[0], WINDOW[1] + 1):  # 旋转窗逐帧。
        truth = truth_at(points, tick)  # 真值。
        row = by_tick.get(tick)  # 该帧输出。
        if truth is None or row is None or row[2] is None:  # 缺证据或缺输出。
            continue
        distances.append((tick, float(np.hypot(row[2] - truth[0], row[3] - truth[1])), row[1], row[4]))  # 记录。
    if not distances:  # 完全没跟上（无输出）。
        return {"video": payload["video"], "covered": 0, "median": 9e9, "p95": 9e9, "hit_ratio": 0.0,
                "at_17s": "无输出", "pass": False, "worst": []}
    dist_array = np.array([d[1] for d in distances])  # 距离序列。
    hit_ratio = float((dist_array <= radius).mean())  # 命中率。
    median = float(np.median(dist_array))  # 中位数。
    p95 = float(np.percentile(dist_array, 95))  # p95。

    def at_second(second, fps):  # 某秒（按给定帧率）的距离与来源。
        tick = int(round(second * fps))
        truth = truth_at(points, tick)  # 真值。
        row = by_tick.get(tick) or by_tick.get(min(by_tick, key=lambda t: abs(t - tick)))  # 最近帧。
        if row is None or row[2] is None:
            return f"{second:.0f}s(t{tick}) 无输出"
        if truth is None:
            return f"{second:.0f}s(t{tick}) 真值窗外 cen=({row[2]:.0f},{row[3]:.0f}) {row[1]}"
        distance = float(np.hypot(row[2] - truth[0], row[3] - truth[1]))  # 距离。
        bad = row[1] in BAD_SOURCES  # 来源是否算没跟住。
        return (f"{second:.0f}s(t{tick}) d={distance:5.1f} cen=({row[2]:.0f},{row[3]:.0f}) "
                f"truth=({truth[0]:.0f},{truth[1]:.0f}) {row[1]} conf={row[4]:.2f}{'' if not bad else ' [来源不算跟住]'}")

    worst = sorted(distances, key=lambda d: -d[1])[:3]  # 最差三帧。
    passed = (median <= MEDIAN_LIMIT and p95 <= radius and hit_ratio >= HIT_RATIO_LIMIT
              and all(d[2] not in BAD_SOURCES for d in distances if d[1] <= radius))
    return {
        "video": payload["video"], "covered": len(distances), "median": median, "p95": p95,
        "max": float(dist_array.max()), "hit_ratio": hit_ratio,
        "at_17s": at_second(17.0, FPS_TRUE), "at_17s_30": at_second(17.0, FPS_LIVE),
        "at_window_end": at_second(WINDOW[1] / FPS_TRUE, FPS_TRUE),
        "worst": [(int(d[0]), round(d[1], 1), d[2]) for d in worst],  # 最差帧明细。
        "applied": [row[0] for row in payload["frames"] if len(row) > 8 and row[8]],  # 重锚帧。
        "pass": passed,
    }


def main():  # 入口。
    parser = argparse.ArgumentParser()  # 解析器。
    parser.add_argument("tags", nargs="+")  # 一个或多个回放 tag。
    parser.add_argument("--video", default="190829")  # 素材过滤串。
    parser.add_argument("--radius", type=float, default=45.0)  # 命中半径（全尺度像素）。
    args = parser.parse_args()  # 解析。
    points, meta = load_truth()  # 真值。
    print(f"truth: track#{meta['id']} {len(points)} pts t{int(points[0][0])}–{int(points[-1][0])} "
          f"({points[0][1]:.0f},{points[0][2]:.0f})→({points[-1][1]:.0f},{points[-1][2]:.0f}) "
          f"rate={meta['rate_signed_deg_per_2s']}°/2s dir={meta['dir']}  radius={args.radius:.0f}")
    worst_result = 0  # 最差结论。
    for tag in args.tags:  # 逐 tag。
        payloads = load_tag(tag, args.video)  # 找产物。
        if not payloads:
            print(f"[NODATA] {tag}: 无 {TIER} 档产物（先跑 _diag_lie_replay.py {tag} {args.video} 1000）")
            worst_result = max(worst_result, 1)  # 缺数据算失败。
            continue
        for name, payload in payloads:  # 逐素材。
            s = score(payload, points, args.radius)  # 打分。
            worst_result = max(worst_result, 0 if s["pass"] else 1)  # 汇总。
            print(f"[{'PASS' if s['pass'] else 'FAIL'}] {tag} {name}")
            if s["covered"] == 0:  # 完全没跟上。
                print("   旋转窗内无任何跟踪输出")
                continue
            print(f"   窗内覆盖={s['covered']}帧 median={s['median']:.1f}px p95={s['p95']:.1f}px "
                  f"max={s['max']:.1f}px hit(≤{args.radius:.0f})={s['hit_ratio']:.0%}")
            print(f"   17s@45fps: {s['at_17s']}")
            print(f"   17s@30fps: {s['at_17s_30']}")
            print(f"   窗末     : {s['at_window_end']}")
            print(f"   最差帧={s['worst']} applied={s.get('applied')}")
    print(f"\n结论: {'达标' if worst_result == 0 else '未达标'}"
          f"（判据 median≤{MEDIAN_LIMIT} p95≤{P95_LIMIT} hit≥{HIT_RATIO_LIMIT:.0%}）")
    return worst_result


if __name__ == "__main__":
    raise SystemExit(main())
