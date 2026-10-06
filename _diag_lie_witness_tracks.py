# -*- coding: utf-8 -*-
"""旋转见证器「内部轨迹」取证：把生产 RotationWitness 每个检查点的全部团块轨迹（含碎片）原样抄出来，
定位真目标（右下角旋转星形）在生产节奏下到底是「没分割出来 / 关联断裂 / 估不出 ω / 被稀释」哪一种。

背景（实测）：诊断探针 _diag_lie_rotation.py 在 20261001_190829 上的唯一 winner 是
lag12 档 #311：birth=702 death=776 samples=29 end=(594.4,412.6)（区域全尺度） rate=109.7°/2s dir=0.793，
即真目标在**第 702~776 帧**这一段才转。但生产回放在同一时段的 top-1 候选一直是
(660.8,249.9)cum=14.5 → (705,53)cum=12 → (636,365)cum=30.4，**右下角真目标从未进过 top-1**，
所以重锚（t617/647/677）锚到的是别的东西。本脚本用来判定差距的成因。

两条关键口径差（脚本会把它们显式量化出来）：
- 探针**逐帧**关联（lag=12 的光流每帧一次，samples 每帧 +1），生产**每 interval(=5) 帧**才 update 一次，
  于是 min_track_samples=6 在生产侧意味着「必须跨 30 帧持续关联成功」，而 5 帧一跳的质心位移要过
  assoc_gate=16px（处理尺度）这道门；残差星环碎裂+漂移时很容易断成一条又一条新轨迹。
- 生产的 windowed_deg = |累计带符号角| × 60 / max(age,12)：碎片轨迹 age 小看似有利，但 samples<6 直接不进 query()。

用法：
    .venv\\Scripts\\python.exe _diag_lie_witness_tracks.py                 # 默认 190829 + extreme + 全长
    .venv\\Scripts\\python.exe _diag_lie_witness_tracks.py 190829 extreme 600 935   # 只取该 tick 区间的快照

输出（_diag_out/）：
    witness_tracks_<tag>.json   逐检查点的轨迹全集 + 汇总判定
    witness_strip_<tag>.png     真目标旋转期（默认 700~776）的原始帧横条带，肉眼核对哪个星形在转
"""

import glob  # 按素材名过滤录像。
import json  # 快照与汇总落盘。
import os  # 路径拼接。
import sys  # 命令行参数与仓库根导入。
import time  # 计时。

import cv2  # 读录像、拼图。
import numpy as np  # 距离与统计。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

RECORD_DIR = "lie_records"  # 生产录像目录（mp4 已是坐标框裁剪结果，整帧喂入）。
OUT_DIR = "_diag_out"  # 与回放脚本同目录，便于横向比对。
TARGET_FULL = (594.4, 412.6)  # 探针 #311 的轨迹终点（区域全尺度）：肉眼与统计都对这里。
TARGET_RADIUS_FULL = 45.0  # 判定「落在真目标附近」的半径（区域全尺度）。
ROTATE_WINDOW = (702, 776)  # 探针测到的真目标旋转帧区间（含透明相位）。
MAX_SNAPSHOT_TICKS = 4000  # 快照上限（全长 935 帧足够）。


def find_video(name_filter):  # 名字含过滤串的录像，多个取最新。
    paths = sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4")))
    hits = [p for p in paths if name_filter in os.path.basename(p)]
    return hits[-1] if hits else None


def dump_tracks(witness, scale, tick):  # 抄出见证器当前全部轨迹（含已关掉的碎片），坐标换回区域全尺度。
    rows = []  # 结果行。
    for track in witness.tracks:  # 逐条。
        age = float(track.last_seen - track.birth)  # 存在时长（帧）。
        cand = track.candidate(rate_window=witness.rate_window)  # 对外口径（含 windowed/score）。
        rows.append({
            "id": track.id,  # 轨迹编号（递增即代表又新建了一条碎片）。
            "birth": round(float(track.birth), 1),  # 建档时刻。
            "last": round(float(track.last_seen), 1),  # 最后关联时刻。
            "age": round(age, 1),  # 年龄。
            "frames": track.frames,  # 关联成功次数（每检查点最多 +1）。
            "missed": round(witness.tick - track.last_seen, 1),  # 距今多久没被关联上。
            "samples": track.samples,  # ω 采样数。
            "center": [round(float(track.center[0]) / scale, 1), round(float(track.center[1]) / scale, 1)],
            "signed": round(float(track.signed_angle), 1),  # 累计带符号转角。
            "windowed": round(float(cand.windowed_deg), 1),  # 折算 60 帧窗内转角。
            "dir": round(float(cand.direction_consistency), 3),  # 方向一致性。
            "score": round(float(cand.score), 1),  # 质量分。
            "live": witness.tick - track.last_seen <= witness.lost_frames,  # 是否仍存活。
        })
    rows.sort(key=lambda item: item["score"], reverse=True)  # 质量分降序，与 query() 同排序。
    return rows


def dump_clusters(witness, scale):  # 抄出聚合后的证据簇（与 query()/rotation_at() 完全同一口径），坐标换回全尺度。
    rows = []  # 结果。
    clusters = witness._clusters()  # 簇候选（未过门槛）。
    # 成员数：再算一次贪心代价高，改用近似——簇内成员轨迹与簇质心距离在半径内且同向同速，够用且不改生产代码。
    for cand in clusters:
        members = [
            t for t in witness._live_tracks()
            if t.samples >= 1
            and float(np.linalg.norm(t.center - cand.center)) <= max(witness.cluster_radius, 1.0)
            and (t.signed_angle >= 0.0) == (cand.signed_angle >= 0.0)
        ] if witness.cluster_enabled else []  # 逐轨迹口径下无成员概念。
        rows.append({
            "center": [round(float(cand.center[0]) / scale, 1), round(float(cand.center[1]) / scale, 1)],
            "samples": int(cand.samples),  # 合计采样数。
            "age": round(float(cand.age), 1),  # 有效旋转时长。
            "signed": round(float(cand.signed_angle), 1),  # 同向累加转角。
            "windowed": round(float(cand.windowed_deg), 1),  # 折算窗内转角。
            "dir": round(float(cand.direction_consistency), 3),  # 方向一致性。
            "score": round(float(cand.score), 1),  # 质量分。
            "reported": bool(cand.samples >= witness.min_track_samples and cand.age >= witness.min_track_age),  # 能否进 query。
            "members": len(members),  # 邻近同向轨迹数（近似成员数）。
        })
    rows.sort(key=lambda item: item["score"], reverse=True)  # 质量分降序。
    return rows


def main():
    name_filter = sys.argv[1] if len(sys.argv) > 1 else "190829"  # 素材过滤。
    tier = sys.argv[2] if len(sys.argv) > 2 else "extreme"  # 精度档。
    tick_from = int(sys.argv[3]) if len(sys.argv) > 3 else ROTATE_WINDOW[0] - 120  # 快照起始。
    tick_to = int(sys.argv[4]) if len(sys.argv) > 4 else ROTATE_WINDOW[1] + 120  # 快照结束。
    path = find_video(name_filter)
    if path is None:
        print(f"no video matched: {name_filter}")
        return 1
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {path}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 容器帧率（本素材实测 50.2，注意秒/帧换算）。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=tier), logger=None)  # 真实会话。
    interval = session.params.witness_check_interval  # 检查节奏（帧）。
    snapshots = []  # 逐检查点快照。
    strip_frames = []  # 真目标旋转期的原始帧（画横条带用）。
    scale = None  # 处理尺度系数（在首帧 reset 后取）。
    min_samples = 6  # ω 采样下限（RotationWitness.min_track_samples，会话侧未覆写故取默认）。
    last_witness_tick = -1.0  # 上一次快照时的见证器时钟（用于识别检查点）。
    tick = 0
    started = time.perf_counter()
    while tick < MAX_SNAPSHOT_TICKS:
        ok, frame = cap.read()
        if not ok:
            break
        if scale is None:
            session.reset(frame.shape[1], frame.shape[0])  # 建区域（会话内部按 max_process_side 定尺度）。
            scale = float(session.effective_scale)
            witness = getattr(session, "witness", None)  # 首帧就取到，顺带读真实采样下限。
            if witness is not None:
                min_samples = int(witness.min_track_samples)  # 与生产同一判据。
        result = session.update(frame, float(src_fps))  # 真实生产链路。
        witness = getattr(session, "witness", None)
        # 只在「本帧真的跑过一次检查点」时快照：witness.tick 只在检查点前进，比读私有计数稳。
        if witness is not None and witness.tick > last_witness_tick and tick_from <= tick <= tick_to:
            last_witness_tick = float(witness.tick)  # 记录已快照到的检查时钟。
            rows = dump_tracks(witness, scale, tick)
            snapshots.append({
                "tick": tick,
                "witness_tick": round(float(witness.tick), 1),
                "source": result.source,
                "conf": round(float(result.confidence), 3),
                "center": None if result.center is None else [round(float(result.center[0]), 1), round(float(result.center[1]), 1)],
                "next_id": witness.next_id,  # 累计建档数（增速=碎片化速度）。
                "live_count": sum(1 for r in rows if r["live"]),
                "reported_count": sum(1 for r in rows if r["samples"] >= min_samples and r["live"]),
                "near_target": [r for r in rows if r["live"] and np.hypot(r["center"][0] - TARGET_FULL[0], r["center"][1] - TARGET_FULL[1]) <= TARGET_RADIUS_FULL][:4],
                "clusters": dump_clusters(witness, scale)[:5],  # 聚合后的簇 top-5（含未过门槛的）。
                "top": rows[:5],
            })
        if ROTATE_WINDOW[0] - 6 <= tick <= ROTATE_WINDOW[1] + 6 and (tick - ROTATE_WINDOW[0]) % 8 == 0:
            strip_frames.append((tick, frame.copy()))  # 取旋转期代表帧。
        tick += 1
    cap.release()
    if scale is None:
        print("empty video")
        return 1

    # 汇总：真目标附近（右下）在旋转期到底有没有可信轨迹；对比 top-1 都是谁。
    near_hits = []  # 出现在真目标附近的轨迹快照行。
    for snap in snapshots:
        for row in snap["near_target"]:
            near_hits.append({"tick": snap["tick"], **row})
    best_near = max(near_hits, key=lambda item: item["score"], default=None)  # 最强的一条。
    max_id = snapshots[-1]["next_id"] if snapshots else 0  # 建档总数。
    live_max = max((snap["live_count"] for snap in snapshots), default=0)  # 同期存活峰值。
    summary = {
        "video": os.path.basename(path), "tier": tier, "src_fps": round(src_fps, 2), "scale": scale,
        "interval": interval, "min_track_samples": min_samples,
        "target_full": list(TARGET_FULL), "target_radius_full": TARGET_RADIUS_FULL,
        "rotate_window": list(ROTATE_WINDOW),
        "snapshots": len(snapshots),
        "track_ids_created": max_id, "peak_live_tracks": live_max,
        "near_target_rows": len(near_hits),
        "best_near_target": best_near,
        "top1_by_tick": [{"tick": s["tick"], "top": (s["top"][0]["center"] if s["top"] else None),
                          "windowed": (s["top"][0]["windowed"] if s["top"] else None),
                          "samples": (s["top"][0]["samples"] if s["top"] else None)} for s in snapshots],
        # 逐检查点的 top-1 簇（聚合口径）：看真目标到底有没有被聚成一个过门槛的候选、排第几。
        "cluster_top1_by_tick": [
            {"tick": s["tick"],
             "cen": (s["clusters"][0]["center"] if s["clusters"] else None),
             "windowed": (s["clusters"][0]["windowed"] if s["clusters"] else None),
             "samples": (s["clusters"][0]["samples"] if s["clusters"] else None),
             "members": (s["clusters"][0]["members"] if s["clusters"] else None),
             "reported": (s["clusters"][0]["reported"] if s["clusters"] else None),
             "near_truth": [
                 c["center"] for c in s["clusters"]
                 if np.hypot(c["center"][0] - TARGET_FULL[0], c["center"][1] - TARGET_FULL[1]) <= TARGET_RADIUS_FULL
             ][:2]} for s in snapshots
        ],
    }
    tag = os.path.splitext(os.path.basename(path))[0]
    out_json = os.path.join(OUT_DIR, f"witness_tracks_{tag}_{tier}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "snapshots": snapshots}, handle, ensure_ascii=False)

    if strip_frames:  # 拼横条带（等宽缩放后并排，标 tick）。
        h = 200
        tiles = []  # 单帧块。
        for frame_tick, frame in strip_frames:
            w = int(frame.shape[1] * h / frame.shape[0])  # 等宽比。
            tile = cv2.resize(frame, (w, h))  # 缩放。
            cv2.putText(tile, f"t{frame_tick}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 帧号。
            cv2.circle(tile, (int(TARGET_FULL[0]), int(TARGET_FULL[1])), 18, (0, 0, 255), 1)  # 真目标位置（帧坐标=区域全尺度）。
            tiles.append(tile)
        strip = cv2.hconcat(tiles)  # 横拼。
        out_png = os.path.join(OUT_DIR, f"witness_strip_{tag}_{tier}.png")
        cv2.imwrite(out_png, strip)
    else:
        out_png = None

    print(f"{tag} {tier} frames={tick} fps={src_fps:.1f} scale={scale:.3f} wall={time.perf_counter() - started:.1f}s")
    print(f"snapshots={summary['snapshots']} track_ids_created={summary['track_ids_created']} peak_live={summary['peak_live_tracks']} near_target_rows={summary['near_target_rows']}")
    print(f"best_near_target={json.dumps(best_near, ensure_ascii=False) if best_near else 'NONE 真目标半径内没有任何存活轨迹'}")
    print(f"strip -> {out_png}")
    print(f"json -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
