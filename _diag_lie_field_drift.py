# -*- coding: utf-8 -*-
"""用全场形状的位移中位数估计「地形漂移」，再从 17s 答案位置反推它在 9s 应该在哪。

动机：三条回溯路线都断了 —— ① t853 的锐利星形模板回溯到 t820 前匹配分掉到 0.08~0.35（答案在透明相位外观不同）；
② YOLO 在 t853 检不到答案（最近检测偏 57px，揭示相位已不是「透明形状」）；③ 探针 track 最早只到 t624。
但答案图形是画在地形上的，它与周围几十枚形状**一起被相机带着走**。所以不需要看见答案本身：
把周围所有形状的逐帧位移取中位数，就是这一帧的地形漂移 D(t)，反积分即可把答案位置送回 9s。

自检（必须过，否则模型不可信）：反推到 t776 应接近探针 #311 末端 (594.4,412.6)，
到 t702 应接近 #311 起点 (610.6,333.7)。这两个点是旋转期实测，独立于本模型。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_field_drift.py            # 首次会跑检测并落盘缓存
    LIE_DRIFT_RELOAD=1 ...                                                # 强制重跑检测
"""

import json  # 缓存与落盘。
import os  # 路径。
import sys  # 导入 src。

import cv2  # 读帧。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 允许从仓库根导入。

from src.liedetector.detector import TransparentShapeDetector  # YOLO 透明图形检测器。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 目标素材。
LO, HI = 360, 900  # 检测区间（覆盖 9s 前后与 17s 锚点）。
GATE = 30.0  # 形状接龙门限（全场形状间距 ~150px，漂移 ~2.5px/帧）。
MIN_LEN = 120  # 参与统计的最短轨迹（帧）。
ANCHOR_TICK = 853  # 17s 真值锚点。
ANCHOR_POS = np.array([606.4, 401.6])  # 锚点坐标。
CACHE = "_diag_out/yolo_dets_190829.json"  # 检测缓存。
CHECKS = {776: (594.4, 412.6), 702: (610.6, 333.7), 760: (593.7, 412.8)}  # 独立自检点（探针实测）。


def detections():  # 取（或生成）逐帧检测中心列表。
    if os.path.exists(CACHE) and not os.environ.get("LIE_DRIFT_RELOAD"):
        return {int(k): v for k, v in json.load(open(CACHE, encoding="utf-8")).items()}
    cap = cv2.VideoCapture(VIDEO)
    detector = TransparentShapeDetector()  # 一次会话。
    frames = {}  # tick -> [(x, y)]
    tick = 0
    while tick <= HI:
        ok, frame = cap.read()
        if not ok:
            break
        if tick >= LO:
            frames[tick] = [[float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0)]
                            for box, _score in detector.detect(frame)]
        tick += 1
    cap.release()
    json.dump({str(k): v for k, v in frames.items()}, open(CACHE, "w", encoding="utf-8"))  # 落盘。
    return frames


def build_tracks(frames):  # 逐帧最近邻接龙成轨迹（只做正向单链，够用来估漂移）。
    active = []  # 进行中的轨迹 [{'last': tick, 'pos': arr, 'pts': {}}]
    for tick in sorted(frames):  # 按时间推进
        dets = np.array(frames[tick], dtype=np.float64) if frames[tick] else np.zeros((0, 2))  # 本帧检测
        used = np.zeros(len(dets), dtype=bool)  # 本帧已被接走
        for tr in active:  # 先尝试延续
            if len(dets) == 0:
                continue
            if tick - tr["last"] > 3:  # 断太久，冻结
                continue
            dist = np.linalg.norm(dets - tr["pos"], axis=1)  # 距上帧位置
            order = np.argsort(dist)  # 由近及远
            for k in order:
                if used[k] or dist[k] > GATE:
                    break  # 最近的都超门限，后面更远
                used[k] = True
                tr["pts"][tick] = dets[k].copy()
                tr["pos"] = dets[k]
                tr["last"] = tick
                break
        for k in np.nonzero(~used)[0]:  # 剩下的开新轨迹
            active.append({"last": tick, "pos": dets[k], "pts": {tick: dets[k].copy()}})
    tracks = [tr["pts"] for tr in active if len(tr["pts"]) >= MIN_LEN]  # 只留长轨迹
    return tracks


def drift_per_tick(tracks):  # 每帧位移（中位数聚合）：tick -> (dx, dy)，表示从 tick 到 tick+1 的漂移。
    acc = {}  # tick -> [dx, dy] 列表
    for pts in tracks:
        ticks = sorted(pts)
        for a, b in zip(ticks, ticks[1:]):
            if b - a != 1:
                continue
            acc.setdefault(a, []).append(pts[b] - pts[a])
    out = {}
    for tick, vecs in acc.items():
        arr = np.array(vecs)  # 本帧所有形状位移
        out[tick] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), len(arr))  # 中位数 + 样本数
    return out


def main():
    frames = detections()  # 检测。
    tracks = build_tracks(frames)  # 轨迹。
    print(f"检测帧 {len(frames)}，长轨迹 {len(tracks)} 条（>={MIN_LEN} 帧）")
    drift = drift_per_tick(tracks)  # 漂移。
    # 从锚点反积分：truth(t) = ANCHOR - Σ_{s=t}^{852} D(s)
    truth = {ANCHOR_TICK: ANCHOR_POS.copy()}  # 锚点自身
    pos = ANCHOR_POS.copy()
    for t in range(ANCHOR_TICK - 1, LO - 1, -1):
        d = drift.get(t)  # t -> t+1 的漂移
        if d is not None:
            pos = pos - np.array([d[0], d[1]])
        truth[t] = pos.copy()
    print("自检（反推到探针实测点，误差应远小于 30px）：")
    ok_all = True
    for t, want in sorted(CHECKS.items()):
        got = truth.get(t)
        if got is None:
            print(f"   t{t} 无值")
            ok_all = False
            continue
        err = float(np.linalg.norm(got - np.array(want)))
        print(f"   t{t} 反推=({got[0]:6.1f},{got[1]:6.1f}) 探针=({want[0]:6.1f},{want[1]:6.1f}) 误差={err:6.1f}px")
        ok_all = ok_all and err < 40.0
    print("自检结论：", "通过，漂移模型可用于 9s" if ok_all else "不通过，模型不可信，禁止用作真值")
    print("逐 20 帧：tick 漂移(dx,dy,n) 反推答案位置")
    for t in range(LO, ANCHOR_TICK + 1, 20):
        d = drift.get(t)
        dtxt = "无" if d is None else f"({d[0]:+.2f},{d[1]:+.2f}) n={d[2]}"
        print(f"   t{t:4d} drift={dtxt:24s} truth=({truth[t][0]:6.1f},{truth[t][1]:6.1f})")
    json.dump({str(t): [float(v[0]), float(v[1])] for t, v in sorted(truth.items())},
              open("_diag_out/field_drift_truth.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
