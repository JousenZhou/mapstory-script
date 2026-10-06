# -*- coding: utf-8 -*-
"""验收判据：17s（视频播放时间，= 帧 853）跟踪中心必须落在右下角那枚答案星形上。

真值来源（本会话定案，见 _diag_lie_br_corner.py / _diag_lie_star_track.py）：
t840 起右下角出现一枚边界异常锐利的五角星，与周围被水波糊掉的星形明显不同；以 t853 的星形中心
(606.4,401.6) 建 104×104 模板做增量跟踪，t841→t873 匹配分始终 0.81~1.00、轨迹平滑连续，
是这段素材里唯一可信的刚体真值。旋转窗（t702–776）的探针质心 (594,413) 与 t776 跟踪中心
(616,388) 都落在这枚星形 ±20px 内 —— 即答案从头到尾没离开过右下角。

反例（已排除的假真值）：全帧中位光流外推给出「答案左移 200px」，但整条累积曲线在 ±230px 之间
锯齿往复（周期约 440 帧），那是水波行进不是相机平移；t820 之前的模板检索得分只有 0.65（背景水平）。

用法：
    .venv\\Scripts\\python.exe _diag_lie_17s_check.py [tag ...]
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import numpy as np  # 数值。

OUT_DIR = "_diag_out"  # 回放产物目录。
VIDEO_KEY = "20261001_190829"  # 失败复现素材名过滤。
TIER = "extreme"  # 失败复现只用最强档。
TRUTH = {  # tick -> 答案星形中心（增量模板跟踪，匹配分见注释）。
    841: (632.0, 415.0),  # 0.824
    845: (623.0, 411.0),  # 0.888
    849: (613.0, 405.0),  # 0.948
    853: (606.4, 401.6),  # 1.000（视频播放 17.00s）
    857: (597.0, 395.0),  # 0.945
    861: (589.0, 388.0),  # 0.916
    865: (583.0, 381.0),  # 0.878
    869: (577.0, 374.0),  # 0.851
    873: (571.0, 367.0),  # 0.808
}
KEY_TICK = 853  # 硬验收帧：用户口径的「第 17s」。
PASS_PX = 30.0  # 达标半径（区域全尺度像素）：约 0.5 倍星形尺寸。
ROT_TRUTH = (594.4, 412.6)  # 旋转窗末（t776）探针 #311 质心，作旋转窗内的对照真值。
ROT_TICKS = (702, 720, 740, 760, 776)  # 旋转窗内抽查帧。


def load(tag):  # 取某 tag 的回放帧表。
    hits = glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_{VIDEO_KEY}_*.json"))  # 只匹配本素材。
    hits = [h for h in hits if f"_{TIER}_" in h]  # 排除 tag 前缀互相包含的情况。
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {r[0]: r for r in payload["frames"]}, payload


def main():
    tags = sys.argv[1:] or ["cluster", "anchor"]  # 默认对比：改前 vs 缰绳。
    print(f"判据：t{KEY_TICK}（视频 17.00s）跟踪中心距真值 {TRUTH[KEY_TICK]} ≤ {PASS_PX}px")
    for tag in tags:
        loaded = load(tag)
        if loaded is None:
            print(f"[MISS] {tag} 无回放产物，先跑 _diag_lie_replay.py {tag} {VIDEO_KEY} 1000")
            continue
        rows, payload = loaded
        ds, detail = [], []  # 距离与逐帧明细。
        for tick in sorted(TRUTH):
            row = rows.get(tick)
            if row is None or row[2] is None:
                ds.append(np.inf)  # 无输出记作无穷远。
                detail.append(f"t{tick}=--")
                continue
            d = float(np.hypot(row[2] - TRUTH[tick][0], row[3] - TRUTH[tick][1]))  # 全尺度像素距离。
            ds.append(d)
            detail.append(f"t{tick}=({row[2]:.0f},{row[3]:.0f})d{d:.0f}")
        key_row = rows.get(KEY_TICK)
        key_d = ds[sorted(TRUTH).index(KEY_TICK)]  # 17s 那一帧的距离。
        finite = [d for d in ds if np.isfinite(d)]
        median = float(np.median(finite)) if finite else np.inf
        # 旋转窗内的对照：跟踪中心离探针质心有多远（历史判据，防止缰绳把旋转窗内行为改坏）。
        rot = []
        for tick in ROT_TICKS:
            row = rows.get(tick)
            if row is not None and row[2] is not None:
                rot.append(float(np.hypot(row[2] - ROT_TRUTH[0], row[3] - ROT_TRUTH[1])))
        applied = sum(1 for r in payload["frames"] if len(r) > 8 and r[8])  # 见证器重锚帧数。
        sources = payload.get("sources", {})
        verdict = "PASS" if key_d <= PASS_PX else "FAIL"
        print(f"[{verdict}] {tag:10s} 17s距离={key_d:6.1f}  窗内median={median:6.1f}  "
              f"旋转窗距探针median={float(np.median(rot)) if rot else float('nan'):6.1f}  "
              f"重锚帧={applied} rotation={sources.get('rotation', 0)} prediction={sources.get('prediction', 0)}")
        print(f"         {' '.join(detail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
