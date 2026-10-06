# -*- coding: utf-8 -*-
"""验收判据：第 9s（t452）之后不得被烧录准星钉住，且全程单调往右下、17s 结论不回退。

缺陷本体（取证定案，见 _diag_lie_9s_scan.py / _diag_lie_9s_maskopt.py）：
    测谎回放画面里烧进了上一轮操作留下的绿色准星十字，它自己会动，是透明相位里最强的时序残差源
    （长基线相干积分：全场最强轨迹峰全落在准星位置带，比真值处高 3 个数量级）。现状口径下
    t452..612 共 160 帧里跟踪中心与准星中位只差 16.7px、100% 重合、连续不分离——即用户看到的
    「第 9s 丢失，然后平滑地跟着鼠标走」。

真值口径（本会话定案，弃用一切手搓折线）：
    自动重建 9s 窗逐帧真值的三条路线全部被实测证伪——地形中位漂移反积分自检误差 217~246px、
    YOLO 反向接龙只补上 98/303 帧且链上点自相矛盾（x 分量忽正忽负）、DP/Viterbi 解出的路径是
    完美直线 v=(0.8,0.8) 在 t702 偏实测 96px。因此这里只用三条可直接测量的无真值判据：
      ① 与准星分离（准星位置逐帧实测，绿色最大连通块质心）；
      ② 单调右下（x 回退帧占比 / y 上攻帧占比，屏幕 y 向下为正）；
      ③ 命中三个实测锚点 t702/t776（旋转窗探针质心）与 t853（揭示帧星形质心，17s 判据同源）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_check.py [tag ...]   # 默认 curmask nomask
    先跑 .venv\\Scripts\\python.exe -X utf8 _diag_lie_replay.py curmask 190829 1000
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 解码素材帧量准星
import numpy as np  # 数值

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材
OUT_DIR = "_diag_out"  # 回放产物目录
TIER = os.environ.get("NINE_S_TIER", "extreme").strip().lower() or "extreme"  # 只看这一档的回放产物（缺陷在最强档复现；高/极高档用 NINE_S_TIER 复核）
W9 = (452, 612)  # 9s 病灶窗：t452 = 容器 fps 50.207 下的第 9.00s
W_ALL = (452, 776)  # 单调性评估窗：9s 转向 → 抵达右下角
ANCHORS = {  # 实测锚点（区域全尺度）：t702/t776 为旋转窗探针质心，t853 为揭示帧星形质心。
    702: (610.6, 333.7),
    776: (594.4, 412.6),
    853: (606.4, 401.6),
}
OVERLAP_PX = 35.0  # 「与准星重合」判定半径（全尺度）：小于半个目标尺寸。
GREEN_MIN = 90  # 与生产掩膜同一绿色判据（G 下限）。
GREEN_DOMINANCE = 12  # G - max(R,B) 下限。
GREEN_MIN_PX = 15  # 少于此数视为本帧无准星。
# 达标线（对照实测基线：重合 100%、距准星中位 16.7px、x回退 26.2%、y上攻 25.9%、t853 7.4px）：
NEED_SEP_RATIO = 25.0  # 9s 窗内与准星重合的帧占比上限（%）。
NEED_SEP_MEDIAN = 60.0  # 9s 窗内距准星的中位下限（px）。
NEED_ANCHOR_PX = 60.0  # t702/t776 锚点命中半径（px）。
NEED_17S_PX = 30.0  # t853 锚点命中半径（px）：17s 已收官结论，不得回退。


def cursor_centroid(frame, scale):  # 本帧准星质心（区域全尺度）；无准星返回 None。
    proc = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame  # 缩到处理尺度（与生产同一口径）
    b = proc[:, :, 0].astype(np.int16)  # 拆通道：带符号相减防 uint8 回绕
    g = proc[:, :, 1].astype(np.int16)
    r = proc[:, :, 2].astype(np.int16)
    green = (((g > GREEN_MIN) & (g - np.maximum(r, b) > GREEN_DOMINANCE)).astype(np.uint8)) * 255  # 绿主导
    if int(cv2.countNonZero(green)) < GREEN_MIN_PX:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(green, 8)  # 连通块
    if count <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块＝准星本体
    return np.array([cents[k][0] / scale, cents[k][1] / scale])  # 换算回全尺度


def measure_cursors(need, scale):  # 一次性解码素材，量出 need 里每个帧号的准星质心。
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    top = max(need) if need else 0  # 只读到需要的帧
    out, tick = {}, 0  # 帧号 → 质心
    while tick <= top:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick in need:  # 命中需求帧
            got = cursor_centroid(frame, scale)  # 量准星
            if got is not None:
                out[tick] = got
        tick += 1
    cap.release()  # 释放
    return out


def load(tag):  # 取某 tag 的回放产物（帧表 + 元数据）；无则 None。
    hits = [h for h in glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_20261001_190829_*.json")) if f"_{TIER}_" in h]
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {r[0]: r for r in payload["frames"]}, payload


def center_at(rows, tick):  # 取某帧中心（全尺度）；无输出返回 None。
    row = rows.get(tick)
    if row is None or row[2] is None:
        return None
    return np.array([float(row[2]), float(row[3])])


def main():
    tags = sys.argv[1:] or ["curmask", "nomask"]  # 默认：开掩膜 vs 关掩膜
    print(f"判据（tier={TIER}）：9s 窗 t{W9[0]}..{W9[1]} 与准星分离（重合≤{NEED_SEP_RATIO}%、中位≥{NEED_SEP_MEDIAN}px）"
          f" + 锚点 {sorted(ANCHORS)} 命中 + t{W_ALL[0]}..{W_ALL[1]} 单调右下")
    fails = 0  # 未达标 tag 数
    for tag in tags:
        loaded = load(tag)
        if loaded is None:
            print(f"[MISS] {tag} 无回放产物，先跑 _diag_lie_replay.py {tag} 190829 1000")
            fails += 1
            continue
        rows, payload = loaded
        scale = float(payload.get("effective_scale", 0.5))  # 掩膜与量准星共用同一处理尺度
        need = set(range(W9[0], W9[1] + 1))  # 需要量准星的帧
        cursors = measure_cursors(need, scale)  # 逐帧准星质心
        ticks = [t for t in range(W9[0], W9[1] + 1) if t in rows and rows[t][2] is not None and t in cursors]
        dists = np.array([float(np.linalg.norm(center_at(rows, t) - cursors[t])) for t in ticks])  # 距准星
        overlap = float(np.mean(dists < OVERLAP_PX) * 100) if len(dists) else np.nan  # 重合占比
        median = float(np.median(dists)) if len(dists) else np.nan  # 距准星中位
        win = [t for t in range(W_ALL[0], W_ALL[1] + 1) if center_at(rows, t) is not None]  # 单调性窗
        pts = np.array([center_at(rows, t) for t in win])  # 依次位置
        back = float(np.mean(np.diff(pts[:, 0]) < -0.5) * 100) if len(pts) > 2 else np.nan  # x 回退占比
        up = float(np.mean(np.diff(pts[:, 1]) < -0.5) * 100) if len(pts) > 2 else np.nan  # y 上攻占比
        hit = {}  # 锚点命中距离
        for tick, pos in ANCHORS.items():
            c = center_at(rows, tick)
            hit[tick] = np.inf if c is None else float(np.linalg.norm(c - np.array(pos)))
        sources = payload.get("sources", {})
        ok = (overlap <= NEED_SEP_RATIO and median >= NEED_SEP_MEDIAN
              and hit[702] <= NEED_ANCHOR_PX and hit[776] <= NEED_ANCHOR_PX
              and hit[853] <= NEED_17S_PX and back <= 30.0 and up <= 30.0)
        verdict = "PASS" if ok else "FAIL"
        fails += 0 if ok else 1
        print(f"[{verdict}] {tag:9s} 9s重合={overlap:5.1f}% 距准星中位={median:6.1f}px  "
              f"x回退={back:5.1f}% y上攻={up:5.1f}%  "
              f"锚点 t702={hit[702]:6.1f} t776={hit[776]:6.1f} t853={hit[853]:6.1f}px  "
              f"rotation={sources.get('rotation', 0)} prediction={sources.get('prediction', 0)}")
    return 1 if fails else 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
