# -*- coding: utf-8 -*-
"""查清 9s 段跟踪中心「跟的是谁」：把画面里游戏自绘的绿色准星（鼠标叠加层）逐帧检出，
与回放跟踪中心、与白相真值、与右下真值线三方对照。

判据：
  1) 白相（color, t19..336）准星应贴在答案上（玩家用准星指着白星），用来验证准星检测本身可信；
  2) 若透明相 |跟踪中心 - 准星| 长期 < 15px，而跟踪中心同时偏离右下线，
     则说明证据层把 UI 叠加层当成了目标 —— 这是「自信地跟错」，也解释「有时跟得上有时跟不上」
     （准星偶尔恰好路过答案）。

输出：
    _diag_out/cursor_traj_190829.json   逐帧准星中心
    控制台对照表

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_cursor_track.py
"""

import glob  # 找回放产物。
import json  # 读写。
import os  # 路径。

import cv2  # 解码与形态学。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 目标素材（已是区域裁剪结果）。
OUT = "_diag_out/cursor_traj_190829.json"  # 准星轨迹缓存。
REPLAY = sorted(glob.glob("_diag_out/anchor2_extreme_20261001_190829_*.json"))[-1]  # 回放。
GREEN = np.array([0, 200, 0], dtype=np.float32)  # 游戏准星大致色相（BGR），用角度筛选更稳。


def cursor_center(frame):
    """检出本帧绿色准星中心；找不到返回 None。

    准星是叠加层，纯色高饱和绿，地形里没有这种颜色，色相环距离判据足够干净。
    """
    f = frame.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    mask = (g > 130) & (g - r > 45) & (g - b > 45)
    if mask.sum() < 20:
        return None
    m8 = (mask.astype(np.uint8)) * 255
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m8, 8)
    if n <= 1:
        return None
    k = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大连通域（准星本体）。
    if stats[k, cv2.CC_STAT_AREA] < 20:
        return None
    return [float(cent[k][0]), float(cent[k][1])], int(stats[k, cv2.CC_STAT_AREA])


def main():
    rows = {r[0]: r for r in json.load(open(REPLAY, encoding="utf-8"))["frames"]}  # 回放行。
    cap = cv2.VideoCapture(VIDEO)
    traj = {}  # tick -> [x,y]
    tick = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        got = cursor_center(frame)
        if got:
            traj[tick] = got[0]
        tick += 1
    cap.release()
    json.dump({str(k): v for k, v in traj.items()}, open(OUT, "w", encoding="utf-8"))
    print(f"解码 {tick} 帧，准星有效帧 {len(traj)}（{len(traj)/max(tick,1):.0%}）")

    def dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # 分段统计：按 source 分组，看跟踪中心到准星距离
    groups = {}
    for t, row in sorted(rows.items()):
        if row[2] is None or t not in traj:
            continue
        groups.setdefault(row[1], []).append(dist((row[2], row[3]), traj[t]))
    print("\n== 跟踪中心 → 准星 距离分布（按 source 分组）==")
    for src, ds in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        d = np.array(ds)
        print(f"{src:12s} n={len(d):4d} 中位={np.median(d):6.1f} P10={np.percentile(d,10):6.1f} "
              f"P90={np.percentile(d,90):6.1f} <15px占比={np.mean(d<15):5.0%}")

    # 逐帧细看 9s 前后
    print("\n== t430..640 逐帧：跟踪中心 / 准星 / 两者距离 / 准星逐帧位移 ==")
    prev_c = None
    for t in range(430, 641):
        row = rows.get(t)
        if row is None:
            continue
        c = traj.get(t)
        mv = "" if (c is None or prev_c is None) else f"准星mv={dist(c, prev_c):5.1f}"
        if c is not None:
            prev_c = c
        txt = "----" if c is None else f"({c[0]:6.1f},{c[1]:6.1f})"
        dd = "  -  " if c is None or row[2] is None else f"{dist((row[2], row[3]), c):5.1f}"
        print(f"t{t:4d} {row[1]:11s} 跟踪=({row[2]:6.1f},{row[3]:6.1f}) conf={row[4]:.2f} snr={row[5]:5.2f} "
              f"准星={txt} 距={dd} {mv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
