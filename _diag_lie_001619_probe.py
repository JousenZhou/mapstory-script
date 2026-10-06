# -*- coding: utf-8 -*-
"""定案「掩膜开/关逐帧偏差显著」的素材：被遮的那个最大绿色连通块到底是谁（UI 准星 or 目标本体）。

判据（对着「准星足迹掩膜只该遮 UI 叠加层」这条设计约束）：
  1) 颜色纯度：准星是叠加层，纯绿高饱和（g 大、r/b 低）；场景里的绿色通常偏暗、主导度小。
  2) 形状：准星是细十字 → 填充率（面积 / 外接框面积）低、外接框接近正方形；团块状物填充率高。
  3) 与两次回放的中心关系：
     - 绿块长期贴 curoff（关掩膜）中心 ⇒ 关掩膜时本来就在跟准星，掩膜把它推开是修复；
     - 绿块长期贴 curon（开掩膜）中心 ⇒ 掩膜把目标自己挖了，是回归，需要收紧判据。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_001619_probe.py [name_key] [tag_on] [tag_off]
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 解码与连通块。
import numpy as np  # 数值。

OUT_DIR = "_diag_out"  # 回放产物目录。
RECORD_DIR = "lie_records"  # 生产录像目录。
CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"  # 外部参考视频。
TIER = "extreme"  # 只查这个档。
GREEN_MIN = 90  # 与生产同口径。
GREEN_DOM = 12


def load(tag, name):  # 读某 tag 某素材的回放产物 → {tick: row}。
    hits = [h for h in glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_{name}.json")) if f"_{TIER}_" in h]
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        return {r[0]: r for r in json.load(handle)["frames"]}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "20260930_001619_score1.00_ultra.mp4"
    tag_on = sys.argv[2] if len(sys.argv) > 2 else "curon"
    tag_off = sys.argv[3] if len(sys.argv) > 3 else "curoff"
    path = os.path.join(RECORD_DIR, name)
    if not os.path.exists(path):
        path = os.path.join(CASE_DIR, name)
    on, off = load(tag_on, name), load(tag_off, name)
    if on is None or off is None:
        print(f"[MISS] 缺回放产物：{tag_on}/{tag_off} {name}")
        return 1
    hi = max(max(on), max(off))
    cap = cv2.VideoCapture(path)
    print(f"素材 {name}｜比对 {tag_on}(遮) vs {tag_off}(不遮)｜帧 0..{hi}")
    print(f"{'t':>5} {'绿块质心(全尺度)':>20} {'面积':>6} {'w':>5} {'h':>5} {'填充':>5} "
          f"{'g均':>5} {'主导':>5} {'遮中心':>16} {'不遮中心':>16} {'遮-绿':>7} {'不遮-绿':>7}")
    tick = 0
    stats_rows = []  # 汇总用
    eff_scale = None
    while tick <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if eff_scale is None:  # 用回放里第一帧的处理尺度反推 scale
            j = json.load(open(sorted(glob.glob(os.path.join(OUT_DIR, f"{tag_on}_{TIER}_{name}.json")))[-1],
                               encoding="utf-8"))
            eff_scale = float(j.get("effective_scale", 0.5))
        proc = cv2.resize(frame, None, fx=eff_scale, fy=eff_scale)  # 处理尺度
        b = proc[:, :, 0].astype(np.int16)
        g = proc[:, :, 1].astype(np.int16)
        r = proc[:, :, 2].astype(np.int16)
        green = ((g > GREEN_MIN) & (g - np.maximum(r, b) > GREEN_DOM)).astype(np.uint8) * 255
        n, labels, st, cent = cv2.connectedComponentsWithStats(green, 8)
        row_on, row_off = on.get(tick), off.get(tick)
        if n > 1:
            k = int(1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
            area = int(st[k, cv2.CC_STAT_AREA])
            bw, bh = int(st[k, cv2.CC_STAT_WIDTH]), int(st[k, cv2.CC_STAT_HEIGHT])
            fill = area / float(max(1, bw * bh))
            cx, cy = float(cent[k][0]) / eff_scale, float(cent[k][1]) / eff_scale
            gm = float(g[labels == k].mean())
            dm = float((g - np.maximum(r, b))[labels == k].mean())
            don = None if row_on is None or row_on[2] is None else float(np.hypot(row_on[2] - cx, row_on[3] - cy))
            doff = None if row_off is None or row_off[2] is None else float(np.hypot(row_off[2] - cx, row_off[3] - cy))
            tc = f"({row_on[2]:6.1f},{row_on[3]:6.1f})" if row_on and row_on[2] is not None else "     none    "
            tf = f"({row_off[2]:6.1f},{row_off[3]:6.1f})" if row_off and row_off[2] is not None else "     none    "
            print(f"{tick:5d} {f'({cx:7.1f},{cy:7.1f})':>20} {area:6d} {bw:5d} {bh:5d} {fill:5.2f} "
                  f"{gm:5.0f} {dm:5.0f} {tc:>16} {tf:>16} "
                  f"{('%7.1f' % don) if don is not None else '      -'} {('%7.1f' % doff) if doff is not None else '      -'}")
            stats_rows.append((area, fill, gm, dm, don, doff))
        tick += 1
    cap.release()
    if stats_rows:
        a = np.array([s[0] for s in stats_rows], float)
        f = np.array([s[1] for s in stats_rows], float)
        gm = np.array([s[2] for s in stats_rows], float)
        dm = np.array([s[3] for s in stats_rows], float)
        dn = np.array([s[4] if s[4] is not None else np.nan for s in stats_rows], float)
        df = np.array([s[5] if s[5] is not None else np.nan for s in stats_rows], float)
        print("\n== 汇总（绿色最大块 n=%d 帧）==" % len(stats_rows))
        print(f"面积 中位={np.median(a):7.0f} p95={np.percentile(a,95):7.0f} max={a.max():7.0f}")
        print(f"填充率 中位={np.median(f):.2f}  g均值 中位={np.median(gm):.0f}  绿主导 中位={np.median(dm):.0f}")
        print(f"到 遮膜中心 距离：中位={np.nanmedian(dn):6.1f}  <20px占比={np.nanmean(dn<20):5.1%}")
        print(f"到 不遮中心 距离：中位={np.nanmedian(df):6.1f}  <20px占比={np.nanmean(df<20):5.1%}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
