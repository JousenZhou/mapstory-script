# -*- coding: utf-8 -*-
"""从探针落盘的轨迹点列里，找经过答案星形（旋转窗末端）的那条 track，读出它在 9s 段的位置。

用途：YOLO 接龙与外观模板都无法把 17s 的答案回溯到 9s（前者在揭示相位检不到答案，后者外观突变），
而探针里每条 track 都带 [tick,x,y] 点列，是唯一可能横跨 9s→17s 的逐帧对应关系。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_probe_link.py [x y tick_lo tick_hi]
"""

import json  # 读探针。
import sys  # 参数。

import numpy as np  # 数值。

PROBE = "_diag_out/rotation_diag_extreme_20261001_190829_score1.00_extreme.mp4.json"  # 探针落盘。


def main():
    tx = float(sys.argv[1]) if len(sys.argv) > 1 else 594.4  # 目标 x。
    ty = float(sys.argv[2]) if len(sys.argv) > 2 else 412.6  # 目标 y。
    tick_lo = int(sys.argv[3]) if len(sys.argv) > 3 else 760  # 命中窗下界。
    tick_hi = int(sys.argv[4]) if len(sys.argv) > 4 else 800  # 命中窗上界。
    tgt = np.array([tx, ty])  # 目标点。
    data = json.load(open(PROBE, encoding="utf-8"))  # 读入。
    for combo, rows in data["combos"].items():  # 逐参数组合。
        hits = []  # 命中列表。
        for r in rows:
            pts = [p for p in r["path"] if tick_lo <= p[0] <= tick_hi]  # 窗内点。
            if not pts:
                continue
            best = min(pts, key=lambda p: np.linalg.norm(np.array(p[1:]) - tgt))  # 最近点。
            if np.linalg.norm(np.array(best[1:]) - tgt) < 30:  # 30px 内算命中。
                hits.append((r, best))
        if not hits:
            continue
        print(f"== {combo} 命中 {len(hits)}")
        for r, best in sorted(hits, key=lambda h: -h[0]["omega_samples"])[:4]:
            print(f"   id={r['id']} b={r['birth']} d={r['death']} smp={r['omega_samples']} "
                  f"rate={r['rate_signed_deg_per_2s']:.1f} dir={r['dir']:.2f} cum={r['cum_abs_deg']:.1f} "
                  f"命中t{int(best[0])} ({best[1]:.1f},{best[2]:.1f})")
            print("   path:", " ".join(f"{int(p[0])}:({p[1]:.0f},{p[2]:.0f})" for p in r["path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
