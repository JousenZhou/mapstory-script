# -*- coding: utf-8 -*-
"""查探针落盘结果：白相之后（birth>=WHITE_PHASE_END）各参数组合里旋转最强的轨迹，用于确认「答案星形到底在哪」。

用法：
    .venv\\Scripts\\python.exe _diag_lie_probe_query.py [tick_from] [tick_to]
"""

import json  # 读探针结果。
import os  # 路径。
import sys  # 参数。

PROBE = os.path.join("_diag_out", "rotation_diag_extreme_20261001_190829_score1.00_extreme.mp4.json")  # 探针落盘。
COMBOS = ("lag12_floor2.0", "lag12_floor3.0", "lag24_floor2.0", "lag24_floor3.0", "lag36_floor2.0")  # 看几个代表组合。


def main():
    tick_from = int(sys.argv[1]) if len(sys.argv) > 1 else 336  # 白相截止（默认）。
    tick_to = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9  # 上界。
    min_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10  # 采样下限。
    with open(PROBE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for combo in COMBOS:
        rows = [
            r for r in data["combos"].get(combo, [])
            if r["omega_samples"] >= min_samples and r["birth"] >= tick_from and r["death"] <= tick_to + 1
        ]
        rows.sort(key=lambda r: -abs(r["rate_signed_deg_per_2s"]))  # 旋转率降序。
        print(f"== {combo}  n={len(rows)}  窗口[{tick_from},{tick_to}]")
        for r in rows[:8]:
            print(f"   id={r['id']:4d} b={r['birth']:4d} d={r['death']:4d} smp={r['omega_samples']:3d} "
                  f"rate={r['rate_signed_deg_per_2s']:8.1f} dir={r['dir']:.2f} "
                  f"start={r['start']} end={r['end']} path={len(r.get('path', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
