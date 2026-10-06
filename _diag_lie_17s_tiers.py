# -*- coding: utf-8 -*-
"""跨档核对 17s 判据：把 off2（锚点关）与 anchor2（锚点开）在三档上的落点距离一次列全。

判据脚本 _diag_lie_17s_check.py 只盯 extreme（失败复现档），但锚点约束是全局改动，
需要知道 high/ultra 是否同样受益、还是根本没参与（_diag_lie_diff 显示后两档逐帧一致）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_17s_tiers.py [tag ...]
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import numpy as np  # 数值。

OUT_DIR = "_diag_out"  # 回放产物目录。
VIDEO_KEY = "20261001_190829"  # 失败素材。
TIERS = ("high", "ultra", "extreme")  # 三档全查。
SUCCESS_KEY = "20261001_165429"  # 成功素材（对照零干扰）。
TRUTH = {  # tick -> 17s 窗内答案星形中心（区域全尺度像素）。
    841: (632.0, 415.0),
    845: (623.0, 411.0),
    849: (613.0, 405.0),
    853: (606.4, 401.6),
    857: (597.0, 395.0),
    861: (589.0, 388.0),
    865: (583.0, 381.0),
    869: (577.0, 374.0),
    873: (571.0, 367.0),
}
KEY_TICK = 853  # 硬验收帧。
PASS_PX = 30.0  # 达标半径。


def load(tag, tier, key):  # 取某 tag/tier/素材 的回放帧表。
    hits = [p for p in glob.glob(os.path.join(OUT_DIR, f"{tag}_{tier}_{key}_*.json")) if f"_{tier}_" in p]
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {r[0]: r for r in payload["frames"]}, payload


def main():
    tags = sys.argv[1:] or ["off2", "anchor2"]  # 默认：关 vs 开。
    for tag in tags:
        for tier in TIERS:
            for key, label in ((VIDEO_KEY, "失败"), (SUCCESS_KEY, "成功")):
                loaded = load(tag, tier, key)
                if loaded is None:
                    print(f"[MISS] {tag} {tier} {label}")
                    continue
                rows, payload = loaded
                ds = []
                for tick in sorted(TRUTH):
                    row = rows.get(tick)
                    ds.append(np.inf if row is None or row[2] is None else float(np.hypot(row[2] - TRUTH[tick][0], row[3] - TRUTH[tick][1])))
                key_d = ds[sorted(TRUTH).index(KEY_TICK)]
                finite = [d for d in ds if np.isfinite(d)]
                sources = payload.get("sources", {})
                verdict = "PASS" if key_d <= PASS_PX else ("FAIL" if label == "失败" else "N/A")
                print(f"{tag:8s} {tier:8s} {label}  17s={key_d:7.1f}  窗median={float(np.median(finite)) if finite else float('nan'):7.1f}  "
                      f"rotation={sources.get('rotation', 0):3d}  重锚帧={sum(1 for r in payload['frames'] if len(r) > 8 and r[8]):3d}  {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
