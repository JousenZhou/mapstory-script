# -*- coding: utf-8 -*-
"""无真值质量判据（跨素材）：跟踪中心落点处「像不像一颗星」。

用途：准星足迹掩膜会改变若干素材的落点（见 `_diag_lie_cursor_diff.py` 的偏差表），偏差大不等于变差，
需要一个与真值无关、且与掩膜无关的独立观测量来判方向 —— 答案星是画面里的高亮小白点，
所以量「中心周围小窗内的峰值亮度」与「亮核的白度」：
  - peak：15×15（全尺度）灰度最大值；星核应显著高于噪声底。
  - white：peak 处像素的 (min(r,g,b)) 即非彩色程度，星核偏白，UI/地形多偏色。
  - 若「遮膜开」的 peak/white 不低于「遮膜关」，说明掩膜没把落点从星上推开。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_star_score.py [tag_on tag_off] [name_filter]
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 解码。
import numpy as np  # 数值。

OUT_DIR = "_diag_out"  # 回放产物目录。
RECORD_DIR = "lie_records"  # 生产录像目录。
CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"  # 外部参考视频。
TIER = "extreme"  # 档位。
WIN = 15  # 取峰亮度的窗口边长（全尺度像素）。


def load(tag, name):  # 回放产物 → {tick: row}。
    hits = [h for h in glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_{name}.json")) if f"_{TIER}_" in h]
    return {r[0]: r for r in json.load(open(sorted(hits)[-1], encoding="utf-8"))["frames"]} if hits else None


def star_score(frame, x, y):  # 落点周围 (WIN×WIN) 的峰值灰度 + 该峰像素的白度。
    h, w = frame.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    if not (0 <= x0 < w and 0 <= y0 < h):
        return None
    x1, x2 = max(0, x0 - WIN // 2), min(w, x0 + WIN // 2 + 1)
    y1, y2 = max(0, y0 - WIN // 2), min(h, y0 + WIN // 2 + 1)
    patch = frame[y1:y2, x1:x2]  # BGR
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    k = int(np.argmax(gray))
    py, px = divmod(k, gray.shape[1])
    b, g, r = patch[py, px].astype(np.int16)
    return float(gray[py, px]), float(min(b, g, r))  # 峰亮度、白度（min 通道越大越白）


def main():
    tags = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("curon", "curoff")
    filt = sys.argv[3] if len(sys.argv) > 3 else "mp4"
    names = sorted({os.path.basename(h).split(f"_{TIER}_")[1][:-5] for h in
                    glob.glob(os.path.join(OUT_DIR, f"{tags[0]}_{TIER}_*.json"))} |
                   {os.path.basename(h).split(f"_{TIER}_")[1][:-5] for h in
                    glob.glob(os.path.join(OUT_DIR, f"{tags[1]}_{TIER}_*.json"))})
    names = [n for n in names if filt in n]
    print(f"落点星形评分（仅统计两侧落点分歧 ≥10px 的帧）：{tags[0]}(遮) vs {tags[1]}(不遮)｜窗 {WIN}×{WIN} 全尺度")
    print(f"{'素材':38} {'分歧帧':>5} {'峰遮':>6} {'峰不':>6} {'白遮':>6} {'白不':>6} {'胜方':>8}")
    for name in names:
        on, off = load(tags[0], name), load(tags[1], name)
        if on is None or off is None:
            print(f"{name:38} 缺产物（{tags[0] if on is None else ''}{tags[1] if off is None else ''}）")
            continue
        path = os.path.join(RECORD_DIR, name)
        if not os.path.exists(path):
            path = os.path.join(CASE_DIR, name)
        cap = cv2.VideoCapture(path)
        hi = max(max(on), max(off))
        pk, wh = {tags[0]: [], tags[1]: []}, {tags[0]: [], tags[1]: []}
        n_pair = 0
        for t in range(hi + 1):
            ok, frame = cap.read()
            if not ok:
                break
            a, b = on.get(t), off.get(t)
            if a is None or b is None or a[2] is None or b[2] is None:
                continue
            sa, sb = star_score(frame, a[2], a[3]), star_score(frame, b[2], b[3])
            if sa is None or sb is None:
                continue
            if float(np.hypot(a[2] - b[2], a[3] - b[3])) < 10.0:
                continue  # 只比「分歧帧」：两边落点几乎重合时比分数无信息量
            n_pair += 1
            pk[tags[0]].append(sa[0]); pk[tags[1]].append(sb[0])
            wh[tags[0]].append(sa[1]); wh[tags[1]].append(sb[1])
        cap.release()
        if not n_pair:
            print(f"{name:38} 无分歧帧（两侧几乎一致）")
            continue
        m = {k: float(np.median(v)) for k, v in pk.items()}
        q = {k: float(np.median(v)) for k, v in wh.items()}
        dpeak = m[tags[0]] - m[tags[1]]
        dwht = q[tags[0]] - q[tags[1]]
        win = tags[0] if dpeak + dwht >= 0 else tags[1]
        print(f"{name:38} {n_pair:5d} {m[tags[0]]:6.0f} {m[tags[1]]:6.0f} "
              f"{q[tags[0]]:6.0f} {q[tags[1]]:6.0f} {win + 'Δ峰%+.0f/Δ白%+.0f' % (dpeak, dwht):>8}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
