# -*- coding: utf-8 -*-
"""全局模板检索：把 t853 那枚锐利五角星的模板拿去整帧搜，看它在旋转窗（t702–776）里到底落在哪。

目的：判定「17s 的锐利星形」和「旋转窗里被见证器确认的旋转目标」是不是同一个物理对象。
若全局最强峰在 t776 落在探针质心 (594,413) 附近，则两者同源，17s 真值就是这枚星形；
若峰位到处乱跳（周期性歧义），说明模板尺度选得不行，需要换判据。

用法：
    .venv\\Scripts\\python.exe _diag_lie_star_global.py
"""

import cv2  # 匹配。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
ANCHOR_TICK = 853  # 模板来源帧（视频 17.0s）。
ANCHOR_POS = (606.4, 401.6)  # 星形中心（_diag_lie_star_track.py 精修值）。
TICKS = (700, 720, 740, 760, 776, 790, 800, 810, 820, 830, 840, 853, 860, 873)  # 检索帧。
HALF = 52  # 模板半宽。
PROBE_END = (594.4, 412.6)  # 探针 #311 在 t776 的质心。
TRACKER_END = (616.0, 388.0)  # 跟踪器在 t776 的中心。


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    frames = {}  # tick -> gray。
    want = set(TICKS) | {ANCHOR_TICK}
    tick = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if tick in want:
            frames[tick] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 缓存。
            want.discard(tick)
        tick += 1
    cap.release()
    gray = frames[ANCHOR_TICK]  # 锚帧。
    h, w = gray.shape  # 尺寸。
    x0 = int(np.clip(ANCHOR_POS[0] - HALF, 0, w - 2 * HALF))  # 模板左上。
    y0 = int(np.clip(ANCHOR_POS[1] - HALF, 0, h - 2 * HALF))
    template = gray[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()  # 参考模板。
    for t in TICKS:
        if t not in frames:
            continue  # 缺帧。
        res = cv2.matchTemplate(frames[t], template, cv2.TM_CCOEFF_NORMED)  # 全帧匹配。
        peaks = []  # top-k 去重峰。
        work = res.copy()  # 拷贝用于抑制。
        for _ in range(4):
            _, mx, _, loc = cv2.minMaxLoc(work)  # 当前峰。
            cx, cy = loc[0] + HALF, loc[1] + HALF  # 峰中心。
            if any(np.hypot(cx - px[0], cy - px[1]) < 40 for px in peaks):
                pass  # 太近则视为同一峰（下面仍会抑制邻域）。
            peaks.append((cx, cy, float(mx)))  # 记录。
            wx0, wx1 = max(0, loc[0] - 30), min(work.shape[1], loc[0] + 30)  # 抑制窗。
            wy0, wy1 = max(0, loc[1] - 30), min(work.shape[0], loc[1] + 30)
            work[wy0:wy1, wx0:wx1] = -1.0  # 非极大抑制。
        d_probe = np.hypot(peaks[0][0] - PROBE_END[0], peaks[0][1] - PROBE_END[1])
        d_track = np.hypot(peaks[0][0] - TRACKER_END[0], peaks[0][1] - TRACKER_END[1])
        top = "  ".join(f"({p[0]:5.0f},{p[1]:5.0f})={p[2]:.2f}" for p in peaks[:3])  # 摘要。
        print(f"t{t:4d} best->({peaks[0][0]:5.1f},{peaks[0][1]:5.1f}) dProbe={d_probe:5.1f} dTracker={d_track:5.1f} | {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
