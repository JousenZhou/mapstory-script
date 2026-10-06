# -*- coding: utf-8 -*-
"""放大取证：把关键位置裁 200×200 放大 3 倍，肉眼分辨透明答案的 rim 究竟压在哪个图案上。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_zoom.py tick:x:y[:label] ...
例：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_zoom.py 853:606:401:reveal 452:390:57:tracker 452:395:55:truth
"""

import sys  # 参数。

import cv2  # 解码与缩放。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
HALF = 100  # 裁剪半宽（200×200）。
ZOOM = 3  # 放大倍数。


def main():
    specs = []  # (tick, x, y, label)
    for arg in sys.argv[1:]:
        p = arg.split(":")
        specs.append((int(p[0]), float(p[1]), float(p[2]), p[3] if len(p) > 3 else "pt"))
    ticks = sorted({s[0] for s in specs})
    cap = cv2.VideoCapture(VIDEO)
    frames = {}  # tick -> frame
    tick = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if tick in ticks:
            frames[tick] = frame
        if tick > max(ticks):
            break
        tick += 1
    cap.release()
    h, w = next(iter(frames.values())).shape[:2]  # 画面尺寸。
    for t, x, y, label in specs:
        f = frames.get(t)
        if f is None:
            print(f"t{t} 缺帧")
            continue
        x0 = int(np.clip(x - HALF, 0, w - 2 * HALF))  # 裁窗。
        y0 = int(np.clip(y - HALF, 0, h - 2 * HALF))
        crop = f[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()
        cv2.circle(crop, (int(x - x0), int(y - y0)), 12, (0, 0, 255), 2)  # 目标点标红圈。
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)  # 放大。
        out = f"_diag_out/nine_s_marked/zoom_{t}_{label}.png"  # 落盘。
        cv2.imwrite(out, big)
        print(f"{out}  窗=({x0},{y0})-({x0+2*HALF},{y0+2*HALF})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
