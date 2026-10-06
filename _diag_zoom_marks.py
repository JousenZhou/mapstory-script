# -*- coding: utf-8 -*-
"""把标记帧的病灶区裁出来放大，肉眼核对「绿色正十字=跟踪中心」究竟落在准星上还是答案星上。

用法：.venv\\Scripts\\python.exe -X utf8 _diag_zoom_marks.py [tag ...]
"""
import os  # 路径。
import sys  # 参数。

import cv2  # 读图与放大。
import numpy as np  # 数值（locate 用）。

TICK = int(os.environ.get("ZOOM_TICK", "552"))  # 看哪一帧。
BOX = (330, 520, 60, 190)  # 裁剪框 (x0, x1, y0, y1)。
ZOOM = 4  # 放大倍数（近邻，保持像素可辨）
MARKS = {  # 标色→名字（BGR，与 _diag_lie_9s_frames.py 的画法一一对应）
    "跟踪中心(绿正十字)": (0, 255, 0),
    "准星(品红斜叉)": (255, 0, 255),
    "真值线点(黄)": (0, 255, 255),
    "见证器候选(橙)": (0, 128, 255),
}


def locate(im):  # 按颜色找标记质心（精确到像素，不靠肉眼猜）。
    out = {}
    for name, bgr in MARKS.items():
        d = np.abs(im.astype(np.int16) - np.array(bgr, np.int16)).sum(axis=2)
        ys, xs = np.where(d <= 30)  # 命中该色的像素
        out[name] = (float(xs.mean()), float(ys.mean()), int(xs.size)) if xs.size else None
    return out


for tag in (sys.argv[1:] or ["nomask", "curcap"]):
    src = os.path.join("_diag_out", f"nine_s_marked_{tag}", f"m{TICK}.png")
    im = cv2.imread(src)
    if im is None:
        print(f"[MISS] {src}")
        continue
    x0, x1, y0, y1 = BOX
    print(f"\n== {tag} t{TICK} 标记质心 ==")
    for name, got in locate(im).items():
        print(f"  {name:18} " + ("无" if got is None else f"({got[0]:6.1f},{got[1]:6.1f}) 像素{got[2]:5d}"))
    big = cv2.resize(im[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    out = os.path.join("_diag_out", f"nine_s_marked_{tag}", f"z{TICK}.png")
    cv2.imwrite(out, big)
    print(out, big.shape)
