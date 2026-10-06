# -*- coding: utf-8 -*-
"""把「旋转窗结束 → 17s」这段的右下角区域拼成对比图，用肉眼确认答案星形到底有没有离开右下角。

为什么需要：光流/模板外推在波纹水面上给出「答案左移 200px」的结论，但实测那是水波行进的往复运动，
不是刚体平移。唯一能定案的办法是把同一块右下角区域按时间排开，直接看那个星形的相对位置。

用法：
    .venv\\Scripts\\python.exe _diag_lie_br_corner.py
"""

import cv2  # 读帧与拼接。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
BOX = (480, 260, 768, 508)  # 右下角裁剪框（处理尺度像素）。
TICKS = (700, 730, 760, 776, 800, 820, 840, 853, 870)  # 取样帧。
ZOOM = 2  # 放大倍数（低对比度纹理需要放大才看得清）。


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    tiles = []  # (tick, tile)。
    tick = 0
    want = set(TICKS)
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if tick in want:
            x0, y0, x1, y1 = BOX  # 裁剪框。
            tile = frame[y0:y1, x0:x1].copy()  # 裁右下角。
            tile = cv2.resize(tile, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)  # 放大。
            cv2.putText(tile, f"t{tick}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 标帧号。
            tiles.append((tick, tile))  # 收集。
            want.discard(tick)
        tick += 1
    cap.release()
    if not tiles:
        print("no frame sampled")
        return 1
    h = max(t.shape[0] for _, t in tiles)  # 统一行高。
    rows = []  # 每行拼接。
    for i in range(0, len(tiles), 3):
        chunk = tiles[i:i + 3]  # 三个一行。
        pad = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for _, t in chunk]  # 补高。
        rows.append(np.hstack(pad))  # 横向拼。
    sheet = np.vstack(rows)  # 纵向拼。
    out = "_diag_out/br_corner_sheet.png"  # 产物。
    cv2.imwrite(out, sheet)
    print(f"wrote {out} shape={sheet.shape} box={BOX} zoom={ZOOM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
