# -*- coding: utf-8 -*-
"""真值取证：把 190829 录像的若干关键帧原样导出（单帧 + 网格），用来肉眼确认「右下角真目标」在各时刻的实际位置。

为什么需要：探针轨迹 #311 的 start=(610.6,333.7)(帧702) → end=(594.4,412.6)(帧776) 说明答案图形自身在**向下漂移**，
所以「右下角」不是固定坐标，判据必须按时刻取真值。之前把 TARGET 写死成 (594.4,412.6) 会在早期帧上误判。

用法：
    .venv\\Scripts\\python.exe _diag_lie_truth_frames.py                     # 默认关键帧集合
    .venv\\Scripts\\python.exe _diag_lie_truth_frames.py 190829 336 510 853  # 自定义帧号列表

输出（_diag_out/truth/）：
    truth_<素材时间戳>_f<帧号>.png   单帧原图（可放大看细节）
    truth_grid_<素材时间戳>.png      全部关键帧网格（左上角标了帧号与两种秒口径）
"""

import glob  # 素材过滤。
import os  # 路径。
import sys  # 参数。

import cv2  # 读写图像。
import numpy as np  # 拼接。

RECORD_DIR = "lie_records"  # 录像目录。
OUT_DIR = os.path.join("_diag_out", "truth")  # 输出目录。
DEFAULT_TICKS = (200, 300, 336, 400, 460, 510, 560, 617, 660, 702, 740, 776, 820, 853, 900, 930)  # 关键帧：白相末/低置信窗/三次重锚/旋转起止/两个17s口径/结束前。
FPS_LIVE = 30.0  # 生产供帧节拍。
FPS_CONTAINER = 50.207  # 容器标注帧率。
GRID_COLS = 4  # 网格列数。
TILE_HEIGHT = 260  # 网格内单帧缩放高度。


def find_video(name_filter):  # 名字含过滤串的录像，取最新。
    paths = sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4")))
    hits = [p for p in paths if name_filter in os.path.basename(p)]
    return hits[-1] if hits else None


def main():  # 入口。
    name_filter = sys.argv[1] if len(sys.argv) > 1 else "190829"  # 素材过滤。
    ticks = [int(value) for value in sys.argv[2:]] or list(DEFAULT_TICKS)  # 帧号集合。
    path = find_video(name_filter)  # 找素材。
    if path is None:
        print(f"no video matched: {name_filter}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)  # 建目录。
    cap = cv2.VideoCapture(path)  # 打开。
    wanted = set(ticks)  # 需要的帧号集合。
    saved = {}  # tick -> 图。
    tick = 0
    while tick <= max(wanted):  # 顺序读到最大帧（不用 CAP_PROP_POS_FRAMES，编码器 seek 不可靠）。
        ok, frame = cap.read()
        if not ok:
            break
        if tick in wanted:  # 命中关键帧。
            saved[tick] = frame.copy()  # 存下。
        tick += 1
    cap.release()  # 释放。
    tag = os.path.splitext(os.path.basename(path))[0].split("_")[1]  # 用时间戳做短 tag。
    for frame_tick, frame in sorted(saved.items()):  # 单帧落盘。
        cv2.imwrite(os.path.join(OUT_DIR, f"truth_{tag}_f{frame_tick:04d}.png"), frame)  # 原图。
    tiles = []  # 网格块。
    for frame_tick in ticks:  # 按给定顺序。
        frame = saved.get(frame_tick)
        if frame is None:  # 素材没那么长。
            continue
        height, width = frame.shape[:2]  # 原尺寸。
        tile_width = int(width * TILE_HEIGHT / height)  # 等宽比。
        tile = cv2.resize(frame, (tile_width, TILE_HEIGHT))  # 缩放。
        label = f"t{frame_tick}  {frame_tick / FPS_LIVE:.1f}s@30  {frame_tick / FPS_CONTAINER:.1f}s@50"  # 三种口径同标。
        cv2.rectangle(tile, (0, 0), (tile_width, 18), (0, 0, 0), -1)  # 黑底标签条。
        cv2.putText(tile, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)  # 标签。
        tiles.append(tile)  # 入列。
    if tiles:  # 拼网格（行内等宽不一致时用右侧补边，hconcat 要求同高，已统一高度）。
        rows = []  # 行图。
        for start in range(0, len(tiles), GRID_COLS):  # 分行。
            row_tiles = tiles[start:start + GRID_COLS]  # 本行。
            width_max = max(tile.shape[1] for tile in row_tiles)  # 本行最大宽。
            padded = []  # 补边后的行块。
            for tile in row_tiles:  # 逐块。
                if tile.shape[1] < width_max:  # 需要补。
                    pad = np.zeros((tile.shape[0], width_max - tile.shape[1], 3), np.uint8)  # 黑边。
                    tile = cv2.hconcat([tile, pad])  # 补齐。
                padded.append(tile)  # 入列。
            rows.append(cv2.hconcat(padded))  # 行拼接。
        grid = cv2.vconcat(rows)  # 行堆叠。
        out = os.path.join(OUT_DIR, f"truth_grid_{tag}.png")
        cv2.imwrite(out, grid)  # 落盘。
        print(f"grid -> {out} ({grid.shape[1]}x{grid.shape[0]}, {len(tiles)} tiles)")
    print(f"saved {len(saved)} frames -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
