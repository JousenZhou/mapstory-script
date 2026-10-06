# -*- coding: utf-8 -*-
"""时序切片接触表：把同一片区域在若干帧上拼成一张图，用眼睛定案「9s 前后答案到底在哪」。

为什么现在需要它：
    本会话到目前为止的所有结论都来自代理量（时序残差峰、见证器簇、模板匹配分、准星位置），
    而它们已经互相否证过三次：模板回溯在 t441..t565 把星形钳在画面底边 y=456 上（分数掉到 0.08），
    残差最优峰说答案在右下、准星却又和跟踪中心 100% 重合。代理量打架时只有画面能定案。

做法：
    取一块矩形区域（默认右下 368×258），按给定帧号列表逐帧裁剪，等比放大后拼成网格；
    每格左上角标注帧号与秒数，另把该帧的绿色准星位置（若有）画成青色叉，便于分辨「鼠标」与「地形」。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_sheet.py [x0 y0 x1 y1] [zoom] t1,t2,...,tn
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与拼接
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
FPS = 50.207  # 帧率


def reticle_px(frame):  # 亮绿准星质心（全尺度）；无准星返回 None。与取证脚本同判据。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 170) & (g - r > 45) & (g - b > 45)).astype(np.uint8) * 255  # 纯绿
    if int(cv2.countNonZero(mask)) < 20:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    return float(cents[k][0]), float(cents[k][1])  # 质心


def main():
    box = [int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else [400, 250, 768, 508])]  # 区域 x0y0x1y1
    zoom = float(sys.argv[5]) if len(sys.argv) > 5 else 1.6  # 放大倍数
    ticks = [int(t) for t in sys.argv[6].split(",")] if len(sys.argv) > 6 else list(range(380, 470, 10))  # 帧列表
    out = f"_diag_out/sheet_{box[0]}_{box[1]}_{ticks[0]}_{ticks[-1]}.png"  # 产物名
    want = set(ticks)  # 目标帧集合
    tiles = {}  # tick -> 裁剪图
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    tick = 0  # 帧号
    while want and tick <= max(want):  # 顺序解码到最晚一帧
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick in want:  # 命中
            tile = frame[box[1]:box[3], box[0]:box[2]].copy()  # 裁剪
            rp = reticle_px(frame)  # 准星（全尺度）
            if rp is not None:  # 有准星则画青色叉（换算到裁剪坐标）
                cx, cy = int(rp[0] - box[0]), int(rp[1] - box[1])
                if 0 <= cx < tile.shape[1] and 0 <= cy < tile.shape[0]:  # 在切片内
                    cv2.drawMarker(tile, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 18, 1)  # 叉
            tile = cv2.resize(tile, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)  # 放大
            cv2.putText(tile, f"t{tick} {tick / FPS:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # 标注
            tiles[tick] = tile  # 入表
            want.discard(tick)  # 已完成
        tick += 1
    cap.release()  # 释放
    if not tiles:
        print("无帧可拼")
        return 1
    th, tw = max(v.shape[0] for v in tiles.values()), max(v.shape[1] for v in tiles.values())  # 单格尺寸
    cols = min(4, len(tiles))  # 列数
    rows = (len(tiles) + cols - 1) // cols  # 行数
    canvas = np.full((rows * th, cols * tw, 3), 40, np.uint8)  # 画布（深灰分隔）
    for i, t in enumerate(sorted(tiles)):  # 按帧号排列
        r, c = divmod(i, cols)  # 行列
        tile = tiles[t]  # 取格
        canvas[r * th:r * th + tile.shape[0], c * tw:c * tw + tile.shape[1]] = tile  # 粘贴
    cv2.imwrite(out, canvas)  # 落盘
    print(f"区域 {box} 放大 {zoom}x 帧 {sorted(tiles)}  →  {out}  ({canvas.shape[1]}x{canvas.shape[0]})")
    print("青色叉＝该帧绿色准星位置（若无叉则准星不在本切片内）。")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
