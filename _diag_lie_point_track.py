# -*- coding: utf-8 -*-
"""通用增量模板点跟踪：从指定 tick 的指定位置建模板，向单方向逐帧接龙，输出该点的轨迹。

与 _diag_lie_star_track.py 的区别：那个脚本写死了 17s 锚点与 700..900 区间；这里把起点、方向、
步长、搜索半径都做成参数，用来回答「t452 那枚星形自己往哪走」这类前推问题。
模板取 104×104（约 1.7 个格胞），STEP 小、SEARCH 小，靠局部外观逐帧传递，不做全局检索。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_point_track.py <start_tick> <x> <y> <step方向:+1/-1> [stop_tick] [STEP] [SEARCH]
例（从 9s 那枚星前推到旋转窗）：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_point_track.py 452 390 50 +1 720 2 10
"""

import sys  # 参数。

import cv2  # 匹配。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 目标素材。
HALF = 52  # 模板半宽。


def to_gray(frame):  # 转灰度并抹掉「绿色准星」叠加层（它是跟着鼠标跑的 UI 层，不是地形，留在模板里会污染匹配）。
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))  # 拆通道。
    mask = (g > 140) & (g - r > 50) & (g - b > 50)  # 绿像素。
    filled = frame.copy()  # 副本，不改原帧。
    if mask.any():  # 有才处理。
        blur = cv2.medianBlur(frame, 5)  # 用邻域中位填。
        filled[mask] = blur[mask]  # 绿色像素换成周围地形颜色。
    return cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)  # 转灰度。


def main():
    start_tick = int(sys.argv[1])  # 起点帧。
    start = np.array([float(sys.argv[2]), float(sys.argv[3])])  # 起点坐标。
    direction = 1 if sys.argv[4].startswith("+") else -1  # 前推 / 回溯。
    stop = int(sys.argv[5]) if len(sys.argv) > 5 else (start_tick + direction * 300)  # 终止帧。
    step_n = int(sys.argv[6]) if len(sys.argv) > 6 else 2  # 每步步长（帧）。
    search = int(sys.argv[7]) if len(sys.argv) > 7 else 10  # 每步搜索半径（像素）。

    cap = cv2.VideoCapture(VIDEO)
    lo, hi = min(start_tick, stop), max(start_tick, stop)  # 需要解码的区间。
    frames = {}  # tick -> 灰度。
    tick = 0
    while tick <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if tick >= lo:
            frames[tick] = to_gray(frame)  # 抹绿后存灰度。
        tick += 1
    cap.release()
    if start_tick not in frames:
        print(f"起点 t{start_tick} 不在解码区间")
        return 1

    h, w = frames[start_tick].shape  # 帧尺寸。
    x0 = int(np.clip(start[0] - HALF, 0, w - 2 * HALF))  # 模板左上。
    y0 = int(np.clip(start[1] - HALF, 0, h - 2 * HALF))
    template = frames[start_tick][y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()  # 参考模板。
    print(f"起点 t{start_tick} ({start[0]:.1f},{start[1]:.1f}) 模板 {2*HALF}x{2*HALF} 方向 {direction:+d} "
          f"步长 {step_n} 搜索 {search}px")

    cur = start.copy()  # 当前位置。
    t = start_tick
    scores = []  # 匹配分序列，用于判断轨迹何时失效。
    while (lo <= t + direction * step_n <= hi):
        t += direction * step_n  # 推进。
        gray = frames[t]  # 目标帧。
        ox, oy = cur[0] - HALF, cur[1] - HALF  # 参考左上。
        sx0 = int(np.clip(ox - search, 0, max(0, w - 2 * HALF)))  # 搜索窗。
        sy0 = int(np.clip(oy - search, 0, max(0, h - 2 * HALF)))
        sx1 = int(np.clip(ox + search + 2 * HALF, 2 * HALF, w))
        sy1 = int(np.clip(oy + search + 2 * HALF, 2 * HALF, h))
        big = gray[sy0:sy1, sx0:sx1]  # 子图。
        if big.shape[0] < 2 * HALF or big.shape[1] < 2 * HALF:
            print(f"t{t:4d} 贴边跳过")
            continue
        res = cv2.matchTemplate(big, template, cv2.TM_CCOEFF_NORMED)  # 匹配。
        _, mx, _, loc = cv2.minMaxLoc(res)  # 最优。
        new = np.array([sx0 + loc[0] + HALF, sy0 + loc[1] + HALF], dtype=np.float64)  # 新中心。
        scores.append(mx)  # 记录。
        d = new - cur  # 本步位移。
        rel = new - start  # 相对起点位移。
        flag = ""
        if mx < 0.5:
            flag = "  <分低，轨迹可能已失效>"
        print(f"t{t:4d} sec={t/50.207:5.2f} pos=({new[0]:6.1f},{new[1]:6.1f}) score={mx:.3f} "
              f"step=({d[0]:+5.1f},{d[1]:+5.1f}) rel=({rel[0]:+6.1f},{rel[1]:+6.1f}){flag}")
        cur = new
    if scores:
        arr = np.array(scores)  # 分数统计。
        print(f"步数 {len(arr)} 分数 min={arr.min():.3f} 中位={np.median(arr):.3f} "
              f"末10步均值={arr[-10:].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
