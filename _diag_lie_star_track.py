# -*- coding: utf-8 -*-
"""定案 17s 的「答案星形」真值：从 t853 那枚轮廓极清晰的五角星出发，向两侧增量模板跟踪它的轨迹。

依据：肉眼对比右下角时间切片（_diag_lie_br_corner.py）发现，t840/t853 有一枚边界异常锐利的五角星，
与周围被水波糊掉的星形明显不同 —— 这才是解法要锁的目标。之前用「全帧中位光流」外推得到的「答案左移
200px」其实是水波行进的往复运动（整条累积曲线在 ±230px 之间锯齿摆动），不能当刚体真值。

做法：以 t853 的星形中心建 104×104 模板（约 1.7 个格胞，比整星大一圈但小于周期重复尺度），
每 4 帧一步、每步搜索 ±18px，向 t700 与 t900 双向增量跟踪，并打印匹配分与相对 t853 的位移。

用法：
    .venv\\Scripts\\python.exe _diag_lie_star_track.py
"""

import os  # 环境变量（LIE_STAR_LO 把回溯下界拉到 9s 之前取真值）。
import cv2  # 匹配。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
ANCHOR_TICK = 853  # 真值锚点帧（视频播放 17.0s）。
ANCHOR_POS = (605.0, 410.0)  # 锚点位置（右下角切片目测的星形中心，下面会局部精修）。
HALF = 52  # 模板半宽。
STEP = 4  # 步长（帧）。
SEARCH = 18  # 每步搜索半径（像素）。
LO, HI = int(os.environ.get("LIE_STAR_LO", "700")), 900  # 跟踪区间（默认只覆盖 17s 窗，回溯 9s 用 LIE_STAR_LO=400）。


def refine(gray, pos, half=40):  # 用「亮边缘像素质心」精修星形中心（星形轮廓是全图最亮的连续闭合边）。
    x, y = int(round(pos[0])), int(round(pos[1]))  # 整数中心。
    patch = gray[y - half:y + half, x - half:x + half].astype(np.float32)  # 取样。
    thr = np.quantile(patch, 0.90)  # 只看最亮的 10%。
    w, h = gray.shape[1], gray.shape[0]  # 尺寸。
    xs, ys, vs = [], [], []  # 加权点。
    for j in range(patch.shape[0]):
        for i in range(patch.shape[1]):
            v = patch[j, i]  # 像素值。
            if v >= thr:  # 亮边。
                xs.append(x - half + i)  # 绝对坐标。
                ys.append(y - half + j)
                vs.append(v - thr)  # 权重。
    if not vs:
        return np.array([x, y], dtype=np.float64)  # 兜底。
    vs = np.array(vs, dtype=np.float64)  # 权重数组。
    cx = float((np.array(xs) * vs).sum() / vs.sum())  # 加权质心 x。
    cy = float((np.array(ys) * vs).sum() / vs.sum())  # 加权质心 y。
    return np.array([np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)], dtype=np.float64)


def grab(cap, tick, cache):  # 顺序读并缓存灰度帧。
    while tick not in cache:
        ok, frame = cap.read()
        if not ok:
            return None
        cache[tick] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 存灰度。
        tick = max(LO, min(tick, HI))  # 保持区间内。
        if len(cache) > 8:  # 只保留少量帧，避免占内存。
            for k in sorted(cache.keys())[:-8]:
                del cache[k]  # 丢弃最旧。
        break
    return cache.get(tick)


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    frames = {}  # tick -> gray（全量缓存，每 100 帧约 39MB 灰度，500 帧可接受）。
    tick = 0
    while tick <= HI:
        ok, frame = cap.read()
        if not ok:
            break
        if tick >= LO:
            frames[tick] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 缓存。
        tick += 1
    cap.release()
    print(f"loaded ticks {min(frames)}..{max(frames)} n={len(frames)}")

    pos = refine(frames[ANCHOR_TICK], ANCHOR_POS)  # 精修锚点。
    print(f"锚点精修 t{ANCHOR_TICK} -> ({pos[0]:.1f},{pos[1]:.1f})")
    h, w = frames[ANCHOR_TICK].shape  # 帧尺寸。
    x0 = int(np.clip(pos[0] - HALF, 0, w - 2 * HALF))  # 模板左上。
    y0 = int(np.clip(pos[1] - HALF, 0, h - 2 * HALF))
    template = frames[ANCHOR_TICK][y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()  # 参考模板。

    for direction in (-1, 1):  # 先回溯到旋转窗，再前推到场景结束。
        cur = pos.copy()  # 当前位置。
        t = ANCHOR_TICK  # 当前帧。
        while LO <= t + direction * STEP <= HI:
            t += direction * STEP  # 前进。
            gray = frames[t]  # 目标帧。
            ox, oy = cur[0] - HALF, cur[1] - HALF  # 参考左上角。
            sx0 = int(np.clip(ox - SEARCH, 0, max(0, w - 2 * HALF)))  # 搜索窗。
            sy0 = int(np.clip(oy - SEARCH, 0, max(0, h - 2 * HALF)))
            sx1 = int(np.clip(ox + SEARCH + 2 * HALF, 2 * HALF, w))
            sy1 = int(np.clip(oy + SEARCH + 2 * HALF, 2 * HALF, h))
            big = gray[sy0:sy1, sx0:sx1]  # 子图。
            if big.shape[0] < 2 * HALF or big.shape[1] < 2 * HALF:
                continue  # 贴边跳过。
            res = cv2.matchTemplate(big, template, cv2.TM_CCOEFF_NORMED)  # 匹配。
            _, mx, _, loc = cv2.minMaxLoc(res)  # 最优。
            cur = np.array([sx0 + loc[0] + HALF, sy0 + loc[1] + HALF], dtype=np.float64)  # 中心。
            if True:  # 每步都打印，轨迹形态才是定案依据。
                d = cur - pos  # 相对锚点位移。
                print(f"t{t:4d} sec={t/50.207:5.2f} star=({cur[0]:6.1f},{cur[1]:6.1f}) score={mx:.3f} rel=({d[0]:+6.1f},{d[1]:+6.1f}) |d|={np.hypot(*d):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
