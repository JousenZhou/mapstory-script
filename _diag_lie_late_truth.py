# -*- coding: utf-8 -*-
"""旋转结束后（t776→结尾）答案星形的独立真值：用**跨多个格胞的大模板**逐帧跟踪图案位移。

为什么不用小模板/相位相关：画面是周期 ~60px 的星形格子，60~120px 的模板会匹配到相邻格胞（实测 −61px
正好一个格胞），整帧相位相关在本素材上响应度 0.03~0.27、解算跳变，都不可信。取 220×220（约 3.5 个格胞）
的模板，图案组合唯一，再限制每帧搜索 ±24px 走增量跟踪，累积不了大误差。

起点用 t776 的跟踪器中心（旋转证据在 t702–776 已确认答案在此处），之后完全由图像自己说话。

用法：
    .venv\\Scripts\\python.exe _diag_lie_late_truth.py [start_tick] [end_tick]
"""

import sys  # 参数。

import cv2  # 匹配。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
HALF = 110  # 模板半宽（220×220，约 3.5 个格胞，组合唯一）。
SEARCH = 24  # 每帧搜索半径（像素）。
START_TICK = 776  # 起点帧（旋转窗末）。
START_POS = (616.0, 388.0)  # 起点位置（t776 跟踪器中心，旋转证据确认的答案）。


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else START_TICK  # 起点帧。
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 880  # 终点帧。
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    frames = []  # (tick, gray)。
    tick = 0
    pos = np.array(START_POS, dtype=np.float64)  # 当前答案位置估计。
    template = None  # 参考模板。
    best_pos = None  # 当前帧匹配到的位置。
    score_min = 1.0  # 区间内最低匹配分。
    while tick <= end:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度。
        if tick == start:  # 建模板。
            h, w = gray.shape  # 尺寸。
            x0 = int(np.clip(pos[0] - HALF, 0, w - 2 * HALF))  # 列起点。
            y0 = int(np.clip(pos[1] - HALF, 0, h - 2 * HALF))  # 行起点。
            template = gray[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].copy()  # 参考块。
            best_pos = pos.copy()  # 初值。
            print(f"t{tick} 建模板 origin=({x0},{y0}) size={template.shape[1]}x{template.shape[0]} pos=({pos[0]:.0f},{pos[1]:.0f})")
        elif template is not None:  # 增量匹配。
            h, w = gray.shape  # 尺寸。
            tw, th = template.shape[1], template.shape[0]  # 模板尺寸。
            # 以「上一帧匹配位置对应的模板左上角」为中心开搜索窗。
            ox = best_pos[0] - HALF  # 参考左上角 x。
            oy = best_pos[1] - HALF  # 参考左上角 y。
            sx0 = int(np.clip(ox - SEARCH, 0, max(0, w - tw)))  # 搜索区左。
            sy0 = int(np.clip(oy - SEARCH, 0, max(0, h - th)))  # 搜索区上。
            sx1 = int(np.clip(ox + SEARCH + tw, tw, w))  # 搜索区右。
            sy1 = int(np.clip(oy + SEARCH + th, th, h))  # 搜索区下。
            big = gray[sy0:sy1, sx0:sx1]  # 搜索区。
            if big.shape[0] < th or big.shape[1] < tw:  # 贴边保护。
                pass
            else:
                res = cv2.matchTemplate(big, template, cv2.TM_CCOEFF_NORMED)  # 匹配。
                _, mx, _, loc = cv2.minMaxLoc(res)  # 最优。
                best_pos = np.array([sx0 + loc[0] + HALF, sy0 + loc[1] + HALF], dtype=np.float64)  # 中心。
                score_min = min(score_min, float(mx))  # 记录最低分。
                if tick % 8 == 0 or tick >= end - 4:  # 稀疏打印。
                    print(f"t{tick:4d} sec={tick/50.207:5.2f} 答案真值=({best_pos[0]:6.1f},{best_pos[1]:6.1f}) score={mx:.3f} "
                          f"相对起点=({best_pos[0]-START_POS[0]:+6.1f},{best_pos[1]-START_POS[1]:+6.1f})")
        tick += 1
    cap.release()
    print(f"区间最低匹配分={score_min:.3f}（<0.3 说明模板跟丢，真值不可用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
