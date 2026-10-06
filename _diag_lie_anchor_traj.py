# -*- coding: utf-8 -*-
"""把「旋转确认过的答案位置」用局部光流递推到旋转结束之后的时刻，得到 17s 处答案星形的真值。

为什么需要：旋转窗（t702–776）之后目标不再转，见证器/探针都失去判据，而画面是低对比度波浪星形格子，
人眼也认不准哪个是答案。唯一可信的外推是：答案在 t776 的位置已被旋转证据锁定，此后它随场景一起漂移，
漂移量 = 该点邻域内的局部光流累积。用相邻帧（短基线，Farneback 可靠）取「以当前估计为中心的圆盘内中位
位移」逐步递推，并用正推/反推互检控制累积误差。

用法：
    .venv\\Scripts\\python.exe _diag_lie_anchor_traj.py
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。

import cv2  # 光流。
import numpy as np  # 数值。

VIDEO = os.path.join("lie_records", "20261001_190829_score1.00_extreme.mp4")  # 失败复现素材。
OUT_DIR = "_diag_out"  # 产物目录。
START = (776, (616.0, 388.0))  # 旋转末帧 + 该帧跟踪器中心（旋转证据确认的答案位置）。
DISK = 45.0  # 局部流取样圆盘半径（像素）。
STEP = 2  # 递推步长（帧）：2 帧基线兼顾可靠与耗时。
PROBE_END = (776, (594.4, 412.6))  # 探针 #311 的窗末质心，作对照起点。


def read_all(cap, limit=935):  # 顺序读全帧灰度（比反复 seek 稳）。
    grays = {}  # tick -> gray。
    tick = 0
    while tick < limit:
        ok, frame = cap.read()
        if not ok:
            break
        grays[tick] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 存灰度。
        tick += 1
    return grays, tick


def flow_of(g0, g1):  # 相邻帧光流。
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 2, 5, 1.2, 0).astype(np.float32)


def local_shift(flow, pos, disk=DISK):  # 圆盘内位移中位数（稳健、抗个别噪声像素）。
    h, w = flow.shape[:2]  # 尺寸。
    x, y = pos  # 当前中心。
    x0, x1 = max(0, int(x - disk)), min(w, int(x + disk) + 1)  # 列范围。
    y0, y1 = max(0, int(y - disk)), min(h, int(y + disk) + 1)  # 行范围。
    patch = flow[y0:y1, x0:x1].reshape(-1, 2)  # 取样。
    if len(patch) < 30:
        return np.zeros(2)  # 太小就当作不动。
    return np.median(patch, axis=0)  # 中位位移。


def propagate(grays, start_tick, start_pos, target_tick):  # 从 start 递推到 target（可正可反）。
    pos = np.array(start_pos, dtype=np.float64)  # 当前位置。
    direction = 1 if target_tick >= start_tick else -1  # 递推方向。
    tick = start_tick  # 当前帧。
    while (tick < target_tick) if direction > 0 else (tick > target_tick):
        nxt = tick + direction * STEP  # 下一帧。
        if nxt not in grays or tick not in grays:
            break
        g0, g1 = (grays[tick], grays[nxt]) if direction > 0 else (grays[nxt], grays[tick])  # 保证 0->1 顺序。
        shift = local_shift(flow_of(g0, g1), pos) * direction  # 圆盘内中位位移。
        pos = pos + shift  # 递推。
        tick = nxt  # 前进。
    return pos, tick


def replay_center(tag, tick):  # 某 tag 回放在该帧的输出中心。
    for path in glob.glob(os.path.join(OUT_DIR, f"{tag}_extreme_20261001_190829_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload["frames"]:
            if row[0] == tick:
                return row[1], (None if row[2] is None else (row[2], row[3]))
    return None, None


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    grays, total = read_all(cap)  # 全帧灰度。
    cap.release()
    print(f"loaded {total} frames")
    targets = (790, 800, 820, 840, 853, 860, 870, 880)
    # 正推（从旋转末 t776 往前）与反推（从各目标帧回到 t776）互检：两者一致才可信。
    print(f"正推起点 t{START[0]} pos={START[1]}")
    for t in targets:
        pos, used = propagate(grays, START[0], START[1], t)  # 正推。
        back, _ = propagate(grays, t, pos, START[0])  # 从正推结果反推回起点，检验累积误差。
        probe_pos, _ = propagate(grays, PROBE_END[0], PROBE_END[1], t)  # 对照：从探针质心起推。
        source, center = replay_center("cluster", t)  # 回放输出。
        err = None if center is None else round(float(np.hypot(center[0] - pos[0], center[1] - pos[1])), 1)
        roundtrip = round(float(np.hypot(back[0] - START[1][0], back[1] - START[1][1])), 1)  # 往返闭合差。
        print(f"t{t:4d} sec={t/50.207:5.2f} 答案外推=({pos[0]:6.1f},{pos[1]:6.1f}) 往返闭合={roundtrip:5.1f}px "
              f"探针起点外推=({probe_pos[0]:6.1f},{probe_pos[1]:6.1f}) 跟踪输出={None if center is None else (round(center[0]), round(center[1]))} src={source} 偏差={err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
