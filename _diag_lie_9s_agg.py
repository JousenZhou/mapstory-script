# -*- coding: utf-8 -*-
"""同口径比较三条候选轨迹在证据得分体上的累计得分，判定「全局证据」到底能不能把跟踪器拉回正确路线。

为什么必须同口径（前一版比较是错的）：
    _diag_lie_9s_dp.py 里「锁得分」是在跟踪中心 ±10px 邻域内取极大，而 DP 路径只能落在 4px 网格上、
    还要受匀速约束。取极大本身就是向上偏置，两者直接比大小对锁系统性有利，得出的「锁均分更高」不可信。
    本脚本把三条轨迹都用同一个双线性采样器在同一得分体（角度已在建体时取极大）上取值，才是公平对照。

三条候选：
    lock ：回放里跟踪中心的实际轨迹（现状）。
    dp   ：_diag_lie_9s_dp.py 解出的匀速 Viterbi 路径（全局证据最优的平滑路径）。
    bounce：物理解释假设——答案在白相末以 (0.18,-1.38)px/帧向上运动，约 t450 撞到 y≈52 后转向右下，
            用「转向点 + t702 旋转窗实测锚 (610.6,333.7)」两点定一条直线（用户描述的「一直往右下」）。

判读：
    若 bounce/dp 的累计得分明显高于 lock，说明证据里本来就有更强的正确峰，病灶是「局部搜索 + 运动先验
    下限太小」导致出不去，可以用全局复核接管；若三者接近甚至 lock 更高，说明透明相的单帧证据根本分不开
    正确答案与地形假峰，任何基于得分的接管都会误伤别的素材，修复必须换判别量（旋转一致性等）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_agg.py [turn_tick] [turn_y]
"""

import os  # 路径。
import sys  # 参数。

import numpy as np  # 数值。

NPZ = "_diag_out/nine_s_dp.npz"  # 得分体（_diag_lie_9s_dp.py scan 产物）。
PATH_NPZ = "_diag_out/nine_s_dp_path.npz"  # DP 路径（solve 产物）。
ANCHOR_TICK, ANCHOR_POS = 702, np.array([610.6, 333.7])  # 旋转窗实测锚（独立来源）。


def sample(surface, xs, ys, pos, p_scale):  # 在全尺度坐标处双线性采样得分体（越界返回 nan）。
    gx = (pos[0] * p_scale - xs[0]) / (xs[1] - xs[0])  # 网格 x 坐标。
    gy = (pos[1] * p_scale - ys[0]) / (ys[1] - ys[0])  # 网格 y 坐标。
    if not (0 <= gx <= len(xs) - 1 and 0 <= gy <= len(ys) - 1):  # 越界不外推。
        return float("nan")  # 缺测。
    i0, j0 = int(np.floor(gx)), int(np.floor(gy))  # 左下角。
    i1, j1 = min(i0 + 1, len(xs) - 1), min(j0 + 1, len(ys) - 1)  # 右下角/右上角。
    fx, fy = gx - i0, gy - j0  # 小数权重。
    v00 = surface[i0, j0] * (1 - fx) + surface[i1, j0] * fx  # 沿 x 插值。
    v01 = surface[i0, j1] * (1 - fx) + surface[i1, j1] * fx  # 沿 x 插值（另一行）。
    return float(v00 * (1 - fy) + v01 * fy)  # 沿 y 插值。


def main():
    turn_tick = float(sys.argv[1]) if len(sys.argv) > 1 else 450.0  # 假设的转向帧。
    turn_y = float(sys.argv[2]) if len(sys.argv) > 2 else 52.0  # 假设的转向高度（全尺度）。
    data = np.load(NPZ)  # 读得分体。
    surf = data["surfaces"].astype(np.float32)  # (T,nx,ny)。
    xs, ys, meta = data["xs"], data["ys"], data["meta"]  # 轴与元数据。
    p_scale = float(data["scale"])  # 处理尺度。
    ticks = meta[:, 0].astype(int)  # 帧号。
    path = np.load(PATH_NPZ, allow_pickle=True)  # DP 路径。
    dp_pts = path["pts"]  # 全尺度。
    lock_pts = meta[:, 1:3]  # 跟踪中心。
    # 白相末速度（由 _diag_lie_9s_linefit.py 实测）：+(0.179,-1.383)px/帧，据此定转向点位置。
    color_end = lock_pts[0]  # t440 附近中心仍在正确轨迹上（白相刚结束不久，偏差小）。
    turn_pos = np.array([float(color_end[0]) + 0.179 * (turn_tick - ticks[0]), turn_y])  # 转向点（全尺度）。
    v_bounce = (ANCHOR_POS - turn_pos) / (ANCHOR_TICK - turn_tick)  # 转向后恒速。
    bounce_pts = turn_pos + v_bounce * (ticks - turn_tick)[:, None]  # 候选直线（t<turn 用同一速度反推）。
    print(f"转向假设 t{turn_tick:.0f} @ ({turn_pos[0]:.1f},{turn_pos[1]:.1f})  转向后速度 "
          f"({v_bounce[0]:.2f},{v_bounce[1]:.2f})px/帧 |速度|={np.linalg.norm(v_bounce):.2f}")
    rows = []  # 汇总。
    for k, t in enumerate(ticks):  # 逐采样帧。
        s_lock = sample(surf[k], xs, ys, lock_pts[k], p_scale)  # 锁轨迹得分。
        s_dp = sample(surf[k], xs, ys, dp_pts[k], p_scale) if k < len(dp_pts) else float("nan")  # DP 轨迹得分。
        s_bn = sample(surf[k], xs, ys, bounce_pts[k], p_scale)  # 转向直线得分。
        rows.append((t, s_lock, s_dp, s_bn))  # 记录。
        print(f"t{t:4d} 锁=({lock_pts[k,0]:6.1f},{lock_pts[k,1]:6.1f}) {s_lock:5.2f} | "
              f"DP=({dp_pts[k,0]:6.1f},{dp_pts[k,1]:6.1f}) {s_dp:5.2f} | "
              f"转向=({bounce_pts[k,0]:6.1f},{bounce_pts[k,1]:6.1f}) {s_bn:5.2f} | "
              f"Δ(DP-锁)={s_dp-s_lock:+5.2f} Δ(转向-锁)={s_bn-s_lock:+5.2f}")
    arr = np.array(rows, dtype=np.float64)[:, 1:]  # (T,3)：三条轨迹的逐帧同口径得分。
    win = (ticks >= turn_tick) & (ticks <= 702)  # 关键窗：转向后到锚点。
    for name, col in (("锁 lock", 0), ("DP", 1), ("转向直线", 2)):  # 汇总。
        full = np.nanmean(arr[:, col])  # 全程均分。
        key = np.nanmean(arr[win, col])  # 关键窗均分。
        print(f"{name:10s} 全程均分 {full:5.2f}  关键窗 t{turn_tick:.0f}..{ANCHOR_TICK} 均分 {key:5.2f}")
    # 每条轨迹的「逐帧最优」参照：同一得分体在整帧上的最大值，用来给得分定标。
    ceil_vals = [float(np.nanmax(s)) for s in surf]  # 每帧全图最大。
    print(f"参照：逐帧全图最大得分 均 {np.mean(ceil_vals):.2f}（同口径下任何轨迹都不可能超过它）")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定。
    raise SystemExit(main())
