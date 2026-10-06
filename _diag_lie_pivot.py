# -*- coding: utf-8 -*-
"""独立仲裁「星形的真正中心在哪」：用长基线光流把目标视运动拟合成了刚体旋转，解出瞬时旋转中心（pivot）。

为什么需要它：判据真值用的是探针 #311 的**证据团块质心**，而跟踪器输出的是**形状模板中心**，两者差一个
稳定的 28px 矢量（几乎纯 -y）。谁是星形的真正中心，靠看图不可靠（画面是周期性星形格子，人眼会看错格）。
物理上唯一无歧义的定义是：答案星形在旋转，它绕自身几何中心转，所以**瞬时旋转中心 = 星形中心**。

刚体视运动 = 平移 t（相机/整体漂移）+ 绕 c 转 θ：d(p) = (R(θ)-I)(p-c) + t。
t 由目标盘外的静止环形带光流中位数给出（背景不转），剩下 θ、c 共 3 个未知数，用 IRLS 高斯-牛顿拟合。

用法：
    .venv\\Scripts\\python.exe _diag_lie_pivot.py
"""

import json  # 读真值。
import os  # 路径。

import cv2  # 光流。
import numpy as np  # 数值。

from _diag_lie_target_score import OUT_DIR, load_truth, truth_at  # 真值与产物目录。

VIDEO = os.path.join("lie_records", "20261001_190829_score1.00_extreme.mp4")  # 失败复现素材。
DISK = 90.0  # 目标盘半径（全尺度像素）：取旋转证据的范围。
RING = (105.0, 170.0)  # 静止参考环（同半径内其它星形不转，用来定整体平移）。
MIN_MOVE = 1.2  # 判定「在动」的位移下限（像素，扣除平移后）。


def read_frame(cap, tick):  # 定位读帧。
    cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
    ok, frame = cap.read()
    return None if not ok else frame


def farneback(g0, g1):  # 长基线稠密光流 (H,W,2)。
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 5, 7, 1.2, 0).astype(np.float32)


def pivot_fit(flow, guess, disk=DISK, ring=RING, min_move=MIN_MOVE):  # 拟合瞬时旋转中心。
    """返回 dict(pivot, theta_deg, shift, n_used, rms)。guess=(x,y) 为初始猜测中心。"""

    h, w = flow.shape[:2]  # 帧尺寸。
    ys, xs = np.mgrid[0:h, 0:w]  # 像素坐标。
    p = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)  # (N,2)。
    d = flow.reshape(-1, 2).astype(np.float64)  # 位移。
    q = p - np.array(guess, dtype=np.float64)  # 相对猜测中心。
    radius = np.linalg.norm(q, axis=1)  # 距猜测中心距离。
    ring_mask = (radius >= ring[0]) & (radius <= ring[1])  # 静止参考环。
    shift = np.median(d[ring_mask], axis=0) if ring_mask.sum() > 50 else np.zeros(2)  # 整体平移（稳健中位）。
    inside = radius <= disk  # 目标盘。
    motion = d - shift  # 扣除平移后的相对运动。
    move = inside & (np.linalg.norm(motion, axis=1) >= min_move)  # 在动的像素。
    n_used = int(move.sum())  # 样本数。
    if n_used < 120:  # 证据太少，不可信。
        return {"pivot": None, "theta_deg": None, "shift": shift.tolist(), "n_used": n_used, "rms": None}
    pts = p[move]  # (M,2)。
    vec = motion[move]  # (M,2)。
    # 参数 (θ, cx, cy)：残差 r = vec - [(R-I)(p-c)]；对 c 的导数是 (I-R)，对 θ 的导数是 dR/dθ·(p-c)。
    theta = 0.0  # 初值。
    center = np.array(guess, dtype=np.float64)  # 初值。
    rms = None  # 残差。
    for _ in range(40):  # 高斯-牛顿迭代。
        cos, sin = np.cos(theta), np.sin(theta)  # 三角。
        rot = np.array([[cos, -sin], [sin, cos]])  # R。
        a = rot - np.eye(2)  # R-I。
        rel = pts - center  # p-c。
        model = rel @ a.T  # (R-I)(p-c)。
        res = vec - model  # 残差。
        rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))  # RMS。
        jac = np.zeros((len(pts) * 2, 3))  # 雅可比。
        drot = np.array([[-sin, -cos], [cos, -sin]])  # dR/dθ。
        dm = rel @ drot.T  # d model/dθ。
        jac[0::2, 0] = dm[:, 0]  # x 对 θ。
        jac[1::2, 0] = dm[:, 1]  # y 对 θ。
        neg_a = -(rot - np.eye(2))  # d model/d c = -(R-I)。
        jac[0::2, 1] = neg_a[0, 0]  # x 对 cx。
        jac[0::2, 2] = neg_a[0, 1]  # x 对 cy。
        jac[1::2, 1] = neg_a[1, 0]  # y 对 cx。
        jac[1::2, 2] = neg_a[1, 1]  # y 对 cy。
        # IRLS：Huber 权重压制环外杂散与另一颗星形的边缘。
        r = res.ravel()  # 残差向量。
        scale = 1.345 * max(np.median(np.abs(r - np.median(r))), 1e-3)  # 稳健尺度。
        wgt = np.minimum(1.0, scale / np.maximum(np.abs(r), 1e-6))  # Huber 权。
        sw = np.sqrt(wgt)[:, None]  # 权重开方。
        try:  # 法方程可能奇异（θ≈0 时 c 不可辨）。
            step, *_ = np.linalg.lstsq(jac * sw, r * np.sqrt(wgt), rcond=None)  # 解增量。
        except np.linalg.LinAlgError:  # 数值失败就停。
            break
        theta += float(step[0])  # 更新角。
        center = center + step[1:3]  # 更新中心。
        if abs(step[0]) < 1e-4 and np.linalg.norm(step[1:3]) < 1e-2:  # 收敛。
            break
    return {
        "pivot": [round(float(center[0]), 1), round(float(center[1]), 1)],
        "theta_deg": round(float(np.degrees(theta)), 2),
        "shift": [round(float(shift[0]), 2), round(float(shift[1]), 2)],
        "n_used": n_used,
        "rms": round(rms, 2) if rms is not None else None,
    }


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    points, meta = load_truth()  # 真值折线。
    print(f"probe#{meta['id']} rate={meta['rate_signed_deg_per_2s']}°/2s dir={meta['dir']}")
    # 基线对：跨 ~20/32/74 帧，旋转角分别约 25°/40°/90°（按 110°/2s≈1.22°/帧 估）。
    pairs = [(744, 764), (730, 762), (702, 744), (702, 776), (720, 760), (752, 776)]
    for a, b in pairs:
        fa, fb = read_frame(cap, a), read_frame(cap, b)
        if fa is None or fb is None:
            print(f"t{a}-{b} 读帧失败")
            continue
        ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)  # 灰度。
        gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        mid = truth_at(points, (a + b) // 2) or truth_at(points, a)  # 猜测中心用真值。
        flow = farneback(ga, gb)  # 光流。
        fit = pivot_fit(flow, mid)  # 拟合。
        truth_b = truth_at(points, b)  # 帧 b 的真值位置（旋转后的中心）。
        print(f"t{a}->t{b} guess={None if mid is None else (round(mid[0],1), round(mid[1],1))} "
              f"pivot={fit['pivot']} theta={fit['theta_deg']}° shift={fit['shift']} n={fit['n_used']} rms={fit['rms']} "
              f"| probe@t{b}={None if truth_b is None else (round(truth_b[0],1), round(truth_b[1],1))} "
              f"Δpivot-probe={None if fit['pivot'] is None or truth_b is None else round(float(np.hypot(fit['pivot'][0]-truth_b[0], fit['pivot'][1]-truth_b[1])),1)}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
