# -*- coding: utf-8 -*-
"""检验「答案在透明相走直线」这条真值假设：用白相强测量拟合运动模型，外推到透明相，与独立锚点对账。

为什么值得先验这条假设：
    粒子预测里没有场景光流补偿项（shape_tracking 里只有自带恒速状态），跟踪器全靠自身历史估速度；
    而用户描述的期望轨迹是「一直往右下」——一条直线。若白相（游戏亲自描出答案、位置精确）本身
    就是高精度直线运动，那么把这条直线外推过透明相就是可用的真值表；若外推还能落在旋转窗实测锚
    t702≈(610.6,333.7) 附近，假设就被两个独立来源同时证实，9s 的病灶即可定性为「跟丢了直线运动」。

做法：
    1) 生产路径逐帧推进，记录 source=='color' 的帧的中心与速度（白相强测量）；
    2) 对白相做一阶/二阶最小二乘拟合，看残差（残差小说明直线成立）；
    3) 用白相末段拟合外推 t337..t900，逐帧对比回放跟踪中心；
    4) 自检：外推在 t702 处与独立锚点的距离。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_linefit.py [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码录像。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
TIER = "extreme"  # 生产默认档。
ANCHOR_TICK, ANCHOR_POS = 702, np.array([610.6, 333.7])  # 旋转窗实测锚点（独立来源）。


def fit(points, times, order):  # 最小二乘多项式拟合，返回 (t)->pos 的预测函数与残差。
    span = max(float(np.ptp(times)), 1.0)  # 时间跨度（归一化用，NumPy2 下 ndarray.ptp 已移除）。
    center_t = float(times.mean())  # 时间中心。
    design = np.vander((times - center_t) / span, order + 1)  # 设计矩阵（归一化时间）。
    coef, *_ = np.linalg.lstsq(design, points, rcond=None)  # 各维系数。
    resid = points - design @ coef  # 残差 (N,2)。

    def predict(t):  # 外推。
        d = np.vander(np.atleast_1d((np.asarray(t, dtype=np.float64) - center_t) / span), order + 1)  # 同规格设计行。
        return (d @ coef).reshape(-1, 2)  # 预测位置。

    return predict, np.linalg.norm(resid, axis=1)  # 预测函数与残差模。


def main():
    t_to = int(sys.argv[1]) if len(sys.argv) > 1 else 900  # 推进到哪一帧。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参。
    cap = cv2.VideoCapture(VIDEO)  # 打开素材。
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率。
    color_t, color_p, rows = [], [], []  # 白相样本与逐帧记录。
    tick = 0  # 帧号。
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧。
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局。
        result = session.update(frame, float(fps))  # 生产路径推进。
        center = np.array(result.center, dtype=np.float64) if result.center is not None else np.full(2, np.nan)  # 中心。
        rows.append((tick, result.source, center))  # 记录。
        if result.source == "color" and np.isfinite(center).all():
            color_t.append(tick)  # 白相帧号。
            color_p.append(center)  # 白相位置。
        tick += 1
    cap.release()  # 释放。
    import json  # 落盘逐帧记录，供后续分析免重跑回放。
    os.makedirs("_diag_out", exist_ok=True)  # 输出目录。
    with open("_diag_out/nine_s_replay_rows.json", "w", encoding="utf-8") as handle:  # 写盘。
        json.dump({str(int(r[0])): [r[1], None if not np.isfinite(r[2]).all() else [round(float(r[2][0]), 2),
                  round(float(r[2][1]), 2)]] for r in rows}, handle, ensure_ascii=False)
    color_t = np.array(color_t, dtype=np.float64)  # 白相时间。
    color_p = np.array(color_p, dtype=np.float64)  # 白相位置。
    print(f"白相样本 {len(color_t)} 帧  t{color_t.min():.0f}..{color_t.max():.0f}")
    print("白相轨迹（每 20 帧一行，括号内为与前一行连线的方向度/速度）：")
    prev = None  # 上一个抽样点。
    for k in range(0, len(color_t), 20):  # 抽样。
        p = color_p[k]  # 位置。
        seg = ""  # 方向速度描述。
        if prev is not None:  # 与前一点连线。
            dlt = p - prev[1]  # 位移。
            dt = color_t[k] - prev[0]  # 帧差。
            ang = float(np.degrees(np.arctan2(dlt[1], dlt[0])))  # 方向（屏幕坐标：y 向下为正）。
            seg = f" 方向{ang:+6.1f}° 速度{np.linalg.norm(dlt)/max(dt,1):.2f}px/帧"
        print(f"  t{color_t[k]:4.0f} ({p[0]:6.1f},{p[1]:6.1f}){seg}")
        prev = (color_t[k], p)  # 推进抽样点。
    for order in (1, 2):  # 一阶（直线）与二阶（抛物线）。
        predict, resid = fit(color_p, color_t, order)  # 拟合。
        print(f"  阶数 {order}：白相残差 中位 {np.median(resid):5.2f} P90 {np.percentile(resid, 90):5.2f} 最大 {resid.max():5.2f}px")
    predict1, _ = fit(color_p, color_t, 1)  # 直线模型。
    # 用白相末段（最后 25% 帧）重新拟合，减少前段变向的污染。
    tail = max(1, len(color_t) // 4)  # 末段长度。
    predict_tail, resid_tail = fit(color_p[-tail:], color_t[-tail:], 1)  # 末段直线。
    print(f"末段直线（t{color_t[-tail]:.0f}..{color_t[-1]:.0f}）残差中位 {np.median(resid_tail):.2f}px  "
          f"速度 {np.round((predict_tail([color_t[-1]]) - predict_tail([color_t[-1]-1]))[0], 3)} px/帧")
    anchor = float(np.linalg.norm(predict_tail([ANCHOR_TICK])[0] - ANCHOR_POS))  # 外推到锚点。
    anchor1 = float(np.linalg.norm(predict1([ANCHOR_TICK])[0] - ANCHOR_POS))  # 全程直线外推到锚点。
    print(f"锚点自检 t{ANCHOR_TICK}：全程直线外推 Δ={anchor1:.1f}px  末段直线外推 Δ={anchor:.1f}px  "
          f"（锚点独立来源：旋转窗实测）")
    print("\n逐帧对照（外推真值 vs 回放跟踪中心）：")
    by_tick = {r[0]: r for r in rows}  # 帧号 -> 记录。
    drift = []  # (t, Δ) 偏离量。
    for t in range(int(color_t.max()) + 1, min(t_to, max(r[0] for r in rows)) + 1, 10):  # 每 10 帧一行。
        rec = by_tick.get(t)  # 该帧记录。
        if rec is None or not np.isfinite(rec[2]).all():
            continue  # 无中心。
        truth = predict_tail([t])[0]  # 外推真值。
        d = float(np.linalg.norm(rec[2] - truth))  # 偏离。
        drift.append((t, d, rec[1]))  # 记录。
        print(f"t{t:4d} 外推=({truth[0]:6.1f},{truth[1]:6.1f}) 跟踪=({rec[2][0]:6.1f},{rec[2][1]:6.1f}) "
              f"Δ={d:6.1f} source={rec[1]}")
    if drift:  # 汇总。
        ds = np.array([x[1] for x in drift])  # 偏离序列。
        first = next((x for x in drift if x[1] > 60.0), None)  # 首次明显偏离。
        print(f"\n偏离统计：中位 {np.median(ds):.1f} P90 {np.percentile(ds, 90):.1f} 最大 {ds.max():.1f}px；"
              f"首次 Δ>60px 在 t{first[0]:d}（{first[2]}）" if first else "\n全程未出现 Δ>60px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
