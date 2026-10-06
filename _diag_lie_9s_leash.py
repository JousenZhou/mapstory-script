# -*- coding: utf-8 -*-
"""逐检查点取证：t612 见证器把锁抢锚到右下之后，为什么几帧内又被 border 弱测量拖回左边。

已知事件（_diag_lie_9s_events.py）：
    t612 source border -> rotation，中心跳变 Δ175px (406,57)->(491,210)   ← 方向正确（右下）
    t613 rotation -> prediction，t619 prediction -> border
    t621 Δ39.8px、t622 Δ39.3px 一路向左回到 (354,219)                     ← 正确锚点被撤销

本脚本把 shape_session 里每检查点覆写的 _witness_trace（current_cum / best / best_dir /
best_samples / anchor / anchor_cum / tol …）与逐帧的 source / center / confidence /
frames_since_reliable 并列打印，用来回答三个问题：
  1) t612 那次抢锚走的是规则①还是规则②，strong 的各分量当时到底是多少；
  2) 抢锚之后为什么没有进入 _witness_anchor_lock（锚点是否曾被自确认过）；
  3) t613-t640 的拖离发生在哪一层：observe_border 的似然加权，还是重撒/重定位。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_leash.py [t_from] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码录像
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
ANCHOR_TICK, ANCHOR_POS = 702, np.array([610.6, 333.7])  # 旋转窗实测锚（全尺度，独立真值来源）
# 转向折线真值参照（_diag_lie_9s_linefit.py 实测 + t702 锚反推）：白相末向上，t≈450 撞顶后转右下。
TURN_TICK, TURN_POS = 450.0, np.array([415.6, 52.0])  # 转向点
V_UP = np.array([0.179, -1.383])  # 白相末向上速度 px/帧
V_DOWN = (ANCHOR_POS - TURN_POS) / (ANCHOR_TICK - TURN_TICK)  # 转向后速度 px/帧


def truth(tick):  # 折线真值（全尺度）；仅作为「往右下走」这一用户口径的量化参照。
    if tick <= TURN_TICK:  # 上升段：从转向点按向上速度反推
        return TURN_POS + V_UP * (tick - TURN_TICK)
    return TURN_POS + V_DOWN * (tick - TURN_TICK)  # 下降段


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 430  # 起始帧
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 760  # 结束帧
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    last_trace_frame = -1  # 已打印的检查点帧号
    prev_center = None  # 上一帧中心
    rows = []  # (tick, x, y, conf, source) 供末尾判定
    print(f"{'t':>5} {'phase':>10} {'center':>16} {'conf':>5} {'fsr':>4} | witness@checkpoint")
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        tracker = session.tracker  # 当前跟踪器
        center = np.array(result.center, dtype=np.float64) if result.center is not None else None  # 全尺度中心
        trace = session._witness_trace  # 诊断快照（可能为 None）
        new_check = bool(trace) and int(trace["frame"]) != last_trace_frame  # 本帧是否为新检查点（构造初期快照为空 dict）
        if new_check:
            last_trace_frame = int(trace["frame"])  # 记录
        if tick >= t_from:  # 打印窗口
            cx = f"({center[0]:7.1f},{center[1]:7.1f})" if center is not None else "        --none--"  # 中心
            conf = f"{result.confidence:5.2f}" if center is not None else "  -- "  # 置信度
            fsr = f"{tracker.frames_since_reliable:4d}" if tracker is not None else "   - "  # 自上次可靠帧数
            info = ""  # 见证器列
            if new_check:  # 检查点：打印完整读数
                best = trace["best"]  # [x, y, cum]（处理尺度）
                bs = f"best=({best[0]:.0f},{best[1]:.0f}){best[2]:.1f}°" if best else "best=None"  # 最优候选
                an = trace["anchor"]  # 锚点
                asx = f"anchor=({an[0]:.0f},{an[1]:.0f})" if an else "anchor=None"  # 锚点
                move = ""  # 与 t702 锚的相对方向
                if center is not None:  # 用独立真值判方向
                    d = ANCHOR_POS - center  # 指向正确锚点的向量
                    move = f" Δ锚={np.linalg.norm(d):5.1f} 右下分量={'+' if d[0] > 0 and d[1] > 0 else '?'}"
                info = (f"W{int(trace['frame']):4d} cur={trace['current_cum']:5.1f}°/{trace['current_dir']:.2f} "
                        f"{bs} dir={trace['best_dir']:.2f} n={trace['best_samples']}s age={trace['best_age']:.0f} "
                        f"ncand={trace['n_cand']} {asx} acum={trace['anchor_cum']:5.1f} lock={int(session._witness_anchor_lock)}"
                        f" lowrun={session._witness_low_run} cd={session._witness_cooldown}{move}")
            jump = ""  # 跳变标记
            if center is not None and prev_center is not None:  # 位移
                step = float(np.linalg.norm(center - prev_center))  # 本帧位移
                if step > 12.0:  # 明显大于目标自身速度
                    jump = f"  <<< 位移{step:.1f}px"
            print(f"{tick:5d} {str(result.source):>10} {cx} {conf} {fsr} | err={float(np.linalg.norm(center - truth(tick))):6.1f} {info}{jump}")
        if center is not None:
            prev_center = center  # 推进
            rows.append((tick, center[0], center[1], float(result.confidence), str(result.source)))  # 记录
        tick += 1
    cap.release()  # 释放
    arr = np.array([(r[0], r[1], r[2]) for r in rows], dtype=np.float64)  # (N,3)
    win = arr[(arr[:, 0] >= 612) & (arr[:, 0] <= 660)]  # 抢锚后 48 帧
    if len(win) > 1:  # 净位移
        v = (win[-1, 1:] - win[0, 1:]) / max(len(win) - 1, 1)  # 平均速度
        print(f"\n抢锚后 t612..t{int(win[-1, 0])} 净位移 ({win[-1, 1] - win[0, 1]:.1f},{win[-1, 2] - win[0, 2]:.1f})px"
              f" 平均速度 ({v[0]:.2f},{v[1]:.2f})px/帧；t702 正确锚在 ({ANCHOR_POS[0]:.1f},{ANCHOR_POS[1]:.1f})")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
