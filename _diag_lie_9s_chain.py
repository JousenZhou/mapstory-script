# -*- coding: utf-8 -*-
"""把「证据图全图最优」抽成逐帧真值链，同时量出当前锁与该最优的得分差——既是 9s 真值表，也是修复判据。

背景（本脚本要验证的结论）：
    全图网格扫描显示证据图里答案轮廓的最强匹配从 t440 (413,65) 连续右下走到 t760 (641,377)，
    而回放跟踪中心从 t520 起偏到它左上方 100~213px，直到 t720 才重新合流。
    也就是说：证据是对的，粒子的局部搜索跟丢了（星形地形是周期的，到处都有得分相近的假峰）。

做法：
    1) 生产路径逐帧推进 ShapeTrackSession，旁路抓本帧证据图与模板尺度；
    2) 对证据图做全图网格打分（位置 × 角度），取互不重叠的局部峰 top-N；
    3) 用「恒速预测 + 关联门限 + 得分优先」把峰串成链（链从白相外推位置起步）；
    4) 同帧量「跟踪中心 30px 邻域内的最佳得分」= 当前锁得分，算 margin。

判据输出：链位置/得分、锁位置/得分、margin、链是否比锁更平滑。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_chain.py [t_from] [t_to] [step]
"""

import json  # 落盘真值表。
import os  # 路径。
import sys  # 参数。
from collections import namedtuple  # 峰记录。

import cv2  # 解码录像。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
OUT_JSON = "_diag_out/nine_s_chain.json"  # 真值链落盘。
TIER = "extreme"  # 生产默认档。
SEED_TICK, SEED_POS = 440, np.array([413.0, 65.0])  # 链起步：白相外推 + 全图扫描双重确认的位置（全尺度）。
GRID = 5.0  # 位置网格步长（处理尺度像素）。
ANGLE_STEP = 12.0  # 角度步长（度）。
TOP_N = 40  # 每帧保留的局部峰数。
PEAK_SUPPRESS = 24.0  # 峰互斥半径（全尺度像素）：同一目标的多角度重复峰只留一个。
GATE = 18.0  # 链关联门限（全尺度像素）：答案自身速度约 1.4px/帧，留足余量。
LOCK_RADIUS = 34.0  # 「当前锁得分」的统计半径（全尺度像素）：跟踪中心邻域内最佳峰。

Peak = namedtuple("Peak", "pos angle score")  # 一个候选峰（pos 为全尺度）。

CAPTURED = {}  # observe_border 旁路抓到的证据图与跟踪器。
_ORIG = st.ParticleShapeTracker.observe_border  # 原实现。


def _patched(self, temporal_map):  # 抓证据图后原样转发，不改变生产行为。
    CAPTURED["evidence"] = st._to_numpy_array(temporal_map)  # (H,W) float，兼容 torch cuda 张量。
    CAPTURED["tracker"] = self  # 跟踪器引用。
    return _ORIG(self, temporal_map)


st.ParticleShapeTracker.observe_border = _patched  # 装旁路。


def peaks(ev, template, play_height, scale, p_scale):  # 全图网格打分并做非极大值抑制，返回按得分降序的峰（全尺度坐标）。
    rx = template.nominal_size * 0.5  # 贴边留出的半径（处理尺度）。
    xs = np.arange(rx, ev.shape[1] - rx, GRID, dtype=np.float32)  # 位置网格。
    ys = np.arange(rx, min(play_height, ev.shape[0]) - rx, GRID, dtype=np.float32)
    period = float(template.symmetry_period or 360.0)  # 对称周期。
    angles = np.arange(0.0, period, ANGLE_STEP, dtype=np.float32)  # 角度网格。
    X, Y, A = np.meshgrid(xs, ys, angles, indexing="ij")  # 三轴。
    states = np.zeros((X.size, 6), dtype=np.float32)  # 只用 x,y,angle 三维。
    states[:, 0] = X.ravel()  # 列。
    states[:, 1] = Y.ravel()  # 行。
    states[:, 4] = A.ravel()  # 角。
    out = st.score_shape_contours(ev, states, template, p_scale, play_height)  # 打分。
    scores = np.asarray(out.scores)  # 得分。
    order = np.argsort(scores)[::-1]  # 降序。
    got = []  # 抑制后的峰。
    for j in order[:4000]:  # 只在高分段里做非极大值抑制，避免 Python 层遍历全部假设。
        if scores[j] < 3.0:  # 低到没有区分意义的峰不再取。
            break
        s = states[j]  # 假设。
        pos = np.array([s[0] / scale, s[1] / scale])  # 全尺度位置。
        if any(np.linalg.norm(pos - g.pos[0:2]) < PEAK_SUPPRESS for g in got):  # 已被更强的峰压制。
            continue
        got.append(Peak(pos, float(s[4]), float(scores[j])))  # 收峰。
        if len(got) >= TOP_N:
            break
    return got


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 400  # 起始帧。
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 780  # 结束帧。
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # 抽链步长（帧）。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参。
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率。
    chain = {}  # tick -> [x, y, score, lock_score, tracker_x, tracker_y]
    pos, vel = SEED_POS.copy(), np.zeros(2)  # 链状态：位置与恒速预测。
    tick = 0
    print(f"seed t{SEED_TICK} ({pos[0]:.1f},{pos[1]:.1f}) 步长 {step} 门限 {GATE}")
    while tick <= t_to:
        ok, frame = cap.read()
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局。
        CAPTURED.clear()  # 重置旁路。
        result = session.update(frame, float(fps))  # 生产路径推进。
        if tick >= t_from and (tick - t_from) % step == 0 and CAPTURED.get("evidence") is not None and session.template is not None:
            ev, tracker = CAPTURED["evidence"], CAPTURED["tracker"]  # 证据图与跟踪器。
            scale = session.effective_scale  # 处理尺度。
            play_height = int(frame.shape[0] * session.params.play_height_ratio * scale)  # 有效高度。
            got = peaks(ev, session.template, play_height, scale, tracker.scale)  # 本帧峰。
            if tick == t_from:  # 起步帧直接按种子对齐，避免首帧就串错。
                near = [p for p in got if np.linalg.norm(p.pos - SEED_POS) < 40.0]  # 种子附近。
                if near:
                    pos, vel = near[0].pos.copy(), np.zeros(2)  # 用种子峰的得分作基准。
            pred = pos + vel  # 恒速预测。
            cand = [p for p in got if np.linalg.norm(p.pos - pred) <= GATE]  # 门内候选。
            pick = max(cand, key=lambda p: p.score) if cand else None  # 门内取最强。
            lock = [p for p in got if result.center is not None and np.linalg.norm(p.pos - np.array(result.center)) <= LOCK_RADIUS]
            lock_score = max((p.score for p in lock), default=float("nan"))  # 当前锁邻域最佳得分。
            if pick is not None:
                vel = 0.75 * vel + 0.25 * (pick.pos - pos)  # 更新恒速。
                pos = pick.pos.copy()  # 链前进。
                used, score = pick.pos, pick.score
            else:
                pos = pred  # 接不上就外推。
                used, score = pred, float("nan")
            tc = np.array(result.center) if result.center is not None else np.array([np.nan] * 2)  # 跟踪中心。
            chain[tick] = [round(float(used[0]), 2), round(float(used[1]), 2), round(float(score), 3),
                           round(float(lock_score), 3), round(float(tc[0]), 1), round(float(tc[1]), 1),
                           0 if pick is not None else 1]  # 末位 1 = 补帧。
            d = np.linalg.norm(used - tc)  # 链与跟踪中心的距离。
            print(f"t{tick:4d} 链=({used[0]:6.1f},{used[1]:6.1f}) 得分={score:5.2f}{'补' if pick is None else '  '} "
                  f"锁得分={lock_score:5.2f} margin={score - lock_score:+5.2f} 跟踪=({tc[0]:6.1f},{tc[1]:6.1f}) "
                  f"Δ链跟踪={d:6.1f} conf={result.confidence:.2f} snr={result.border_snr:.2f} 峰数={len(got)}")
        tick += 1
    cap.release()
    with open(OUT_JSON, "w", encoding="utf-8") as handle:  # 落盘真值链。
        json.dump({"seed": [SEED_TICK, SEED_POS.tolist()], "gate": GATE,
                   "chain": {str(k): v for k, v in sorted(chain.items())}}, handle, ensure_ascii=False)
    rows = [v for v in chain.values() if v[6] == 0]  # 非补帧。
    gaps = sum(1 for v in chain.values() if v[6] == 1)  # 补帧数。
    print(f"\n链覆盖 {len(chain)} 帧，其中补帧 {gaps}；峰值步长中位 "
          f"{np.median([np.linalg.norm(np.array(chain[a][:2]) - np.array(chain[b][:2])) for a, b in zip(sorted(chain)[:-1], sorted(chain)[1:])]):.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
