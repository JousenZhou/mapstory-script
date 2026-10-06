# -*- coding: utf-8 -*-
"""决定性对照：把画面上的绿色准星从时序证据里挖掉，看透明相的全图最优峰到底属于准星还是答案。

为什么做这个实验（本段取证的直接产物）：
    _diag_lie_9s_cursor.py 实测：t452..612 跟踪中心与准星质心 100% 重合（中位 17px，最小 0.1px），
    t612 之后两者才分道（中位 162px）。准星是画面里对比度最高、运动最平滑的目标，时序残差对它
    必然给出强响应——如果 border 相的峰主要来自准星，那么「跟到右上」根本不是运动模型的问题，
    而是证据被准星污染；此时任何调粒子先验的修法都是在两个假峰之间挑一个。

判读：
    挖掉准星后，若原峰（准星处）明显塌陷、且新的最优峰落在「右下」方向并逐帧连续移动
    → 准星是干扰源，修复方向：证据/打分阶段剔除准星（颜色掩膜），并复核 17s 素材不受影响；
    若挖掉后剩余峰仍散落在地形上、没有稳定往右下走的峰
    → 准星无罪（只是与答案同行），回到方向反转 + 局部搜索出不去这条线。

做法：
    与 _diag_lie_9s_dp.py 同一套打分口径（score_shape_contours + 角度取最大），只是每帧算两遍
    得分面：原始证据 vs 抹掉准星（本帧与上一帧准星圆盘并集，膨胀覆盖运动拖影）后的证据。
    另附峰值非极大抑制后的前 3 名，用来观察「峰到底在不在动」。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_mask.py [t_from] [t_to] [step]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与形态学
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TIER = "extreme"  # 生产默认档
GRID = 4.0  # 位置网格步长（处理尺度像素）
ANGLE_STEP = 18.0  # 角度步长（度）
PAD = 26.0  # 准星圆盘半径（处理尺度像素）：准星本体 17px + 运动拖影


def reticle_full(frame):  # 亮绿准星质心（全尺度像素）；找不到返回 None。与 _diag_lie_9s_cursor.py 同判据。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 170) & (g - r > 45) & (g - b > 45)).astype(np.uint8) * 255  # 纯绿
    if int(cv2.countNonZero(mask)) < 20:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    return float(cents[k][0]), float(cents[k][1])  # 质心


def surface(ev, template, play_height, xs, ys, angles, scale):  # 全图打分 → (nx,ny) 角度最大值。
    X, Y, A = np.meshgrid(xs, ys, angles, indexing="ij")  # 三轴
    states = np.zeros((X.size, 6), dtype=np.float32)  # 假设矩阵
    states[:, 0] = X.ravel()  # 列
    states[:, 1] = Y.ravel()  # 行
    states[:, 4] = A.ravel()  # 角
    out = st.score_shape_contours(ev, states.astype(np.float32), template, scale, play_height)  # 打分
    return np.asarray(out.scores, dtype=np.float32).reshape(len(xs), len(ys), len(angles)).max(axis=2)  # 角度取最大


def top_peaks(surf, xs, ys, k=3, suppress=24.0):  # 非极大抑制取前 k 个峰（返回全尺度坐标与得分）。
    s = surf.copy()  # 工作副本
    out = []  # 结果
    for _ in range(k):  # 逐个取峰
        if not np.isfinite(s).any():  # 全被抑制
            break
        i, j = np.unravel_index(int(np.nanargmax(s)), s.shape)  # 当前最大
        out.append((float(xs[i] * 2.0), float(ys[j] * 2.0), float(s[i, j])))  # 全尺度坐标（处理尺度 0.5）
        lo_i, hi_i = max(0, i - int(suppress / GRID)), min(len(xs), i + int(suppress / GRID) + 1)  # 抑制窗
        lo_j, hi_j = max(0, j - int(suppress / GRID)), min(len(ys), j + int(suppress / GRID) + 1)  # 抑制窗
        s[lo_i:hi_i, lo_j:hi_j] = -np.inf  # 抹平邻域
    return out  # 前 k 峰


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 起始帧
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 700  # 结束帧
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 10  # 采样步长
    captured = {}  # 旁路抓证据
    origin = st.ParticleShapeTracker.observe_border  # 原实现

    def patched(self, temporal_map):  # 拦截：只抓证据图，不改行为
        captured["ev"] = st._to_numpy_array(temporal_map)  # 证据 (H,W)
        captured["tr"] = self  # 跟踪器
        return origin(self, temporal_map)

    st.ParticleShapeTracker.observe_border = patched  # 装桩
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    prev_ret = None  # 上一帧准星（处理尺度）
    rows = []  # 汇总：(tick, 准星处原始分, 准星处挖除分, 原始全图最大, 挖除后全图最大)
    print(f"{'t':>5} {'准星(全尺度)':>18} {'跟踪中心':>18} | {'原始:准星峰/全图峰':>22} {'挖除:准星处/全图峰':>22}  挖除后前三峰（全尺度）")
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        rf = reticle_full(frame)  # 准星（全尺度）
        rp = None if rf is None else (rf[0] * 0.5, rf[1] * 0.5)  # 处理尺度
        if tick >= t_from and tick % step == 0 and "ev" in captured and result.center is not None:  # 采样帧
            ev = captured["ev"]  # 原始证据
            tr = captured["tr"]  # 跟踪器
            template = tr.template  # 模板
            ph = int(tr.play_height) if hasattr(tr, "play_height") else int(ev.shape[0] * 0.94)  # 有效高度
            scale = float(tr.scale)  # 缩放
            rx = float(template.nominal_size) * 0.5  # 边界留白
            xs = np.arange(rx, ev.shape[1] - rx, GRID, dtype=np.float32)  # x 轴
            ys = np.arange(rx, min(ph, ev.shape[0]) - rx, GRID, dtype=np.float32)  # y 轴
            period = float(template.symmetry_period or 360.0)  # 对称周期
            angles = np.arange(0.0, period, ANGLE_STEP, dtype=np.float32)  # 角度轴
            s_raw = surface(ev, template, ph, xs, ys, angles, scale)  # 原始得分面
            ev_masked = ev.copy()  # 挖除副本
            for p in (rp, prev_ret):  # 本帧与上一帧准星都挖（覆盖拖影）
                if p is not None:
                    cv2.circle(ev_masked, (int(round(p[0])), int(round(p[1]))), int(PAD), 0.0, -1)  # 填零
            s_mask = surface(ev_masked, template, ph, xs, ys, angles, scale)  # 挖除后得分面
            # 准星所在网格点（挖前/挖后同一点对比）
            gi = int(np.argmin(np.abs(xs - (rp[0] if rp else -999))))  # 最近 x 格
            gj = int(np.argmin(np.abs(ys - (rp[1] if rp else -999))))  # 最近 y 格
            on_ret_raw, on_ret_mask = float(s_raw[gi, gj]), float(s_mask[gi, gj])  # 准星处得分
            peaks = top_peaks(s_mask, xs, ys)  # 挖除后前三峰
            pk = " ".join(f"({x:.0f},{y:.0f}){v:.1f}" for x, y, v in peaks)  # 文本
            rs = f"({rf[0]:.0f},{rf[1]:.0f})" if rf else "  --none--"  # 准星
            print(f"{tick:5d} {rs:>18} ({result.center[0]:6.1f},{result.center[1]:6.1f}) | "
                  f"{on_ret_raw:8.2f} / {np.nanmax(s_raw):6.2f}  {on_ret_mask:8.2f} / {np.nanmax(s_mask):6.2f}  {pk}")
            rows.append((tick, on_ret_raw, on_ret_mask, float(np.nanmax(s_raw)), float(np.nanmax(s_mask))))  # 记录
        if rp is not None:
            prev_ret = rp  # 推进准星历史
        tick += 1
    st.ParticleShapeTracker.observe_border = origin  # 卸桩
    cap.release()  # 释放
    a = np.array(rows, dtype=np.float64)  # (N,5)
    if len(a):  # 汇总
        print(f"\n准星处得分：原始均 {a[:, 1].mean():.2f} → 挖除后 {a[:, 2].mean():.2f}（降 {a[:, 1].mean() - a[:, 2].mean():.2f}）")
        print(f"全图最大得分：原始均 {a[:, 3].mean():.2f} → 挖除后 {a[:, 4].mean():.2f}")
        drop = a[:, 1] - a[:, 2]  # 每帧准星处塌陷量
        print(f"准星处塌陷量：中位 {np.median(drop):.2f} 最大 {drop.max():.2f}；"
              f"挖除后全图峰 == 原在准星处的帧占比 {np.mean(np.abs(a[:, 2] - a[:, 4]) < 0.05)*100:.0f}%")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
