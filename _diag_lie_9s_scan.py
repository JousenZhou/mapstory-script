# -*- coding: utf-8 -*-
"""9s 定因：绕开粒子滤波的局部搜索，对**证据图**做全图暴力扫描，看答案轮廓的全局最强匹配到底在哪。

为什么要这么做：回放里 9s 段 border 弱测量 snr 4.5~8.25、conf≈1.00，却把中心拖向左侧——
「自信地跟错」只有两种成因：
    A 证据图里答案的 rim 本来就在全局最强处，但粒子群没覆盖到（搜索窗/重定位问题）；
    B 证据图里最强处根本不是答案（残差被水波/准星等干扰主导），那么任何搜索策略都救不了，
      得改证据本身。
本脚本把 (x, y, angle, scale) 在整个证据图上网格化打分，直接给出 A/B 判据。

取证产物：
    _diag_out/nine_s_scan.log            每 10 帧：全局 top-3 / 跟踪器位姿得分 / 右下线最优 / 准星处得分
    _diag_out/nine_s_scan/t{tick}.png    证据图热图 + 四种标记，肉眼复核

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_scan.py [t_from] [t_to] [step]
"""

import json  # 读准星缓存。
import os  # 路径。
import sys  # 参数。

import cv2  # 解码与画图。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
CURSOR = "_diag_out/cursor_traj_190829.json"  # 准星轨迹（对照，判断证据是否被 UI 层主导）。
OUT_DIR = "_diag_out/nine_s_scan"  # 产物目录。
TIER = "extreme"  # 生产默认档。
A_TICK, A_POS = 452, np.array([395.0, 55.0])  # 右下线起点（全尺度）。
B_TICK, B_POS = 702, np.array([610.6, 333.7])  # 右下线终点（全尺度）。
GRID = 6.0  # 位置网格步长（处理尺度像素）。
ANGLE_STEP = 9.0  # 角度步长（度），折进对称周期域。

CAPTURED = {}  # observe_border 旁路抓到的证据图与跟踪器。
_ORIG_OBSERVE = st.ParticleShapeTracker.observe_border  # 原实现。


def _patched_observe_border(self, temporal_map):  # 记录本帧证据图（下载成 NumPy）后原样转发。
    CAPTURED["evidence"] = st._to_numpy_array(temporal_map)  # 证据图 (H,W) float，兼容 torch cuda 张量。
    CAPTURED["tracker"] = self  # 跟踪器引用，取粒子状态与角度。
    return _ORIG_OBSERVE(self, temporal_map)


st.ParticleShapeTracker.observe_border = _patched_observe_border  # 装上旁路。


def scan(evidence, template, play_height, xs, ys, angles, scale):  # 网格化全图打分，返回展平后的假设与得分。
    X, Y, A = np.meshgrid(xs, ys, angles, indexing="ij")  # 三轴网格。
    states = np.zeros((X.size, 6), dtype=np.float32)  # 6 维状态，只用 x,y,angle。
    states[:, 0] = X.ravel()  # 列坐标。
    states[:, 1] = Y.ravel()  # 行坐标。
    states[:, 4] = A.ravel()  # 角度。
    out = st.score_shape_contours(evidence, states, template, scale, play_height)  # 打分（薄壳返回 NumPy）。
    return states, np.asarray(out.scores), np.asarray(out.coverage)


def best_at(evidence, template, play_height, pos, tracker_angle, scale):  # 指定位置（全尺度）上的最佳角度得分。
    angles = np.arange(0.0, template.symmetry_period if template.symmetry_period else 360.0, ANGLE_STEP)  # 对称域内扫角。
    states = np.zeros((len(angles), 6), dtype=np.float32)  # 状态。
    states[:, 0] = pos[0] * scale  # 处理尺度 x。
    states[:, 1] = pos[1] * scale  # 处理尺度 y。
    states[:, 4] = angles  # 角度。
    out = st.score_shape_contours(evidence, states, template, scale, play_height)  # 打分。
    k = int(np.argmax(out.scores))  # 最优。
    return float(out.scores[k]), float(angles[k])


def ev_peak(ev, scale):  # 证据图自身最强处：高响应二值化后取包含全局最大点的连通域质心（返回全尺度）。
    finite = ev[np.isfinite(ev)]  # 有效像素。
    if finite.size < 16:
        return None
    mask = (np.nan_to_num(ev, nan=0.0) >= np.quantile(finite, 0.995)).astype(np.uint8) * 255  # 高分位二值化。
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)  # 连通域。
    if n <= 1:
        return None
    iy, ix = np.unravel_index(int(np.nanargmax(ev)), ev.shape)  # 全局最大点所在标签。
    k = int(lab[iy, ix])  # 峰所属连通域。
    if k == 0:
        k = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大点落在缝隙里时退而取最大域。
    return np.array([cent[k][0] / scale, cent[k][1] / scale])  # 全尺度质心。


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 起始帧。
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 640  # 结束帧。
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 10  # 扫描间隔。
    os.makedirs(OUT_DIR, exist_ok=True)
    cur = {int(k): v for k, v in json.load(open(CURSOR, encoding="utf-8")).items()}  # 准星。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参。
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率。
    vel = (B_POS - A_POS) / (B_TICK - A_TICK)  # 右下线速度。
    tick = 0
    while tick <= t_to:
        ok, frame = cap.read()
        if not ok:
            break
        CAPTURED.clear()  # 每帧重置旁路。
        if tick == 0:  # 只在首帧建局（与生产回放驱动一致，避免反复 reset 清掉光流历史）。
            session.reset(frame.shape[1], frame.shape[0])  # 建立尺度与历史队列。
        result = session.update(frame, float(fps))  # 生产路径推进。
        if tick >= t_from and (tick - t_from) % step == 0 and CAPTURED.get("evidence") is not None:
            ev = CAPTURED["evidence"]  # 证据图。
            tracker = CAPTURED["tracker"]  # 跟踪器。
            template = session.template  # 模板。
            scale = session.effective_scale  # 处理尺度。
            h, w = ev.shape  # 证据尺寸。
            play_height = int(frame.shape[0] * session.params.play_height_ratio * scale)  # 有效高度。
            xs = np.arange(template.nominal_size * 0.5, w - template.nominal_size * 0.5, GRID, dtype=np.float32)  # 位置网格。
            ys = np.arange(template.nominal_size * 0.5, min(play_height, h) - template.nominal_size * 0.5, GRID, dtype=np.float32)
            period = float(template.symmetry_period or 360.0)  # 对称周期。
            angles = np.arange(0.0, period, ANGLE_STEP, dtype=np.float32)  # 角度网格。
            states, scores, cover = scan(ev, template, play_height, xs, ys, angles, tracker.scale)  # 全图扫描。
            order = np.argsort(scores)[::-1]  # 降序。
            peak = ev_peak(ev, scale)  # 证据图自身最强处（不依赖模板）。
            line = A_POS + vel * (tick - A_TICK)  # 本帧右下线预测点。
            sl, _sa = best_at(ev, template, play_height, line, None, tracker.scale)  # 线上证人。
            tc = result.center  # 跟踪中心（全尺度）。
            st_tr, sa_tr = best_at(ev, template, play_height, tc, None, tracker.scale) if tc is not None else (float("nan"), 0)
            cu = cur.get(tick)  # 准星。
            sc_cu, _ = best_at(ev, template, play_height, np.array(cu), None, tracker.scale) if cu else (float("nan"), 0)
            print(f"\nt{tick:4d} 证据 {w}x{h} scale={scale:.3f} 假设数={len(scores)} 周期={period:.0f}° "
                  f"粒子尺度={tracker.scale:.3f} conf={result.confidence:.2f} snr={result.border_snr:.2f}")
            for j in order[:5]:  # 全局最强假设。
                p = states[j]  # 假设。
                print(f"   全局#{int(np.where(order == j)[0][0])+1} pos=({p[0]/scale:6.1f},{p[1]/scale:6.1f}) "
                      f"ang={p[4]:5.1f} score={scores[j]:6.3f} cover={cover[j]:5.2f}")
            print(f"   跟踪中心 pos=({tc[0]:6.1f},{tc[1]:6.1f}) 该处最佳 score={st_tr:6.3f} ang={sa_tr:5.1f}")
            print(f"   右下线点 pos=({line[0]:6.1f},{line[1]:6.1f}) 该处最佳 score={sl:6.3f}")
            if cu:
                print(f"   准星点   pos=({cu[0]:6.1f},{cu[1]:6.1f}) 该处最佳 score={sc_cu:6.3f}")
            if peak is not None:  # 证据峰与三方参照的距离：峰跟准星走 ⇒ UI 残差主导。
                d_pk = np.hypot(peak[0] - tc[0], peak[1] - tc[1]) if tc is not None else float("nan")
                d_pc = np.hypot(peak[0] - cu[0], peak[1] - cu[1]) if cu else float("nan")
                d_pl = np.hypot(peak[0] - line[0], peak[1] - line[1])
                print(f"   证据峰   pos=({peak[0]:6.1f},{peak[1]:6.1f}) 强度={float(ev.max()):5.2f} "
                      f"Δ跟踪={d_pk:6.1f} Δ准星={d_pc:6.1f} Δ右下线={d_pl:6.1f}")
            # 存证据图热图与标记，肉眼复核。
            heat = np.clip(ev / max(float(np.percentile(ev, 99.5)), 1e-6), 0, 1)  # 归一化。
            heat = (cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))  # 热图。
            heat = cv2.resize(heat, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_NEAREST)  # 回全尺度。
            top = states[order[0]] / scale  # 全局最优（全尺度）。
            cv2.circle(heat, tuple(top[:2].astype(int)), 14, (0, 255, 0), 2)  # 绿=全局最优。
            if tc is not None:
                cv2.drawMarker(heat, tuple(tc.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)  # 黄=跟踪中心。
            cv2.circle(heat, tuple(line.astype(int)), 14, (255, 255, 0), 2)  # 青=右下线点。
            if peak is not None:
                cv2.circle(heat, tuple(peak.astype(int)), 8, (255, 0, 0), 2)  # 蓝=证据峰。
            if cu:
                cv2.drawMarker(heat, (int(cu[0]), int(cu[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)  # 品=准星。
            cv2.imwrite(os.path.join(OUT_DIR, f"e{tick}.png"), heat)
        tick += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
