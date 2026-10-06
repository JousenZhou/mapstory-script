# -*- coding: utf-8 -*-
"""长基线相干积分判据（track-before-detect）：透明相的证据里到底有没有目标？

为什么要做（上一步把路堵死了一半）：
    _diag_lie_9s_scores.py 用生产打分器逐帧比「锁 / 真值 / 全场」，结论是真值处得分高于锁的
    只有 25.5% —— 单帧证据分不出目标。但单帧分辨率本来就不够：目标是连续数百帧的匀速直线
    运动，水波与全局闪烁是空间不相干的随机场。把证据沿「候选匀速轨迹」积起来，噪声按
    sqrt(L) 收敛、目标不收敛，这是弱目标检测的标准做法。积完能不能把真值顶出来是可证伪判据：
      (i)  最优轨迹落在真值附近 ⇒ 证据里有目标（以相干形式存在），缺的是一个能做长基线全局
           搜索/接管的机制 —— 可修；
      (ii) 最优轨迹到处乱跑、真值轨迹排名平平 ⇒ 证据里真的没有目标，得换信息源。

做法（不动生产逻辑，只旁路取证据）：
    钩 tracker.observe_border 收 processing 尺度的时序残差图，逐帧另存 tracker.estimate()
    的锁位置/速度；收完后对若干参考时刻做 (起点网格 × 速度网格) 的轨迹平均打分，
    输出全场前几名峰、真值轨迹得分及排名分位、锁轨迹得分及排名分位。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_tbd.py [t_from] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.shape_tracking import _to_numpy_array  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 准星仍在目标上的最后可信点（全尺度）
END_TICK, END_POS = 853.0, np.array([606.4, 401.6])  # 结尾揭示的答案星
V_TRUTH = (END_POS - TURN_POS) / (END_TICK - TURN_TICK)  # 真值速度（右下，全尺度 px/帧）
POS_STEP = 8  # 起点位置网格步长（处理尺度像素）
VEL_STEP = 0.05  # 速度网格步长（处理尺度像素/帧）
VEL_LIMIT = 0.45  # 速度网格半径（处理尺度像素/帧）：覆盖真值速度 (0.23,0.42) 的一半量级
WINDOWS = (40, 80, 120)  # 积分长度（帧）：同一目标在不同基线下的可检性
REFS = (500, 560, 610)  # 参考时刻（帧）：都落在卡死窗 t452..612 内


def truth(tick):  # 真值折线（全尺度）
    return TURN_POS + V_TRUTH * (tick - TURN_TICK)


def nearest_velocity(vv_list, vx, vy):  # 取与给定速度最近的速度假设下标
    return int(np.argmin([abs(a - vx) + abs(b - vy) for a, b in vv_list]))


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 420  # 采集起点
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 640  # 采集终点
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    BOX = {"origin": None}  # 原 observe_border（用容器避免闭包只读）
    tick_box = {"t": 0}  # 当前帧号（钩子里要用）
    frames = {}  # 帧号 -> 证据图（处理尺度 float32）
    locks = {}  # 帧号 -> (锁中心 x,y, 速度 vx,vy) 处理尺度
    scale_box = {"sc": None}  # 处理尺度

    def hooked(self, temporal_map):  # 只旁路取证据，不改行为
        arr = np.nan_to_num(_to_numpy_array(temporal_map), nan=0.0, posinf=0.0, neginf=0.0)
        frames[tick_box["t"]] = arr.astype(np.float32)  # 存本帧证据
        return BOX["origin"](self, temporal_map)  # 原样执行

    tracker_ref = {}  # 跟踪器实例（首帧后取）

    def ensure_hook(tr):  # 跟踪器延迟创建：换实例就重装钩
        if tr is not None and tracker_ref.get("obj") is not tr:
            if BOX["origin"] is None:
                BOX["origin"] = type(tr).observe_border  # 保存原实现（类级）
                setattr(type(tr), "observe_border", hooked)  # 类级替换
            tracker_ref["obj"] = tr  # 记录实例

    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        tick_box["t"] = tick  # 供钩子索引
        ensure_hook(session.tracker)  # 装钩
        result = session.update(frame, float(fps))  # 推进
        tr = session.tracker  # 跟踪器
        if tr is not None and tr.template is not None and result.center is not None:
            c, v, _ = tr.estimate()  # 锁的位姿与速度（处理尺度）
            locks[tick] = (float(c[0]), float(c[1]), float(v[0]), float(v[1]))  # 记录
            if scale_box["sc"] is None:
                scale_box["sc"] = float(session.effective_scale)  # 尺度
        tick += 1
    cap.release()  # 释放
    if BOX["origin"] is not None and session.tracker is not None:  # 卸钩
        setattr(type(session.tracker), "observe_border", BOX["origin"])  # 还原

    sc = scale_box["sc"] or 0.5  # 尺度兜底
    have = sorted(frames)  # 有证据的帧
    if not have:
        print("没有采到任何证据帧：observe_border 在采集窗内没被调用")
        return 1
    print(f"采集 {t_from}..{t_to}：证据帧 {len(have)}（{have[0]}..{have[-1]}），"
          f"锁帧 {len(locks)}，处理尺度 {sc:g}，证据尺寸 {frames[have[0]].shape}")
    vals = np.concatenate([frames[t].ravel() for t in have[:60]])  # 前 60 帧的像素分布
    print(f"证据值分布：min={vals.min():.2f} p50={np.percentile(vals, 50):.2f} "
          f"p90={np.percentile(vals, 90):.2f} p99={np.percentile(vals, 99):.2f} max={vals.max():.2f}")

    any_ev = frames[have[0]]  # 任取一帧定尺寸
    h, w = any_ev.shape[:2]  # 证据尺寸
    xs = np.arange(POS_STEP, w - POS_STEP, POS_STEP, dtype=np.float32)  # 起点 x 网格
    ys = np.arange(POS_STEP, h - POS_STEP, POS_STEP, dtype=np.float32)  # 起点 y 网格
    XX, YY = np.meshgrid(xs, ys)  # 展开起点
    X0 = XX.ravel().astype(np.float32)  # 起点 x 向量
    Y0 = YY.ravel().astype(np.float32)  # 起点 y 向量
    n_pos = X0.size  # 起点数
    vs = np.arange(-VEL_LIMIT, VEL_LIMIT + 1e-6, VEL_STEP, dtype=np.float32)  # 速度分量网格
    VV = [(float(a), float(b)) for b in vs for a in vs]  # 速度假设全集
    print(f"假设空间：起点 {n_pos} × 速度 {len(VV)} = {n_pos * len(VV)} 条轨迹/窗")

    def nearest_pos(x, y):  # 起点网格上的最近点下标
        return int(np.argmin(np.abs(X0 - float(x)) + np.abs(Y0 - float(y))))  # 曼哈顿距离最近

    for L in WINDOWS:  # 逐种基线长度
        for ref in REFS:  # 逐种参考时刻
            ts = [t for t in range(ref - L // 2, ref + L // 2 + 1) if t in frames]  # 窗内可用帧
            if len(ts) < max(20, L // 2):
                continue  # 帧数不足的窗跳过
            best_map = np.zeros((len(VV), n_pos), np.float32)  # 逐速度假设的轨迹平均证据
            for vi, (vx, vy) in enumerate(VV):  # 逐速度假设
                for t in ts:  # 逐帧积分：位置 = 参考时刻位置 + v*(t-ref)
                    dt = float(t - ref)  # 时间偏移
                    xi = np.rint(X0 + vx * dt).astype(np.int32)  # 采样列
                    yi = np.rint(Y0 + vy * dt).astype(np.int32)  # 采样行
                    np.clip(xi, 0, w - 1, out=xi)  # 越界钳边
                    np.clip(yi, 0, h - 1, out=yi)  # 越界钳边
                    best_map[vi] += frames[t][yi, xi]  # 累加
            best_map /= float(len(ts))  # 取均值（不同基线长度才可比）
            score = best_map.ravel()  # 全假设得分向量
            tp = truth(ref) * sc  # 真值在该时刻的位置（处理尺度）
            tv = V_TRUTH * sc  # 真值速度（处理尺度）
            row_t = nearest_velocity(VV, tv[0], tv[1])  # 真值速度行
            truth_v = float(best_map[row_t, nearest_pos(tp[0], tp[1])])  # 真值轨迹（速度+位置都钉在真值上）
            truth_best = float(np.max(best_map[row_t]))  # 只钉真值速度、起点取该行最优（宽松口径）
            rank = float((score > truth_v).mean() * 100.0)  # 真值得分的排名分位（越小越好）
            rank_best = float((score > truth_best).mean() * 100.0)  # 宽松口径的排名分位
            lk = locks.get(ref)  # 该时刻的锁
            if lk is not None:
                row_l = nearest_velocity(VV, lk[2], lk[3])  # 锁速度行
                pos_l = nearest_pos(lk[0], lk[1])  # 锁起点
                lock_v = float(best_map[row_l, pos_l])  # 锁轨迹得分
                lock_rank = float((score > lock_v).mean() * 100.0)  # 锁的排名分位
            else:
                lock_v, lock_rank = float("nan"), float("nan")  # 无锁
            order = np.argsort(score)[::-1]  # 从高到低
            tops = []  # 前 5 名（同峰去重）
            for idx in order:
                vi, pi = divmod(int(idx), n_pos)  # 还原假设
                if any(np.hypot(X0[pi] - a[0], Y0[pi] - a[1]) < 24 and abs(VV[vi][0] - a[2]) + abs(VV[vi][1] - a[3]) < 0.12
                       for a in tops):
                    continue  # 与已收录的峰太近
                tops.append((float(X0[pi]), float(Y0[pi]), VV[vi][0], VV[vi][1], float(score[idx])))  # 记录
                if len(tops) >= 5:
                    break  # 够了
            print(f"\n窗 L={L} ref=t{ref}（{len(ts)} 帧） p50={np.percentile(score, 50):.2f} "
                  f"p99={np.percentile(score, 99):.2f} p99.9={np.percentile(score, 99.9):.2f} max={score.max():.2f}")
            print(f"  真值轨迹 ({tp[0]:.0f},{tp[1]:.0f}) v=({tv[0]:.2f},{tv[1]:.2f}) 得分 {truth_v:.2f} 分位 P{rank:.3f}"
                  f"　宽松（同速最佳起点）{truth_best:.2f} 分位 P{rank_best:.3f}")
            if lk is not None:
                print(f"  锁  轨迹 ({lk[0]:.0f},{lk[1]:.0f}) v=({lk[2]:.2f},{lk[3]:.2f}) 得分 {lock_v:.2f} 分位 P{lock_rank:.2f}")
            for x, y, vx, vy, s in tops:  # 打印前几名
                d = float(np.hypot(x - tp[0], y - tp[1])) * (1.0 / sc)  # 与真值距离（全尺度）
                print(f"    峰 ({x:.0f},{y:.0f}) v=({vx:.2f},{vy:.2f}) 得分 {s:.2f} 距真值 {d:.1f}px")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
