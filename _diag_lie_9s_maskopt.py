# -*- coding: utf-8 -*-
"""证据侧掩膜形状寻优：哪种「摘准星」掩膜能把准星的相干优势压到答案之下？

背景（本缺陷已取证的三条事实）：
    ① _diag_lie_9s_tbd.py：透明相 t452..612 里全场最强相干轨迹峰落在准星带上（得分 9.4-11.6，
       >p99.9），真值折线只有 1.28-4.39（分位 P2-P30）⇒ 跟踪器锁的是准星。
    ② _diag_lie_9s_suppress.py：在证据图上挖整盘（半径 9-26 处理像素）能把 9s 窗从
       109-121px 修到 35.9-48.7px，但 sup9/16/26 全部把 17s 判据从 7.4px 打到 89.8-299.6px；
       sup26 直接崩溃（conf 0.10）。⇒ 盘太大把答案星自己的残差环一起挖掉了。
    ③ _diag_lie_9s_witness.py：9s 之后头 100 帧里，答案团块的旋转读数只有 2.9-10.9°
       （门槛 witness_min_cum_deg≈20 且 dir≥0.8），到 t≈570 才出现 82.1° 的强旋转簇。
       ⇒ 旋转见证器在 t452..570 结构性地无法接管，这段时间只能靠证据。

    所以掩膜形状是唯一要解的题：既要杀掉准星的相干峰，又要保住答案星的残差环
    （准星十字 34×34 全尺度 = 17×17 处理尺度、细线条；答案星 ~60 全尺度 = 30 处理尺度）。
    本脚本不跑粒子滤波器（混沌、一次一分钟、看不出因果），只在证据层做可证伪判据：

    对每种掩膜 M，在窗 t470..590 / t540..620 上做长基线相干积分，输出三个数：
        S_准星：沿「逐帧实测的准星质心轨迹」积分（真准星，不是速度假设）
        S_答案：沿真值折线积分（起点在 ±24px 内取最优，容忍真值模型误差）
        全场峰：粗网格全局搜索的最强相干轨迹，看它落在答案还是准星上
    判据：S_答案 ≥ S_准星 且 全场峰距真值 ≤ 40px 的掩膜才算「能摘干净」；
         同时看掩膜挖掉的证据质量占比，防止「把整张图挖黑」这种自欺解。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_maskopt.py [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与形态学
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.shape_tracking import _to_numpy_array  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 顶部转向点（全尺度）
CORNER_TICK, CORNER_POS = 776.0, np.array([594.4, 412.6])  # 右下角驻留点（全尺度）
V_DOWN = (CORNER_POS - TURN_POS) / (CORNER_TICK - TURN_TICK)  # 右下段真值速度（全尺度 px/帧）
GREEN_LOOSE = 12  # 绿色判据：g - max(r,b) 门槛
POS_STEP = 12  # 全局搜索起点网格步长（处理尺度像素）
VEL_STEP = 0.1  # 全局搜索速度步长
VEL_LIMIT = 0.5  # 全局搜索速度半径
WINDOWS = ((470, 590), (540, 620))  # 两个积分窗（都在卡死窗内）
NEAR_PX = 24.0  # 「落在某个目标附近」的起点半径（处理尺度像素）

# 掩膜口径：名称 -> (类型, 参数)。foot=本帧足迹膨胀 k；lag=近 K 帧足迹并集再膨胀 k；disk=质心圆盘 r。
VARIANTS = [
    ("none", ("none", 0)),
    ("foot_d3", ("foot", 3)),
    ("foot_d5", ("foot", 5)),
    ("foot_d7", ("foot", 7)),
    ("foot_d9", ("foot", 9)),
    ("lag2_d3", ("lag", (2, 3))),
    ("lag4_d3", ("lag", (4, 3))),
    ("lag4_d5", ("lag", (4, 5))),
    ("disk6", ("disk", 6)),
    ("disk9", ("disk", 9)),
    ("disk13", ("disk", 13)),
]


def truth(tick):  # 真值折线（全尺度）
    if tick <= TURN_TICK:
        return TURN_POS
    if tick <= CORNER_TICK:
        return TURN_POS + V_DOWN * (tick - TURN_TICK)
    return CORNER_POS


def green_footprint(frame):  # 准星足迹：最大绿色连通块掩膜 (0/1) + 质心 + 面积；无则 None
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))  # 拆通道
    mask = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8)  # 绿主导
    if int(mask.sum()) < 15:
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask * 255, 8)  # 连通块
    if count <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    area = int(stats[k, cv2.CC_STAT_AREA])  # 面积
    box = (int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT]))  # 外接框（判形状）
    return (labels == k).astype(np.uint8), (float(cents[k][0]), float(cents[k][1])), area, box  # 结果


def build_keep(ft, history, kind, param, shape):  # 本帧的保留掩膜（1=留证据，0=挖掉）
    keep = np.ones(shape, np.float32)  # 默认全留
    if kind == "none" or ft is None:  # 不挖 / 本帧没准星
        return keep if kind != "none" else keep
    mask, (cx, cy), _area, _box = ft  # 足迹
    if kind == "foot":  # 本帧足迹膨胀
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (param, param))  # 结构元
        hole = cv2.dilate(mask, ker)  # 膨胀
    elif kind == "lag":  # 近 K 帧足迹并集再膨胀
        look, ksize = param  # 回看帧数与结构元
        union = mask.copy()  # 并集
        for older in history[:look]:  # 旧帧足迹
            if older is not None and older.shape == union.shape:
                union |= older  # 合并
        hole = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))  # 膨胀
    elif kind == "disk":  # 质心圆盘
        hole = np.zeros(shape, np.uint8)  # 空掩膜
        cv2.circle(hole, (int(round(cx)), int(round(cy))), int(param), 1, -1)  # 画盘
    else:
        return keep  # 未知口径不挖
    if hole.shape != shape:  # 尺度不符
        return keep
    keep[hole > 0] = 0.0  # 挖掉
    return keep


def main():
    t_to = int(sys.argv[1]) if len(sys.argv) > 1 else 640  # 采集终点
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    store = {"ev": {}, "foot": {}, "cursor": {}}  # 帧号 -> 证据 / 足迹 / 质心（处理尺度）
    origin_update = None  # 原实现对齐器 update

    def patched_update(frame):  # 旁路取证：证据原样返回，只把图和准星足迹存下来
        ev = origin_update(frame)  # 原证据
        if ev is not None:  # 有证据
            arr = np.nan_to_num(_to_numpy_array(ev.normalized), nan=0.0, posinf=0.0, neginf=0.0)  # 下载
            store["ev"][tick_box["t"]] = arr.astype(np.float32)  # 存证据
            got = green_footprint(frame)  # 准星足迹（frame 已是处理尺度）
            store["foot"][tick_box["t"]] = None if got is None else got[0]  # 只存掩膜
            store["cursor"][tick_box["t"]] = None if got is None else (got[1], got[2], got[3])  # 质心+面积+框
        return ev

    tick_box = {"t": 0}  # 当前帧号
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:  # 建局后装桩（aligner 实例在 reset 里不变）
            session.reset(frame.shape[1], frame.shape[0])  # 建局
            origin_update = session.aligner.update  # 原实现
            session.aligner.update = patched_update  # 装桩
        tick_box["t"] = tick  # 帧号
        session.update(frame, float(fps))  # 推进（只为驱动证据链，结论不采）
        tick += 1
    cap.release()  # 释放
    if session.aligner is not None and origin_update is not None:
        session.aligner.update = origin_update  # 卸桩

    have = sorted(store["ev"])  # 有证据的帧
    if not have:
        print("没采到证据帧")
        return 1
    h, w = store["ev"][have[0]].shape[:2]  # 证据尺寸
    areas = [store["cursor"][t][1] for t in have if store["cursor"].get(t)]  # 准星面积序列
    boxes = [store["cursor"][t][2] for t in have if store["cursor"].get(t)]  # 准星外接框序列
    print(f"采集：证据帧 {len(have)}（{have[0]}..{have[-1]}）证据尺寸 {w}x{h}")
    if areas:
        print(f"准星检出 {len(areas)}/{len(have)} 帧 面积 p10={np.percentile(areas, 10):.0f} "
              f"p50={np.median(areas):.0f} p90={np.percentile(areas, 90):.0f} "
              f"外接框 p50={int(np.median([b[0] for b in boxes]))}x{int(np.median([b[1] for b in boxes]))}")

    sc = float(session.effective_scale)  # 处理尺度
    print("\n读法：比值 = 答案附近最强相干轨迹 / 准星附近最强相干轨迹，>1 才算把准星摘到答案之下；"
          "挖掉质量% 同时报出，防止「把整张图挖黑」的自欺解。")
    for name, (kind, param) in VARIANTS:  # 逐掩膜口径
        for (lo, hi) in WINDOWS:  # 逐窗
            ts = [t for t in range(lo, hi + 1) if t in store["ev"]]  # 窗内帧
            if len(ts) < 60:
                continue  # 帧数不足
            # 1) 逐帧构造掩膜后的证据；同时记录被挖掉的质量占比
            maps, removed = [], []  # 掩膜后证据 / 挖掉比例
            for t in ts:  # 逐帧
                ft = None  # 本帧足迹四元组
                cur = store["cursor"].get(t)  # 准星
                mask = store["foot"].get(t)  # 足迹掩膜
                if mask is not None and cur is not None:  # 有准星
                    ft = (mask, cur[0], cur[1], cur[2])  # 组装
                history = []  # 近几帧足迹（供 lag 口径）
                back = 1  # 回看偏移
                while back <= 4:  # 最多取 4 帧
                    older = store["foot"].get(t - back)  # 旧掩膜
                    if older is not None:
                        history.append(older)  # 计入并集
                    back += 1  # 继续回看
                keep = build_keep(ft, history, kind, param, (h, w))  # 保留掩膜
                ev = store["ev"][t] * keep  # 掩膜后证据
                maps.append(ev)  # 存
                total = float(store["ev"][t].sum())  # 原质量
                removed.append(0.0 if total <= 0 else 1.0 - float(ev.sum()) / total)  # 挖掉占比
            removed_pct = float(np.mean(removed) * 100.0)  # 平均占比
            # 2) 粗网格全局搜索：把「起点 × 速度」全假设的相干积分一次算完，后续读法全从它取
            xs = np.arange(POS_STEP, w - POS_STEP, POS_STEP, dtype=np.float32)  # 起点网格
            ys = np.arange(POS_STEP, h - POS_STEP, POS_STEP, dtype=np.float32)  # 起点网格
            XX, YY = np.meshgrid(xs, ys)  # 展开
            X0, Y0 = XX.ravel(), YY.ravel()  # 向量
            n_pos = X0.size  # 起点数
            vs = np.arange(-VEL_LIMIT, VEL_LIMIT + 1e-6, VEL_STEP, dtype=np.float32)  # 速度分量
            mid = (lo + hi) // 2  # 参考时刻（两条轨迹与全场搜索共用同一时间基准）
            best_map = np.zeros((len(vs), len(vs), n_pos), np.float32)  # 得分张量
            for t, ev in zip(ts, maps):  # 逐帧
                dt = float(t - mid)  # 时间偏移
                for iv, vy in enumerate(vs):  # 逐速度 y
                    yi = np.rint(Y0 + vy * dt).astype(np.int32)  # 采样行
                    np.clip(yi, 0, h - 1, out=yi)  # 钳边
                    for iu, vx in enumerate(vs):  # 逐速度 x
                        xi = np.rint(X0 + vx * dt).astype(np.int32)  # 采样列
                        np.clip(xi, 0, w - 1, out=xi)  # 钳边
                        best_map[iv, iu] += ev[yi, xi]  # 累加
            best_map /= float(len(ts))  # 均值
            # 3) 两个竞争者各取「附近最强轨迹」：质心本身被挖成 0 也能公平比（漏网残差在邻域）
            tp_mid = truth(mid) * sc  # 窗中真值（处理尺度）
            cmid = store["cursor"].get(mid)  # 窗中准星
            near_t = np.hypot(X0 - tp_mid[0], Y0 - tp_mid[1]) <= NEAR_PX  # 起点落在答案附近
            near_c = (np.full(n_pos, False) if cmid is None else
                      np.hypot(X0 - cmid[0][0], Y0 - cmid[0][1]) <= NEAR_PX)  # 落在准星附近
            b_truth = float(best_map[:, :, near_t].max()) if bool(near_t.any()) else 0.0  # 答案侧最强
            b_cursor = float(best_map[:, :, near_c].max()) if bool(near_c.any()) else 0.0  # 准星侧最强
            # 4) 钉在真值速度上的读数（去掉「换个速度蒙对」的自由度）
            iv_t = int(np.argmin(np.abs(vs - float(V_DOWN[1] * sc))))  # 真值速度 y 最近行
            iu_t = int(np.argmin(np.abs(vs - float(V_DOWN[0] * sc))))  # 真值速度 x 最近列
            s_truth = float(best_map[iv_t, iu_t, near_t].max()) if bool(near_t.any()) else 0.0  # 真值速度下答案最强
            flat = best_map.ravel()  # 摊平
            top = int(np.argmax(flat))  # 冠军下标
            iv, iu, pi = np.unravel_index(top, best_map.shape)  # 还原
            px, py = float(X0[pi]), float(Y0[pi])  # 峰位置（处理尺度）
            pvx, pvy = float(vs[iu]), float(vs[iv])  # 峰速度
            d_truth = float(np.hypot(px - tp_mid[0], py - tp_mid[1])) / sc  # 峰距真值（全尺度）
            d_cursor = float("nan") if cmid is None else float(np.hypot(px - cmid[0][0], py - cmid[0][1])) / sc  # 峰距准星
            ratio = b_truth / b_cursor if b_cursor > 1e-6 else float("inf")  # 答案/准星强度比
            print(f"{name:>9} t{lo}..{hi} 准星最强={b_cursor:6.2f} 答案最强={b_truth:6.2f} 比值={ratio:6.2f} "
                  f"答案@真值速={s_truth:5.2f} 全场峰={float(flat[top]):6.2f} 峰距真值={d_truth:7.1f} "
                  f"峰距准星={d_cursor:7.1f} 挖掉质量={removed_pct:5.1f}%  "
                  f"峰({px / sc:5.0f},{py / sc:5.0f}) v=({pvx / sc:+.2f},{pvy / sc:+.2f})")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
