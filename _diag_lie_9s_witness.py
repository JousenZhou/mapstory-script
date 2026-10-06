# -*- coding: utf-8 -*-
"""逐检查点取证：9s 丢失期间，旋转见证器手里到底有没有「答案」这张牌。

为什么要看这个（前面八次都栽在证据图上）：
    _diag_lie_9s_tbd.py 证明透明相里全场最强相干源是烧进录像的绿色十字准星（得分 >p99.9，
    真值折线只有 P2-P30），基线口径 t452..612 跟踪中心距准星中位 16.7px、100% 重合。
    但在证据图上动手（整盘置零）代价太大：sup9/16/26 全部把 17s 判据从 7.4px 打到
    89.8-299.6px，且 rotation 帧数从 79 掉到 0-3——准星与答案在 9s 之前是重合的，
    挖准星等于把答案自己的残差环一起挖掉。

    换一条不动信息的路：旋转见证器只认「在转的团块」，准星是纯平移、不转，
    所以它是天然免疫准星的检测器（17s 那次就是靠它从顶部脱困）。本脚本要回答：
      1) t452..612 每个检查点，witness.query() 的候选清单里，是否存在落在真值折线
         附近（≤40px 全尺度）的旋转团块？若有 → 修 9s 不需要碰证据，只需要给见证器
         一个「准星脚下的锁不可信」的抢锚理由；
      2) 若候选只有准星位置 → 旋转口径在 9s 段拿不到答案，必须回到证据侧；
      3) 顺带记录准星脚下（跟踪中心）的 current_cum 读数：它若不为 0，规则②
         static_now 就永远不成立，这是「高置信误锁无法被推翻」的直接解释。

用法：
    .venv\Scripts\python.exe -X utf8 _diag_lie_9s_witness.py [t_from] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 顶部转向点（全尺度）
CORNER_TICK, CORNER_POS = 776.0, np.array([594.4, 412.6])  # 右下角驻留点（全尺度）
V_DOWN = (CORNER_POS - TURN_POS) / (CORNER_TICK - TURN_TICK)  # 右下段真值速度
GREEN_LOOSE = 12  # 处理尺度绿色判据：g - max(r,b) 门槛


def truth(tick):  # 真值折线（全尺度）：t430 顶部 → t776 右下角 → 之后驻留
    if tick <= TURN_TICK:
        return TURN_POS
    if tick <= CORNER_TICK:
        return TURN_POS + V_DOWN * (tick - TURN_TICK)
    return CORNER_POS


def green_centroid(frame):  # 最大绿色连通块质心（处理尺度）；无则 None
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))  # 拆通道
    mask = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8) * 255  # 绿主导
    if int(cv2.countNonZero(mask)) < 15:
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    return (float(cents[k][0]), float(cents[k][1]), int(stats[k, cv2.CC_STAT_AREA]))  # 质心+面积


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 打印起
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 620  # 打印止
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    scale = None  # 处理尺度
    tick = 0  # 帧号
    last_check = -1  # 已打印检查点
    n_check = 0  # 检查点计数
    n_truth_cand = 0  # 「候选清单里有答案附近团块」的检查点数
    n_cursor_cand = 0  # 「候选清单里有准星附近团块」的检查点数
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
            scale = session.effective_scale  # 取实际缩放
        result = session.update(frame, float(fps))  # 推进
        trace = session._witness_trace  # 检查点快照
        if trace and int(trace["frame"]) != last_check:  # 新检查点（本帧刚评估过）
            last_check = int(trace["frame"])  # 记录
            if tick < t_from:  # 打印窗之外只推进节奏，不做取证
                tick += 1
                continue  # 下一帧
            proc = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # 处理尺度帧
            got = green_centroid(proc)  # 准星
            center = np.array(result.center, dtype=np.float64) if result.center is not None else None  # 全尺度中心
            cursor_full = None if got is None else np.array([got[0] / scale, got[1] / scale])  # 准星全尺度
            # query() 是纯读，检查点之后到下一检查点之间状态不变，故此刻取到的就是刚才那份候选清单。
            cands = session.witness.query() if session.witness is not None else []  # 候选（处理尺度）
            d_c, hits = [], []  # 逐候选距离
            for cd in cands[:4]:  # 前四名
                p = np.array([float(cd.center[0]), float(cd.center[1])]) / scale  # 全尺度
                dt = float(np.linalg.norm(p - truth(tick)))  # 距真值
                dc = 9999.0 if cursor_full is None else float(np.linalg.norm(p - cursor_full))  # 距准星
                hits.append((p, dt, dc, float(cd.windowed_deg), float(cd.direction_consistency),
                             int(cd.samples), float(cd.age)))  # 记录
                if dt <= 40.0:
                    d_c.append("答案")  # 落在真值附近
                if dc <= 40.0:
                    d_c.append("准星")  # 落在准星附近
            n_check += 1  # 检查点 +1
            if any(x == "答案" for x in d_c):
                n_truth_cand += 1  # 本检查点手里有答案
            if any(x == "准星" for x in d_c):
                n_cursor_cand += 1  # 本检查点手里有准星
            dcen = 9999.0 if cursor_full is None or center is None else float(np.linalg.norm(center - cursor_full))  # 中心距准星
            if center is None:  # 无输出也要记账
                print(f"W t{tick:4d} 本检查点无跟踪中心 ncand={trace['n_cand']}")
                tick += 1
                continue
            print(f"W t{tick:4d} conf={result.confidence:.2f} src={str(result.source):>10} "
                  f"中心距准星={dcen:6.1f} cur={trace['current_cum']:5.1f}°/{trace['current_dir']:.2f} "
                  f"ncand={trace['n_cand']} 有答案={n_truth_cand} 有准星={n_cursor_cand}")
            print(f"     中心({center[0]:6.1f},{center[1]:6.1f}) 真值({truth(tick)[0]:6.1f},{truth(tick)[1]:6.1f}) "
                  f"准星({'--' if cursor_full is None else f'{cursor_full[0]:.0f},{cursor_full[1]:.0f}'})")
            for i, h in enumerate(hits):  # 候选明细
                tag = "答案" if h[1] <= 40.0 else ("准星" if h[2] <= 40.0 else "其它")  # 归类
                print(f"     #{i + 1} ({h[0][0]:6.1f},{h[0][1]:6.1f}) 距真值={h[1]:6.1f} 距准星={h[2]:6.1f} "
                      f"{h[3]:6.1f}° dir={h[4]:.2f} n={h[5]} age={h[6]:.0f} → {tag}")
        elif tick >= t_from and session.tracker is not None and result.center is not None:  # 非检查点只报行踪
            if tick % 10 == 0:  # 抽样
                c = np.array(result.center, dtype=np.float64)  # 中心
                print(f"    t{tick:4d} {str(result.source):>10} ({c[0]:6.1f},{c[1]:6.1f}) 距真值={np.linalg.norm(c - truth(tick)):6.1f}")
        tick += 1
    cap.release()  # 释放
    print(f"\n检查点共 {n_check} 个：手里有答案（≤40px）{n_truth_cand} 个，手里有准星（≤40px）{n_cursor_cand} 个")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
