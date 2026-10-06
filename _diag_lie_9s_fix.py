# -*- coding: utf-8 -*-
"""端到端验证：把绿色准星从时序残差证据里剔除后重跑整局，看 9s 之后是否不再被准星吸走。

取证依据（本脚本不重复论证，只验证结论能否兑现）：
    _diag_lie_9s_cursor.py：t452..612 跟踪中心与准星质心 100% 重合（中位 17px、最小 0.1px），
                            t612 之后两者才分开（中位 162px）——丢失窗正是「与准星同行」窗。
    _diag_lie_9s_mask.py  ：在准星处打分，原始均分 5.32（同帧全图上限 7.31）；把准星圆盘填零后
                            该处只剩 -0.03，而剩余图里的最优峰逐帧往右下走且分数更高（6.6~7.9）。
                            即：准星自己就是一个和答案同量级的假峰，且答案的峰本来更强。

两种剔除口径都要试，因为准星经常压在答案上，挖太狠会把答案一起挖掉：
    tight：只填「最近 N 帧准星绿色像素并集 + 少量膨胀」——N 取时序基线上限（lag=4），
           因为残差是 t 与 t-lag 两帧之差，准星在任一帧经过的像素都会污染本帧证据。
    disk ：以准星质心为圆心的整盘填零（对照用，半径同 _diag_lie_9s_mask.py）。

判读：
    与准星的重合比例应从 100% 显著下降，且 t452 之后的轨迹应持续往右下（与转向折线同向），
    而不是回到准星上；若 tight 口径就能做到，生产改动取 tight（对答案证据伤害最小）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_fix.py [tight|disk] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 形态学与连通块
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import (  # noqa: E402
    ShapeTrackParams, ShapeTrackSession, SOURCE_COLOR,  # 会话与相位常量。
)
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 450.0, np.array([415.6, 52.0])  # 转向点（全尺度）
ANCHOR_TICK, ANCHOR_POS = 702, np.array([610.6, 333.7])  # 旋转窗实测锚（全尺度）
V_UP = np.array([0.179, -1.383])  # 白相末向上速度 px/帧
V_DOWN = (ANCHOR_POS - TURN_POS) / (ANCHOR_TICK - TURN_TICK)  # 转向后速度 px/帧
HISTORY = 5  # 准星历史长度（覆盖 temporal_lags 最长基线 4 + 当前帧）
DISK_R = 26  # disk 口径的填零半径（处理尺度像素）

STATE = {"mode": "tight", "hist": []}  # 运行期参数：剔除口径 + 准星绿色像素掩膜历史
_ORIGIN_ALIGNER_UPDATE = None  # aligner.update 原实现句柄（main 里保存）


def truth(tick):  # 转向折线真值参照（全尺度）
    return TURN_POS + (V_UP * (tick - TURN_TICK) if tick <= TURN_TICK else V_DOWN * (tick - TURN_TICK))


def green_mask(frame):  # 亮绿准星的像素掩膜（处理尺度，uint8 0/255）；无准星返回 None。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 170) & (g - r > 45) & (g - b > 45)).astype(np.uint8) * 255  # 纯绿判据（与取证脚本一致）
    if int(cv2.countNonZero(mask)) < 20:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块＝准星本体
    return (labels == k).astype(np.uint8) * 255, (float(cents[k][0]), float(cents[k][1]))  # 掩膜 + 质心


def patched_update(self, frame):  # 拦截 aligner.update：算完证据后按口径剔除准星污染
    ev = _ORIGIN_ALIGNER_UPDATE(self, frame)  # 先走原逻辑
    if ev is None or frame is None or frame.ndim != 3:  # 无证据或非彩色帧
        return ev
    got = green_mask(frame)  # 准星掩膜
    if got is None:  # 本帧没有准星：历史照常老化（不留陈旧污染）
        STATE["hist"].append(None)  # 占位
        STATE["hist"] = STATE["hist"][-HISTORY:]  # 定长
        return ev
    mask, center = got  # 掩膜与质心（处理尺度）
    STATE["hist"].append(mask)  # 入队
    STATE["hist"] = STATE["hist"][-HISTORY:]  # 定长
    union = np.zeros_like(mask)  # 并集
    for m in STATE["hist"]:  # 逐帧并起来：残差对「任一基线帧被准星覆盖」的像素都敏感
        if m is not None:
            union = cv2.bitwise_or(union, m)  # 或
    if STATE["mode"] == "tight":  # 紧口径：仅膨胀 2px
        kill = cv2.dilate(union, np.ones((5, 5), np.uint8))  # 覆盖抗锯齿边
    else:  # 盘口径：整盘填零
        kill = np.zeros_like(union)  # 空图
        cv2.circle(kill, (int(round(center[0])), int(round(center[1]))), DISK_R, 255, -1)  # 画盘
    try:  # normalized 可能是显存张量或 OpenVINO 静态缓冲：下载→改→回传，失败则原样返回
        arr = np.asarray(st._to_numpy_array(ev.normalized), dtype=np.float32).copy()  # 下载副本
    except Exception:  # noqa: BLE001  # 后端不支持时不改写证据，行为等同未修复
        return ev
    arr[kill > 0] = 0.0  # 剔除
    ev.normalized = arr  # 回填（下游 score_shape_contours 接受 CPU 数组）
    return ev


def main():
    global _ORIGIN_ALIGNER_UPDATE  # 原实现句柄
    STATE["mode"] = sys.argv[1] if len(sys.argv) > 1 else "tight"  # 口径
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧
    _ORIGIN_ALIGNER_UPDATE = st.DenseTemporalAligner.update  # 保存
    st.DenseTemporalAligner.update = patched_update  # 装桩
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    rows = []  # (tick, x, y, conf, 距准星, 距真值, source)
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        if result.center is not None:  # 有输出
            got = green_mask(st.resize_for_processing(frame, session.effective_scale))  # 准星（处理尺度）
            rc = None if got is None else np.array([got[1][0] / session.effective_scale, got[1][1] / session.effective_scale])  # 准星全尺度
            c = np.array(result.center, dtype=np.float64)  # 跟踪中心
            rows.append((tick, c[0], c[1], float(result.confidence),
                         float(np.linalg.norm(c - rc)) if rc is not None else np.nan,
                         float(np.linalg.norm(c - truth(tick))), str(result.source)))  # 记录
        tick += 1
    cap.release()  # 释放
    st.DenseTemporalAligner.update = _ORIGIN_ALIGNER_UPDATE  # 卸桩
    a = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], dtype=np.float64)  # 数值列
    print(f"口径={STATE['mode']}  共 {len(a)} 帧有输出")
    print(f"{'窗':>14} {'帧数':>5} {'距准星中位':>10} {'重合<35px%':>10} {'距真值中位':>10} {'距真值末值':>10} {'conf中位':>9}")
    for lo, hi in ((19, 344), (344, 452), (452, 612), (612, 702), (702, 800)):  # 分段
        seg = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]  # 子集
        if len(seg) == 0:
            continue
        dc = seg[:, 4]  # 距准星
        print(f"t{lo:4d}..{hi:4d} {len(seg):5d} {np.nanmedian(dc):10.1f} "
              f"{np.nanmean(dc < 35.0) * 100:10.1f} {np.median(seg[:, 5]):10.1f} {seg[-1, 5]:10.1f} {np.median(seg[:, 3]):9.2f}")
    # 关键判据：9s 之后是否「一直往右下」——x、y 应单调不减（允许小抖动），且末段接近 t702 真值锚。
    seg = a[(a[:, 0] >= 452) & (a[:, 0] <= 702)]  # 丢失窗
    if len(seg) > 10:  # 有数据
        net = seg[-1, 1:3] - seg[0, 1:3]  # 净位移
        back = float(np.mean(np.diff(seg[:, 1]) < -0.5) * 100)  # x 回退帧占比
        print(f"\nt452..702 净位移 ({net[0]:.1f},{net[1]:.1f})px  x 回退帧占比 {back:.1f}%  "
              f"末位置 ({seg[-1, 1]:.1f},{seg[-1, 2]:.1f}) vs 真值 ({ANCHOR_POS[0]:.1f},{ANCHOR_POS[1]:.1f})")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
