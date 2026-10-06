# -*- coding: utf-8 -*-
"""验证一个具体怀疑：透明相里跟踪器是不是跟到了画面上的绿色准星（reticle），而不是答案星。

为什么要查（本脚本由画面直接引出）：
    _diag_out/fail_frames/f0500.png、f0620.png 显示：整幅是静态星形地形，唯一明显运动的
    除了半透明答案，还有一个亮绿色准星图标。border 相的时序残差对「任何运动物体」一视同仁，
    而亮绿准星边缘锐利、对比极高、移动平滑——是残差图里最强的目标之一。
    用户症状「丢失然后平滑到右上」与准星轨迹（t540 (328,135) → t700 (620,77)：一路往右、偏上）
    高度吻合，因此必须先量出重合度，再谈修跟踪逻辑。

判读：
    若 |跟踪中心 - 准星| 在丢失窗内普遍小于一个目标尺寸（≈35px），说明证据图被准星污染，
    修复方向是在证据/打分里剔除准星（颜色掩膜 + 尺寸形状不符），而不是调运动先验；
    若距离一直很大，则准星无罪，回到「方向反转 + 局部搜索出不去」这条线。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_cursor.py [t_from] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与颜色阈值
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
CURSOR = "_diag_out/cursor_traj_190829.json"  # 已提取的准星轨迹（逐帧、区域全尺度）


def reticle(frame):  # 亮绿准星质心与外接尺寸（全尺度像素）；找不到返回 None。
    # 不能取全图绿像素质心：地形的黄褐色调在 HSV 上也会漏进绿色相，实测散点外接框达 375x242。
    # 改取最大连通块：准星是一个实心小圆环，必然是单一紧挨的团块。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 170) & (g - r > 45) & (g - b > 45)).astype(np.uint8) * 255  # 纯绿：亮度高且明显压倒红蓝
    n = int(cv2.countNonZero(mask))  # 命中像素数
    if n < 20:  # 太少视为不存在
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块的标号
    x, y, w, h, area = stats[k]  # 外接框与面积
    cx, cy = cents[k]  # 质心
    return float(cx), float(cy), float(w), float(h), int(area)  # 质心+外接宽高+面积


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 330  # 起始帧
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 820  # 结束帧
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    hits = []  # (tick, 距离) 统计用
    import json  # 交叉校验用
    ref = json.load(open(CURSOR, encoding="utf-8")) if os.path.exists(CURSOR) else {}  # 已提取的准星轨迹（用于校验本检测器）
    ref_d = []  # (tick, |检测-已有轨迹|)
    print(f"{'t':>5} {'跟踪中心':>18} {'准星质心':>18} {'尺寸w,h,px':>14} {'距离':>7}")
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进生产路径
        if tick >= t_from and result.center is not None:  # 打印窗口
            r = reticle(frame)  # 准星
            if r is None:  # 本帧没找到准星
                print(f"{tick:5d} ({result.center[0]:7.1f},{result.center[1]:7.1f}) {'--none--':>18}")
            else:
                d = float(np.hypot(result.center[0] - r[0], result.center[1] - r[1]))  # 距离
                rv = ref.get(str(tick))  # 已有轨迹的同帧准星
                tag = "" if rv is None else f" 已有准星=({rv[0]:.0f},{rv[1]:.0f})"  # 校验列
                if rv is not None:
                    ref_d.append((tick, float(np.hypot(r[0] - rv[0], r[1] - rv[1]))))  # 记录检测器偏差
                hits.append((tick, d))  # 记录
                flag = "  <<< 重合" if d < 35.0 else ""  # 一个目标尺寸内
                print(f"{tick:5d} ({result.center[0]:7.1f},{result.center[1]:7.1f}) ({r[0]:7.1f},{r[1]:7.1f}) "
                      f"{r[2]:4.0f},{r[3]:<4.0f},{int(r[4]):<5d} {d:7.1f}{flag}{tag}")
        tick += 1
    cap.release()  # 释放
    arr = np.array(hits, dtype=np.float64)  # (N,2)
    if ref_d:  # 检测器与已有准星轨迹的一致性（先证明自己量的准星是对的）
        rd = np.array(ref_d, dtype=np.float64)[:, 1]  # 偏差列
        print(f"\n[校验] 本检测器与 {CURSOR} 的偏差：中位 {np.median(rd):.1f}px 最大 {rd.max():.1f}px")
    if len(arr):  # 汇总
        for lo, hi in ((330, 452), (452, 612), (612, 702), (702, 820)):  # 分段
            seg = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]  # 子集
            if len(seg):  # 有数据
                near = float(np.mean(seg[:, 1] < 35.0))  # 重合比例
                print(f"t{lo}..{hi}: |跟踪-准星| 中位 {np.median(seg[:, 1]):6.1f}px 最小 {seg[:, 1].min():6.1f} "
                      f"重合(<35px)比例 {near*100:5.1f}%")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
