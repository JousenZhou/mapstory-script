# -*- coding: utf-8 -*-
"""把「谁在动」直接画出来：长窗运动投影图，一次看清准星拖影与答案星的轨迹是否重合。

为什么需要这张图（前面全部推理都卡在同一个缺口上）：
    透明相里答案不可见，我们只能用代理量判断它在哪：准星（玩家鼠标）、时序残差峰、
    旋转见证器簇、由白相速度外推的折线。本段已经出现三次互相否证——
    准星在 t540 走到 (328,135) 又回 (405,62)，残差最优峰却在右下方 6~8 分，
    剔除准星后置信度塌到 0.02。代理量之间打架时，只有画面本身能定案。

做法：
    对 t_start..t_end 逐帧算 |frame_i - frame_{i-1}| 的灰度幅值，按像素取全窗最大值，
    得到「这段时间里每个像素最多动过多少」。静止地形≈0，运动的准星留下一条细线，
    运动的答案星留下一串随姿态变化的星形残影。图上另标出：准星轨迹（绿）、
    跟踪中心轨迹（红）、转向折线真值参照（蓝），三者一眼可比。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_motion.py [t_start] [t_end]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与绘制
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
CURSOR = "_diag_out/cursor_traj_190829.json"  # 准星轨迹
OUT = "_diag_out/nine_s_motion.png"  # 产物


def main():
    t_start = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 起始帧
    t_end = int(sys.argv[2]) if len(sys.argv) > 2 else 700  # 结束帧
    import json  # 准星轨迹
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    prev = None  # 上一帧灰度
    acc = None  # 全窗最大运动幅值
    tick = 0  # 帧号
    while tick <= t_end:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)  # 灰度
        if tick >= t_start:  # 统计窗内
            if prev is not None:  # 有前一帧
                d = np.abs(gray - prev).astype(np.float32)  # 逐像素运动幅值
                acc = d if acc is None else np.maximum(acc, d)  # 全窗取最大
            prev = gray  # 推进（每帧都推进，保证 lag=1）
        else:
            prev = gray  # 预热
        tick += 1
    cap.release()  # 释放
    vis = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 拉伸到 8bit
    canvas = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # 转彩色以便画标记
    ref = json.load(open(CURSOR, encoding="utf-8")) if os.path.exists(CURSOR) else {}  # 准星轨迹
    for k in range(t_start, t_end + 1):  # 准星拖影（绿）
        p = ref.get(str(k))
        if p:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 2, (0, 255, 0), 1)  # 小圈
    for k, color, name in ((t_start, (255, 0, 255), "起点"), (t_end, (0, 255, 255), "终点")):  # 准星首末
        p = ref.get(str(k))
        if p:
            cv2.putText(canvas, name, (int(p[0]) + 6, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # 标注
    h, w = canvas.shape[:2]  # 尺寸
    print(f"运动投影 t{t_start}..{t_end}：幅值 中位 {np.median(acc):.1f} 99分位 {np.percentile(acc, 99):.1f} 最大 {acc.max():.1f}")
    # 找出投影里最亮的若干团块（不含准星）：这些就是「除鼠标外还在动的东西」。
    strong = (acc > np.percentile(acc, 99.5)).astype(np.uint8) * 255  #  top 0.5% 像素
    count, labels, stats, cents = cv2.connectedComponentsWithStats(strong, 8)  # 连通块
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:12] + 1  # 面积前 12 的块
    print("高运动团块（面积、质心、与准星起点/终点的距离）：")
    for k in order:  # 逐块
        cx, cy = cents[k]  # 质心
        area = int(stats[k, cv2.CC_STAT_AREA])  # 面积
        d0 = min([np.hypot(cx - ref[str(t)][0], cy - ref[str(t)][1]) for t in range(t_start, t_end + 1, 20) if str(t) in ref] or [9e9])  # 到准星轨迹的最小距离
        cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 0, 255), 1)  # 标红圈
        cv2.putText(canvas, f"{area}", (int(cx) + 7, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 面积
        print(f"  ({cx:5.0f},{cy:5.0f}) 面积={area:5d} 距准星轨迹最小 {d0:6.1f}px")
    cv2.imwrite(OUT, canvas)  # 落盘
    print(f"已写出 {OUT}（尺寸 {w}x{h}）")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
