# -*- coding: utf-8 -*-
"""把准星扫过的像素从运动投影里剔除，剩下的唯一高运动轨迹就是答案（本段定案图）。

为什么这是最后一个代理量之争：
    _diag_out/nine_s_motion.png（全窗最大运动幅值）里只有一条亮带，而它正好沿着准星轨迹——
    于是「答案在准星处」与「答案被准星的强对比掩盖」两种解释都能吻合，投影图本身分不出来。
    本脚本把每一帧准星覆盖过的像素在差分里强制置零（准星在 t 或 t-1 出现过的地方都不计入），
    等于把鼠标从画面里擦掉再问「还有谁在动」。

判读：
    若擦掉准星后出现另一条连续的高运动带 → 那条带就是答案，方向即用户说的「一直往右下」，
      并且它此前被准星的强残差压住；修复方向=证据里剔除准星（颜色掩膜），且要按「紧口径」
      只擦鼠标本体，别把答案一起擦掉（_diag_lie_9s_fix.py 的 disk 口径已证明擦太狠会崩）。
    若擦掉准星后什么都不剩（只剩地形级的弱边）→ 答案在透明相确实几乎不产生残差，
      那么整条 border 弱测量路线都不该被信任，9s 之后应当改用别的信息源。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_erase.py [t_start] [t_end] [erase_radius]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与绘制
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
OUT = "_diag_out/nine_s_erase.png"  # 产物


def reticle_px(frame):  # 亮绿准星质心（本帧像素坐标）；无准星返回 None。与取证脚本同判据。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 170) & (g - r > 45) & (g - b > 45)).astype(np.uint8) * 255  # 纯绿
    if int(cv2.countNonZero(mask)) < 20:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    return float(cents[k][0]), float(cents[k][1])  # 质心


def main():
    t_start = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 起始帧
    t_end = int(sys.argv[2]) if len(sys.argv) > 2 else 700  # 结束帧
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 22  # 擦除半径（像素）：准星本体 17 + 抗锯齿
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    prev = None  # 上一帧灰度
    acc = None  # 擦除准星后的全窗最大运动幅值
    last_ret = None  # 上一帧准星质心（用于覆盖拖影）
    tick = 0  # 帧号
    while tick <= t_end:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)  # 灰度
        if tick >= t_start and prev is not None:  # 统计窗内且有前一帧
            d = np.abs(gray - prev).astype(np.float32)  # 逐像素运动幅值
            kill = np.zeros(d.shape, np.uint8)  # 擦除掩膜
            for p in (reticle_px(frame), last_ret):  # 本帧与上一帧准星都擦（覆盖拖影）
                if p is not None:
                    cv2.circle(kill, (int(round(p[0])), int(round(p[1]))), radius, 1, -1)  # 画盘
            d[kill > 0] = 0.0  # 擦掉准星
            acc = d if acc is None else np.maximum(acc, d)  # 全窗取最大
        rp = reticle_px(frame)  # 准星（用于下一帧的拖影擦除）
        if rp is not None:
            last_ret = rp  # 推进
        prev = gray  # 推进
        tick += 1
    cap.release()  # 释放
    hi = float(np.percentile(acc, 99.9))  # 高运动门限
    qs = [np.percentile(acc, q) for q in (90, 99, 99.9, 99.99)]  # 擦除后的幅值分布（判断还剩多少信号）
    print(f"擦除后幅值分位：90={qs[0]:.1f} 99={qs[1]:.1f} 99.9={qs[2]:.1f} 99.99={qs[3]:.1f} 最大={acc.max():.1f}")
    clip = float(sys.argv[4]) if len(sys.argv) > 4 else 45.0  # 低幅值截断：准星顶高了分位，全窗最大已不可用
    vis = cv2.normalize(np.clip(acc, 0, clip), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 按弱信号截断拉伸
    canvas = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # 彩色画布
    strong = (acc > clip).astype(np.uint8) * 255  # 高运动像素（弱信号口径）
    n = cv2.connectedComponentsWithStats(strong, 8)  # 连通块
    count, labels, stats, cents = n  # 解包
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:14] + 1 if count > 1 else []  # 面积前 14
    print(f"擦除准星后的运动投影 t{t_start}..{t_end}（半径 {radius}px，截断 {clip:.0f}）：")
    print("剩余高运动团块（按面积）：")
    for k in order:  # 逐块
        cx, cy = cents[k]  # 质心
        area = int(stats[k, cv2.CC_STAT_AREA])  # 面积
        cv2.circle(canvas, (int(cx), int(cy)), 8, (0, 0, 255), 1)  # 红圈
        cv2.putText(canvas, str(area), (int(cx) + 9, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)  # 面积
        print(f"  ({cx:5.0f},{cy:5.0f}) 面积={area:5d}")
    cv2.imwrite(OUT, canvas)  # 落盘
    print(f"已写出 {OUT}")
    return 0


last_ret = None  # 占位（已改为 main 内局部变量）

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
