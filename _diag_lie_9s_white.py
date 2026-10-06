# -*- coding: utf-8 -*-
"""逐帧追出「白色答案星」的真实轨迹（t 白相 + 淡出尾段），作为 9s 缺陷的地理真值。

画面定案（_diag_out/sheet_0_0_30_360.png / sheet_0_0_800_920.png）：
    · 场景是一整片互相雷同的星形浮雕地形，答案星在白相被描成纯白轮廓，别的时间与周围星形无法目视区分；
    · 绿色准星＝鼠标标记，白相时它一直压在答案星上（玩家点得对），t780 之后停在右上角空处，
      t860 出现「失败」字样 —— 本素材是一局失败录像，「往右上」正是上一轮丢目标后鼠标的下场；
    · 所以：准星轨迹不是真值，白相答案星才是。此前把 (415.6,52)@t450 当转向点是拿跟踪器自己的输出当真值，
      几何上根本站不住（y=52 在画面顶部，而白相末期答案在 y≈310–345 的中下部）。

做法：
    纯白判据（三通道都高且互相差得少）→ 连通块 → 只保留「面积/外接框」像一枚星形的块 →
    质心即答案星。逐帧打印，并给出相邻帧位移，方便肉眼判断它到底是连续移动还是换行跳变。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_white.py [t_from] [t_to] [step]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与形态学
import numpy as np  # 数值

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
FPS = 50.207  # 帧率


def white_stars(frame):  # 本帧所有「像星形」的纯白连通块：[(cx, cy, area, w, h)]。
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]  # 拆通道
    lo = np.minimum(np.minimum(b.astype(np.int16), g.astype(np.int16)), r.astype(np.int16))  # 三通道最小＝白度
    sat = np.maximum(np.maximum(b, g), r).astype(np.int16) - lo.astype(np.int16)  # 饱和度（白应很低）
    mask = ((lo > 190) & (sat < 40)).astype(np.uint8) * 255  # 亮且不饱和
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 补轮廓小断口
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    out = []  # 结果
    for k in range(1, count):  # 逐块
        area = int(stats[k, cv2.CC_STAT_AREA])  # 面积
        w, h = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])  # 外接框
        if area < 120 or w < 12 or h < 12 or w > 120 or h > 120:  # 星形量级
            continue
        fill = area / float(w * h)  # 实心度：星形≈0.35，文字/方块≈0.7+
        if fill > 0.62:  # 太实心的是倒计时数字或文字，不是星
            continue
        out.append((float(cents[k][0]), float(cents[k][1]), area, w, h))  # 记录
    return out


def reticle_px(frame):  # 亮绿准星质心；无准星返回 None。
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
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 起始帧
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 470  # 结束帧
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # 抽样步长
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    tick = 0  # 帧号
    prev = None  # 上一命中帧的 (tick, 位置)
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick >= t_from and tick % step == 0:  # 抽样窗内
            stars = white_stars(frame)  # 白色星形
            rp = reticle_px(frame)  # 准星
            if stars:  # 有星
                stars.sort(key=lambda s: -s[2])  # 面积降序
                best = stars[0]  # 最大星
                mv = "" if prev is None else f" Δ=({best[0] - prev[1][0]:+6.1f},{best[1] - prev[1][1]:+6.1f})/{tick - prev[0]:d}f"  # 相对上次命中
                rc = "" if rp is None else f" 准星=({rp[0]:5.0f},{rp[1]:5.0f}) 距星={np.hypot(rp[0] - best[0], rp[1] - best[1]):5.1f}"  # 准星关系
                print(f"t{tick:4d} {tick / FPS:5.2f}s 星={len(stars)} 主星=({best[0]:5.1f},{best[1]:5.1f}) 面积={best[2]:4d} 框={best[3]}x{best[4]}{rc}{mv}"
                      + (f"  其余={[(round(s[0]), round(s[1]), s[2]) for s in stars[1:4]]}" if len(stars) > 1 else ""))
                prev = (tick, (best[0], best[1]))  # 推进
            else:  # 无星
                rc = "" if rp is None else f" 准星=({rp[0]:5.0f},{rp[1]:5.0f})"  # 仍报准星
                print(f"t{tick:4d} {tick / FPS:5.2f}s 无白色星{rc}")
        tick += 1
    cap.release()  # 释放
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
