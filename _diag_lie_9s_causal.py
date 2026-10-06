# -*- coding: utf-8 -*-
"""因果实验：把画面里的绿色准星精灵整体平移一段距离再喂给会话，看跟踪器跟不跟着走。

为什么必须先做这个实验（否则修复方向可能是错的）：
    录像里的准星＝上一轮脚本的鼠标位置，而上一轮跑的就是同一套算法、同一份素材、同一个随机种子。
    于是「回放跟踪中心与准星 100% 重合」这件事有两种完全解释不了的相容读法：
      A 吸引子：准星在时序残差里是个强假峰，跟踪器被它吸住 → 修复＝把准星从证据里摘掉；
      B 影子：跟踪器走的是自己那套算法必然走的那条路，准星只是这条路的可视化，
              两者高度重合是因为回放本来就在复现同一轮计算 → 摘掉准星什么也修不好。
    前一段的 inpaint 与本段的 prevfill 都不但没让跟踪器离开准星，反而贴得更紧（t612..787 距准星
    中位 5.3px），这已经是 B 的强信号，但还不够——必须做干预实验。

判读：
    跟踪中心跟随「平移后的假准星」→ A 成立，摘准星是正确修复方向（但要换成不产生残差的摘法）。
    跟踪中心仍留在准星原位（或去别处）→ B 成立，9s 的丢失与准星无关，
    真正要修的是「目标转向时跟踪器出不去自己所在的局部」这一层。

做法：
    在 resize_for_processing 之后：用绿色掩膜把准星抠出来（原位置 inpaint 补掉），
    把它贴到偏移 (dx,dy) 处理像素处。掩膜用「绿主导 ∪ 圆盘」以带上黑描边。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_causal.py [dx] [dy] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 抠图与贴图
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 准星仍在目标上的最后可信点（全尺度）
END_TICK, END_POS = 853.0, np.array([606.4, 401.6])  # 结尾揭示的答案星
V_TRUTH = (END_POS - TURN_POS) / (END_TICK - TURN_TICK)  # 真值速度（右下）
GREEN_LOOSE = 12  # 处理尺度绿色判据
DISK_R = 11  # 带上黑描边的圆盘半径（处理像素）

DX = [0]  # 平移量（处理像素）
DY = [0]  # 平移量
_ORIGIN_RESIZE = [None]  # 原实现


def sprite_mask(frame):  # 准星掩膜（绿主导 ∪ 质心圆盘）与质心；无准星返回 (None, None)。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    green = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8)  # 绿主导
    if int(green.sum(dtype=np.int64)) < 15:  # 未检出
        return None, None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(green * 255, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None, None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块
    m = (labels == k).astype(np.uint8)  # 只取准星那一块
    disk = np.zeros_like(m)  # 圆盘（补上描边与光晕）
    cv2.circle(disk, (int(round(cents[k][0])), int(round(cents[k][1]))), DISK_R, 1, -1)  # 画盘
    return np.maximum(m, disk), (float(cents[k][0]), float(cents[k][1]))  # 并集掩膜 + 质心


def patched_resize(frame, scale):  # 抠掉准星并贴到偏移处
    out = _ORIGIN_RESIZE[0](frame, scale)  # 原缩放
    if out is None or out.ndim != 3 or (DX[0] == 0 and DY[0] == 0):  # 无平移则等价基线
        return out
    m, c = sprite_mask(out)  # 掩膜与质心
    if m is None:  # 无准星
        return out
    patch = out.copy()  # 精灵原样（含其覆盖到的地形，够用）
    out = cv2.inpaint(out, (m * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)  # 原位补掉
    h, w = out.shape[:2]  # 尺寸
    ys, xs = np.nonzero(m)  # 掩膜像素
    dy, dx = int(DY[0]), int(DX[0])  # 平移
    tx, ty = xs + dx, ys + dy  # 目标坐标
    keep = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)  # 越界裁剪
    out[ty[keep], tx[keep]] = patch[ys[keep], xs[keep]]  # 贴到新位置
    return out


def truth(tick):  # 真值折线（全尺度）
    return TURN_POS + V_TRUTH * (tick - TURN_TICK)


def main():
    DX[0] = int(sys.argv[1]) if len(sys.argv) > 1 else -60  # 处理像素偏移
    DY[0] = int(sys.argv[2]) if len(sys.argv) > 2 else 90  # 处理像素偏移
    t_to = int(sys.argv[3]) if len(sys.argv) > 3 else 900  # 结束帧
    _ORIGIN_RESIZE[0] = ss.resize_for_processing  # 保存原实现
    ss.resize_for_processing = patched_resize  # 装桩
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    rows = []  # (tick, x, y, conf, 距原准星, 距假准星, 距真值)
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进（帧里已被换成假准星）
        if result.center is not None:  # 有输出
            proc = cv2.resize(frame, None, fx=session.effective_scale, fy=session.effective_scale)  # 未打桩的原始缩放帧
            _, c = sprite_mask(proc)  # 原准星质心（处理尺度）
            sc = session.effective_scale or 1.0  # 尺度
            orig = None if c is None else np.array([c[0] / sc, c[1] / sc])  # 原位置（全尺度）
            fake = None if c is None else np.array([(c[0] + DX[0]) / sc, (c[1] + DY[0]) / sc])  # 假位置
            cc = np.array(result.center, dtype=np.float64)  # 跟踪中心
            rows.append((tick, cc[0], cc[1], float(result.confidence),
                         float(np.linalg.norm(cc - orig)) if orig is not None else np.nan,
                         float(np.linalg.norm(cc - fake)) if fake is not None else np.nan,
                         float(np.linalg.norm(cc - truth(tick)))))  # 记录
        tick += 1
    cap.release()  # 释放
    ss.resize_for_processing = _ORIGIN_RESIZE[0]  # 卸桩
    if not rows:
        print("无任何输出")
        return 1
    a = np.array(rows, dtype=np.float64)  # (N,7)
    print(f"平移量 ({DX[0]},{DY[0]}) 处理像素 = ({DX[0] * 2:.0f},{DY[0] * 2:.0f}) 全尺度（准星已不在原位，原位被补洞）")
    print(f"{'窗':>13} {'帧数':>5} {'距原准星':>9} {'距假准星':>9} {'贴假<35px%':>11} {'贴原<35px%':>11} {'距真值':>8} {'conf':>6}")
    for lo, hi in ((19, 336), (336, 452), (452, 612), (612, 787), (787, 860)):  # 分段
        seg = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]  # 子集
        if len(seg) == 0:
            continue
        print(f"t{lo:4d}..{hi:4d} {len(seg):5d} {np.nanmedian(seg[:, 4]):9.1f} {np.nanmedian(seg[:, 5]):9.1f} "
              f"{np.nanmean(seg[:, 5] < 35.0) * 100:11.1f} {np.nanmean(seg[:, 4] < 35.0) * 100:11.1f} "
              f"{np.median(seg[:, 6]):8.1f} {np.median(seg[:, 3]):6.2f}")
    seg = a[(a[:, 0] >= 452) & (a[:, 0] <= 853)]  # 关键窗
    if len(seg) > 10:  # 判读
        d_fake, d_orig = float(np.nanmedian(seg[:, 5])), float(np.nanmedian(seg[:, 4]))
        verdict = "A：准星是吸引子（跟踪器跟到了假准星）" if d_fake < d_orig and d_fake < 60 else \
            "B：准星只是影子（跟踪器没跟假准星）"
        print(f"\nt452..853 距假准星中位 {d_fake:.1f}px vs 距原准星中位 {d_orig:.1f}px  → {verdict}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
