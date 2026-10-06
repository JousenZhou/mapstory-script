# -*- coding: utf-8 -*-
"""9s 缺陷修复验证：把绿色准星（上一轮失败录像里停在画面上的鼠标标记）从时序证据里干净地摘掉。

画面定案（本段用 _diag_lie_9s_sheet.py 亲眼看过，代理量之前的推断有错，记录如下）：
    · 场景是一整片互相雷同的星形浮雕；答案星在白相被描成实心白，t0..t250 钉在屏幕中心 (384,254)，
      t255..t335 以 1.36px/帧 匀速往上走并淡出，t335 之后肉眼完全不可见（透明相）。
    · 绿色准星＝鼠标标记。白相时它一直压在答案星上（上一轮跟对了），t335..t420 继续沿同一条
      向上直线走（这就是「跟得上」的部分）；t420 之后目标转向右下，上一轮没跟上，鼠标冲到
      顶部 y≈45 徘徊，t780 起停在右上角 (740,25)，t860 出现「失败」。
    · 所以：t452 之前准星≈目标真值，t452 之后准星＝失败轨迹本身。用户说的「丢失后平滑到右上」
      就是这条鼠标轨迹；「一直往右下」是答案星的真实去向（t841 揭示于 (632,415)，t853 (606,402)）。
    · 生产环境同理且更糟：鼠标由本脚本移到自己的跟踪点上 → 准星永远画在跟踪点 → 一旦滑到
      准星上就自我确认、再也回不来。这解释了「有时能跟上有时不能」。

为什么上一段的 cv2.inpaint 反而更糟（nine_s_fix2_inpaint.log：t612..787 距准星中位 17px、conf 0.46）：
    准星是绿芯＋黑描边的精灵，只按绿色像素 inpaint 会留下一圈暗边在动；更根本的是
    TELEA 补洞的填充值每帧不同，洞本身在相邻帧之间就有残差 —— 对时序残差跟踪器来说
    「每帧不一样的补丁」＝新的人造目标。

正确做法（prevfill）：
    把准星圆盘（含描边，半径 11 处理像素）用「上一帧已清理画面」的同一位置回填。
    于是该处在相邻帧之间逐像素相同 → 残差恰好为 0，且不会引入任何闪烁；准星走过的路径
    变成一段静止的地形拖影，对时序证据完全隐形。答案星自己的残差不受影响。

口径：
    none     ：现状基线。
    inpaint  ：上一段的做法（对照，预期仍失败，留作证据）。
    prevfill ：本段的修复候选。

判据（用户口径「9s 之后不丢失、一直往右下」）：
    真值折线取两个都已被画面证实的端点：准星仍在目标上的 (410,48)@t430，与结尾揭示的 (606.4,401.6)@t853。
    期望：t452..853 与真值中位偏差明显小于基线、末段接近右下角、且不再与准星重合。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_fix3.py [none|inpaint|prevfill] [t_to] [逐帧打印起始帧]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 形态学、连通块、inpaint
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402  # 桩要打在这里（会话模块是直接 import 绑定）
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 准星仍在目标上的最后一个可信点（全尺度）
END_TICK, END_POS = 853.0, np.array([606.4, 401.6])  # 结尾揭示的答案星（_diag_lie_17s_check.py 已验证）
V_TRUTH = (END_POS - TURN_POS) / (END_TICK - TURN_TICK)  # 真值速度 (0.464,0.836) px/帧，右下
DISK_R = 11  # 准星圆盘半径（处理尺度像素）：本体 34×34 全像素→8.5，加黑描边与光晕留余量
GREEN_LOOSE = 12  # 处理尺度绿色判据：g - max(r,b) 门槛（缩放混色，比取证脚本松）

MODE = ["none"]  # 运行期口径
_ORIGIN_RESIZE = None  # resize_for_processing 原实现句柄
_PREV_CLEAN = [None]  # 上一帧已清理画面（prevfill 的回填源）


def truth(tick):  # 真值折线（全尺度）
    return TURN_POS + V_TRUTH * (tick - TURN_TICK)


def green_centroid(frame):  # 最大绿色连通块质心（处理尺度）；无准星返回 None。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8) * 255  # 绿主导
    if int(cv2.countNonZero(mask)) < 15:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块＝准星本体
    return float(cents[k][0]), float(cents[k][1])  # 质心


def patched_resize(frame, scale):  # 在源头摘掉准星：下游（白色检测/时序残差/见证器灰度）都拿到干净画面
    out = _ORIGIN_RESIZE(frame, scale)  # 原缩放
    if MODE[0] == "none" or out is None or out.ndim != 3:  # 基线：登记后原样返回
        _PREV_CLEAN[0] = None if out is None else out.copy()  # 无清理，登记本帧
        return out
    c = green_centroid(out)  # 准星质心
    kill = np.zeros(out.shape[:2], np.uint8)  # 圆盘掩膜
    if c is not None:  # 有准星才挖
        cv2.circle(kill, (int(round(c[0])), int(round(c[1]))), DISK_R, 1, -1)  # 含黑描边与光晕
    if MODE[0] == "prevfill" and int(cv2.countNonZero(kill)) > 0:  # 时序自洽回填：该处相邻帧逐像素相同 → 残差恰为 0
        prev = _PREV_CLEAN[0]  # 上一帧的清理结果（永不包含准星像素，因为是递归回填）
        if prev is not None and prev.shape == out.shape:  # 有可用回填源
            out = out.copy()  # 不改调用方传进来的数组
            out[kill > 0] = prev[kill > 0]  # 回填
    elif int(cv2.countNonZero(kill)) > 0:  # inpaint 口径：空间补洞（对照，预期仍失败）
        out = cv2.inpaint(out, kill * 255, 3, cv2.INPAINT_TELEA)  # 补掉
    _PREV_CLEAN[0] = out.copy()  # 本帧清理结果成为下一帧的回填源
    return out


def main():
    global _ORIGIN_RESIZE  # 原实现句柄
    MODE[0] = sys.argv[1] if len(sys.argv) > 1 else "prevfill"  # 口径
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧
    t_print = int(sys.argv[3]) if len(sys.argv) > 3 else -1  # 逐帧打印起始帧（-1 不打明细）
    _ORIGIN_RESIZE = ss.resize_for_processing  # 保存
    ss.resize_for_processing = patched_resize  # 装桩
    _PREV_CLEAN[0] = None  # 清理源复位
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    rows = []  # (tick, x, y, conf, 距准星, 距真值)
    src_seq = []  # (tick, source) 便于看相位切换点
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        if result.center is not None:  # 有输出
            proc = cv2.resize(frame, None, fx=session.effective_scale, fy=session.effective_scale)  # 独立缩一次取准星（不经桩）
            sc = green_centroid(proc)  # 准星（处理尺度）
            rc = None if sc is None else np.array([sc[0] / session.effective_scale, sc[1] / session.effective_scale])  # 全尺度
            c = np.array(result.center, dtype=np.float64)  # 跟踪中心
            rows.append((tick, c[0], c[1], float(result.confidence),
                         float(np.linalg.norm(c - rc)) if rc is not None else np.nan,
                         float(np.linalg.norm(c - truth(tick)))))  # 记录
            src_seq.append((tick, str(result.source)))  # 相位
            if t_print >= 0 and t_print <= tick <= t_print + 250:  # 逐帧明细窗
                print(f"  t{tick:4d} {str(result.source):>10} ({c[0]:6.1f},{c[1]:6.1f}) "
                      f"conf={result.confidence:.2f} 距准星={rows[-1][4]:7.1f} 距真值={rows[-1][5]:6.1f}")
        tick += 1
    cap.release()  # 释放
    ss.resize_for_processing = _ORIGIN_RESIZE  # 卸桩
    if not rows:
        print("无任何输出")
        return 1
    a = np.array(rows, dtype=np.float64)  # (N,6)
    from collections import Counter  # 相位统计
    print(f"\n口径={MODE[0]}  有输出帧 {len(a)}  相位分布={dict(Counter(s for _, s in src_seq))}")
    print(f"{'窗':>13} {'帧数':>5} {'距准星中位':>10} {'重合<35px%':>10} {'距真值中位':>10} {'距真值末值':>10} {'conf中位':>8}")
    for lo, hi in ((19, 336), (336, 452), (452, 612), (612, 787), (787, 860)):  # 分段
        seg = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]  # 子集
        if len(seg) == 0:
            continue
        print(f"t{lo:4d}..{hi:4d} {len(seg):5d} {np.nanmedian(seg[:, 4]):10.1f} "
              f"{np.nanmean(seg[:, 4] < 35.0) * 100:10.1f} {np.median(seg[:, 5]):10.1f} {seg[-1, 5]:10.1f} {np.median(seg[:, 3]):8.2f}")
    seg = a[(a[:, 0] >= 452) & (a[:, 0] <= 853)]  # 关键窗：9s 到结尾揭示
    if len(seg) > 10:  # 判据
        net = seg[-1, 1:3] - seg[0, 1:3]  # 净位移
        back = float(np.mean(np.diff(seg[:, 1]) < -0.5) * 100)  # x 回退帧占比
        up = float(np.mean(np.diff(seg[:, 2]) < -0.5) * 100)  # y 上攻帧占比（往右上的特征）
        ok_flag = "PASS" if np.median(seg[:, 5]) < 60.0 and net[0] > 100 and net[1] > 200 else "FAIL"
        print(f"\nt452..853 净位移 ({net[0]:.1f},{net[1]:.1f})px  x回退 {back:.1f}%  y上攻 {up:.1f}%  "
              f"末位置 ({seg[-1,1]:.1f},{seg[-1,2]:.1f}) vs 真值 ({END_POS[0]:.0f},{END_POS[1]:.0f})  中位偏差 {np.median(seg[:,5]):.1f}px  → {ok_flag}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
