# -*- coding: utf-8 -*-
"""9s 缺陷的修复验证：把游戏画在鼠标位置的绿色准星从画面源头抹掉，看跟踪是否回到「一直往右下」。

诊断结论（本脚本只验证，论证在 nine_s_cursor.log / nine_s_mask.log）：
    t890（回合结束后）画面里出现的是白色手型系统光标，而整局中只看到绿色准星——
    准星是游戏在「当前鼠标位置」画的十字标记，录像把它烧进了画面，于是回放里存在
    一个「上一轮跟踪输出」的实体：
      · 高对比、纯绿、边缘锐利、逐帧运动 → 时序残差里自评分 5.32 的强峰（同帧全图上限 7.31）；
      · 生产环境同样有它：脚本把鼠标移向跟踪点 → 准星就画在跟踪点上 → 正反馈，永不脱离。
    实测：t452..612 跟踪中心与准星 100% 重合（中位 17px、最小 0.1px）；把准星圆盘从证据里
    填零后该处得分塌到 -0.03，而剩余图里最优峰在右下方且更高（6.6~7.9）。

两种口径：
    none   ：现状基线。
    inpaint：在 resize_for_processing 之后、所有下游消费之前用 cv2.inpaint 把绿色像素补掉。
             残差里没有本体也没有拖影，白色强测量与见证器灰度同时受益。
    （tight 只抠本体——拖影仍在，实测照旧贴住准星；disk 整盘填零——把答案一起挖掉，conf 塌到 0.02。
      两种都已否证，不再保留。）

判据（用户口径「9s 之后一直往右下」）：
    真值折线 (415.6,52)@t450 → (606,401)@t787，其中 t787 端点是 17s 缺陷收官时验证到 4px 的真值。
    期望：t452..787 与真值的中位偏差显著下降、末段接近 (606,401)、且不再与准星重合。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_fix2.py [none|inpaint] [t_to] [t_from_print]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 形态学、连通块、inpaint
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402  # 会话模块：桩要打在这里（它是直接 import 绑定）
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 450.0, np.array([415.6, 52.0])  # 转向点（全尺度）
END_TICK, END_POS = 787.0, np.array([606.0, 401.0])  # 17s 真值（已验证到 4px）
V_UP = np.array([0.179, -1.383])  # 白相末向上速度 px/帧
V_DOWN = (END_POS - TURN_POS) / (END_TICK - TURN_TICK)  # 转向后速度 px/帧（右下）
GREEN_LOOSE = 12  # 处理尺度下绿色判据：g - max(r,b) 的门槛（缩放会混色，比取证脚本更松）
DILATE_K = 5  # 掩膜膨胀核（覆盖抗锯齿边）

MODE = ["none"]  # 运行期口径
_ORIGIN_RESIZE = None  # resize_for_processing 原实现句柄


def truth(tick):  # 真值折线（全尺度）
    return TURN_POS + (V_UP * (tick - TURN_TICK) if tick <= TURN_TICK else V_DOWN * (tick - TURN_TICK))


def green_mask(frame):  # 准星绿色像素掩膜与质心（输入为处理尺度 BGR）；无准星返回 None。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8) * 255  # 绿主导
    if int(cv2.countNonZero(mask)) < 15:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块＝准星本体
    return (labels == k).astype(np.uint8) * 255, (float(cents[k][0]), float(cents[k][1]))  # 掩膜 + 质心


def patched_resize(frame, scale):  # 在源头抹掉准星：下游（白色检测/时序残差/见证器灰度）都拿到干净画面
    out = _ORIGIN_RESIZE(frame, scale)  # 原缩放
    if MODE[0] != "inpaint" or out is None or out.ndim != 3:  # 非本口径则原样返回
        return out
    got = green_mask(out)  # 准星掩膜
    if got is None:  # 无准星
        return out
    mask = cv2.dilate(got[0], np.ones((DILATE_K, DILATE_K), np.uint8))  # 膨胀覆盖抗锯齿边
    return cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)  # 补掉


def main():
    global _ORIGIN_RESIZE  # 原实现句柄
    MODE[0] = sys.argv[1] if len(sys.argv) > 1 else "inpaint"  # 口径
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧
    t_print = int(sys.argv[3]) if len(sys.argv) > 3 else -1  # 逐帧打印起始帧（-1 不打印明细）
    _ORIGIN_RESIZE = ss.resize_for_processing  # 保存
    ss.resize_for_processing = patched_resize  # 装桩（会话内引用的是 shape_session 模块属性）
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    rows = []  # (tick, x, y, conf, 距准星, 距真值, source码)
    src_map = {}  # source 字符串 → 编码
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        if result.center is not None:  # 有输出
            proc = cv2.resize(frame, None, fx=session.effective_scale, fy=session.effective_scale)  # 独立缩一次取准星（不经桩，保持取证口径）
            got = green_mask(proc)  # 准星
            rc = None if got is None else np.array([got[1][0] / session.effective_scale, got[1][1] / session.effective_scale])  # 全尺度
            c = np.array(result.center, dtype=np.float64)  # 跟踪中心
            code = src_map.setdefault(str(result.source), len(src_map))  # 相位编码
            rows.append((tick, c[0], c[1], float(result.confidence),
                         float(np.linalg.norm(c - rc)) if rc is not None else np.nan,
                         float(np.linalg.norm(c - truth(tick))), code))  # 记录
            if t_print >= 0 and t_print <= tick <= t_print + 200:  # 逐帧明细窗
                print(f"  t{tick:4d} {str(result.source):>10} ({c[0]:6.1f},{c[1]:6.1f}) "
                      f"conf={result.confidence:.2f} 距准星={rows[-1][4]:7.1f} 距真值={rows[-1][5]:6.1f}")
        tick += 1
    cap.release()  # 释放
    ss.resize_for_processing = _ORIGIN_RESIZE  # 卸桩
    if not rows:
        print("无任何输出")
        return 1
    a = np.array([r[:6] for r in rows], dtype=np.float64)  # (N,6)
    codes = {v: k for k, v in src_map.items()}  # 编码 → source 名
    print(f"\n口径={MODE[0]}  有输出帧 {len(a)}  相位分布：" +
          ", ".join(f"{codes[int(v)]}={int((np.array([r[6] for r in rows]) == v).sum())}" for v in sorted(src_map.values())))
    print(f"{'窗':>13} {'帧数':>5} {'距准星中位':>10} {'重合<35px%':>10} {'距真值中位':>10} {'距真值末值':>10} {'conf中位':>8}")
    for lo, hi in ((19, 344), (344, 452), (452, 612), (612, 787), (787, 900)):  # 分段
        seg = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]  # 子集
        if len(seg) == 0:
            continue
        print(f"t{lo:4d}..{hi:4d} {len(seg):5d} {np.nanmedian(seg[:, 4]):10.1f} "
              f"{np.nanmean(seg[:, 4] < 35.0) * 100:10.1f} {np.median(seg[:, 5]):10.1f} {seg[-1, 5]:10.1f} {np.median(seg[:, 3]):8.2f}")
    seg = a[(a[:, 0] >= 452) & (a[:, 0] <= 787)]  # 关键窗
    if len(seg) > 10:  # 判据
        net = seg[-1, 1:3] - seg[0, 1:3]  # 净位移
        back = float(np.mean(np.diff(seg[:, 1]) < -0.5) * 100)  # x 回退帧占比
        ok_flag = "PASS" if np.median(seg[:, 5]) < 60.0 and net[0] > 100 and net[1] > 200 else "FAIL"
        print(f"\nt452..787 净位移 ({net[0]:.1f},{net[1]:.1f})px  x回退 {back:.1f}%  "
              f"末位置 ({seg[-1,1]:.1f},{seg[-1,2]:.1f}) vs 真值 ({END_POS[0]:.0f},{END_POS[1]:.0f})  → {ok_flag}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
