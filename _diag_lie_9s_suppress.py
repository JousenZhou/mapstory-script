# -*- coding: utf-8 -*-
"""9s 缺陷修复验证（证据侧摘准星）：不改画面，只把准星圆盘处的时序残差证据压掉。

为什么换成证据侧下手（前面三次都栽在改画面上）：
    _diag_lie_9s_tbd.py 的长基线相干积分给出决定性线索：透明相里全场最强的相干轨迹峰
    全都落在 (336-416, 58-112) 全尺度那条带上——正是准星精灵的位置；而真值折线上的
    相干得分只有 P2-P30 分位，被准星甩开 3 个数量级。基线口径 t452..612 跟踪中心
    距准星中位 16.7px、100% 重合、连续 160 帧不分离，也印证了同一件事：
    回放画面里有一个「上一轮鼠标轨迹」烧进去的高对比绿色十字，它自己就在动，
    时序残差里它是最强的那个目标，跟踪器锁的其实是它。
    之前的 inpaint / prevfill / 平移因果实验都在改画面：补洞值逐帧不同（TELEA）或补丁是
    静止块，等于在准星原处新造一个残差源，所以「摘掉准星」反而贴得更紧，实验结论被污染。
    证据侧压制不留任何新残差：圆盘内残差置零 = 「此处无信息」，既不造峰也不挖真答案以外的东西。

口径：
    none   ：现状基线。
    sup12  ：以准星质心为圆心、半径 12 处理像素（=24 全尺度）圆盘内证据置零。
    sup20 ：半径 20 处理像素（=40 全尺度）——连准星光晕与拖影一起屏蔽，看代价。

判据（用户口径「9s 之后不丢失、一直往右下」）：
    真值折线 (410,48)@t430 → (606.4,401.6)@t853（端点为结尾揭示的答案星，17s 缺陷已验证到 4px）。
    期望：t452..787 距真值中位显著下降、净位移朝右下、且不再与准星重合。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_suppress.py [none|sup12|sup20] [t_to] [t_print_from]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码、画圆盘掩膜
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.shape_tracking import ParticleShapeTracker, RotationWitness, _to_numpy_array  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
# 真值锚点（全尺度）：t430 = 白相末期目标转到画面顶部；t776 = 17s 收官时验证过的旋转窗探针质心。
# 为什么中间可以当直线匀速：白相实测速度 1.36px/帧，t430→t776 直线距离 408px，346 帧恰好走得完，
# 任何弯曲都需要更高速度或更晚到达，与锚点矛盾。t776 之后答案在右下角驻留（17s 判据另校 t853）。
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 顶部转向点
CORNER_TICK, CORNER_POS = 776.0, np.array([594.4, 412.6])  # 右下角驻留点（旋转窗末探针质心）
# 实测锚点（全尺度）：t702/t776 为旋转窗探针质心，t853 为揭示帧星形质心（17s 缺陷收官时校到 4px）。
ANCHORS = ((702, 610.6, 333.7), (776, 594.4, 412.6), (853, 606.4, 401.6))
V_DOWN = (CORNER_POS - TURN_POS) / (CORNER_TICK - TURN_TICK)  # 右下段真值速度
GREEN_LOOSE = 12  # 处理尺度绿色判据：g - max(r,b) 门槛（缩放会混色，比取证脚本松）
# 口径背书（_diag_lie_9s_maskopt.py 的证据层寻优结论）：
#   准星足迹 p50=127px、外接框 17x17（处理尺度），沿它自己膨胀 3-9px 就能把「准星最强相干轨迹」
#   从 7.87/10.12 压到 6.13/3.66，而「答案最强相干轨迹」始终 6.25/6.27 一分不掉（挖掉质量仅 2-3.7%），
#   比值由 0.62-0.79（准星赢）翻到 1.02-1.71（答案赢）。
#   而整盘口径 disk6 比值只有 0.94（压不住），半径再大就开始挖掉答案星的残差环（sup9/16/26 实拍回退 17s）。
#   → 所以只保留足迹口径：fd3/5/7/9 = 本帧足迹膨胀 3/5/7/9；fd7l4 = 近 4 帧并集再膨胀 7。
DISKS = {"none": 0, "fd3": 0, "fd5": 0, "fd7": 0, "fd9": 0, "fd7l4": 0,
         "sup9": 9, "sup16": 16}  # 口径→盘半径（sup*）或 0（足迹口径）
FOOT = {"fd3": (1, 3), "fd5": (1, 5), "fd7": (1, 7), "fd9": (1, 9), "fd7l4": (4, 7)}  # (回看几帧, 膨胀核奇数)
MODE = ["none"]  # 运行期口径
_INJECT = os.environ.get("LIE_MASK_SCOPE", "all")  # 注入层：all=证据出厂（默认）/border=只遮弱测量/witness=只遮见证器
_ORIGIN_OBSERVE = None  # observe_border 原实现（scope=border 时用）
_ORIGIN_WITNESS_UPDATE = None  # witness.update 原实现（scope=witness 时用）
_ORIGIN_RESIZE = None  # resize_for_processing 原实现句柄
_DISK = {"xy": None}  # 本帧准星质心（处理尺度）
_TRAIL = {"masks": []}  # 最近几帧的绿色足迹（供多 lag 残差覆盖）


def truth(tick):  # 真值折线（全尺度）：t430 顶部 → t776 右下角 → 之后驻留
    if tick <= TURN_TICK:
        return TURN_POS  # 白相段本脚本不评
    if tick <= CORNER_TICK:
        return TURN_POS + V_DOWN * (tick - TURN_TICK)  # 右下段直线
    return CORNER_POS  # 驻留


def green_mask(frame):  # 准星绿色像素掩膜与质心（处理尺度 BGR）；未检出返回 None。
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)  # 拆通道
    mask = ((g > 90) & (g - np.maximum(r, b) > GREEN_LOOSE)).astype(np.uint8) * 255  # 绿主导
    if int(cv2.countNonZero(mask)) < 15:  # 未检出
        return None
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)  # 连通块
    if count <= 1:  # 只有背景
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 最大块＝准星本体
    return (labels == k).astype(np.uint8), (float(cents[k][0]), float(cents[k][1]))  # 掩膜 + 质心


def record_disk(frame, scale):  # 桩：只读，量出本帧准星质心与绿色足迹，不改画面
    out = _ORIGIN_RESIZE(frame, scale)  # 原缩放
    got = green_mask(out) if out is not None and out.ndim == 3 else None  # 量准星
    if got is None:  # 无准星：足迹清空（避免拿旧位置误压）
        _DISK["xy"] = None
        _TRAIL["masks"] = []  # 足迹也断掉
    else:  # 有准星：记录质心并把本帧足迹压入环形列表
        _DISK["xy"] = got[1]  # 质心
        _TRAIL["masks"] = ([got[0]] + _TRAIL["masks"])[: max(FOOT.get(MODE[0], (1, 3))[0], 1)]  # 保留最近 K 帧
    return out


def foot_mask(shape):  # 足迹掩膜：最近 K 帧绿色像素的并集再膨胀（十字是细线条，不挖整盘）
    look, kernel = FOOT[MODE[0]]  # 本口径的回看帧数与膨胀核
    masks = _TRAIL["masks"][:look]  # 取用几帧
    if not masks:  # 无足迹
        return None
    union = masks[0].copy()  # 并集
    for m in masks[1:]:  # 逐帧合并
        if m.shape == union.shape:
            union |= m  # 按位或
    if kernel > 1:  # 膨胀覆盖抗锯齿边与拖影
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    h, w = shape[:2]  # 证据尺寸
    if union.shape[:2] != (h, w):  # 尺度不一致则放弃（宁可不误压）
        return None
    return union.astype(np.float32)  # 0/1 浮点掩膜


def suppress(norm):  # 圆盘或足迹内置零：证据侧「此处无信息」，后端张量原样保持在后端
    radius = DISKS[MODE[0]]  # 盘口径半径（足迹口径为 0）
    c = _DISK["xy"]  # 准星质心
    if norm is None or (radius <= 0 and MODE[0] not in FOOT):  # 不压制
        return norm
    arr = _to_numpy_array(norm)  # 下载看尺寸与类型（torch 时这一步是 D2H）
    h, w = arr.shape[:2]  # 证据尺寸
    if MODE[0] in FOOT:  # 足迹口径：只压准星自己的像素
        foot = foot_mask(arr.shape)  # 最近 K 帧足迹并集
        if foot is None:  # 无足迹则本帧不压
            return norm
        keep = 1.0 - foot  # 足迹内 0、足迹外 1
    else:  # 整盘口径
        if radius <= 0 or c is None:  # 不压制
            return norm
        mask = np.zeros((h, w), np.uint8)  # 圆盘掩膜
        cv2.circle(mask, (int(round(c[0])), int(round(c[1]))), radius, 1, -1)  # 画盘
        if mask.max() == 0:  # 盘在图外
            return norm
        keep = (1 - mask).astype(np.float32)  # 盘内 0、盘外 1
    keep = keep.astype(np.float32)  # 统一类型
    if isinstance(norm, np.ndarray):  # NumPy 后端
        return (arr * keep).astype(norm.dtype, copy=False)  # 直接乘
    try:  # torch 路径：同设备同类型乘掩膜，生成新张量（不写静态输出缓冲）
        from torch import from_numpy  # 取 torch.from_numpy
        return norm * from_numpy(keep).to(norm.device, norm.dtype)  # 盘内清零、盘外不动
    except Exception:  # 后端不支持就交回下载后乘好的 numpy（探针口径，允许退化）
        return (arr * keep).astype(np.float32)


def main():
    global _ORIGIN_RESIZE, _ORIGIN_OBSERVE, _ORIGIN_WITNESS_UPDATE  # 原实现句柄
    MODE[0] = sys.argv[1] if len(sys.argv) > 1 else "sup12"  # 口径
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧
    t_print = int(sys.argv[3]) if len(sys.argv) > 3 else -1  # 逐帧打印起始帧（-1 不打印）
    _ORIGIN_RESIZE = ss.resize_for_processing  # 保存
    ss.resize_for_processing = record_disk  # 装桩（会话内引用模块属性）
    print(f"注入层={_INJECT}")
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    origin_aligner_update = session.aligner.update  # 对齐器原实现（实例上包一层）

    def patched_aligner_update(frame):  # 证据出厂后再压制准星：下游（弱测量 + 见证器）都拿到干净证据
        ev = origin_aligner_update(frame)  # 原证据
        if ev is not None and MODE[0] != "none" and _INJECT == "all":  # 仅出厂档
            ev.normalized = suppress(ev.normalized)  # 足迹/盘内置零
        return ev

    session.aligner.update = patched_aligner_update  # 装桩
    if _INJECT == "border":  # 只遮弱测量：见证器保持原始证据（候选修复 G）
        _ORIGIN_OBSERVE = ParticleShapeTracker.observe_border

        def patched_observe_border(self, temporal_map):  # 类级打桩：入参遮，见证器读的原图不动
            if MODE[0] != "none":
                temporal_map = suppress(temporal_map)  # 准星足迹内置零后再喂粒子
            return _ORIGIN_OBSERVE(self, temporal_map)

        ParticleShapeTracker.observe_border = patched_observe_border
    elif _INJECT == "witness":  # 只遮见证器：弱测量保持原始证据（反向对照）
        _ORIGIN_WITNESS_UPDATE = RotationWitness.update

        def patched_witness_update(self, evidence_np, gray_np, dt=None):  # 类级打桩：只改证据入参
            if MODE[0] != "none":
                evidence_np = suppress(evidence_np)
            return _ORIGIN_WITNESS_UPDATE(self, evidence_np, gray_np, dt)

        RotationWitness.update = patched_witness_update
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    rows = []  # (tick, x, y, conf, 距准星, 距真值, source码)
    src_map = {}  # source 字符串 → 编码
    last_wit = -1  # 已打印的见证器检查点帧号（逐检查点取证用）
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        result = session.update(frame, float(fps))  # 推进
        wit = getattr(session, "_witness_trace", None)  # 本检查点的见证器快照（未到检查点则与上一帧同帧号）
        if wit is not None and 0 <= t_print <= tick <= t_print + 400 and wit["frame"] != last_wit:  # 逐检查点取证
            last_wit = wit["frame"]  # 记号
            inv = 1.0 / session.effective_scale  # 处理尺度 → 全尺度
            bc = wit["best"]  # 最优候选（处理尺度）
            best_txt = "无" if bc is None else (
                f"({bc[0] * inv:6.1f},{bc[1] * inv:6.1f}){bc[2]:6.1f}°/dir{wit['best_dir']:.2f}"
                f"/n{wit['best_samples']:2d}/age{wit['best_age']:3.0f}")  # 最优候选描述
            anc = wit["anchor"]  # 锚点（处理尺度）
            anchor_txt = "无" if anc is None else f"({anc[0] * inv:6.1f},{anc[1] * inv:6.1f})"  # 锚点描述
            # 脚下簇现场复取（cluster_at 纯读）：半径与簇心是「自确认」唯一未被日志覆盖的两个量。
            tr = session.tracker  # 跟踪器
            rad = 12.0 if tr is None else max(float(tr.size_ref) * 0.6, 12.0)  # 与生产同口径的查询半径
            cc = None if tr is None else session.witness.cluster_at(np.asarray(wit["center"], dtype=np.float32), rad)  # 脚下簇
            cc_txt = "无" if cc is None else f"({cc.center[0] * inv:6.1f},{cc.center[1] * inv:6.1f}){cc.windowed_deg:5.1f}°"  # 脚下簇心
            d_cb = "-" if (cc is None or bc is None) else f"{np.linalg.norm(cc.center - np.asarray(bc[:2], dtype=np.float32)) * inv:5.1f}"  # 脚下簇心↔最优（全尺度）
            print(f"  W t{wit['frame']:4d} conf={result.confidence:.2f} src={str(result.source):>8} "
                  f"脚下={wit['current_cum']:5.1f}°/dir{wit['current_dir']:.2f}/n{wit['current_samples']:2d} "
                  f"最优={best_txt} ncand={wit['n_cand']:2d} 锚={anchor_txt} "
                  f"锚读数={wit.get('anchor_cum', 0.0):6.1f}°/峰{wit.get('anchor_peak', 0.0):6.1f}° "
                  f"自确认={int(wit['anchor_confirmed'])} tol={wit['tol']:.1f} "
                  f"中心=({wit['center'][0] * inv:6.1f},{wit['center'][1] * inv:6.1f}) "
                  f"size_ref={12.0 if tr is None else tr.size_ref:5.1f} 半径={rad:5.1f} 脚下簇={cc_txt} Δ脚优={d_cb}")
        if result.center is not None:  # 有输出
            proc = cv2.resize(frame, None, fx=session.effective_scale, fy=session.effective_scale)  # 独立缩一次取准星
            got = green_mask(proc)  # 准星掩膜与质心
            c_cursor = None if got is None else got[1]  # 质心（处理尺度）
            rc = None if c_cursor is None else np.array([c_cursor[0] / session.effective_scale, c_cursor[1] / session.effective_scale])  # 全尺度
            c = np.array(result.center, dtype=np.float64)  # 跟踪中心
            code = src_map.setdefault(str(result.source), len(src_map))  # 相位编码
            rows.append((tick, c[0], c[1], float(result.confidence),
                         float(np.linalg.norm(c - rc)) if rc is not None else np.nan,
                         float(np.linalg.norm(c - truth(tick))), code))  # 记录
            if 0 <= t_print <= tick <= t_print + 200:  # 逐帧明细窗
                print(f"  t{tick:4d} {str(result.source):>10} ({c[0]:6.1f},{c[1]:6.1f}) "
                      f"conf={result.confidence:.2f} 距准星={rows[-1][4]:7.1f} 距真值={rows[-1][5]:6.1f}")
        tick += 1
    cap.release()  # 释放
    ss.resize_for_processing = _ORIGIN_RESIZE  # 卸桩
    if _ORIGIN_OBSERVE is not None:
        ParticleShapeTracker.observe_border = _ORIGIN_OBSERVE
    if _ORIGIN_WITNESS_UPDATE is not None:
        RotationWitness.update = _ORIGIN_WITNESS_UPDATE
    if not rows:
        print("无任何输出")
        return 1
    a = np.array([r[:6] for r in rows], dtype=np.float64)  # (N,6)
    codes = {v: k for k, v in src_map.items()}  # 编码 → source 名
    srcs = np.array([r[6] for r in rows], dtype=np.int64)  # 相位序列
    print(f"\n口径={MODE[0]}（半径 {DISKS[MODE[0]]}px / 足迹 {FOOT.get(MODE[0])}）  有输出帧 {len(a)}  相位分布：" +
          ", ".join(f"{codes[int(v)]}={int((srcs == v).sum())}" for v in sorted(src_map.values())))
    print(f"{'窗':>13} {'帧数':>5} {'距准星中位':>10} {'重合<35px%':>10} {'距真值中位':>10} {'距真值末值':>10} {'conf中位':>8}")
    for lo, hi in ((19, 344), (344, 452), (452, 612), (612, 787), (787, 900)):  # 分段
        seg = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]  # 子集
        if len(seg) == 0:
            continue
        print(f"t{lo:4d}..{hi:4d} {len(seg):5d} {np.nanmedian(seg[:, 4]):10.1f} "
              f"{np.nanmean(seg[:, 4] < 35.0) * 100:10.1f} {np.median(seg[:, 5]):10.1f} {seg[-1, 5]:10.1f} {np.median(seg[:, 3]):8.2f}")
    seg = a[(a[:, 0] >= 452) & (a[:, 0] <= 776)]  # 关键窗：9s 转向 → 抵达右下角
    if len(seg) > 10:  # 判据
        net = seg[-1, 1:3] - seg[0, 1:3]  # 净位移
        dx = np.diff(seg[:, 1])  # 逐帧 x 增量
        dy = np.diff(seg[:, 2])  # 逐帧 y 增量（屏幕 y 向下为正）
        back = float(np.mean(dx < -0.5) * 100)  # x 回退帧占比
        up = float(np.mean(dy < -0.5) * 100)  # y 上攻帧占比
        d = seg[:, 5]  # 距真值
        ok_flag = ("PASS" if np.median(d) < 45.0 and back < 12.0 and up < 12.0 and net[0] > 100 and net[1] > 250
                   else "FAIL")
        print(f"\nt452..776 净位移 ({net[0]:.1f},{net[1]:.1f})px  x回退 {back:.1f}%  y上攻 {up:.1f}%  "
              f"距真值中位 {np.median(d):.1f}px p90 {np.percentile(d, 90):.1f}px 最大 {d.max():.1f}px  "
              f"末位置 ({seg[-1,1]:.1f},{seg[-1,2]:.1f}) vs 右下角 ({CORNER_POS[0]:.0f},{CORNER_POS[1]:.0f})  → {ok_flag}")
        corner = a[(a[:, 0] >= 700) & (a[:, 0] <= 776)]  # 抵达窗
        if len(corner):  # 到得早不早：t700..776 应已在右下角附近
            dc = np.hypot(corner[:, 1] - CORNER_POS[0], corner[:, 2] - CORNER_POS[1])  # 距右下角
            print(f"t700..776 距右下角中位 {np.median(dc):.1f}px 最小 {dc.min():.1f}px（≤ 45px 为按时抵达）")
        reveal = a[a[:, 0] == 853]  # 17s 硬判据帧
        if len(reveal):  # 不得回退 17s 已收官结论
            dr = float(np.hypot(reveal[-1, 1] - 606.4, reveal[-1, 2] - 401.6))  # 距揭示星形
            print(f"t853 距揭示答案星 (606.4,401.6) = {dr:.1f}px（17s 判据 ≤ 30px）")
    # 无真值三判据（前段定案：自动重建 9s 窗逐帧真值的三条路线——地形中位漂移反积分 / YOLO 反向接龙 /
    # DP 得分体——全部被实测证伪（自检误差 96~246px、链上点自相矛盾），于是不再与手搓折线比对，
    # 改用三条可证伪的几何判据：① 与准星不重合（准星位置可直接量）② 单调右下 ③ 命中三个实测锚点。
    print("\n[无真值判据]")
    for tt, ax, ay in ANCHORS:  # 逐锚点命中（锚点为取证实测，非手推）
        r = a[a[:, 0] == tt]  # 该帧记录
        if len(r):
            print(f"  锚点 t{tt} 实测 ({ax:6.1f},{ay:6.1f}) → 中心 ({r[-1,1]:6.1f},{r[-1,2]:6.1f}) 距 {float(np.hypot(r[-1,1]-ax, r[-1,2]-ay)):6.1f}px")
    win = a[(a[:, 0] >= 452) & (a[:, 0] <= 900)]  # 9s 起至结尾的全程单调性
    if len(win) > 10:
        dx = np.diff(win[:, 1])
        dy = np.diff(win[:, 2])
        print(f"  t452..900 x回退 {float(np.mean(dx < -0.5) * 100):.1f}%  y上攻 {float(np.mean(dy < -0.5) * 100):.1f}%  "
              f"净位移 ({win[-1,1]-win[0,1]:.1f},{win[-1,2]-win[0,2]:.1f})  距准星<35px 占比 {float(np.nanmean(win[:, 4] < 35.0) * 100):.1f}%")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
