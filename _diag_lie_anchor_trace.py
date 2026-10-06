# -*- coding: utf-8 -*-
"""逐检查点取证：在**未加缰绳**的基线轨迹上，打印见证器内部四个量（当前位置转角 / 最优候选 / 中心 / 门槛）。

为什么需要：第一次尝试的锚点规则「只要存在过门槛的候选就拿它当锚」把跟踪器在 t612 抢到了
(585,162)（画面顶部），一路毁到结尾——因为最优候选经常是远处的干扰团块。要设计正确的锚点规则，
必须先看清每个检查点上「跟踪器自己脚下到底有没有旋转」「候选离自己多远」，这两个量生产链路里看不到。

用法：
    .venv\\Scripts\\python.exe _diag_lie_anchor_trace.py [tick_from] [tick_to]
"""

import os  # 环境变量（LIE_TRACE_TIER 换档取证）。
import sys  # 参数。
import time  # 计时。

import cv2  # 读帧。
import numpy as np  # 数值。

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # 生产会话。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 失败复现素材。
TRUTH_853 = (606.4, 401.6)  # 17s 真值（区域全尺度），用于给每行标注「离真值还有多远」。


def main():
    tick_from = int(sys.argv[1]) if len(sys.argv) > 1 else 300  # 起始帧。
    tick_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧。
    tier = os.environ.get("LIE_TRACE_TIER", "extreme") or "extreme"  # 取证档位：默认生产档 extreme。
    params = ShapeTrackParams(precision_tier=tier)  # 生产默认参数（含簇聚合，但关掉缰绳）。
    params.witness_anchor_enabled = False  # 只看基线轨迹的内部量，不让锚点改变跟踪器。
    session = ShapeTrackSession(params=params, logger=None)  # 会话。
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 采集帧率。
    tick = 0
    inited = False  # 会话只 reset 一次（首帧），否则学习缓存被反复清空。
    started = time.perf_counter()
    while tick < tick_to:
        ok, frame = cap.read()
        if not ok:
            break
        if not inited:  # 首帧建会话。
            session.reset(frame.shape[1], frame.shape[0])
            inited = True
        result = session.update(frame, float(fps))  # 逐帧推进。
        trace = session._witness_trace or {}  # 检查点快照（非检查点沿用上一次的）。
        if tick >= tick_from and trace and tick % 5 == 0:  # 只在检查点附近打印。
            scale = session.effective_scale  # 处理尺度 -> 全尺度。
            center = None if result.center is None else np.asarray(result.center, dtype=np.float64)  # 全尺度中心。
            best = trace.get("best")  # 最优候选（处理尺度）。
            best_full = None if best is None else [best[0] / scale, best[1] / scale, best[2]]  # 换算回全尺度。
            anchor = trace.get("anchor")  # 锚点（处理尺度）。
            d_center = -1.0 if center is None else float(np.hypot(center[0] - TRUTH_853[0], center[1] - TRUTH_853[1]))
            d_best = -1.0 if (center is None or best_full is None) else float(np.hypot(center[0] - best_full[0], center[1] - best_full[1]))
            ctext = "    none    " if center is None else f"({center[0]:6.1f},{center[1]:6.1f})"  # 中心文本。
            btext = "      none      " if best_full is None else f"({best_full[0]:6.1f},{best_full[1]:6.1f})"  # 候选文本。
            atext = "none" if anchor is None else f"({anchor[0] / scale:6.1f},{anchor[1] / scale:6.1f})"  # 锚点文本。
            print(f"t{trace['frame']:4d} src={result.source:10s} c={ctext} dTruth={d_center:6.1f} "
                  f"curCum={trace['current_cum']:6.1f}/{trace.get('current_dir', 0.0):4.2f}/{trace.get('current_samples', 0):3d} "
                  f"best={btext} bestCum={best[2] if best else -1.0:6.1f} "
                  f"dir={trace.get('best_dir', 0.0):4.2f} samp={trace.get('best_samples', 0):3d} age={trace.get('best_age', 0.0):6.1f} "
                  f"dBest={d_best:6.1f} ncand={trace['n_cand']:2d} tol={trace['tol'] / scale:5.1f} "
                  f"anchor={atext} aCum={trace.get('anchor_cum', 0.0):6.1f}/{trace.get('anchor_peak', 0.0):6.1f}")
        tick += 1
    cap.release()
    print(f"done ticks={tick} wall={time.perf_counter() - started:.1f}s scale={session.effective_scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
