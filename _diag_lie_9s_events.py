# -*- coding: utf-8 -*-
"""9s 前后到底发生了什么「事件」：把会话级重置、模板重学、跟踪器重建、相位切换、滞回门限的触发时刻全部打出来。

动机：
    用户说「有时可以跟上有时不可以」——同一份录像在 ceiling 模式下是确定的（random_seed 固定），
    所以不确定只能来自「事件型」分支：某帧触发了 reset/重学/重锚，差一帧就走到另一条路。
    本脚本把这些事件连同逐帧相位/中心/置信度一并列出，直接看 9s（t452）附近有没有这类事件。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_events.py [t_from] [t_to]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码录像。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
TIER = "extreme"  # 生产默认档。

EVENTS = []  # (tick, 事件名, 详情)。
TICK = [0]  # 当前帧号（供事件打时间戳）。

_ORIGIN_RESET = ShapeTrackSession.reset  # 原实现。
_ORIGIN_INIT = st.ParticleShapeTracker.__init__  # 原实现。
_ORIGIN_LEARN = getattr(ShapeTrackSession, "_learn_template", None)  # 模板学习入口（可能不存在）。


def _reset(self, width, height, *a, **kw):  # 会话重置。
    EVENTS.append((TICK[0], "session.reset", f"{width}x{height}"))  # 记录。
    return _ORIGIN_RESET(self, width, height, *a, **kw)  # 转发。


def _init(self, *a, **kw):  # 跟踪器构造。
    EVENTS.append((TICK[0], "tracker.__init__", ""))  # 记录。
    return _ORIGIN_INIT(self, *a, **kw)  # 转发。


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 300  # 起始帧。
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 900  # 结束帧。
    ShapeTrackSession.reset = _reset  # 装桩。
    st.ParticleShapeTracker.__init__ = _init  # 装桩。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参。
    cap = cv2.VideoCapture(VIDEO)  # 打开素材。
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率。
    prev_source = None  # 上一帧相位。
    prev_center = None  # 上一帧中心。
    tick = 0  # 帧号。
    print("逐帧事件（相位变化 / 中心跳变 >25px）：")
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧。
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局。
        TICK[0] = tick  # 时间戳。
        result = session.update(frame, float(fps))  # 生产路径推进。
        center = np.array(result.center, dtype=np.float64) if result.center is not None else None  # 中心。
        if result.source != prev_source:  # 相位切换。
            EVENTS.append((tick, "source", f"{prev_source} -> {result.source}"))  # 记录。
            prev_source = result.source  # 推进。
        if center is not None and prev_center is not None:  # 跳变检测。
            jump = float(np.linalg.norm(center - prev_center))  # 位移。
            if jump > 25.0:  # 明显跳变（远大于目标自身速度 ~1.4px/帧）。
                EVENTS.append((tick, "jump", f"Δ={jump:.1f}px ({prev_center[0]:.0f},{prev_center[1]:.0f})->({center[0]:.0f},{center[1]:.0f})"))  # 记录。
        if center is not None:
            prev_center = center  # 推进。
        tick += 1
    cap.release()  # 释放。
    for t, name, detail in EVENTS:  # 全量事件。
        if t >= t_from or name in ("session.reset", "tracker.__init__", "source"):
            flag = "  <<< 9s 窗" if t_from <= t <= t_to and name in ("session.reset", "tracker.__init__", "jump", "source") else ""
            print(f"t{t:4d} {name:16s} {detail}{flag}")
    print(f"\n共 {len(EVENTS)} 条事件；素材 t452≈9.0s（fps={fps:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
