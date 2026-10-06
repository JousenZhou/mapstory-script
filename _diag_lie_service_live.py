# -*- coding: utf-8 -*-
"""离线驱动「服务实况解题链路」：把 lie_records 录像按墙钟节拍喂进 LieDetectorService._solve，
用生产同一段代码产出真实的 `Lie solve diag` 日志行（含 witness=(x,y)cum=..deg）与
draw_shape_overlay 叠加画面，验证低置信窗口里 source 是否切到 rotation、候选圆圈与光标是否落到
真正在旋转的图形上。

与 _diag_lie_replay.py 的分工：回放脚本只跑算法层 session；这里跑 GUI 生产链路 service._solve
（每秒 diag 限频、overlay 12FPS 节流、光标分步追赶全部生效），只把四个 IO 点换成内存替身——
窗口采集(_capture) / 触发与坐标框匹配(_find_trigger、_get_region_box) / 真实鼠标(_move) /
UI 推送(_update_vision)，算法与绘制一行不改。

节拍：无窗口可克隆时 _solve 走 pump=None 回退路径，旧 `_idle_sleep(33ms)+采集` 的串行节拍在离线
替身下会叠两层等待，故把 `_idle_sleep` 置空、统一由 fake_capture 按墙钟 deadline 供帧，
实测 fps 即等于目标帧率（与生产 pump 30FPS 供帧同口径）；diag 里的 wait 段因此恒为 0。

用法：
    .venv\\Scripts\\python.exe _diag_lie_service_live.py                     # 默认失败录像 190829 + extreme + 30FPS
    .venv\\Scripts\\python.exe _diag_lie_service_live.py 165429 high 30      # 素材名过滤 / 精度档 / 目标帧率
    .venv\\Scripts\\python.exe _diag_lie_service_live.py 190829 extreme 1200 20  # 第四个参数=最多帧数

输出（_diag_out/service_live/）：
    diag_<tag>.log      每秒一条的 Lie solve diag 原文（与 logs/ok-script.log 同格式）
    trace_<tag>.json    逐 tick 的 source/conf/center/witness/cum/applied（生产结果字段的完整还原）
    overlay_*.png       关键 tick 的叠加画面（候选首现、转角跨阈值、重锚、周期性快照）
    path_<tag>.png      末帧底图上叠加全部见证候选轨迹（看候选圆圈是否连成旋转目标的路径）
"""

import glob  # 按素材名过滤 lie_records 下的 mp4。
import json  # 逐 tick 轨迹与汇总落盘。
import logging  # 挂 handler 捕获 service 的 diag 日志行。
import os  # 路径拼接。
import sys  # 命令行参数与本地包导入路径。
import threading  # 服务构造需要退出事件。
import time  # 墙钟节拍调度（模拟 30FPS 采集线程供帧）。

import cv2  # 读录像、写叠加图。
import numpy as np  # 坐标运算。
from unittest.mock import patch  # 只替换 IO 点。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.liedetector.service as service_module  # noqa: E402  patch 模块常量与 ShapeTrackSession。
from src.liedetector.service import LieDetectorService  # noqa: E402  被测的生产服务。
from src.liedetector.shape_session import ShapeTrackSession  # noqa: E402  记录用子类基类。

RECORD_DIR = "lie_records"  # 生产录像目录（mp4 本身已是【测谎坐标框】裁剪结果）。
OUT_DIR = os.path.join("_diag_out", "service_live")  # 输出目录。
CUM_MARKS = (20.0, 40.0, 60.0, 80.0)  # 候选折算窗内转角每跨过这些阈值存一张叠加图，看旋转证据如何攒起来。
PERIODIC_SAVE = 150  # 另按固定 tick 间隔存图，保证全程有画面可查。
MAX_SAVES = 24  # 存图上限，防止刷屏占盘。
LOW_CONF = 0.80  # 低置信窗口判据，与 ShapeTrackParams.witness_recover_conf 同口径。


class RecordingSession(ShapeTrackSession):  # 只加记录的会话子类：算法/参数/状态机全沿用生产实现，逐 tick 抄出 witness 诊断字段。
    trace = None  # 由 run() 注入的记录列表。

    def update(self, crop, fps):  # 生产 update 之后追加一条记录，不改任何返回值。
        result = super().update(crop, fps)  # 真实结果。
        witness = getattr(result, "witness_center", None)  # 最优旋转候选中心（区域全尺度）。
        center = result.center  # 当前跟踪中心。
        RecordingSession.trace.append({
            "tick": len(RecordingSession.trace) + 1,  # tick 从 1 起，与 _solve 的 tick 同口径（本素材区域恒定）。
            "source": result.source,  # 来源标签。
            "conf": round(float(result.confidence), 3),  # 置信度。
            "center": None if center is None else [round(float(center[0]), 1), round(float(center[1]), 1)],
            "witness": None if witness is None else [round(float(witness[0]), 1), round(float(witness[1]), 1)],
            "cum": round(float(getattr(result, "witness_cum_deg", 0.0)), 1),  # 候选折算窗内转角（度）。
            "applied": bool(getattr(result, "witness_applied", False)),  # 本 tick 是否重锚。
            "snr": round(float(result.border_snr), 2),  # 边界信噪比。
            "whites": int(result.white_candidates),  # 白色候选数。
        })
        return result  # 原样返回。


class DiagCapture(logging.Handler):  # 捕获 service logger 的 `Lie solve diag` 行，等同于 logs/ok-script.log 里会看到的内容。

    def __init__(self, sink):  # sink 为收集列表。
        super().__init__(level=logging.INFO)  # 只收 INFO 及以上。
        self.sink = sink  # 输出列表。

    def emit(self, record):  # 命中关键字就原文入列。
        try:
            message = record.getMessage()  # 渲染后的文本。
        except Exception:  # 极端情况下格式化失败不影响解题。
            return
        if "Lie solve diag" in message:  # 只留解题诊断行。
            self.sink.append(f"{time.strftime('%H:%M:%S')} {message}")  # 带时刻入列。


def find_video(name_filter):  # 返回名字含过滤串的录像路径，多个取最新。
    paths = sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4")))
    hits = [p for p in paths if name_filter in os.path.basename(p)]
    return hits[-1] if hits else None


def run(path, tier, target_fps, max_frames):  # 核心：以 target_fps 的墙钟节拍把 path 喂进 service._solve。
    cap = cv2.VideoCapture(path)  # 打开录像。
    ok, first = cap.read()  # 首帧即触发帧。
    if not ok:  # 打不开。
        cap.release()
        return None
    height, width = first.shape[:2]  # 帧尺寸（区域已裁好）。
    interval = 1.0 / max(1.0, float(target_fps))  # 目标帧间隔（秒）。
    state = {"index": 0, "due": time.perf_counter(), "moves": [], "canvas": None, "saves": 0,
             "marks": [False] * len(CUM_MARKS), "witness_seen": False, "rotation_saved": 0}  # 运行期共享状态。
    trace = []  # 逐 tick 记录。
    diag_lines = []  # 捕获的 diag 日志行。
    RecordingSession.trace = trace  # 注入记录列表。

    from types import SimpleNamespace  # 假标注框。
    fake_trigger = SimpleNamespace(x=6, y=6, width=90, height=26)  # 触发框只用于绘制，位置随意。
    fake_region = SimpleNamespace(x=0, y=0, width=width, height=height)  # 坐标框=整帧：录像已是裁剪结果，不能再裁。

    def fake_capture():  # 替代窗口采集：按墙钟节拍推进一帧，落后节拍则重置基准不疯狂追赶。
        state["due"] += interval  # 下一帧应交付时刻。
        delay = state["due"] - time.perf_counter()  # 还差多少秒。
        if delay > 0:  # 未到时刻：真睡，使 _solve 里的 dt 与生产一致。
            time.sleep(delay)
        else:
            state["due"] = time.perf_counter()  # 处理慢了：重置基准，避免后续连续空转。
        state["index"] += 1  # 帧计数。
        if state["index"] >= max_frames:  # 达到长度上限：置急停让循环收尾（trace 已完整）。
            service._abort_event.set()
            return None
        ok_frame, frame = cap.read()  # 读下一帧。
        if not ok_frame:  # 素材耗尽：同样置急停收尾。
            service._abort_event.set()
            return None
        return frame  # 交给解题循环。

    def fake_move(x, y):  # 替代真实鼠标：只记录光标目标点。
        state["moves"].append((len(trace), int(x), int(y)))

    def fake_vision(canvas):  # 替代 UI 推送：缓存最新画面并按关键时刻存盘。
        state["canvas"] = canvas  # 最新叠加图。
        if not trace or state["saves"] >= MAX_SAVES:  # 无记录或存满。
            return
        rec = trace[-1]  # 当前 tick 记录（update 先于 overlay 发生）。
        reason = None  # 存图原因。
        if rec["source"] == "rotation" and state["rotation_saved"] < 2:  # 重锚 tick：最多存两张。
            reason = "rotation"
            state["rotation_saved"] += 1
        elif rec["witness"] is not None and not state["witness_seen"]:  # 候选首现。
            reason = "witness-first"
        else:  # 转角跨阈值 / 周期快照。
            for i, mark in enumerate(CUM_MARKS):  # 依次检查阈值。
                if not state["marks"][i] and rec["cum"] >= mark:  # 首次跨过。
                    state["marks"][i] = True  # 标记已存。
                    reason = f"cum{int(mark)}"
                    break
            if reason is None and rec["tick"] % PERIODIC_SAVE == 0:  # 周期快照。
                reason = "periodic"
        if reason is not None:  # 需要存图。
            state["witness_seen"] = True  # 候选首现标记（任何有候选的帧之后都算见过）。
            tag = os.path.splitext(os.path.basename(path))[0].split("_")[1]  # 用时间戳做 tag。
            out = os.path.join(OUT_DIR, f"overlay_{tag}_{tier}_t{rec['tick']:04d}_{reason}.png")  # 文件名带 tick 与原因。
            cv2.imwrite(out, canvas)  # 写盘。
            state["saves"] += 1  # 计数。

    service = LieDetectorService(threading.Event())  # 与单测一致：直接构造、不起线程。
    handler = DiagCapture(diag_lines)  # diag 行收集器。
    std_logger = service_module.logger.logger  # ok.Logger 包装的是标准库 logger，handler 要挂在内层 logging.Logger 上。
    std_logger.addHandler(handler)  # 挂到服务 logger 上，捕获与 logs/ok-script.log 同格式的 diag 行。
    try:
        with patch.object(service_module, "LIE_MAX_SECONDS", 10.0 ** 7), \
                patch.object(service_module, "LIE_MAX_TICKS", 10 ** 7), \
                patch.object(service_module, "ShapeTrackSession", RecordingSession), \
                patch.object(service, "_ensure_in_front", lambda: None), \
                patch.object(service, "_find_trigger", lambda frame, name, threshold: fake_trigger), \
                patch.object(service, "_get_region_box", lambda frame, name: fake_region), \
                patch.object(service, "_capture", fake_capture), \
                patch.object(service, "_move", fake_move), \
                patch.object(service, "_idle_sleep", lambda seconds: None), \
                patch.object(service, "_update_vision", fake_vision):
            outcome = service._solve(first, "测谎触发", "测谎坐标框", 0.75, precision_tier=tier)  # 跑生产解题循环。
    finally:
        std_logger.removeHandler(handler)  # 摘掉 handler。
        cap.release()  # 释放录像。
    return {"video": os.path.basename(path), "tier": tier, "target_fps": target_fps,
            "outcome": outcome, "frames": state["index"] + 1, "size": [width, height],
            "diag": diag_lines, "trace": trace, "moves": state["moves"], "saves": state["saves"],
            "last_canvas": state["canvas"], "feed_trace_steps": len(service._last_feed_trace)}


def summarize(data):  # 汇总关键结论：低置信窗口、重锚时刻、候选落点与光标终点。
    trace = data["trace"]  # 逐 tick 记录。
    sources = {}  # source 分布。
    for rec in trace:  # 统计。
        sources[rec["source"]] = sources.get(rec["source"], 0) + 1
    transparent = [r for r in trace if r["source"] not in ("color", "waiting")]  # 透明阶段（见证器才生效的帧）。
    low = [r for r in transparent if r["conf"] < LOW_CONF]  # 透明阶段的低置信窗口。
    applied = [r for r in trace if r["applied"]]  # 重锚 tick。
    rotation = [r for r in trace if r["source"] == "rotation"]  # source=rotation 的 tick。
    with_witness = [r for r in transparent if r["witness"] is not None]  # 有候选的 tick。
    last_center = next((r["center"] for r in reversed(trace) if r["center"]), None)  # 最后一次跟踪中心。
    last_cursor = data["moves"][-1] if data["moves"] else None  # 光标最后位置（区域坐标=画面坐标）。
    summary = {
        "sources": sources,  # 各 source 次数。
        "transparent_ticks": len(transparent),  # 透明阶段帧数。
        "transparent_min_conf": min([r["conf"] for r in transparent], default=None),  # 透明阶段最低置信。
        "low_conf_ticks": len(low),  # 透明阶段低置信帧数。
        "low_conf_range": [low[0]["tick"], low[-1]["tick"]] if low else None,  # 低置信窗口起止 tick。
        "witness_first_tick": with_witness[0]["tick"] if with_witness else None,  # 候选首现。
        "witness_max_cum": max([r["cum"] for r in with_witness], default=0.0),  # 候选最大转角。
        "applied_ticks": [r["tick"] for r in applied],  # 重锚 tick 列表。
        "applied_detail": applied[:6],  # 前几条重锚现场。
        "rotation_ticks": [r["tick"] for r in rotation],  # source=rotation 的 tick。
        "last_center": last_center,  # 跟踪终点。
        "last_cursor": last_cursor,  # 光标终点。
        "diag_lines": len(data["diag"]),  # diag 行数（≈运行秒数）。
    }
    return summary


def draw_path(data, out_path):  # 末帧底图上画出全部候选中心与跟踪中心轨迹，直观核对候选圆圈是否落在旋转目标上。
    frames = cv2.VideoCapture(data["video_full"])  # 重开录像取末帧。
    total = int(frames.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 素材总帧数（部分编码器报值不准，仅作上限参考）。
    target_frame = data["frames"] - 1  # 目标：最后一次喂入的帧号。
    if 0 < total < target_frame:  # 报值可信且更短则夹住。
        target_frame = total - 1
    base = None  # 待取底图。
    for _ in range(target_frame + 1):  # 逐帧读到目标帧（比 CAP_PROP_POS_FRAMES 定位稳，编码器_seek 不可靠）。
        ok, frame = frames.read()  # 读一帧。
        if not ok:  # 提前读完。
            break
        base = frame  # 滚动保留最新。
    frames.release()  # 释放。
    if base is None:  # 取不到就不画。
        return False
    canvas = base.copy()  # 复制。
    prev_w = None  # 上一候选点。
    for i, rec in enumerate(data["trace"]):  # 逐 tick。
        if rec["witness"] is None:  # 无候选。
            continue
        point = (int(rec["witness"][0]), int(rec["witness"][1]))  # 候选中心。
        shade = int(255 * i / max(1, len(data["trace"])))  # 时间渐变（暗=早，亮=晚）。
        cv2.circle(canvas, point, 2, (0, 160, 255), -1)  # 橙色小点。
        if prev_w is not None:  # 连线段。
            cv2.line(canvas, prev_w, point, (0, int(80 + shade / 2), 255), 1)
        prev_w = point  # 滚动。
    prev_c = None  # 上一跟踪点。
    for rec in data["trace"]:  # 画跟踪中心轨迹。
        if rec["center"] is None:  # 无输出。
            continue
        point = (int(rec["center"][0]), int(rec["center"][1]))  # 跟踪中心。
        color = (255, 0, 255) if rec["source"] == "rotation" else (0, 220, 0)  # 重锚帧紫红、其余绿色。
        cv2.circle(canvas, point, 1, color, -1)  # 点。
        if prev_c is not None:  # 连线。
            cv2.line(canvas, prev_c, point, color, 1)
        prev_c = point  # 滚动。
    cv2.imwrite(out_path, canvas)  # 写盘。
    return True


def main():  # 命令行入口。
    name_filter = sys.argv[1] if len(sys.argv) > 1 else "190829"  # 素材过滤。
    tier = sys.argv[2] if len(sys.argv) > 2 else "extreme"  # 精度档。
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None  # 可能是帧率或帧数。
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None  # 另一个参数。
    target_fps, max_frames = 30.0, 1400  # 默认按生产 30FPS 节拍、最多 1400 帧。
    for value in (arg3, arg4):  # 解析两个可选参数：>100 当帧数，否则当帧率。
        if value is None:  # 未给。
            continue
        number = float(value)  # 转数值。
        if number > 100:  # 视为帧数上限。
            max_frames = int(number)
        else:  # 视为目标帧率。
            target_fps = number
    path = find_video(name_filter)  # 找素材。
    if path is None:  # 没找到。
        print(f"no video matched: {name_filter}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)  # 建输出目录。
    started = time.perf_counter()  # 计时。
    data = run(path, tier, target_fps, max_frames)  # 跑实况链路。
    if data is None:  # 打开失败。
        print(f"cannot open {path}")
        return 1
    data["video_full"] = path  # 供画图取末帧。
    summary = summarize(data)  # 汇总。
    tag = os.path.splitext(os.path.basename(path))[0]  # 素材全名。
    with open(os.path.join(OUT_DIR, f"trace_{tag}_{tier}.json"), "w", encoding="utf-8") as handle:  # 逐 tick。
        json.dump({"summary": summary, "trace": data["trace"]}, handle, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, f"diag_{tag}_{tier}.log"), "w", encoding="utf-8") as handle:  # diag 原文。
        handle.write("\n".join(data["diag"]) + "\n")
    draw_path(data, os.path.join(OUT_DIR, f"path_{tag}_{tier}.png"))  # 轨迹图。
    print(f"{tag} {tier} fps={target_fps} outcome={data['outcome']} frames={data['frames']} "
          f"wall={time.perf_counter() - started:.1f}s saves={data['saves']}")
    print(f"sources={summary['sources']}")
    print(f"transparent={summary['transparent_ticks']} min_conf={summary['transparent_min_conf']} "
          f"low_conf_ticks={summary['low_conf_ticks']} range={summary['low_conf_range']} "
          f"witness_first={summary['witness_first_tick']} witness_max_cum={summary['witness_max_cum']}")
    print(f"applied_ticks={summary['applied_ticks']} rotation_ticks={summary['rotation_ticks']}")
    print(f"last_center={summary['last_center']} last_cursor={summary['last_cursor']} diag_lines={summary['diag_lines']}")
    for line in data["diag"]:  # 原样打出每秒 diag，便于直接读 witness=/source=。
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
