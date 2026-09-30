# -*- coding: utf-8 -*-
"""解测谎录像回放基准：逐帧落盘 source/center/confidence/snr 与分段耗时，供 GPU 化改造前后对比。

用法（同一份回放逻辑，只是 tag 不同，保证苹果对苹果）：
    python _diag_lie_replay.py baseline      # 改造前基线
    python _diag_lie_replay.py torch         # 改造后
    python _diag_lie_diff.py baseline torch  # 对比两次结果
    python _diag_lie_replay.py t5 2239       # 第二个参数是素材名过滤，只跑名字含 2239 的

对照实验（隔离「数组库差异」与「算法差异」）：
    set LIE_REPLAY_FORCE_BACKEND=numpy && python _diag_lie_replay.py nfacade
    把显卡后端槽位换成 CPU 门面、但 is_gpu 仍为 True（精度档不被 clamp），
    于是它与真正显卡后端的唯一区别就是数组库本身。
    若 baseline↔nfacade 的偏差与 baseline↔torch 同量级，说明偏差来自粒子滤波的
    混沌放大（1e-6 的得分差翻转 top-K 选择 → RNG 消耗错位），而不是新后端的 bug。

素材：
    - D:\\workspace\\新建文件夹\\maoxiandao\\test_case\\*.mp4（外部参考视频，整帧喂入）
    - lie_records\\*.mp4（生产录像，**本身已是边车 region 裁剪后的区域**，整段喂入，不能再裁一次）
"""

import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402

CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"
RECORD_DIR = "lie_records"
OUT_DIR = "_diag_out"
TIERS = ("high", "ultra", "extreme")  # 含默认档 extreme（最强），回归必须覆盖它。
MAX_FRAMES = 400  # 每段录像最多回放帧数，控制总耗时。
WARMUP_FRAMES = 10  # 预热帧不计入耗时统计（含后端初始化与首帧分配）。
FORCE_BACKEND = os.environ.get("LIE_REPLAY_FORCE_BACKEND", "").strip().lower()  # 对照实验开关，见模块 docstring。


def install_forced_backend():
    """对照实验：把显卡后端槽位换成 CPU 门面，但让它仍被当成显卡后端（is_gpu=True）。

    这样精度档不会被 clamp 回落 medium，粒子数/对照组/粗扫/lag 全部与显卡路径一致，
    唯一的变量就是底层数组库（numpy 门面 vs torch 门面 vs CuPy）。

    光流引擎也要一起换：``DenseTemporalAligner`` 是按 ``is_gpu`` 选 ``TorchFarnebackFlow`` 的，
    而后者构造时要从门面取裸 ``torch`` 模块，CPU 门面没有。这里换成同参数的
    ``Cv2FarnebackFlow``——**算法与显卡侧同为 Farneback**（不是 DIS），于是对照实验隔离出来的
    仍然只是「数组库 + 设备」这一个变量，不会把 DIS/Farneback 的算法差异混进来。
    """

    if FORCE_BACKEND != "numpy":  # 未开启对照模式。
        return
    from src.liedetector import gpu_shape_backend as gsb
    from src.liedetector import shape_tracking as st
    from src.liedetector.torch_array import numpy_api
    from src.liedetector.torch_flow import Cv2FarnebackFlow

    gsb.gpu_backend_available = lambda: True  # 探测恒为可用，保住 high/ultra 档。
    gsb._gpu_backend = gsb.ShapeScoreBackend(numpy_api(), "numpy")  # 槽位里放 CPU 门面。
    gsb.ShapeScoreBackend.is_gpu = property(lambda self: True)  # 强制按显卡对待，跳过档位 clamp。
    st.TorchFarnebackFlow = lambda xp, **params: Cv2FarnebackFlow(**params)  # 同算法的 CPU Farneback 顶替显卡引擎。


def collect_videos():
    """收集全部待回放素材，返回 [(显示名, 路径, region 或 None)]。"""

    items = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, "*.mp4"))):
        items.append((os.path.basename(path), path, None))
    # lie_records 的 mp4 由 LieRecorder.write 在入队前就按【测谎区域标注】裁好了，边车里的 region
    # 是**整帧坐标系**下的记录值（宽高恒等于视频宽高），不是「还需要再裁的区域」。
    # 早先这里把 region 原样返回，回放与 _diag_perf_probe 于是又裁了一次：
    # frame[138:624, 317:1045] 落在 728x486 的帧上被 numpy 越界夹成 411x348，
    # 实测区域凭空缩小，effective_scale=min(process_scale, max_process_side/长边) 被 process_scale
    # 直接绑定 —— 「最高档抬分辨率」的改动在诊断里静默失效，量出来的耗时不代表生产。故一律返回 None。
    for path in sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4"))):
        items.append((os.path.basename(path), path, None))
    return items


def replay(path, region, tier):
    """按指定精度档回放一段录像，返回逐帧结果与耗时统计。"""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    session = None
    frames = []
    timings = []
    tick = 0
    while tick < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame
        if region is not None:
            x, y, w, h = region
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                break
        if session is None:
            session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=tier), logger=None)
            session.reset(crop.shape[1], crop.shape[0])
        started = time.perf_counter()
        result = session.update(crop, float(src_fps))
        elapsed = time.perf_counter() - started
        if tick >= WARMUP_FRAMES:
            timings.append(elapsed * 1000.0)
        center = result.center
        frames.append([
            tick,
            result.source,
            None if center is None else round(float(center[0]), 3),
            None if center is None else round(float(center[1]), 3),
            round(float(result.confidence), 4),
            round(float(result.border_snr), 4),
        ])
        tick += 1
    cap.release()
    if session is None:
        return None
    sources = {}
    for row in frames:
        sources[row[1]] = sources.get(row[1], 0) + 1
    return {
        "frames": frames,
        "sources": sources,
        "processed": tick,
        "effective_scale": round(float(session.effective_scale), 5),
        "backend": session.backend.name,
        "engine": getattr(getattr(session, "aligner", None), "engine_name", ""),
        "timing_ms": {
            "mean": round(float(np.mean(timings)), 2) if timings else None,
            "p95": round(float(np.percentile(timings, 95)), 2) if timings else None,
            "max": round(float(np.max(timings)), 2) if timings else None,
        },
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    name_filter = sys.argv[2] if len(sys.argv) > 2 else ""  # 只跑名字含此串的素材，省时。
    install_forced_backend()  # 对照实验开关（默认空操作）。
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = [item for item in collect_videos() if name_filter in item[0]]
    if not videos:
        print("no videos found, 未找到任何待回放素材")
        return 1
    for name, path, region in videos:
        for tier in TIERS:
            started = time.perf_counter()
            data = replay(path, region, tier)
            if data is None:
                print(f"SKIP {name} {tier} 无法打开或无有效帧")
                continue
            out = os.path.join(OUT_DIR, f"{tag}_{tier}_{name}.json")
            payload = {"tag": tag, "tier": tier, "video": name, "region": list(region) if region else None}
            payload.update(data)
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            print(
                f"{tag} {tier} {name} frames={data['processed']} backend={data['backend']} engine={data['engine']} "
                f"mean={data['timing_ms']['mean']}ms p95={data['timing_ms']['p95']}ms "
                f"sources={data['sources']} wall={time.perf_counter() - started:.1f}s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
