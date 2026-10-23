# -*- coding: utf-8 -*-
"""测谎触发录像器：每次触发把【测谎区域标注】的原始画面录成 mp4 + json 边车历史记录。

设计要点（沿用实战验证过的机制）：
- 常开：每次测谎触发都录；滚动保留最近 LIE_RECORD_KEEP 组（mp4+json），旧的自动删除。
- 帧率按游戏真实帧率录制：独立采集线程以 LIE_RECORD_MAX_FPS 为上限自由抓帧，
  与上一帧完全相同的画面去重跳过（游戏没渲染新帧就不写）。
- 容器帧率==实测真实帧率（回放绝不加速）：VideoWriter 延迟到标定完成才开流——
  先缓冲 LIE_RECORD_CALIB_FRAMES 帧（或最多等 LIE_RECORD_CALIB_MAX_WAIT 秒），
  用首末帧真实时间跨度算出实测帧率，再以该帧率建流并回灌缓冲帧。
- 只录【测谎区域标注】区域（crop 固定），输出尺寸恒定，回放验证直接整帧喂管线。
- 所有对外方法吞异常：录像问题绝不外溢到解题主流程。

结果 outcome ∈ success（结算成功）/ failure（3 秒未出现成功标注）/ timeout（解题兜底超时）/
aborted（用户急停）/ abandoned（后端未就绪等放弃）。
"""
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

from ok import Logger

logger = Logger.get_logger(__name__)

LIE_RECORD_DIR = "lie_records"  # 录像目录（相对项目根，.gitignore 已忽略）。
LIE_RECORD_KEEP = 50  # 滚动保留最近多少组录像（mp4+json 各一），超出按 mtime 删除最旧。
LIE_RECORD_FPS = 30  # 兜底/标称帧率：无法标定时用它建流。
LIE_RECORD_MAX_FPS = 60  # 自采集抓帧节拍上限（同时是容器帧率夹取上限）。
LIE_RECORD_MIN_FPS = 5.0  # 容器帧率夹取下限：防止开局卡顿把标定帧率算得过低导致回放慢放。
LIE_RECORD_CALIB_FRAMES = 20  # 开流标定所需的最少互异帧数。
LIE_RECORD_CALIB_SPAN = 0.8  # 开流标定所需的最小首末帧时间跨度（秒）。
LIE_RECORD_CALIB_MAX_WAIT = 2.0  # 标定的最长等待（秒）：帧太慢也用已缓冲帧开流。
LIE_RECORD_FOURCC = "mp4v"  # mp4 编码 fourcc，OpenCV 内置。
LIE_RECORD_QUEUE_MAX = 128  # 有界队列容量：写线程跟不上时满即丢帧，绝不回压采集。
_JOIN_TIMEOUT = 5.0  # 收尾等待采集/写线程退出的超时秒数。

# 结果中文标签：历史下拉与验证页签展示用。
LIE_OUTCOME_LABELS = {
    "success": "成功",
    "failure": "失败",
    "timeout": "超时",
    "aborted": "急停",
    "abandoned": "放弃",
}


def _clamp_fps(value):
    """把标定帧率夹到 [MIN_FPS, MAX_FPS]；非法/非正回退标称帧率。"""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return float(LIE_RECORD_FPS)
    if fps <= 0:
        return float(LIE_RECORD_FPS)
    return max(LIE_RECORD_MIN_FPS, min(fps, float(LIE_RECORD_MAX_FPS)))


def _read_json(path):
    """按 UTF-8 读 JSON，失败返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class LieRecorder:
    """一次测谎触发的录像器：start 拉起采集/写线程，stop 收尾写边车。单次使用（一局一个实例）。"""

    def __init__(self):
        self._lock = threading.Lock()  # 保护 start/stop 状态切换。
        self._writer = None  # cv2.VideoWriter：标定出真实帧率后才开流。
        self._fps_written = None  # 实际建流用的容器帧率。
        self._open_failed = False  # 开流失败标志：置位后不再重试。
        self._calib = []  # 标定缓冲：开流前暂存 (ts, frame)。
        self._queue = None  # 有界帧队列。
        self._thread = None  # 写线程句柄。
        self._capture = None  # 独立采集回调（返回 BGR 帧或 None）。
        self._capture_release = None  # 采集资源释放回调。
        self._cap_thread = None  # 自采集线程句柄。
        self._running = False  # 采集/写线程运行标志。
        self._path = None  # mp4 文件路径（起流失败保持 None）。
        self._width = 0  # 输出宽（裁剪区域宽或整帧宽）。
        self._height = 0
        self._crop = None  # 录像区域 (x, y, w, h)（整帧坐标系）；None 录整帧。触发时按【测谎区域标注】确定，全程固定。
        self._frames = 0  # 已写入帧数。
        self._dropped = 0  # 队列满丢帧数。
        self._meta = {}  # 边车附加元数据（score/backend/conf 等）。
        self._start_iso = ""  # 起录时间 ISO。
        self._first_ts = None  # 首帧写入时间戳（perf_counter）。
        self._last_ts = None  # 末帧写入时间戳。
        self._last_enqueued = None  # 上一帧已入队的裁剪帧：去重基准。

    # ------------------------------------------------------------------ 起停

    def start(self, shape, meta=None, crop=None, capture=None, capture_release=None):
        """起录：建目录、拉起写线程（传 capture 时再拉起自采集线程）。

        shape 为整帧 (高, 宽)；crop 为 (x,y,w,h) 录像区域，None 录整帧；
        capture/capture_release 为独立采集回调与释放回调（推荐传入，与解题抢帧解耦）。
        """
        with self._lock:
            try:
                if self._writer is not None or self._running:  # 已起录，幂等返回。
                    return
                height, width = int(shape[0]), int(shape[1])
                if height <= 0 or width <= 0:
                    logger.warning(f"Lie record skipped, invalid shape {shape}. 测谎录像跳过：分辨率非法。")
                    return
                self._meta = dict(meta or {})
                self._crop = self._resolve_crop(crop, width, height)
                self._width, self._height = (self._crop[2], self._crop[3]) if self._crop else (width, height)
                os.makedirs(LIE_RECORD_DIR, exist_ok=True)
                self._path = self._build_path()
                self._frames = 0
                self._dropped = 0
                self._first_ts = None
                self._last_ts = None
                self._fps_written = None
                self._open_failed = False
                self._calib = []
                self._last_enqueued = None
                self._start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._queue = queue.Queue(maxsize=LIE_RECORD_QUEUE_MAX)
                self._capture = capture if callable(capture) else None
                self._capture_release = capture_release if callable(capture_release) else None
                self._running = True
                self._thread = threading.Thread(target=self._consume, name="LieRecorder", daemon=True)
                self._thread.start()
                if self._capture is not None:
                    self._cap_thread = threading.Thread(target=self._capture_loop, name="LieRecorderCap", daemon=True)
                    self._cap_thread.start()
                logger.info(f"Lie record started: {self._path} {self._width}x{self._height} crop={self._crop}. 测谎录像已开始。")
            except Exception as e:
                logger.warning(f"Lie record start failed: {e}. 测谎录像启动失败，本局不录。")
                self._running = False
                self._release_writer()
                self._release_capture()
                self._path = None

    def stop(self, outcome="abandoned"):
        """收尾：停采集线程 → 排空写线程 → 关流 → 释放采集 → 写边车 → 滚动保留。"""
        with self._lock:
            if not self._running and self._writer is None and not self._calib:
                self._release_capture()
                return
            self._running = False
            cap_thread = self._cap_thread
            self._cap_thread = None
            if cap_thread is not None:  # 先停采集（不再入队），再收写线程。
                cap_thread.join(timeout=_JOIN_TIMEOUT)
            try:
                if self._queue is not None:
                    self._queue.put_nowait(None)  # 停止哨兵。
            except Exception:
                pass
            if self._thread is not None:
                self._thread.join(timeout=_JOIN_TIMEOUT)
                self._thread = None
            frames = self._frames
            dropped = self._dropped
            path = self._path if frames > 0 else None  # 0 帧不落地任何文件。
            self._release_writer()
            self._release_capture()
            if frames > 0:
                self._write_sidecar(outcome, frames)
            self._prune()
            if path:
                logger.info(f"Lie record stopped: {path} outcome={outcome} frames={frames} dropped={dropped} "
                            f"fps={self._fps_written} measured_fps={self._measured_fps(frames)}. 测谎录像已结束。")

    # ------------------------------------------------------------------ 录帧

    def write(self, frame):
        """被动模式：外部逐帧喂一帧 BGR 画面。自采集模式（start 传了 capture）下空转。"""
        if self._capture is not None:  # 录像由独立采集线程驱动，手动喂帧忽略。
            return
        self._enqueue(frame)

    def _capture_loop(self):
        """自采集线程：以 MAX_FPS 为上限自由抓帧（独立采集实例），去重后入队。"""
        min_interval = 1.0 / float(LIE_RECORD_MAX_FPS)
        next_t = time.perf_counter()
        while self._running:
            try:
                frame = self._capture()
            except Exception:
                frame = None
            if frame is not None:
                self._enqueue(frame)
            next_t += min_interval
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:  # 落后于节拍：重置基准，不疯狂补采。
                next_t = time.perf_counter()

    def _enqueue(self, frame):
        """裁帧、去重、打时间戳后非阻塞入队；未起录/重复帧/队列满时安全跳过。"""
        if not self._running or self._queue is None or frame is None:
            return
        try:
            ts = time.perf_counter()
            item = frame
            crop = self._crop
            if crop is not None:
                x, y, w, h = crop
                fh, fw = frame.shape[:2]
                x0, y0 = min(max(0, x), fw), min(max(0, y), fh)
                x1, y1 = min(max(0, x + w), fw), min(max(0, y + h), fh)
                if x1 <= x0 or y1 <= y0:  # 区域完全越界：跳过，保持输出尺寸恒定。
                    return
                item = frame[y0:y1, x0:x1].copy()  # 拷贝保证内存连续且不牵住整帧。
            if self._last_enqueued is not None and np.array_equal(self._last_enqueued, item):
                return  # 游戏这一拍没渲染新帧：去重。
            self._queue.put_nowait((ts, item))
            self._last_enqueued = item
        except queue.Full:
            self._dropped += 1
        except Exception:
            pass

    def _consume(self):
        """写线程：开流前缓冲标定，标定出真实帧率后开流回灌，随后流式写入。"""
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break
                self._maybe_open(force=False)  # 帧来得慢时超过最长等待也要开流。
                continue
            if item is None:
                self._queue.task_done()
                break
            try:
                self._handle_item(item)
            except Exception as e:
                logger.warning(f"Lie record consume failed: {e}. 测谎录像写帧失败，已跳过。")
            finally:
                self._queue.task_done()
        self._maybe_open(force=True)  # 退出前把标定缓冲强制开流写完（录像很短的情况）。

    def _handle_item(self, item):
        ts, frame = item
        if self._writer is None:
            if self._open_failed:
                return
            self._calib.append(item)
            self._maybe_open(force=False)
            return
        self._write_frame(ts, frame)

    def _maybe_open(self, force=False):
        """标定并开流：缓冲帧数与跨度足够（或 force/超最长等待）时按实测帧率建流并回灌。"""
        if self._writer is not None or self._open_failed or not self._calib:
            return
        first_ts = self._calib[0][0]
        span = self._calib[-1][0] - first_ts
        ready = force or (len(self._calib) >= LIE_RECORD_CALIB_FRAMES and span >= LIE_RECORD_CALIB_SPAN) \
            or (time.perf_counter() - first_ts >= LIE_RECORD_CALIB_MAX_WAIT)
        if not ready:
            return
        fps = self._calib_fps(self._calib)
        if self._open_writer(fps) is None:
            self._open_failed = True
            self._calib = []
            return
        for ts, frame in self._calib:
            self._write_frame(ts, frame)
        self._calib = []

    def _calib_fps(self, items):
        """由缓冲首末帧时间跨度估算真实帧率，夹到合法区间；样本不足回退标称帧率。"""
        if len(items) >= 2:
            span = items[-1][0] - items[0][0]
            if span > 1e-6:
                return _clamp_fps((len(items) - 1) / span)
        return float(LIE_RECORD_FPS)

    def _open_writer(self, fps):
        """以指定容器帧率开流；失败置空返回 None。"""
        try:
            writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*LIE_RECORD_FOURCC),
                                     float(fps), (int(self._width), int(self._height)))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter not opened")
            self._writer = writer
            self._fps_written = float(fps)
            return writer
        except Exception as e:
            logger.warning(f"Lie record open writer failed: {e}. 测谎录像开流失败，本局不落地。")
            self._writer = None
            return None

    def _write_frame(self, ts, frame):
        """写入单帧：尺寸不符则缩放，维护首末帧时间戳与帧计数。"""
        writer = self._writer
        if writer is None:
            return
        try:
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height))
            writer.write(frame)
            self._frames += 1
            if self._first_ts is None:
                self._first_ts = ts
            self._last_ts = ts
        except Exception as e:
            logger.warning(f"Lie record write frame failed: {e}. 测谎录像写单帧失败。")

    # ------------------------------------------------------------------ 收尾工具

    def _release_writer(self):
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception as e:
                logger.warning(f"Lie record release writer failed: {e}. 关闭录像写入流失败。")

    def _release_capture(self):
        release = self._capture_release
        self._capture_release = None
        if release is not None:
            try:
                release()
            except Exception as e:
                logger.warning(f"Lie record release capture failed: {e}. 释放录像独立采集失败。")

    def _measured_fps(self, frames):
        """整段实测平均帧率 = (帧数-1)/首末帧真实时间跨度；样本不足回退容器帧率。"""
        if frames >= 2 and self._first_ts is not None and self._last_ts is not None:
            span = self._last_ts - self._first_ts
            if span > 1e-6:
                return round((frames - 1) / span, 3)
        return round(float(self._fps_written or LIE_RECORD_FPS), 3)

    def _write_sidecar(self, outcome, frames):
        """写与 mp4 同名的 .json 边车。"""
        if not self._path:
            return
        try:
            fps_written = float(self._fps_written or LIE_RECORD_FPS)
            measured_fps = self._measured_fps(frames)
            duration = round(frames / fps_written, 3) if fps_written > 0 else 0.0
            real_duration = 0.0
            if self._first_ts is not None and self._last_ts is not None and self._last_ts >= self._first_ts:
                real_duration = round(self._last_ts - self._first_ts, 3)
            data = {
                "video": os.path.basename(self._path),
                "path": self._path,
                "timestamp": self._start_iso,
                "outcome": str(outcome or "abandoned"),
                "frames": int(frames),
                "fps": round(fps_written, 3),
                "measured_fps": measured_fps,
                "duration": duration,
                "real_duration": real_duration,
                "width": int(self._width),
                "height": int(self._height),
                "region": list(self._crop) if self._crop else None,
                "meta": dict(self._meta),
            }
            json_path = os.path.splitext(self._path)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Lie record write sidecar failed: {e}. 测谎录像边车写入失败。")

    def _prune(self):
        """滚动保留：按 mp4/json 成对的最大 mtime 倒序，只留最近 LIE_RECORD_KEEP 组。"""
        try:
            entries = {}  # stem -> 最大 mtime。
            for name in os.listdir(LIE_RECORD_DIR):
                stem, ext = os.path.splitext(name)
                if ext.lower() not in (".mp4", ".json"):
                    continue
                try:
                    mtime = os.path.getmtime(os.path.join(LIE_RECORD_DIR, name))
                except OSError:
                    continue
                entries[stem] = max(entries.get(stem, 0.0), mtime)
            ordered = sorted(entries.items(), key=lambda kv: kv[1], reverse=True)
            for stem, _mtime in ordered[LIE_RECORD_KEEP:]:
                for ext in (".mp4", ".json"):
                    try:
                        os.remove(os.path.join(LIE_RECORD_DIR, stem + ext))
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Lie record prune failed: {e}. 测谎录像滚动清理失败。")

    def _build_path(self):
        """生成录像文件名：YYYYmmdd_HHMMSS.mp4，重名追加序号。"""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(LIE_RECORD_DIR, stamp)
        path = base + ".mp4"
        index = 1
        while os.path.exists(path):
            path = f"{base}_{index}.mp4"
            index += 1
        return path

    @staticmethod
    def _resolve_crop(crop, frame_w, frame_h):
        """把录像区域夹到整帧范围内，返回 (x,y,w,h)；无效/退化返回 None（录整帧）。"""
        if crop is None:
            return None
        try:
            x, y, w, h = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
        except (TypeError, ValueError, IndexError):
            return None
        x = max(0, min(x, frame_w))
        y = max(0, min(y, frame_h))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)


# ------------------------------------------------------------------ 历史记录 API（供验证页签）

def list_records(folder=None):
    """扫描目录下的 .json 边车，返回按时间倒序（新->旧）的记录 dict 列表。"""
    folder = folder or LIE_RECORD_DIR  # 调用时解析，允许测试重定向目录。
    records = []
    try:
        names = os.listdir(folder)
    except OSError:
        return records
    for name in names:
        if not name.endswith(".json"):
            continue
        data = _read_json(os.path.join(folder, name))
        if not isinstance(data, dict):
            continue
        video_name = data.get("video") or (os.path.splitext(name)[0] + ".mp4")
        data["video"] = video_name
        data.setdefault("timestamp", "")
        data.setdefault("outcome", "")
        data.setdefault("meta", {})
        try:
            data["_mtime"] = os.path.getmtime(os.path.join(folder, name))
        except OSError:
            data["_mtime"] = 0.0
        records.append(data)
    records.sort(key=lambda r: r.get("_mtime", 0.0), reverse=True)
    return records


def delete_record(video_name, folder=None):
    """删除一条历史录像：按 mp4 文件名取同名 stem，成对删除 mp4+json，返回是否删掉了 mp4。"""
    folder = folder or LIE_RECORD_DIR  # 调用时解析，允许测试重定向目录。
    if not video_name:
        return False
    stem = os.path.splitext(os.path.basename(video_name))[0]
    removed = False
    for ext in (".mp4", ".json"):
        path = os.path.join(folder, stem + ext)
        try:
            if os.path.exists(path):
                os.remove(path)
                removed = removed or ext == ".mp4"
        except OSError:
            pass
    return removed
