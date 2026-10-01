"""测谎触发录像记录：把「触发确认 -> 解除」全过程的原始帧录成 mp4，并写一条 JSON 边车记录。

设计要点：
- 常开：每次测谎触发都录，无需开关；滚动保留最近 LIE_RECORD_KEEP 组（mp4+json），旧的自动删除。
- 绝不阻塞求解：录帧走「有界队列 + 独立写线程」，队列满即丢帧，解测谎主循环永不等录像。
- 采集与求解彻底分离：service 为录像器单独构建一个独立的 BitBlt 采集实例（自带 lock/DC/contexts），不再与解题
  循环共用同一把 get_frame 锁——重档光流把解题循环拖到 ~10FPS 时，录像采集线程仍按自己的节拍抓帧，不被饿死。
- 帧数按游戏当前真实帧率录制：录像采集线程以 LIE_RECORD_MAX_FPS 为上限自由抓帧，并对「与上一帧完全相同」的画面
  去重（说明游戏这一拍没有渲染新帧）。于是写入的帧数==游戏真实互异帧数、写入帧率==游戏当前真实帧率，且不会有
  重复帧（重复帧会让复算端相邻帧光流残差≈0、被误判「结算静止画面」而提前触发 scene_ended）。
- 容器帧率==实测真实帧率（回放绝不加速）：VideoWriter 延迟到标定完成才开流——先缓冲 LIE_RECORD_CALIB_FRAMES 帧
  （或最多等 LIE_RECORD_CALIB_MAX_WAIT 秒），用首末帧真实时间跨度算出实测帧率，再以该帧率建流并回灌缓冲帧。于是
  「容器时长 = 帧数 / 容器帧率 ≈ 真实录制墙钟时长」，用任意影视软件播放都是游戏真实速度。收尾把实测帧率、容器帧率、
  容器时长、真实墙钟跨度一并写进边车（measured_fps / fps / duration / real_duration），并对明显漂移告警便于核对。
- 录【测谎区域标注】区域的原始像素（裁剪到区域、不叠加标注）：service 循环里的 frame 全程是 _capture 返回的干净副本，
  叠加绘制都在 frame.copy() 上做；录像器按触发时确定的区域框裁剪每帧后入队，验证页签能拿它按同一套算法复算。
- 全流程吞异常仅 logger.warning：录像失败（编码/磁盘/权限）绝不影响解测谎主流程，
  与报警音、后端预热等既有「最佳努力」守卫风格一致。
"""

from __future__ import annotations

import json  # 边车记录读写。
import os  # 录像目录与文件路径。
import queue  # 有界帧队列：生产者（采集线程）与消费者（写线程）解耦。
import tempfile  # mp4v 编码器可用性探测的临时文件目录。
import threading  # 异步写线程与自采集线程。
import time  # 时间戳、时长、标定与滚动保留排序。

import cv2  # VideoWriter 编码 mp4。
import numpy as np  # 采集去重：判定相邻帧是否完全相同（游戏是否渲染了新帧）。

from ok.util.logger import Logger  # 框架日志器，与 dashboard_store 一致。

logger = Logger.get_logger(__name__)

LIE_RECORD_DIR = "lie_records"  # 录像目录（相对项目根，已在 .gitignore 忽略，不入库）。
LIE_RECORD_KEEP = 50  # 滚动保留最近多少组录像（mp4+json 各一），超出按 mtime 删除最旧。
LIE_RECORD_FPS = 30  # 兜底/标称帧率：无法标定（不足两帧或首末帧时间跨度≈0，如瞬时连写）时用它建流，保证仍有合理容器帧率。
LIE_RECORD_MAX_FPS = 60  # 自采集抓帧节拍上限（同时是容器帧率夹取上限）：覆盖常见游戏刷新率，又限制 PrintWindow/BitBlt 的 CPU 成本。
LIE_RECORD_MIN_FPS = 5.0  # 容器帧率夹取下限：防止开局卡顿把标定帧率算得过低导致回放慢放。
LIE_RECORD_CALIB_FRAMES = 20  # 开流标定所需的最少互异帧数（配合最小时间跨度估算真实帧率）。
LIE_RECORD_CALIB_SPAN = 0.8  # 开流标定所需的最小首末帧时间跨度（秒）：窗口拉长到近 1 秒，把开局 solver 光流尚未吃满 CPU 时的偏快帧率一并平均进来，避免标定过估导致回放偏快。
LIE_RECORD_CALIB_MAX_WAIT = 2.0  # 标定的最长等待（秒）：帧来得太慢时也用已缓冲帧开流，避免迟迟不落地。
LIE_RECORD_CALIB_MIN_SPAN = 0.05  # 计算标定帧率所需的最小首末帧跨度（秒）：低于它视为瞬时连写（无法反映真实速率），回退标称帧率。
LIE_RECORD_FOURCC = "mp4v"  # mp4 编码 fourcc，OpenCV 内置无需额外依赖。
LIE_RECORD_QUEUE_MAX = 128  # 有界队列容量（约 2 秒@60FPS 缓冲）：写线程跟不上传者满即丢帧，绝不回压采集/求解。
_JOIN_TIMEOUT = 10.0  # stop 时等待线程排空并退出的最长秒数，超时也继续收尾（守护线程随进程退出）。


def _to_float(value, default=0.0):  # 宽松转 float：非法值回退默认，边车字段与文件名都靠它兜底。
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_fps(fps):  # 把标定帧率夹到 [LIE_RECORD_MIN_FPS, LIE_RECORD_MAX_FPS]；非法/非正回退标称帧率。
    value = _to_float(fps, 0.0)
    if value <= 0.0:
        return float(LIE_RECORD_FPS)
    return max(LIE_RECORD_MIN_FPS, min(value, float(LIE_RECORD_MAX_FPS)))


def _safe_mtime(path):  # 取文件修改时间，失败返回 0.0（滚动保留与倒序排序用）。
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _read_json(path):  # 按 UTF-8 读 JSON，失败返回 None（与 dashboard_store 的容错读取惯例一致）。
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_MP4V_OK = None  # mp4v 编码器可用性探测结果缓存（None 表示尚未探测）。


def mp4v_available():  # 探测本机 mp4v 编码器是否可用（结果缓存）：延迟开流后起录时不再立即建流，测试与自检用它判断能否真录 mp4。
    global _MP4V_OK
    if _MP4V_OK is not None:  # 已探测过，直接返回缓存。
        return _MP4V_OK
    probe_path = None
    writer = None
    try:
        probe_path = os.path.join(tempfile.gettempdir(), f"_lie_mp4v_probe_{os.getpid()}.mp4")  # 临时探测文件。
        fourcc = cv2.VideoWriter_fourcc(*LIE_RECORD_FOURCC)  # 与正式录像同一 fourcc。
        writer = cv2.VideoWriter(probe_path, fourcc, float(LIE_RECORD_FPS), (16, 16))  # 建一个极小的探测流。
        ok = bool(writer.isOpened())  # 能否打开编码器后端。
        if ok:  # 打开成功再写一帧确认后端真能编码（部分后端惰性失败）。
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        _MP4V_OK = ok
    except Exception:  # 探测异常按不可用处理。
        _MP4V_OK = False
    finally:
        if writer is not None:
            try:
                writer.release()  # 释放探测流。
            except Exception:
                pass
        if probe_path is not None:
            try:
                if os.path.exists(probe_path):
                    os.remove(probe_path)  # 清理探测文件。
            except OSError:
                pass
    return _MP4V_OK


class LieRecorder:
    """一次测谎触发的录像器：start 拉起采集/写线程，采集线程按真实帧率抓帧去重后入队，写线程标定真实帧率后开流写入，
    stop 收尾并写边车记录。

    单次使用（一局一个实例）。所有对外方法都吞异常，保证录像问题不外溢到解测谎主流程。
    """

    def __init__(self):
        self._lock = threading.Lock()  # 保护 start/stop 状态切换，避免重入。
        self._writer = None  # cv2.VideoWriter：延迟到标定出真实帧率才开流，未开流或已收尾时为 None。
        self._fps_written = None  # 实际建流用的容器帧率（标定所得实测帧率，或兜底 LIE_RECORD_FPS）；影视软件按它播放。
        self._open_failed = False  # 开流失败标志：置位后不再重试、不产出任何文件。
        self._calib = []  # 标定缓冲：开流前暂存 (ts, frame)，标定完成后回灌写入并清空。
        self._queue = None  # 有界帧队列。
        self._thread = None  # 异步写线程句柄。
        self._capture = None  # 独立采集回调（可调用，返回一帧 BGR 或 None）：传入则进自采集模式，起线程按真实节拍抓帧。
        self._capture_release = None  # 采集资源释放回调（独立 BitBlt 实例的 DC/bitmap 清理），stop 时调用。
        self._cap_thread = None  # 自采集线程句柄（仅自采集模式非空）。
        self._running = False  # 采集/写线程是否应在跑：stop 置 False 后两线程排空即退出。
        self._path = None  # mp4 文件路径，起流失败时保持 None（write/stop 自动空转）。
        self._json_path = None  # 边车 JSON 路径，stop 写完记录。
        self._width = 0  # 录像宽（像素）：有裁剪区域时为区域宽，否则为整帧宽。
        self._height = 0  # 录像高（像素）：有裁剪区域时为区域高，否则为整帧高。
        self._crop = None  # 录像裁剪区域 (x, y, w, h)（整帧坐标系）；None 表示录整帧。触发时按【测谎区域标注】确定，全程固定保证输出尺寸恒定。
        self._frames = 0  # 已成功写入的帧数（写线程内累加，stop join 后读取）。
        self._dropped = 0  # 队列满被丢弃的帧数，收尾日志用于判断写线程是否跟得上。
        self._score = 0.0  # 触发匹配分，写进文件名与边车。
        self._tier = "high"  # 精度档 key，写进文件名与边车。
        self._start_iso = ""  # 起录时间 ISO 字符串，写进边车。
        self._first_ts = None  # 首帧写入时间戳（perf_counter）：与末帧之差为真实采集时长，用于算 measured_fps 与 real_duration。
        self._last_ts = None  # 末帧写入时间戳。
        self._last_enqueued = None  # 上一帧已入队的裁剪帧：自采集去重用（与之完全相同则跳过，不写重复帧）。

    # ------------------------------------------------------------------ 起停

    def start(self, shape, meta=None, crop=None, capture=None, capture_release=None):  # 起录：建目录、拉起写线程（传 capture 时再拉起自采集线程）；VideoWriter 延迟到写线程标定出真实帧率才开流。shape 为整帧 (高, 宽)；crop 为 (x,y,w,h) 录像区域，None 表示录整帧；capture 为可选采集回调；capture_release 为可选的采集资源释放回调。
        with self._lock:
            try:
                if self._writer is not None or self._running:  # 已起录，幂等返回。
                    return
                height, width = int(shape[0]), int(shape[1])  # 解包分辨率。
                if height <= 0 or width <= 0:  # 非法分辨率（截图异常等）不录。
                    logger.warning(f"Lie record skipped, invalid shape {shape}. 测谎录像跳过：分辨率非法 {shape}。")
                    return
                meta = dict(meta or {})  # 拷贝元数据，避免外部改动影响录像。
                self._score = _to_float(meta.get("score"), 0.0)  # 触发分。
                self._tier = str(meta.get("tier") or "high").strip() or "high"  # 精度档。
                self._width, self._height = width, height  # 先按整帧记录分辨率，随后按裁剪区域收敛。
                self._crop = self._resolve_crop(crop, width, height)  # 解析并夹取录像区域：非法/越界退回整帧（None）。
                if self._crop is not None:  # 有有效裁剪区域：输出分辨率改为区域尺寸。
                    self._width, self._height = self._crop[2], self._crop[3]
                os.makedirs(LIE_RECORD_DIR, exist_ok=True)  # 建录像目录（已存在则跳过）。
                self._path = self._build_path()  # 生成不重名的 mp4 路径（文件在开流时才真正创建）。
                self._frames = 0  # 帧计数清零。
                self._dropped = 0  # 丢帧计数清零。
                self._first_ts = None  # 写入时间跨度复位（供 measured_fps / real_duration 计算）。
                self._last_ts = None
                self._fps_written = None  # 容器帧率待定：写线程标定后写入。
                self._open_failed = False  # 复位开流失败标志。
                self._calib = []  # 清空标定缓冲。
                self._last_enqueued = None  # 复位去重基准。
                self._start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")  # 起录时间戳。
                self._queue = queue.Queue(maxsize=LIE_RECORD_QUEUE_MAX)  # 有界队列。
                self._capture = capture if callable(capture) else None  # 采集回调：非空则进自采集模式（write 转为空转）。
                self._capture_release = capture_release if callable(capture_release) else None  # 采集资源释放回调（独立采集实例的 DC 清理）。
                self._running = True  # 允许采集/写线程运行与入队。
                self._thread = threading.Thread(target=self._consume, name="LieRecorder", daemon=True)  # 守护写线程随进程退出。
                self._thread.start()  # 拉起写线程。
                if self._capture is not None:  # 自采集模式：起独立采集线程，按真实节拍抓帧喂录像，与被光流运算拖慢的解题循环解耦。
                    self._cap_thread = threading.Thread(target=self._capture_loop, name="LieRecorderCap", daemon=True)  # 守护采集线程随进程退出。
                    self._cap_thread.start()  # 拉起采集线程。
                logger.info(f"Lie record started: {self._path} {self._width}x{self._height} crop={self._crop} score={self._score:.2f} tier={self._tier} self_capture={self._capture is not None} independent_capture={self._capture is not None and self._capture_release is not None}. 测谎录像已开始（容器帧率按实测真实帧率标定）。")
            except Exception as e:  # 起录任何异常都不能影响解测谎。
                logger.warning(f"Lie record start failed: {e}. 测谎录像启动失败，本局不录。")
                self._running = False  # 复位，write/stop 空转。
                self._release_writer()  # 清理可能的半成品写入流。
                self._release_capture()  # 释放可能已建的独立采集实例。
                self._path = None

    def _build_path(self):  # 生成录像文件名：YYYYmmdd_HHMMSS_score{触发分:.2f}_{档位}.mp4，重名则追加序号。
        stamp = time.strftime("%Y%m%d_%H%M%S")  # 起录时间戳（秒级）。
        tier = "".join(ch for ch in self._tier if ch.isalnum()) or "high"  # 档位只保留字母数字，避免非法文件名字符。
        base = os.path.join(LIE_RECORD_DIR, f"{stamp}_score{self._score:.2f}_{tier}")  # 不含扩展名的基名。
        path = base + ".mp4"  # 首选路径。
        index = 1  # 重名序号。
        while os.path.exists(path):  # 同一秒内多次触发（极少见）时避免覆盖。
            path = f"{base}_{index}.mp4"  # 追加序号。
            index += 1
        return path

    def _resolve_crop(self, crop, frame_w, frame_h):  # 把传入的录像区域夹取到整帧范围内，返回 (x,y,w,h)；无效/退化返回 None（录整帧）。
        if crop is None:  # 未指定区域：录整帧。
            return None
        try:  # 区域可能是非法结构，宽松解析。
            x, y, w, h = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
        except (TypeError, ValueError, IndexError):  # 结构非法。
            return None  # 按录整帧处理。
        x = max(0, min(x, frame_w))  # 左上角夹到帧内。
        y = max(0, min(y, frame_h))
        w = min(w, frame_w - x)  # 宽高不超过右/下边界。
        h = min(h, frame_h - y)
        if w <= 0 or h <= 0:  # 夹取后区域退化（完全越界等）：录整帧兜底。
            return None
        return (x, y, w, h)

    # ------------------------------------------------------------------ 录帧

    def write(self, frame):  # 被动模式：外部逐帧喂一帧 BGR 画面。自采集模式（start 传了 capture）下为空转，避免与采集线程重复喂帧。
        if self._capture is not None:  # 自采集模式：录像由独立采集线程驱动，手动 write 忽略。
            return
        self._enqueue(frame)

    def _enqueue(self, frame):  # 采集/写帧统一入口：按固定裁剪区域裁帧、去重、打采集时间戳后非阻塞入队；未起录/已停/重复帧/队列满时安全跳过（绝不阻塞采集与求解）。
        if not self._running or self._queue is None or frame is None:  # 未在录像或无帧。
            return
        try:
            ts = time.perf_counter()  # 采集时刻：写线程记录首末帧时间跨度算真实帧率 measured_fps 与 real_duration。
            item = frame  # 默认录整帧。
            crop = self._crop  # 触发时确定的固定录像区域。
            if crop is not None:  # 只录【测谎区域标注】区域。
                x, y, w, h = crop
                fh, fw = frame.shape[:2]  # 当前帧尺寸（窗口中途变化时按新尺寸再夹一次）。
                x0 = min(max(0, x), fw)
                y0 = min(max(0, y), fh)
                x1 = min(max(0, x + w), fw)
                y1 = min(max(0, y + h), fh)
                if x1 <= x0 or y1 <= y0:  # 区域完全越界：本帧无可录内容，跳过（不写整帧，保持输出尺寸恒定）。
                    return
                item = frame[y0:y1, x0:x1].copy()  # 裁剪并拷贝：保证内存连续（VideoWriter 要求）且不牵住整帧内存。
            if self._last_enqueued is not None and np.array_equal(self._last_enqueued, item):  # 与上一帧完全相同：游戏这一拍没渲染新帧，去重跳过（写入帧数==真实互异帧数，且杜绝重复帧破坏复算光流）。
                return
            self._queue.put_nowait((ts, item))  # 非阻塞入队 (采集时间戳, 帧)：每帧为独立副本，写线程跟不上传者满即丢帧。
            self._last_enqueued = item  # 更新去重基准（仅在成功入队后）。
        except queue.Full:  # 写线程暂时跟不上。
            self._dropped += 1  # 记一次丢帧，宁可丢帧也不能回压采集/求解。
        except Exception:  # 入队/裁剪其它异常（队列已关闭等）。
            pass  # 吞掉，录像问题不外溢。

    def _capture_loop(self):  # 自采集线程主体：以 LIE_RECORD_MAX_FPS 为上限自由抓帧（独立采集实例，不与解题循环共用锁），去重后入队——写入帧率即游戏真实互异帧率。
        min_interval = 1.0 / float(LIE_RECORD_MAX_FPS)  # 抓帧节拍上限对应的最小帧间隔（秒）。
        next_t = time.perf_counter()  # 下一帧的节拍点（deadline 调度避免累积漂移）。
        while self._running:  # stop 置 False 后退出。
            try:
                frame = self._capture()  # 独立抓一帧（自带 lock/DC，返回副本，线程安全）。
            except Exception:  # 采集异常（窗口失效等）不中断录像线程。
                frame = None
            if frame is not None:  # 抓到画面才入队（取不到则跳过该节拍，不补帧；去重在 _enqueue 内做）。
                self._enqueue(frame)
            next_t += min_interval  # 推进节拍点。
            delay = next_t - time.perf_counter()  # 距下一节拍点的剩余秒数。
            if delay > 0:  # 未落后：睡到下一节拍点，把抓帧率限制在上限内。
                time.sleep(delay)
            else:  # 落后于节拍（采集慢/机器负载）：重置基准，避免恢复后疯狂补采；实测帧率会如实反映到容器帧率。
                next_t = time.perf_counter()

    def _consume(self):  # 写线程主体：开流前把帧缓冲进 _calib，标定出真实帧率后开流并回灌，随后原样流式写入（不补帧、不去重——去重在入队侧做）。
        while True:
            try:
                item = self._queue.get(timeout=0.2)  # 取一帧，超时就复检是否该退出或该按最长等待开流。
            except queue.Empty:  # 队列暂时空。
                if not self._running:  # stop 已请求且队列排空。
                    break  # 退出写线程。
                self._maybe_open_from_calib(force=False)  # 帧来得慢时，超过最长等待也要用已缓冲帧开流。
                continue
            if item is None:  # 停止哨兵。
                self._queue.task_done()  # 标记处理完成。
                break  # 退出写线程。
            try:
                self._handle_item(item)  # 缓冲或写入单帧。
            except Exception as e:  # 单帧处理失败不中断写线程。
                logger.warning(f"Lie record consume failed: {e}. 测谎录像写线程处理单帧失败，已跳过。")
            finally:
                self._queue.task_done()  # 无论成败都标记该帧处理完成。
        self._maybe_open_from_calib(force=True)  # 退出前：把仍在标定缓冲里的帧强制开流写完（录像很短、没等到标定触发就收尾的情况）。

    def _handle_item(self, item):  # 处理一帧：未开流则进标定缓冲（并尝试触发开流），已开流则直接写入。
        ts, frame = item
        if self._writer is None:  # 尚未开流。
            if self._open_failed:  # 开流已失败：丢弃，不再缓冲。
                return
            self._calib.append(item)  # 暂存待标定。
            self._maybe_open_from_calib(force=False)  # 攒够帧/跨度或超过最长等待就开流。
            return
        self._write_frame(ts, frame)  # 已开流：原样写入。

    def _maybe_open_from_calib(self, force=False):  # 标定并开流：缓冲帧数与时间跨度足够（或 force/超过最长等待）时，用实测帧率建流并把缓冲帧回灌写入。
        if self._writer is not None or self._open_failed or not self._calib:  # 已开流/已失败/无缓冲帧：无需处理。
            return
        first_ts = self._calib[0][0]  # 缓冲首帧采集时刻。
        span = self._calib[-1][0] - first_ts  # 缓冲首末帧时间跨度。
        count = len(self._calib)  # 缓冲帧数。
        ready = force or (count >= LIE_RECORD_CALIB_FRAMES and span >= LIE_RECORD_CALIB_SPAN) or (time.perf_counter() - first_ts >= LIE_RECORD_CALIB_MAX_WAIT)
        if not ready:  # 尚未到标定时机。
            return
        fps = self._calib_fps(self._calib)  # 由缓冲帧估算真实帧率。
        if self._open_writer_at_fps(fps) is None:  # 建流失败。
            self._open_failed = True  # 标记失败，后续不再重试。
            self._calib = []  # 丢弃缓冲。
            return
        for ts, frame in self._calib:  # 回灌缓冲帧（此时 _writer 已就绪）。
            self._write_frame(ts, frame)
        self._calib = []  # 清空缓冲，之后直接流式写入。

    def _calib_fps(self, items):  # 由标定缓冲的首末帧时间跨度估算真实帧率，夹取到合法区间；不足两帧或跨度≈0（瞬时连写）回退标称帧率。
        count = len(items)
        if count >= 2:
            span = items[-1][0] - items[0][0]
            if span >= LIE_RECORD_CALIB_MIN_SPAN:  # 跨度够大才可信。
                return _clamp_fps((count - 1) / span)
        return float(LIE_RECORD_FPS)  # 无法标定：回退标称帧率（如被动模式下的瞬时连写）。

    def _open_writer_at_fps(self, fps):  # 以指定容器帧率开流；成功返回 VideoWriter，失败置空并返回 None。
        if not self._path:  # 无有效路径（起录已失败）。
            return None
        try:
            fourcc = cv2.VideoWriter_fourcc(*LIE_RECORD_FOURCC)  # mp4v 编码。
            writer = cv2.VideoWriter(self._path, fourcc, float(fps), (self._width, self._height))  # 开流：尺寸取输出分辨率（区域或整帧），帧率取实测真实帧率。
            if not writer.isOpened():  # 编码器不可用/路径不可写。
                logger.warning(f"Lie record VideoWriter failed to open: {self._path} @{fps:.3f}fps. 测谎录像无法打开写入流，本局不录。")
                writer.release()  # 释放半成品。
                return None
            self._writer = writer  # 记录写入流。
            self._fps_written = float(fps)  # 记录实际容器帧率（写边车与收尾日志用）。
            return writer
        except Exception as e:  # 建流异常。
            logger.warning(f"Lie record open writer failed: {e}. 测谎录像建流异常，本局不录。")
            return None

    def _write_frame(self, ts, frame):  # 写入单帧：尺寸不符则缩放，维护首末帧时间戳与帧计数，吞异常。
        try:
            if self._writer is None or frame is None:  # 未开流或无帧。
                return
            if frame.shape[:2] != (self._height, self._width):  # 中途窗口尺寸变化。
                frame = cv2.resize(frame, (self._width, self._height))  # 缩放到起流分辨率，VideoWriter 要求尺寸恒定。
            if self._first_ts is None:  # 首帧写入时刻：真实时间轴原点。
                self._first_ts = ts
            self._last_ts = ts  # 末帧写入时刻：与首帧之差即真实采集时长。
            self._writer.write(frame)  # 原样写入当前真实帧（不补帧、不写重复帧）。
            self._frames += 1  # 累计已写帧数（==真实互异采集帧数）。
        except Exception as e:  # 单帧写入失败（编码异常等）。
            logger.warning(f"Lie record write frame failed: {e}. 测谎录像写入单帧失败，已跳过该帧。")

    def stop(self, outcome="solved"):  # 收尾：停采集线程 -> 排空并回灌写线程 -> 关流 -> 释放采集实例 -> 写边车 JSON -> 滚动保留。outcome ∈ success/failure/solved/timeout/gone/abandoned/aborted。
        with self._lock:
            if not self._running and self._writer is None and not self._calib:  # 从未起录或已收尾。
                self._release_capture()  # 仍幂等释放采集资源。
                return
            self._running = False  # 通知采集线程与写线程排空后退出，write 也随之空转。
            cap_thread = self._cap_thread  # 取自采集线程句柄。
            self._cap_thread = None  # 先摘引用，避免重复 join。
            if cap_thread is not None:  # 先等采集线程停抓（不再入队），再收写线程，避免收尾期间还有新帧挤进来。
                cap_thread.join(timeout=_JOIN_TIMEOUT)
            try:
                if self._queue is not None:
                    self._queue.put_nowait(None)  # 投停止哨兵；队列满则失败也无妨，写线程排空后靠 _running 判定退出。
            except Exception:  # 队列已关闭或满。
                pass
            if self._thread is not None:  # 等写线程把已入队帧（含标定缓冲回灌）写完再关流，避免丢尾帧或在释放后写入。
                self._thread.join(timeout=_JOIN_TIMEOUT)
                self._thread = None
            frames = self._frames  # join 后读取，写线程已结束，计数为最终值。
            dropped = self._dropped
            measured_fps = self._measured_fps(frames)  # 整段实测平均帧率（供日志与边车）。
            path = self._path if frames > 0 else None  # 仅真正写入过帧才算产出录像（0 帧不落地任何文件）。
            self._release_writer()  # 关闭并释放 mp4 写入流。
            self._release_capture()  # 释放独立采集实例的 GDI 资源（DC/bitmap）。
            if frames > 0:  # 有帧才写边车。
                self._write_sidecar(outcome, frames)
            self._prune()  # 滚动保留最近 LIE_RECORD_KEEP 组。
            if path:  # 起流成功且写过帧才记收尾日志。
                logger.info(f"Lie record stopped: {path} outcome={outcome} frames={frames} dropped={dropped} fps={self._fps_written} measured_fps={measured_fps}. 测谎录像已结束。")

    def _release_writer(self):  # 关闭并释放 VideoWriter，吞异常。
        writer = self._writer  # 取当前写入流。
        self._writer = None  # 先置空，避免写线程或重复调用再用它。
        if writer is not None:
            try:
                writer.release()  # 冲刷缓冲并关闭文件。
            except Exception as e:  # 释放异常。
                logger.warning(f"Lie record release writer failed: {e}. 测谎录像关闭写入流失败。")

    def _release_capture(self):  # 释放独立采集实例（其 GDI 资源），吞异常；幂等（释放后置空回调）。
        release = self._capture_release  # 取释放回调。
        self._capture_release = None  # 先摘引用，避免重复释放。
        if release is not None:
            try:
                release()  # 释放独立 BitBlt 实例的 DC/bitmap。
            except Exception as e:  # 释放异常不影响收尾。
                logger.warning(f"Lie record release capture failed: {e}. 测谎录像释放独立采集实例失败。")

    # ------------------------------------------------------------------ 边车与滚动保留

    def _measured_fps(self, frames):  # 整段实测平均帧率 = (帧数-1)/首末帧真实时间跨度；不足两帧或跨度≈0 时回退容器帧率。
        if frames >= 2 and self._first_ts is not None and self._last_ts is not None:
            span = self._last_ts - self._first_ts  # 真实采集时长（秒）。
            if span > 1e-6:  # 跨度有效才做除法，避免瞬时连写导致除零/爆表。
                return round((frames - 1) / span, 3)
        if self._fps_written:  # 无法由跨度估算时回退实际容器帧率。
            return round(float(self._fps_written), 3)
        return float(LIE_RECORD_FPS)

    def _write_sidecar(self, outcome, frames):  # 写与 mp4 同名的 .json 边车：时间戳/触发分/精度档/结果/帧数/容器帧率/实测帧率/容器时长/真实墙钟/分辨率/区域。
        if not self._path:  # 未起流成功，无录像可记。
            return
        try:
            fps_written = float(self._fps_written) if self._fps_written else float(LIE_RECORD_FPS)  # 实际容器帧率（==标定所得游戏真实帧率）。
            json_path = os.path.splitext(self._path)[0] + ".json"  # 同名 .json。
            duration = round(frames / fps_written, 3) if (frames and fps_written > 0) else 0.0  # 容器时长（秒）= 帧数 / 容器帧率 ≈ 真实录制墙钟（回放不加速）。
            measured_fps = self._measured_fps(frames)  # 整段实测平均帧率（应≈fps；明显偏离说明标定期与整段速率有漂移）。
            real_duration = 0.0  # 首末帧真实时间跨度（诊断用）。
            if self._first_ts is not None and self._last_ts is not None and self._last_ts >= self._first_ts:
                real_duration = round(self._last_ts - self._first_ts, 3)
            data = {  # 边车记录字段。
                "video": os.path.basename(self._path),  # mp4 文件名，供 list_records 与目录无关地重建路径。
                "path": self._path,  # 录制时的路径（相对项目根）。
                "timestamp": self._start_iso,  # 起录时间 ISO。
                "score": round(self._score, 4),  # 触发匹配分。
                "tier": self._tier,  # 精度档 key。
                "outcome": str(outcome or "solved"),  # 结束原因。
                "frames": int(frames),  # 已写帧数（==游戏真实互异帧数，去重、不补帧）。
                "fps": round(fps_written, 3),  # 实际容器帧率（VideoWriter 建流用，==标定所得真实帧率；影视软件按它播放）。
                "measured_fps": measured_fps,  # 整段实测平均帧率：正常应≈fps（明显偏离说明游戏帧率中途漂移）。
                "duration": duration,  # 容器时长（秒）≈ 真实录制墙钟。
                "real_duration": real_duration,  # 首末帧真实时间跨度（秒），与 duration 应基本一致，偏差大即回放速度有出入。
                "width": int(self._width),  # 分辨率宽（有裁剪区域时为区域宽）。
                "height": int(self._height),  # 分辨率高（有裁剪区域时为区域高）。
                "region": list(self._crop) if self._crop else None,  # 录像裁剪区域 [x,y,w,h]（整帧坐标系），None 表示录整帧。
            }
            with open(json_path, "w", encoding="utf-8") as f:  # 写边车。
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._json_path = json_path  # 记录边车路径。
            if real_duration > 0.5 and measured_fps > 0 and self._fps_written:  # 时长忠实度自检：标定帧率（开局估算）与整段实测帧率偏差过大，说明游戏帧率中途漂移，告警便于排查（不影响已录文件）。
                drift = abs(float(self._fps_written) - measured_fps) / measured_fps
                if drift > 0.08:
                    logger.warning(f"Lie record fps drift {drift:.1%}: calibrated={float(self._fps_written):.2f} measured={measured_fps:.2f} duration={duration}s real={real_duration}s. 测谎录像标定帧率与整段实测帧率偏差偏大（游戏帧率中途漂移），回放速度可能略有出入。")
        except Exception as e:  # 边车写失败不影响已录好的 mp4。
            logger.warning(f"Lie record write sidecar failed: {e}. 测谎录像写边车记录失败。")

    def _prune(self):  # 滚动保留：按 mp4/json 成对的最大 mtime 倒序，只留最近 LIE_RECORD_KEEP 组，其余删除。
        try:
            if not os.path.isdir(LIE_RECORD_DIR):  # 目录不存在无从清理。
                return
            stems = {}  # 基名 -> 该组（mp4+json）最大 mtime。
            for name in os.listdir(LIE_RECORD_DIR):
                stem, ext = os.path.splitext(name)
                if ext.lower() in (".mp4", ".json"):  # 只统计录像相关文件。
                    mtime = _safe_mtime(os.path.join(LIE_RECORD_DIR, name))
                    stems[stem] = max(stems.get(stem, 0.0), mtime)  # 同组取较新时间。
            ordered = sorted(stems.items(), key=lambda kv: kv[1], reverse=True)  # 由新到旧。
            for stem, _ in ordered[LIE_RECORD_KEEP:]:  # 超出保留数的旧记录。
                for ext in (".mp4", ".json"):  # mp4 与 json 一并删。
                    path = os.path.join(LIE_RECORD_DIR, stem + ext)
                    try:
                        if os.path.exists(path):
                            os.remove(path)  # 删除旧录像文件。
                    except OSError as e:  # 单个文件删除失败不影响其余清理。
                        logger.warning(f"Lie record prune failed for {path}: {e}. 测谎录像滚动清理删除文件失败。")
        except Exception as e:  # 清理整体失败不影响录像本身。
            logger.warning(f"Lie record prune error: {e}. 测谎录像滚动清理异常。")


def list_records(folder=LIE_RECORD_DIR):  # 扫描目录下的 .json 边车，返回按时间倒序（新->旧）的记录 dict 列表，供 UI 历史下拉。
    records = []  # 结果列表。
    try:
        if not os.path.isdir(folder):  # 目录不存在返回空。
            return records
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):  # 只读边车。
                continue
            json_path = os.path.join(folder, name)
            data = _read_json(json_path)
            if not isinstance(data, dict):  # 脏数据跳过。
                continue
            video_name = data.get("video") or (os.path.splitext(name)[0] + ".mp4")  # mp4 文件名，缺失则按边车名推断。
            video_path = os.path.join(folder, video_name)  # 与 folder 拼接，使传入临时目录（测试）也正确。
            if not os.path.exists(video_path):  # mp4 已被清理或从未写成，跳过该记录。
                continue
            data["video"] = video_name  # 规范化字段。
            data["path"] = video_path  # 覆盖为该 folder 下的实际路径，UI 直接用它复算。
            data["_mtime"] = _safe_mtime(json_path)  # 排序键：边车修改时间。
            records.append(data)
    except Exception as e:  # 扫描失败返回已收集的部分，不抛给 UI。
        logger.warning(f"Lie record list failed: {e}. 测谎录像历史列举失败。")
    records.sort(key=lambda item: item.get("_mtime", 0.0), reverse=True)  # 按时间倒序，最新录像排最前。
    return records


def delete_record(path, folder=LIE_RECORD_DIR):  # 删除一条历史录像：按 mp4 路径取同名 stem，成对删除 folder 下的 mp4+json，返回是否删掉了 mp4。
    try:
        stem = os.path.splitext(os.path.basename(str(path)))[0]  # 取录像基名（与 _prune/list_records 同口径，兼容传入临时目录）。
        if not stem:  # 空基名（非法路径）不删。
            return False
        removed_mp4 = False  # 是否成功删掉 mp4（作为返回值）。
        for ext in (".mp4", ".json"):  # mp4 与边车 json 一并删，与滚动保留一致。
            target = os.path.join(folder, stem + ext)
            try:
                if os.path.exists(target):
                    os.remove(target)  # 删除该文件。
                    if ext == ".mp4":
                        removed_mp4 = True  # 标记 mp4 已删。
            except OSError as e:  # 单个文件删除失败不影响其余。
                logger.warning(f"Lie record delete failed for {target}: {e}. 测谎录像删除文件失败。")
        return removed_mp4
    except Exception as e:  # 删除整体异常不抛给 UI。
        logger.warning(f"Lie record delete error: {e}. 测谎录像删除异常。")
        return False
