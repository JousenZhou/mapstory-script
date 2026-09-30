"""测谎触发录像记录：把「触发确认 -> 解除」全过程的原始帧录成 mp4，并写一条 JSON 边车记录。

设计要点：
- 常开：每次测谎触发都录，无需开关；滚动保留最近 LIE_RECORD_KEEP 组（mp4+json），旧的自动删除。
- 绝不阻塞求解：录帧走「有界队列 + 独立写线程」，队列满即丢帧，30FPS 的解测谎主循环永不等录像。
- 真实时间轴：重档解测谎时实际采集常慢于标称 30FPS（如 ultra 约 11FPS），写线程按每帧采集时间戳把帧落到
  对应帧号，慢于标称处用上一帧补齐空档，使录像时长==真实时长，回放（验证页签按源帧率播放）不被压缩加速。
- 录【测谎区域标注】区域的原始像素（裁剪到区域、不叠加标注）：service 循环里的 frame 全程是 _capture
  返回的干净副本，叠加绘制都在 frame.copy() 上做；录像器按触发时确定的区域框裁剪每帧后入队，验证页签能拿它
  按同一套算法复算（裁剪录像无弹窗标题，复算探测不中即自动进裁剪模式，整帧作为图形区域）。
- 全流程吞异常仅 logger.warning：录像失败（编码/磁盘/权限）绝不影响解测谎主流程，
  与报警音、后端预热等既有「最佳努力」守卫风格一致。
"""

from __future__ import annotations

import json  # 边车记录读写。
import os  # 录像目录与文件路径。
import queue  # 有界帧队列：生产者（求解线程）与消费者（写线程）解耦。
import threading  # 异步写线程。
import time  # 时间戳、时长与滚动保留排序。

import cv2  # VideoWriter 编码 mp4。

from ok.util.logger import Logger  # 框架日志器，与 dashboard_store 一致。

logger = Logger.get_logger(__name__)

LIE_RECORD_DIR = "lie_records"  # 录像目录（相对项目根，已在 .gitignore 忽略，不入库）。
LIE_RECORD_KEEP = 50  # 滚动保留最近多少组录像（mp4+json 各一），超出按 mtime 删除最旧。
LIE_RECORD_FPS = 30  # 录像标称帧率：写线程按采集时间戳把帧对齐到该帧率的真实时间轴，采集慢于此则用上一帧补齐，保证回放时长==真实时长（不被压缩加速）。
LIE_RECORD_FOURCC = "mp4v"  # mp4 编码 fourcc，OpenCV 内置无需额外依赖。
LIE_RECORD_QUEUE_MAX = 64  # 有界队列容量（约 2 秒缓冲）：写线程跟不上传者满即丢帧，绝不回压求解线程。
_JOIN_TIMEOUT = 10.0  # stop 时等待写线程排空并退出的最长秒数，超时也继续收尾（守护线程随进程退出）。


def _to_float(value, default=0.0):  # 宽松转 float：非法值回退默认，边车字段与文件名都靠它兜底。
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


class LieRecorder:
    """一次测谎触发的录像器：start 起流与写线程，write 入队原始帧，stop 收尾并写边车记录。

    单次使用（一局一个实例）。所有对外方法都吞异常，保证录像问题不外溢到解测谎主流程。
    """

    def __init__(self):
        self._lock = threading.Lock()  # 保护 start/stop 状态切换，避免重入。
        self._writer = None  # cv2.VideoWriter，未起流或已收尾时为 None。
        self._queue = None  # 有界帧队列。
        self._thread = None  # 异步写线程句柄。
        self._running = False  # 写线程是否应在跑：stop 置 False 后写线程排空即退出。
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

    # ------------------------------------------------------------------ 起停

    def start(self, shape, meta=None, crop=None):  # 起流：建目录、开 VideoWriter、拉起异步写线程。shape 为整帧 (高, 宽)；crop 为 (x,y,w,h) 录像区域，None 表示录整帧。
        with self._lock:
            try:
                if self._writer is not None or self._running:  # 已起流，幂等返回。
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
                self._path = self._build_path()  # 生成不重名的 mp4 路径。
                fourcc = cv2.VideoWriter_fourcc(*LIE_RECORD_FOURCC)  # mp4v 编码。
                writer = cv2.VideoWriter(self._path, fourcc, float(LIE_RECORD_FPS), (self._width, self._height))  # 开流：尺寸取输出分辨率（区域或整帧）。
                if not writer.isOpened():  # 编码器不可用/路径不可写。
                    logger.warning(f"Lie record VideoWriter failed to open: {self._path}. 测谎录像无法打开写入流，本局不录。")
                    writer.release()  # 释放半成品。
                    self._path = None  # 标记未起流，write/stop 空转。
                    return
                self._writer = writer  # 记录写入流。
                self._frames = 0  # 帧计数清零。
                self._dropped = 0  # 丢帧计数清零。
                self._start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")  # 起录时间戳。
                self._queue = queue.Queue(maxsize=LIE_RECORD_QUEUE_MAX)  # 有界队列。
                self._running = True  # 允许写线程运行与 write 入队。
                self._thread = threading.Thread(target=self._consume, name="LieRecorder", daemon=True)  # 守护写线程随进程退出。
                self._thread.start()  # 拉起写线程。
                logger.info(f"Lie record started: {self._path} {self._width}x{self._height}@{LIE_RECORD_FPS}fps crop={self._crop} score={self._score:.2f} tier={self._tier}. 测谎录像已开始。")
            except Exception as e:  # 起流任何异常都不能影响解测谎。
                logger.warning(f"Lie record start failed: {e}. 测谎录像启动失败，本局不录。")
                self._running = False  # 复位，write/stop 空转。
                self._release_writer()  # 清理可能的半成品写入流。
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

    def write(self, frame):  # 入队一帧 BGR 画面（带采集时间戳；有裁剪区域时只入队该区域）；未起流、已停止或队列满时安全跳过（绝不阻塞求解）。
        if not self._running or self._queue is None or frame is None:  # 未在录像或无帧。
            return
        try:
            ts = time.perf_counter()  # 采集时刻：写线程据此把帧落到真实时间轴对应帧号，慢采集时补齐避免回放加速。
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
            self._queue.put_nowait((ts, item))  # 非阻塞入队 (采集时间戳, 帧)：capture 每帧返回独立副本，写线程跟不上传者满即丢帧。
        except queue.Full:  # 写线程暂时跟不上。
            self._dropped += 1  # 记一次丢帧，宁可丢帧也不能回压 30FPS 求解。
        except Exception:  # 入队/裁剪其它异常（队列已关闭等）。
            pass  # 吞掉，录像问题不外溢。

    def _consume(self):  # 写线程主体：按采集时间戳把帧落到真实时间轴对应帧号，慢于标称帧率处用上一帧补齐空档，使录像时长==真实时长（回放不加速）。
        t0 = None  # 首帧采集时刻，真实时间轴原点。
        fps = float(LIE_RECORD_FPS)  # 标称帧率。
        last = None  # 上一帧（已缩放到输出尺寸），用于补齐采集空档。
        while True:
            try:
                item = self._queue.get(timeout=0.5)  # 取一帧，超时就复检是否该退出。
            except queue.Empty:  # 队列暂时空。
                if not self._running:  # stop 已请求且队列排空。
                    break  # 退出写线程。
                continue  # 否则继续等下一帧。
            if item is None:  # 停止哨兵。
                self._queue.task_done()  # 标记处理完成。
                break  # 退出写线程。
            try:
                ts, frame = item  # 解包 (采集时间戳, 帧)。
                if self._writer is not None:  # 写入流仍在。
                    if frame.shape[:2] != (self._height, self._width):  # 中途窗口尺寸变化。
                        frame = cv2.resize(frame, (self._width, self._height))  # 缩放到起流分辨率，VideoWriter 要求尺寸恒定。
                    if t0 is None:  # 首帧确立时间轴原点。
                        t0 = ts
                    slot = int(round((ts - t0) * fps))  # 本帧按真实时间应处的帧号。
                    if slot < self._frames:  # 采集快于标称帧率（或已补齐过）：不回退，顺写即可。
                        slot = self._frames
                    pad = last if last is not None else frame  # 补齐用上一帧（画面连续），首帧无上一帧则用自身。
                    while self._frames < slot:  # 用上一帧填满 [已写帧号, slot) 的空档，把时间轴撑到真实时长。
                        self._writer.write(pad)
                        self._frames += 1
                    self._writer.write(frame)  # 写入当前帧到 slot。
                    self._frames += 1  # 累计已写帧数（含补齐帧）。
                    last = frame  # 记住当前帧供下次补齐。
            except Exception as e:  # 单帧写入失败（编码异常等）。
                logger.warning(f"Lie record write frame failed: {e}. 测谎录像写入单帧失败，已跳过该帧。")
            finally:
                self._queue.task_done()  # 无论成败都标记该帧处理完成。

    def stop(self, outcome="solved"):  # 收尾：排空写线程 -> 关流 -> 写边车 JSON -> 滚动保留。outcome ∈ success/failure/solved/timeout/gone/abandoned/aborted。
        with self._lock:
            if not self._running and self._writer is None:  # 从未起流或已收尾。
                return
            self._running = False  # 通知写线程排空后退出，write 也随之空转。
            try:
                if self._queue is not None:
                    self._queue.put_nowait(None)  # 投停止哨兵；队列满则失败也无妨，写线程排空后靠 _running 判定退出。
            except Exception:  # 队列已关闭或满。
                pass
            if self._thread is not None:  # 等写线程把已入队帧写完再关流，避免丢尾帧或在释放后写入。
                self._thread.join(timeout=_JOIN_TIMEOUT)
                self._thread = None
            frames = self._frames  # join 后读取，写线程已结束，计数为最终值。
            dropped = self._dropped
            path = self._path
            self._release_writer()  # 关闭并释放 mp4 写入流。
            self._write_sidecar(outcome, frames)  # 写同名 .json 边车记录。
            self._prune()  # 滚动保留最近 LIE_RECORD_KEEP 组。
            if path:  # 起流成功过才记收尾日志。
                logger.info(f"Lie record stopped: {path} outcome={outcome} frames={frames} dropped={dropped}. 测谎录像已结束。")

    def _release_writer(self):  # 关闭并释放 VideoWriter，吞异常。
        writer = self._writer  # 取当前写入流。
        self._writer = None  # 先置空，避免写线程或重复调用再用它。
        if writer is not None:
            try:
                writer.release()  # 冲刷缓冲并关闭文件。
            except Exception as e:  # 释放异常。
                logger.warning(f"Lie record release writer failed: {e}. 测谎录像关闭写入流失败。")

    # ------------------------------------------------------------------ 边车与滚动保留

    def _write_sidecar(self, outcome, frames):  # 写与 mp4 同名的 .json 边车：时间戳/触发分/精度档/结果/时长/帧数/分辨率。
        if not self._path:  # 未起流成功，无录像可记。
            return
        try:
            json_path = os.path.splitext(self._path)[0] + ".json"  # 同名 .json。
            duration = round(frames / float(LIE_RECORD_FPS), 3) if frames else 0.0  # 时长（秒）= 帧数 / 帧率。
            data = {  # 边车记录字段。
                "video": os.path.basename(self._path),  # mp4 文件名，供 list_records 与目录无关地重建路径。
                "path": self._path,  # 录制时的路径（相对项目根）。
                "timestamp": self._start_iso,  # 起录时间 ISO。
                "score": round(self._score, 4),  # 触发匹配分。
                "tier": self._tier,  # 精度档 key。
                "outcome": str(outcome or "solved"),  # 结束原因。
                "frames": int(frames),  # 已写帧数。
                "fps": LIE_RECORD_FPS,  # 帧率。
                "duration": duration,  # 时长（秒）。
                "width": int(self._width),  # 分辨率宽（有裁剪区域时为区域宽）。
                "height": int(self._height),  # 分辨率高（有裁剪区域时为区域高）。
                "region": list(self._crop) if self._crop else None,  # 录像裁剪区域 [x,y,w,h]（整帧坐标系），None 表示录整帧。
            }
            with open(json_path, "w", encoding="utf-8") as f:  # 写边车。
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._json_path = json_path  # 记录边车路径。
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
