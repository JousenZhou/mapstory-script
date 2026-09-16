# 测谎检验页签：用户上传谎言检测器录像，按视频原帧率实时播放（处理跟不上时丢帧保实时，贴近真实采集场景），
# 用 DIS 稠密光流对齐历史帧 + 粒子滤波跟踪透明轮廓（无神经网络），
# 画面叠加轮廓、状态与实测/源帧率，视频模式不发送鼠标，仅验证算法效果。
import os
from pathlib import Path
import queue
import threading
import time

import cv2  # 导入 OpenCV，用于读视频与叠加绘制。
import numpy as np  # 导入 NumPy，用于画面矩阵。
from PySide6.QtCore import Qt, QThread, QTimer, Signal  # 导入 Qt 线程、定时器与信号。
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QSizePolicy, QWidget  # 导入布局与控件。
from qfluentwidgets import BodyLabel, FluentIcon, PushButton, TextEdit  # 导入 Fluent 控件。

from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。
from src.ui.DashboardTab import VisionLabel  # 复用看板页签的自适应画面标签。

TICK_FPS_FALLBACK = 30  # 源帧率缺失时的回退帧率，与参考项目 FPS=30 一致。
TIMEOUT_TICKS = 545  # 超时预算 545 tick@30fps（约 18 秒），与参考项目 solve_shape.rs 一致；实际按源帧率折算成视频时长。
LOCATE_MAX_SIDE = 400  # 全屏弹窗定位的分辨率上限（最长边像素）：超过则先缩帧再匹配，坐标换算回原图。

try:  # 探测 OpenCV 是否包含 DIS 稠密光流，缺失时页签降级为不可运行。
    INFERENCE_AVAILABLE = hasattr(cv2, 'DISOpticalFlow_create')
except Exception:
    INFERENCE_AVAILABLE = False

from src.liedetector.detector import (  # 仅保留多尺度模板定位能力（弹窗标题定位），不再需要 YOLO 检测器。
    TEMPLATE_THRESHOLD, LieDetectorRegion, MultiScaleTemplate, _template_best_score)
from src.liedetector.shape_session import (  # 在线编排。
    ShapeTrackParams, ShapeTrackSession,
    SOURCE_BORDER, SOURCE_COLOR, SOURCE_INTERPOLATED, SOURCE_PREDICTION, SOURCE_SCENE_ENDED)


class _EmitLogger:  # 把检测器内部日志转发成 Qt 信号的适配器。

    def __init__(self, emit):
        self._emit = emit

    def info(self, message):
        self._emit(str(message))

    def __call__(self, message):  # shape_session 的 logger 回调是单参数函数调用。
        self._emit(str(message))


def draw_shape_overlay(frame, region, result, tick, status, timeout_ticks, fps_text="", mode_tag=""):
    """在全帧上叠加：区域框、轮廓、中心点、状态文本（含模式/置信度/对称周期/信噪比）与右上角帧率。"""

    rx, ry, rw, rh = region  # 解包区域坐标。
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色区域框。
    cv2.putText(frame, "LIE REGION", (rx, max(ry - 6, 12)),  # 区域标签。
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    # 颜色约定对齐外部脚本：color 绿、border 黄、prediction 红、interpolated 蓝、scene-ended/未学习 灰。
    source = result.source  # 来源标签。
    if source == SOURCE_COLOR:  # 白色轮廓可见。
        color = (0, 220, 0)  # 绿色。
    elif source == SOURCE_BORDER:  # 边界残差证据。
        color = (0, 220, 255)  # 黄色。
    elif source == SOURCE_INTERPOLATED:  # 回看插值帧。
        color = (255, 170, 0)  # 蓝色。
    elif source == SOURCE_PREDICTION:  # 纯预测外推。
        color = (0, 70, 255)  # 红色。
    else:  # 场景结束或未学习。
        color = (185, 185, 185)  # 灰色。
    if result.contour is not None:  # 有有效轮廓时画轮廓。
        pts = result.contour + np.array([[[rx, ry]]], dtype=np.int32)  # 区域坐标偏移加到全帧坐标。
        cv2.polylines(frame, [pts], True, color, 3, cv2.LINE_AA)  # 3 像素粗线。
    if result.center is not None:  # 有有效中心时画中心点。
        cx, cy = int(result.center[0] + rx), int(result.center[1] + ry)  # 换算到全帧坐标。
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)  # 半径 5 实心圆。
    # 状态文本。
    if result.tracker_alive:  # 跟踪输出有效时显示详细参数。
        text = (f"SHAPE {source} conf={result.confidence:.2f} "
                f"sym={result.symmetry_period:g}deg snr={result.border_snr:.2f}")
    elif source == SOURCE_SCENE_ENDED:  # 场景已结束。
        text = "SHAPE: scene ended"
    else:  # 未学习/等待。
        text = "SHAPE: learning"
    cv2.putText(frame, text, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)  # 状态文本。
    cv2.putText(frame, f"tick {tick}/{timeout_ticks} [{mode_tag}] {status}", (8, 22),  # 左上角状态条（含模式标签）。
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if fps_text:  # 右上角实测处理帧率。
        fw = frame.shape[1]  # 帧宽。
        (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)  # 测量文本宽度。
        cv2.putText(frame, fps_text, (fw - text_w - 8, 24),  # 右对齐到右上角。
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


class LieDetectorWorker(QThread):  # 工作线程：在线分析 + 实时预览，不限速全速处理。

    log_message = Signal(str)  # 日志消息信号。
    algorithm_ready = Signal(str)  # 算法准备就绪信号（参数摘要）。
    finished_result = Signal(bool, str)  # 结束信号：(是否通过, 结果描述)。

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path  # 录像路径。
        self.frame_queue = queue.Queue(maxsize=2)  # 只保留最新帧，避免 UI 积压。
        self._stopped = threading.Event()  # 停止标志。

    def stop(self):  # 请求停止并等待线程退出。
        self._stopped.set()
        self.wait(3000)

    def _push_frame(self, frame):  # 推送标注帧：队列满时丢弃旧帧。
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def run(self):  # 线程主循环：全速逐帧分析 + 实时预览。
        cap = cv2.VideoCapture(self.video_path)  # 打开录像。
        if not cap.isOpened():  # 打开失败。
            self.log_message.emit(f"cannot open video 无法打开视频: {self.video_path}")
            self.finished_result.emit(False, "open failed")
            return
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽。
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高。
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))  # 源帧率：决定播放节拍与算法时间阈值。
        if not (5.0 <= src_fps <= 240.0):  # 部分容器返回 0/1000 等异常值：回退 30。
            src_fps = float(TICK_FPS_FALLBACK)
        frame_interval = 1.0 / src_fps  # 原帧率播放的单帧间隔（秒）。
        timeout_ticks = max(1, int(round(TIMEOUT_TICKS / TICK_FPS_FALLBACK * src_fps)))  # 超时预算按源帧率折算，固定约 18 秒视频时长。
        self.log_message.emit(
            f"video {frame_w}x{frame_h} opened 视频已打开 src_fps={src_fps:.1f} 源帧率 timeout={timeout_ticks} ticks")
        region_finder = LieDetectorRegion()  # 多尺度模板定位器。
        try:  # 前置流程（模板缩小/探测/定位器构造）异常时也要正常收尾，避免线程静默死亡卡死 UI。
            locate_scale = min(1.0, LOCATE_MAX_SIDE / max(frame_w, frame_h, 1))  # 定位缩帧系数：按最长边封顶。
            if locate_scale < 1.0:  # 大帧：定位模板同步缩小同一比例，保持匹配分与坐标系一致。
                for _name in ("title_template", "prepare_template"):  # 标题与“准备中”两个模板。
                    _tpl = getattr(region_finder, _name)  # 原模板（property 读取）。
                    _new_w = max(8, int(round(_tpl.shape[1] * locate_scale)))  # 缩后宽。
                    _new_h = max(8, int(round(_tpl.shape[0] * locate_scale)))  # 缩后高。
                    setattr(region_finder, "_" + _name,  # 写回私有属性（公开 property 只读）。
                            cv2.resize(_tpl, (_new_w, _new_h), interpolation=cv2.INTER_AREA))
                self.log_message.emit(f"locate downscale {locate_scale:.3f} 全屏定位帧与模板同步缩小")
            # 启动探测：读前几帧多尺度匹配标题模板。裁剪的区域录像无标题（得分远低于阈值），
            # 全屏录像开头弹窗通常已在画面内；探测不中即锁定裁剪模式，
            # 避免全屏多尺度模板匹配拖垮 30FPS 节拍。
            crop_mode = False  # 默认全屏模式。
            probe_best = -1.0  # 探测最高匹配分。
            probe_title = None  # 探测到的标题位置。
            for _ in range(5):  # 最多读 5 帧探测。
                ok, probe_frame = cap.read()
                if not ok:  # 读帧失败中断。
                    break
                if locate_scale < 1.0:  # 大帧：探测也在缩帧上做，与模板坐标系一致。
                    probe_frame = cv2.resize(probe_frame, None, fx=locate_scale, fy=locate_scale,
                                             interpolation=cv2.INTER_AREA)
                probe_best = max(probe_best, _template_best_score(probe_frame, region_finder.title_template))
                if probe_best >= TEMPLATE_THRESHOLD:  # 探测命中。
                    probe_title = region_finder.find_title(probe_frame)
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到文件开头。
            if probe_title is not None:  # 命中弹窗标题。
                self.log_message.emit(f"dialog title found in probe, fullscreen mode 探测命中标题，全屏模式")
            else:  # 未命中，锁定裁剪模式。
                crop_mode = True
                self.log_message.emit(
                    f"crop mode: no title in probe (best {probe_best:.3f} < {TEMPLATE_THRESHOLD}) "
                    f"裁剪模式：探测未见弹窗标题（最高分 {probe_best:.3f} 低于阈值），整帧作为图形区域 {frame_w}x{frame_h}")
            prepare_matcher = MultiScaleTemplate(region_finder.prepare_template, TEMPLATE_THRESHOLD)  # "准备中"模板匹配器。
        except Exception as exc:  # 前置流程异常：记录后按失败收尾，绝不静默死亡。
            self.log_message.emit(f"init error 初始化异常: {exc}")
            cap.release()
            self.finished_result.emit(False, f"init error: {exc}")
            return

        # ---- 在线分析初始化 ----
        params = ShapeTrackParams()  # 使用实时档参数。
        logger_adapter = _EmitLogger(self.log_message.emit)  # 日志适配器。
        session = None  # 先置空：收尾日志兼容会话构造失败的情形。
        session = ShapeTrackSession(params=params, logger=logger_adapter)  # 在线编排状态机。
        found_dialog = False  # 是否已定位到弹窗。
        algo_emitted = False  # 是否已发出算法就绪信号。
        result_ok = False  # 最终判定。
        last_target_tick = 0  # 最近一次有效跟踪的 tick。
        result_msg = f"timeout after {timeout_ticks} ticks 超过 {timeout_ticks} tick 未通过"  # 默认超时。
        last_region = None  # 上一帧的区域坐标。
        tick = 0  # 节拍计数。
        user_stopped = False  # 用户是否手动停止。
        fps_frames = 0  # FPS 统计窗口内帧数。
        fps_window_start = time.perf_counter()  # FPS 统计窗口起点。
        fps_value = 0.0  # 最近一次统计出的实测帧率。
        decode_ms = 0.0  # 最近一帧视频解码耗时（毫秒）。
        locate_ms = 0.0  # 最近一帧弹窗定位耗时（毫秒）。
        track_ms = 0.0  # 最近一帧粒子跟踪耗时（毫秒）。
        mode_tag = "CROP" if crop_mode else "FULL"  # 画面左上角显示的模式标签。
        drop_count = 0  # 处理超时累计丢弃的帧数（模拟真实采集掉帧）。
        frame_idx = 0  # 下一个待读帧序号，丢帧快进时推进。
        play_start = 0.0  # 原帧率播放的绝对时间起点，主循环首帧时锁定。
        next_deadline = 0.0  # 当前帧的原帧率播放节拍点（绝对时间）。

        self.log_message.emit(  # 打印实际生效的参数，便于与外部脚本回归对照。
            f"algorithm params: scale={params.process_scale} lags={params.temporal_lags} "
            f"particles={params.particle_count} proposals={params.global_proposals}")

        try:  # ---- 在线分析 + 实时预览（按源帧率播放，处理超时丢帧保实时） ----
            while not self._stopped.is_set() and tick < timeout_ticks:
                next_deadline = play_start + frame_idx * frame_interval  # 本帧的原帧率播放节拍点。
                t_decode = time.perf_counter()  # 解码计时起点。
                ok, frame = cap.read()  # 读下一帧。
                decode_ms = (time.perf_counter() - t_decode) * 1000  # 解码耗时。
                if not ok:  # 视频播完。
                    if crop_mode and last_target_tick > 0 and tick - last_target_tick <= 10:  # 裁剪模式：目标还在被跟踪即通过。
                        result_ok = True
                        result_msg = (
                            f"crop mode: target tracked to tick {last_target_tick} "
                            f"裁剪模式：目标被持续跟踪至第 {last_target_tick} tick，验证通过")
                    else:
                        result_msg = "video ended before result 视频播完仍未出结果"
                    break
                tick += 1
                frame_idx += 1
                if play_start == 0.0:  # 首帧：锁定播放节拍起点。
                    play_start = time.perf_counter()
                    next_deadline = play_start
                if time.perf_counter() > next_deadline:  # 节拍已过：处理跟不上，丢弃落后帧快进追平实时。
                    behind = int((time.perf_counter() - next_deadline) / frame_interval)  # 落后帧数。
                    if behind > 0:
                        drop_count += behind  # 累计丢帧。
                        frame_idx += behind  # 推进读帧游标。
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # 直接 seek，跳过落后帧。
                        if drop_count % 30 < behind:  # 每丢约 1 秒报告一次，避免刷屏。
                            self.log_message.emit(
                                f"tick {tick}: dropped {drop_count} frames 处理超时累计丢帧 {drop_count}，保持实时播放")
                fps_frames += 1  # FPS 统计窗口 +1。
                fps_elapsed = time.perf_counter() - fps_window_start  # 窗口已经过时长。
                if fps_elapsed >= 0.5:  # 每 0.5 秒更新一次实测帧率。
                    fps_value = fps_frames / fps_elapsed  # 实测处理帧率。
                    fps_frames = 0  # 重置窗口。
                    fps_window_start = time.perf_counter()  # 重置窗口起点。
                fps_text = (f"{fps_value:.0f}/{src_fps:.0f} FPS" +  # 右上角：实测处理帧率/源帧率，丢帧时附计数。
                            (f" drop={drop_count}" if drop_count else "")) if fps_value > 0 else ""
                if crop_mode:  # 裁剪模式：整帧即区域。
                    preparing = False
                    region = (0, 0, frame.shape[1], frame.shape[0])
                    locate_ms = 0.0  # 裁剪模式无需模板定位。
                else:  # 全屏模式：多尺度模板定位弹窗（大帧先缩到分辨率上限再匹配）。
                    t_locate = time.perf_counter()  # 定位计时起点。
                    if locate_scale < 1.0:  # 帧超上限：缩帧后匹配。
                        small_frame = cv2.resize(frame, None, fx=locate_scale, fy=locate_scale,
                                                 interpolation=cv2.INTER_AREA)
                    else:
                        small_frame = frame
                    preparing = prepare_matcher.match(small_frame)[0] if found_dialog else False
                    small_region = region_finder.find_region(small_frame)
                    if small_region is None:  # 缩帧上未命中。
                        region = None
                    else:  # 缩帧坐标按 1/locate_scale 换算回原图全尺度。
                        region = tuple(int(round(v / locate_scale)) for v in small_region)
                    locate_ms = (time.perf_counter() - t_locate) * 1000  # 定位耗时。
                if preparing or region is None:  # 弹窗未出现或准备中：不推进求解。
                    if found_dialog and not preparing:  # 弹窗曾出现后消失，判定挑战完成通过。
                        result_ok = True
                        result_msg = f"dialog closed at tick {tick} 弹窗在第 {tick} tick 关闭，判定通过"
                        self.log_message.emit(result_msg)
                        cv2.putText(frame, "PASSED", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        self._push_frame(frame)
                        break
                    status = "preparing..." if preparing else "waiting for dialog..."
                    cv2.putText(frame, f"tick {tick}/{timeout_ticks} [{mode_tag}] {status}", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    if fps_text:  # 等待阶段也显示实测帧率。
                        fw = frame.shape[1]  # 帧宽。
                        (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)  # 文本宽度。
                        cv2.putText(frame, fps_text, (fw - text_w - 8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                    if tick % 30 == 1 and not preparing:  # 诊断：每秒报告标题模板多尺度最高分。
                        if locate_scale < 1.0:  # 缩帧坐标系：诊断也在缩帧上算。
                            diag_frame = cv2.resize(frame, None, fx=locate_scale, fy=locate_scale,
                                                    interpolation=cv2.INTER_AREA)
                        else:
                            diag_frame = frame
                        score = _template_best_score(diag_frame, region_finder.title_template)
                        self.log_message.emit(
                            f"tick {tick}: title match score {score:.3f} (threshold {TEMPLATE_THRESHOLD}) "
                            f"标题模板最高分 {score:.3f}（阈值 {TEMPLATE_THRESHOLD}）")
                    self._push_frame(frame)
                else:  # 弹窗已定位：推进求解。
                    if not found_dialog:  # 首次定位到弹窗。
                        found_dialog = True
                        if not crop_mode:
                            self.log_message.emit(f"dialog located at tick {tick}, region={region} 定位到弹窗区域")
                    rx, ry, rw, rh = region
                    # 区域首次出现或位移超阈值时 reset 状态机（清空光流历史、模板、跟踪器）。
                    if last_region is None or abs(rw - last_region[2]) > 4 or abs(rh - last_region[3]) > 4:
                        session.reset(rw, rh)  # 区域尺寸变化：必须清空全部状态。
                    elif abs(rx - last_region[0]) > 4 or abs(ry - last_region[1]) > 4:
                        session.reset(rw, rh)  # 区域位移：也必须清空，否则坐标偏移。
                    last_region = region
                    crop = frame[ry:ry + rh, rx:rx + rw]  # 裁出图形区域。
                    t_track = time.perf_counter()  # 跟踪计时起点。
                    result = session.update(crop, src_fps)  # 喂入跟踪会话（时间阈值按源帧率换算）。
                    track_ms = (time.perf_counter() - t_track) * 1000  # 粒子跟踪耗时。
                    if not algo_emitted:  # 首次发出算法就绪信号。
                        algo_emitted = True
                        self.algorithm_ready.emit(
                            f"DIS光流+粒子滤波 P={params.particle_count} S={params.process_scale}")
                    if result.source in (SOURCE_COLOR, SOURCE_BORDER, SOURCE_INTERPOLATED):  # 有效跟踪。
                        last_target_tick = tick
                    if tick % 30 == 1:  # 诊断：每秒报告一次流水线状态与分段耗时。
                        self.log_message.emit(
                            f"tick {tick}: source={result.source} conf={result.confidence:.2f} "
                            f"snr={result.border_snr:.2f} radius={result.search_radius:.1f} "
                            f"flow={result.flow_residual} whites={result.white_candidates} "
                            f"src={src_fps:.1f}fps | decode={decode_ms:.0f}ms locate={locate_ms:.0f}ms track={track_ms:.0f}ms")
                    draw_shape_overlay(frame, region, result, tick, "", timeout_ticks, fps_text, mode_tag)  # 叠加绘制。
                    self._push_frame(frame)
                now = time.perf_counter()  # 本帧处理完毕。
                if now < next_deadline:  # 处理快于节拍：休眠到点，保证视频按原帧率播放。
                    time.sleep(next_deadline - now)
        except Exception as exc:  # 流水线异常不拖垮 UI，记录后按失败收尾。
            self.log_message.emit(f"pipeline error 流水线异常: {exc}")
            result_msg = f"pipeline error: {exc}"
        finally:
            cap.release()
        if self._stopped.is_set():  # 用户手动停止。
            result_msg = "stopped by user 用户手动停止"
            user_stopped = True
        # 收尾日志：source 分布与实测帧率（对齐外部脚本 SHAPE done: {...}）；session 未建时降级为空分布。
        source_counts = session.source_counts if session is not None else {}
        self.log_message.emit(
            f"source counts: {source_counts} measured fps≈{fps_value:.1f} 实测帧率")
        if self._stopped.is_set() and not user_stopped:
            result_msg = "stopped by user 用户手动停止"
        self.finished_result.emit(result_ok, result_msg)


class LieDetectorTab(CustomTab):  # 测谎检验页签：上传录像验证谎言检测器求解流水线。

    def __init__(self):
        super().__init__()
        self.icon = FluentIcon.LIBRARY
        self.worker = None
        self.video_path = ""

        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        self.pick_button = PushButton(FluentIcon.FOLDER, "Select Video 选择视频")
        self.pick_button.clicked.connect(self.pick_video)
        self.path_label = BodyLabel("No video selected 未选择视频")
        self.path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.start_button = PushButton(FluentIcon.PLAY, "Start 开始")
        self.start_button.clicked.connect(self.start)
        self.stop_button = PushButton(FluentIcon.PAUSE, "Stop 停止")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.fps_label = BodyLabel("Mode 模式: full-speed 全速不限速")
        self.algorithm_label = BodyLabel("Algorithm 算法: --")
        layout.addWidget(self.pick_button)
        layout.addWidget(self.path_label, 1)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.algorithm_label)
        self.add_card("Video & Run 视频与运行", control)

        if not INFERENCE_AVAILABLE:  # 依赖缺失时禁用运行并提示安装命令。
            self.start_button.setEnabled(False)
            self.path_label.setText("OpenCV DIS optical flow not available 缺少 DIS 稠密光流，请升级 opencv-python")

        self.image_label = VisionLabel()
        self.add_card("Process Vision 运算过程画面", self.image_label, stretch=1)

        self.log_edit = TextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(160)
        self.add_card("Run Log 运行日志", self.log_edit)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)  # 按 30FPS 节拍拉取标注帧显示。

    @property
    def name(self):
        return "LieDetector"

    def append_log(self, message):  # 追加一行运行日志。
        self.log_edit.append(message)

    def pick_video(self):  # 弹出文件对话框选择录像。
        path, _ = QFileDialog.getOpenFileName(
            self, "Select lie detector video 选择谎言检测器录像", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:
            self.video_path = path
            self.path_label.setText(os.path.basename(path))
            self.append_log(f"video selected 已选择视频: {path}")

    def start(self):  # 启动工作线程回放并验证。
        if not self.video_path or not Path(self.video_path).exists():
            self.append_log("please select a video first 请先选择视频")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.log_edit.clear()
        self.worker = LieDetectorWorker(self.video_path, self)
        self.worker.log_message.connect(self.append_log)
        self.worker.algorithm_ready.connect(self.on_algorithm_ready)
        self.worker.finished_result.connect(self.on_finished)
        self.start_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.append_log(f"start verifying at full speed 开始全速验证（画面右上角显示实测 FPS）")
        self.worker.start()

    def stop(self):  # 请求停止工作线程。
        if self.worker is not None and self.worker.isRunning():
            self.append_log("stopping... 正在停止")
            self.worker.stop()

    def on_algorithm_ready(self, summary):  # 算法就绪后更新徽标与日志。
        self.algorithm_label.setText(f"算法 Algorithm: DIS光流+粒子滤波")
        self.append_log(f"algorithm ready 算法就绪: {summary}")

    def on_finished(self, ok, message):  # 工作线程结束：恢复按钮并输出结论。
        self.start_button.setEnabled(INFERENCE_AVAILABLE)
        self.pick_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        verdict = "PASSED 通过" if ok else "NOT PASSED 未通过"
        self.append_log(f"result 结果: {verdict} - {message}")

    def refresh(self):  # 定时从工作线程队列取最新标注帧显示。
        if self.worker is None:
            return
        frame = None
        try:
            while True:  # 丢弃旧帧只保留最新一帧。
                frame = self.worker.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if frame is None:
            return
        self.image_label.set_frame(frame)  # 直接把 BGR 画面矩阵交给标签，由它按当前尺寸预缩放后转图片显示。

    def closeEvent(self, event):  # 页签关闭时确保工作线程退出。
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)
