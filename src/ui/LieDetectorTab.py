# 测谎检验页签：用户上传谎言检测器录像，按视频原帧率实时播放（处理跟不上时丢帧保实时，贴近真实采集场景），
# 用稠密光流对齐历史帧 + 粒子滤波跟踪透明轮廓（有 N 卡走 torch CUDA，否则走 CPU；无神经网络），
# 画面叠加轮廓、状态与实测/源帧率，视频模式不发送鼠标，仅验证算法效果。
import math
import os
from pathlib import Path
import queue
import threading
import time

import cv2  # 导入 OpenCV，用于读视频与叠加绘制。
import numpy as np  # 导入 NumPy，用于画面矩阵。
from PySide6.QtCore import Qt, QThread, QTimer, Signal  # 导入 Qt 线程、定时器与信号。
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QWidget  # 导入布局与控件。
from qfluentwidgets import BodyLabel, ComboBox, FlowLayout, FluentIcon, PushButton, TextEdit  # 导入 Fluent 控件。

from ok import og  # 导入全局对象：改精度/延迟后通知独立测谎服务热更新配置。
from ok.gui.widget.CustomTab import CustomTab  # 导入自定义页签基类。
from src.ui.DashboardTab import (LIE_PRECISION_GPU_ONLY, LIE_PRECISION_TIERS, LIE_TIER_LABELS,  # 复用看板页签的画面标签与精度档定义（单一数据源，避免两处不一致）。
                                 VisionLabel)
from src.ui.spin_wheel_guard import DoubleSpinBox  # 触发延迟数字框用滚轮守卫子类：需点击聚焦后滚轮才生效，避免滚动页面误改数值。
from src.dashboard_store import load_dashboard_config, save_dashboard_config  # 读写看板配置：验证页签与线上服务同档复算，改精度/延迟合并写回 Dashboard.json。
from src.liedetector.gpu_shape_backend import gpu_backend_available  # 探测 torch CUDA+N 卡可用性：决定验证页签精度下拉可选档位与运算后端显示。

TICK_FPS_FALLBACK = 30  # 源帧率缺失时的回退帧率，与参考项目 FPS=30 一致。
TIMEOUT_TICKS = 545  # 算法超时预算下限 545 tick@30fps（约 18 秒），与参考项目 solve_shape.rs 一致；实际 timeout_ticks 取此预算与视频总帧数的较大者（回放整段录像）。
LOCATE_MAX_SIDE = 400  # 全屏弹窗定位的分辨率上限（最长边像素）：超过则先缩帧再匹配，坐标换算回原图。
ENGINE_TAGS = {"torch-farneback": "torch CUDA", "cv2-dis": "CPU DIS", "cv2-farneback": "CPU Farneback"}  # 光流引擎名 -> 徽标短文案（未知引擎直接显原名）。
LIE_OUTCOME_LABELS = {"success": "成功", "failure": "失败", "solved": "已解", "gone": "触发消失",  # 录像边车 outcome -> 中文结果（历史下拉摘要末尾展示，成功/失败来自解测谎结算）。
                      "timeout": "超时", "abandoned": "放弃", "aborted": "急停"}

try:  # 探测 OpenCV 是否包含 DIS 稠密光流，缺失时页签降级为不可运行。
    INFERENCE_AVAILABLE = hasattr(cv2, 'DISOpticalFlow_create')
except Exception:
    INFERENCE_AVAILABLE = False

from src.liedetector.detector import (  # 仅保留多尺度模板定位能力（弹窗标题定位），不再需要 YOLO 检测器。
    TEMPLATE_THRESHOLD, LieDetectorRegion, MultiScaleTemplate, _template_best_score)
from src.liedetector.shape_session import (  # 在线编排。
    ShapeTrackParams, ShapeTrackSession, PRECISION_TIER_DEFAULT, PRECISION_TIER_KEYS,
    SOURCE_BORDER, SOURCE_COLOR, SOURCE_INTERPOLATED, SOURCE_PREDICTION, SOURCE_SCENE_ENDED)
from src.liedetector.recorder import LIE_RECORD_DIR, delete_record, list_records  # 测谎录像历史：列举 lie_records 边车记录供下拉复算，并支持删除选中记录。
from src.liedetector.feed import decode_trace, iter_replay_plan, should_reset  # 逐帧喂入层：回放“仿实时”按实时喂帧日程重喂同一帧子集/同一 dt（与实时同源）；should_reset 为两路共用的区域重置判定。


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


class LieDetectorWorker(QThread):  # 工作线程：在线分析 + 实时预览，按视频原帧率播放（处理超时丢帧保实时）。

    log_message = Signal(str)  # 日志消息信号。
    algorithm_ready = Signal(str, str)  # 算法准备就绪信号：(徽标文案, 日志参数摘要)。
    finished_result = Signal(bool, str)  # 结束信号：(是否通过, 结果描述)。

    def __init__(self, video_path, tier="high", parent=None, delay=0.0, mode="ceiling", sidecar=None):
        super().__init__(parent)
        self.video_path = video_path  # 录像路径。
        self.tier = tier  # 复算精度档：由页签精度下拉传入，与线上服务同档验证。
        self.delay = max(0.0, float(delay))  # 触发延迟秒数：定位到弹窗后先等待再解题，与线上服务一致；<=0 表示不延迟。
        self.mode = str(mode or "ceiling")  # 复算模式：'ceiling'=算法上限逐帧（现有）；'live'=按边车 feed_trace 日程仿实时复现。
        self.sidecar = sidecar or {}  # 选中录像的完整边车 dict（含 feed_trace/tier/fps），仅 'live' 模式使用。
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

    def run(self):  # 线程主循环：按源帧率逐帧分析 + 实时预览（处理超时丢帧保实时）。
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
        budget_ticks = max(1, int(round(TIMEOUT_TICKS / TICK_FPS_FALLBACK * src_fps)))  # 算法超时预算按源帧率折算，约 18 秒。
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 视频总帧数：录像按真实时长补齐后可能长于 18 秒预算。
        # 验证需回放整段录像：取「算法预算」与「总帧数+1 秒余量」的较大者，避免长录像（含触发延迟/慢采集补齐）在片尾判定前被判超时。
        timeout_ticks = max(budget_ticks, frame_count + int(round(src_fps))) if frame_count > 0 else budget_ticks
        self.log_message.emit(
            f"video {frame_w}x{frame_h} opened 视频已打开 src_fps={src_fps:.1f} 源帧率 frames={frame_count} timeout={timeout_ticks} ticks")
        if self.mode == "live":  # 仿实时模式：不逐帧、不按原速播放，而是严格按边车 feed_trace 复现实时的喂帧子集/dt/区域重置（与实时同口径）。
            trace = decode_trace(self.sidecar.get("feed_trace"))
            if not trace:  # 旧记录无喂帧日程：无法仿实时，告警并自动回落逐帧上限模式。
                self.log_message.emit("no feed_trace in sidecar, fallback to ceiling 边车无喂帧日程，自动回落“算法上限(逐帧)”复算")
            else:
                self._run_live_replay(cap, frame_w, frame_h, src_fps, trace)
                return
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
        tier = str(self.tier or PRECISION_TIER_DEFAULT).strip()  # 用页签精度下拉传入的档位复算，与线上服务同档验证。
        if tier not in PRECISION_TIER_KEYS:  # 非法档位（防御性校验）。
            tier = PRECISION_TIER_DEFAULT
        params = ShapeTrackParams(precision_tier=tier)  # 按精度档装配，与线上解测谎同档复算（无 N 卡时高/极高/最强会在会话构造时回落 medium）。
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
        delay_active = False  # 触发延迟等待中：定位到弹窗后先按原速播放 self.delay 秒再解题。
        delay_done = False  # 触发延迟是否已结束（避免区域抖动重置时重复计时）。
        delay_deadline = 0.0  # 触发延迟结束的绝对时间点。

        self.log_message.emit(  # 打印实际生效的参数，便于与外部脚本回归对照。
            f"algorithm params: tier={params.precision_tier} backend={session.backend.name} "
            f"engine={session.aligner.engine_name} scale={params.process_scale} lags={params.temporal_lags} "
            f"particles={params.particle_count} proposals={params.global_proposals} "
            f"flow(levels={params.flow_levels},iterations={params.flow_iterations},winsize={params.flow_winsize})")

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
                else:  # 弹窗已定位：推进求解（触发延迟未到则先按原速等待，模拟线上服务延迟解题）。
                    if not found_dialog:  # 首次定位到弹窗。
                        found_dialog = True
                        if not crop_mode:
                            self.log_message.emit(f"dialog located at tick {tick}, region={region} 定位到弹窗区域")
                        if self.delay > 0 and not delay_done:  # 配置了触发延迟：从定位到弹窗这一刻起计时等待。
                            delay_deadline = time.perf_counter() + self.delay  # 延迟结束时间点。
                            delay_active = True  # 进入延迟等待。
                            self.log_message.emit(
                                f"trigger delay {self.delay:.1f}s, wait before solving 触发延迟 {self.delay:.1f} 秒后再开始解测谎")
                    if delay_active:  # 延迟等待期：只按原速播放预览，不喂求解器。
                        if time.perf_counter() >= delay_deadline:  # 延迟已到，开始解题。
                            delay_active = False
                            delay_done = True
                            self.log_message.emit("trigger delay elapsed, start solving 触发延迟结束，开始解测谎")
                        else:  # 仍在延迟：叠加倒计时状态并跳过本帧求解。
                            remaining = delay_deadline - time.perf_counter()  # 剩余等待秒数。
                            status = f"delaying {math.ceil(remaining)}s"  # 倒计时按整秒显示。
                            cv2.putText(frame, f"tick {tick}/{timeout_ticks} [{mode_tag}] {status}", (8, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                            if fps_text:  # 延迟期也显示实测/源帧率。
                                fw = frame.shape[1]  # 帧宽。
                                (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)  # 文本宽度。
                                cv2.putText(frame, fps_text, (fw - text_w - 8, 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                            self._push_frame(frame)  # 推送延迟期预览帧。
                            now = time.perf_counter()  # 延迟期同样按原帧率节拍播放。
                            if now < next_deadline:
                                time.sleep(next_deadline - now)
                            continue  # 本帧不解题，读下一帧。
                    rx, ry, rw, rh = region
                    # 区域首次出现或位移/尺寸超阈值时 reset 状态机（清空光流历史、模板、跟踪器）——与实时解题共用 feed.should_reset，杜绝两份手写漂移。
                    if should_reset(last_region, region, 4):  # 首帧或未记录过区域、或任一边变化 >4px 才重置（new_region 为空不重置，但此处 region 已有效）。
                        session.reset(rw, rh)  # 重置光流会话：以新尺寸重建，避免上一段历史污染。
                    last_region = region
                    crop = frame[ry:ry + rh, rx:rx + rw]  # 裁出图形区域。
                    t_track = time.perf_counter()  # 跟踪计时起点。
                    result = session.update(crop, src_fps)  # 喂入跟踪会话（时间阈值按源帧率换算）。
                    track_ms = (time.perf_counter() - t_track) * 1000  # 粒子跟踪耗时。
                    if not algo_emitted:  # 首次发出算法就绪信号。
                        algo_emitted = True
                        engine = session.aligner.engine_name  # 会话实际生效的光流引擎（运行期降级后也会跟着变）。
                        tag = ENGINE_TAGS.get(engine, engine)  # 引擎名映射成短文案。
                        self.algorithm_ready.emit(
                            f"光流+粒子滤波 ({tag})",
                            f"{tag}光流+粒子滤波 P={params.particle_count} S={params.process_scale} "
                            f"engine={engine} flow(levels={params.flow_levels},iterations={params.flow_iterations},winsize={params.flow_winsize})")
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

    def _run_live_replay(self, cap, frame_w, frame_h, src_fps, trace):  # 仿实时回放：严格按实时喂帧日程重放——跳到日程指定的 mp4 帧、用该步真实 dt 换算的 fps 喂入、按记录的区域 reset，复现实时的掉帧与时间阈值行为。
        try:  # 任何异常不拖垮 UI，按失败收尾；cap 由 run() 的 finally 统一释放（此处只读到结束）。
            try:  # 起点对齐偏移：录像在触发确认即起录、解题要再等报警+触发延迟才开题，不补这段会在「弹窗未开题」的画面上喂帧、全程错位跑不出轨迹。
                start_frame = max(0, int(self.sidecar.get("feed_start_frame") or 0))
            except (TypeError, ValueError):
                start_frame = 0
            if "feed_start_frame" not in self.sidecar:  # 旧记录无该字段：只能从第 0 帧起喂（含提前段），告警提示结果可能偏差。
                self.log_message.emit("no feed_start_frame in sidecar 边车无首喂帧偏移（旧记录），仿实时从第 0 帧起喂，时间线可能含起录提前段")
            tier = str(self.sidecar.get("tier") or self.tier or PRECISION_TIER_DEFAULT).strip()  # 精度档取边车 tier（与实时同档，忠实复现），缺失回退页签档/默认档。
            if tier not in PRECISION_TIER_KEYS:  # 非法档位防御。
                tier = PRECISION_TIER_DEFAULT
            params = ShapeTrackParams(precision_tier=tier)  # 与线上解测谎同档装配会话。
            logger_adapter = _EmitLogger(self.log_message.emit)  # 日志适配器。
            session = ShapeTrackSession(params=params, logger=logger_adapter)  # 光流粒子滤波在线会话。
            session.reset(frame_w, frame_h)  # 先以整帧尺寸建会话，保证首步未带 reset 也能安全 update（reset 口径为 (w, h)）。
            self.algorithm_ready.emit(
                f"光流+粒子滤波（仿实时 {ENGINE_TAGS.get(session.aligner.engine_name, session.aligner.engine_name)}）",
                f"仿实时复现 tier={params.precision_tier} 日程={len(trace)} 步 src_fps={src_fps:.1f} 起点帧={start_frame}")
            tick = 0  # 已喂入的日程步数。
            last_target_tick = 0  # 最近一次有效跟踪的步号。
            fps_sum = 0.0  # 各步有效帧率累加，收尾求均值（应与实时 diag 的 fps 口径一致）。
            last_index = start_frame - 1  # 最近一步消费的 mp4 帧序号（从起点前算起），用于估算“复现丢弃帧数”。
            last_rect = None  # 最近一次有效子区域：实时 rect=None 的语义是「未采到坐标框、沿用旧区域」，回放同样沿用，绝不退整帧（否则裁取尺寸突变、跟踪器发散）。
            planned = len(trace)  # 计划喂入的日程步数（= 实时真正喂入 session 的帧数）。
            result = None  # 最近一帧跟踪结果（供循环外兜底引用）。
            for step in iter_replay_plan(trace, src_fps, start_frame):  # 按日程逐步驱动：起点=首喂帧偏移，游标按 dt 跳帧复现实时掉帧。
                if self._stopped.is_set():  # 用户手动停止。
                    break
                if step.frame_index > last_index:  # 只在前进到新帧时 seek（避免同一帧重复 seek，日程允许步内不前进的极端情形）。
                    cap.set(cv2.CAP_PROP_POS_FRAMES, step.frame_index)  # 跳到该步对应的 mp4 帧。
                    last_index = step.frame_index
                ok, frame = cap.read()  # 读该帧。
                if not ok:  # 越界或播完。
                    break
                tick += 1
                rx, ry = 0, 0  # 子区域左上角（mp4 帧内绝对坐标：step.rect 已相对裁剪原点=mp4 原点）。
                rw, rh = frame_w, frame_h  # 默认整帧（仅当从未有过 rect 的极端旧记录才用到）。
                if step.rect is not None:  # 该步记录了相对子区域：更新沿用基准并按它复切夹到帧内。
                    last_rect = step.rect
                if last_rect is not None:  # 沿用最近有效区域（含 rect=None 的「未采到坐标框」步，与实时语义一致）。
                    x, y, w, h = last_rect
                    rx = max(0, min(int(x), frame_w - 1))
                    ry = max(0, min(int(y), frame_h - 1))
                    rw = max(1, min(int(w), frame_w - rx))
                    rh = max(1, min(int(h), frame_h - ry))
                crop = frame[ry:ry + rh, rx:rx + rw]  # 裁出图形区域。
                if step.reset:  # 该步实时做过区域重置：复现 session.reset 清空历史。
                    session.reset(crop.shape[1], crop.shape[0])  # reset 口径 (w, h)。
                result = session.update(crop, step.fps)  # 用该步真实 dt 换算的 fps 喂入（与实时同一口径函数）。
                fps_sum += step.fps  # 累计有效帧率。
                if result.source in (SOURCE_COLOR, SOURCE_BORDER, SOURCE_INTERPOLATED):  # 有效跟踪。
                    last_target_tick = tick
                if tick % 30 == 1:  # 诊断：每约 30 步报一次，与实时逐帧口径对照。
                    self.log_message.emit(
                        f"live-replay tick {tick}/{planned}: source={result.source} conf={result.confidence:.2f} "
                        f"fps={step.fps:.1f} reset={step.reset} rect=({rx},{ry},{rw},{rh})")
                draw_shape_overlay(frame, (rx, ry, rw, rh), result, tick, "", planned,
                                   f"LIVE {fps_sum / tick:.0f}/{src_fps:.0f}FPS" if tick else "", "LIVE")  # 叠加绘制并推送预览。
                self._push_frame(frame)
            spanned = max(0, last_index + 1)  # 日程在 mp4 上跨越的总帧数（含起录→开题提前段）。
            dropped = max(0, spanned - planned - start_frame)  # 复现的“实时因处理慢而丢弃的中间帧”估计（扣除提前段，只算解题窗口内）。
            avg_fps = (fps_sum / tick) if tick else 0.0  # 平均有效帧率。
            self.log_message.emit(
                f"live-replay done 仿实时结束：计划喂入 {planned} 帧/实际喂入 {tick} 帧、跨越 {spanned} 帧、复现丢弃约 {dropped} 帧、平均有效 fps={avg_fps:.1f}")
            source_counts = session.source_counts if session is not None else {}  # source 分布。
            self.log_message.emit(f"source counts: {source_counts} 仿实时 source 分布")
            result_ok = last_target_tick > 0  # 只要全程有过有效跟踪即判“可复现跟踪”（与 crop 上限模式判定口径一致）。
            if self._stopped.is_set():
                self.finished_result.emit(False, "stopped by user 用户手动停止（仿实时）")
            else:
                self.finished_result.emit(result_ok, f"live-replay: tracked {last_target_tick}/{planned} 仿实时跟踪至第 {last_target_tick} 步")
        except Exception as exc:  # 流水线异常按失败收尾，不静默死亡。
            self.log_message.emit(f"live-replay error 仿实时异常: {exc}")
            self.finished_result.emit(False, f"live-replay error: {exc}")


class LieDetectorTab(CustomTab):  # 测谎检验页签：上传录像验证谎言检测器求解流水线。

    def __init__(self):
        super().__init__()
        self.icon = FluentIcon.LIBRARY
        self.worker = None
        self.video_path = ""
        self._selected_record = None  # 当前选中历史录像的完整边车 dict（含 feed_trace/tier/fps），仿实时模式据此复现；手选视频时为 None。
        self._history_by_path = {}  # 路径 -> 边车 dict 映射，供 _on_history_selected 取回完整记录（下拉 itemData 仍只存路径）。

        control = QWidget()
        layout = FlowLayout(control, needAni=False)  # 自适应流式布局：控件按可用宽度自动换行，不再全挤在一行。
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)  # 同行控件的水平间距。
        layout.setVerticalSpacing(10)  # 换行后行与行的垂直间距。
        self.pick_button = PushButton(FluentIcon.FOLDER, "选择视频")
        self.pick_button.clicked.connect(self.pick_video)
        self.history_combo = ComboBox()  # 历史录像下拉：列出 lie_records 里的触发录像，选中即可复算验证。
        self.history_combo.setMinimumWidth(220)  # 保证摘要（时间/分/档/结果）可见。
        self.history_combo.activated.connect(self._on_history_selected)  # activated 仅用户点选时触发，程序重建不误触发。
        self.refresh_history_button = PushButton(FluentIcon.SYNC, "刷新")  # 手动刷新历史下拉。
        self.refresh_history_button.clicked.connect(self._reload_history)
        self.delete_history_button = PushButton(FluentIcon.DELETE, "删除")  # 删除当前选中的历史录像（mp4+json），弹确认框防误删。
        self.delete_history_button.clicked.connect(self._delete_selected_history)
        self.precision_label = BodyLabel("精度")  # 精度下拉标签。
        self.precision_combo = ComboBox()  # 复算精度档下拉（低/中等/高/极高，GPU 门控），与看板 Dashboard.json 同键，改动即合并写回并通知服务。
        self.precision_combo.setMinimumWidth(90)  # 保证档位中文可见。
        self.precision_combo.currentIndexChanged.connect(self._persist_lie_settings)  # 用户改档即持久化（加载期由 _loading 守卫屏蔽）。
        self.mode_label = BodyLabel("复算模式")  # 复算模式下拉标签。
        self.mode_combo = ComboBox()  # 复算模式：算法上限(逐帧) / 仿实时(按录像日程)；后者需选中带 feed_trace 的录像。
        self.mode_combo.setMinimumWidth(150)  # 保证模式中文可见。
        self.mode_combo.addItem("算法上限(逐帧)", None, "ceiling")  # 默认：逐帧、按 src_fps 口径，作为算法上限参照。
        self.mode_combo.addItem("仿实时(按录像日程)", None, "live")  # 按选中录像边车的 feed_trace 复现实时喂帧节奏。
        self.backend_label = BodyLabel("运算后端: --")  # 只读显示当前测谎打分后端（GPU/CPU），按显卡可用性刷新。
        self.delay_label = BodyLabel("触发延迟")  # 触发延迟标签。
        self.delay_spin = DoubleSpinBox()  # 触发延迟秒数，与看板 Dashboard.json 同键，改动即合并写回并通知服务。
        self.delay_spin.setRange(0.0, 60.0)  # 延迟范围 0~60 秒，0 表示不延迟立即解题。
        self.delay_spin.setSingleStep(0.5)  # 步长。
        self.delay_spin.setDecimals(1)  # 保留一位小数。
        self.delay_spin.setSuffix(" s")  # 单位后缀，一眼看出是秒数。
        self.delay_spin.valueChanged.connect(self._persist_lie_settings)  # 用户改延迟即持久化（加载期由 _loading 守卫屏蔽）。
        self.path_label = BodyLabel("未选择视频")
        self.path_label.setMinimumWidth(180)  # 流式布局不做拉伸，给个最小宽保证所选视频名可读。
        self.start_button = PushButton(FluentIcon.PLAY, "开始")
        self.start_button.clicked.connect(self.start)
        self.stop_button = PushButton(FluentIcon.PAUSE, "停止")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.fps_label = BodyLabel("模式: 原速播放")
        self.algorithm_label = BodyLabel("算法: --")
        # 成对的「标签+控件」与「开始/停止」各自包进小容器，作为整体参与流式换行，避免标签与其控件被拆到两行。
        layout.addWidget(self.pick_button)
        layout.addWidget(self._flow_group(self.history_combo, self.refresh_history_button, self.delete_history_button))
        layout.addWidget(self._flow_group(self.precision_label, self.precision_combo))
        layout.addWidget(self._flow_group(self.mode_label, self.mode_combo))  # 复算模式（算法上限/仿实时）与精度档并列。
        layout.addWidget(self.backend_label)
        layout.addWidget(self._flow_group(self.delay_label, self.delay_spin))
        layout.addWidget(self._flow_group(self.start_button, self.stop_button))
        layout.addWidget(self.path_label)
        layout.addWidget(self._flow_group(self.fps_label, self.algorithm_label))
        self.add_card("视频与运行", control)

        if not INFERENCE_AVAILABLE:  # 依赖缺失时禁用运行并提示安装命令。
            self.start_button.setEnabled(False)
            self.path_label.setText("缺少 DIS 稠密光流，请升级 opencv-python")

        self.image_label = VisionLabel()
        self.add_card("运算过程画面", self.image_label, stretch=1)

        self.log_edit = TextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(160)
        self.add_card("运行日志", self.log_edit)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)  # 按 30FPS 节拍拉取标注帧显示。
        self._loading = False  # 加载守卫：程序回填控件值期间置 True，避免 _persist_lie_settings 被自身触发回写。
        self._load_lie_settings()  # 从 Dashboard.json 回填精度下拉与触发延迟（含 GPU 门控刷新）。
        self._reload_history()  # 初始填充历史录像下拉（页签每次显示时也会刷新）。
        self._center_status_labels()  # 只读标签与控件等高，流式行内垂直居中（首显示时按真实控件高度再校正一次）。

    @property
    def name(self):
        return "LieDetector"

    def append_log(self, message):  # 追加一行运行日志。
        self.log_edit.append(message)

    def showEvent(self, event):  # 页签每次显示时刷新精度/延迟与历史下拉，让看板改动与刚录好的触发录像即时同步。
        super().showEvent(event)  # 先走父类显示逻辑。
        self._load_lie_settings()  # 从 Dashboard.json 回填精度与延迟（看板页签可能已改动，切回来时同步）。
        self._reload_history()  # 重建历史下拉。
        self._center_status_labels()  # 按显示后的真实控件高度校正只读标签高度，保证流式行内垂直居中。

    def _reload_history(self):  # 用 list_records 重建历史下拉：首项占位，其余每条录像一项（摘要 label，data=mp4 路径）。
        current = self.video_path  # 记录当前视频路径，重建后尽量保持选中。
        records = list_records()  # 扫描 lie_records 边车，按时间倒序（新->旧）。
        self._history_by_path = {rec.get("path"): rec for rec in records}  # 缓存路径->完整边车，供仿实时取 feed_trace/tier/fps。
        self.history_combo.blockSignals(True)  # 重建期间屏蔽信号（activated 本就只在用户点选时发，双保险）。
        self.history_combo.clear()  # 清空旧项。
        self.history_combo.addItem("历史记录", None, None)  # 首项占位，data=None 表示未选具体录像。
        for rec in records:  # 逐条录像填一项。
            self.history_combo.addItem(self._format_history_label(rec), None, rec.get("path"))
        self.history_combo.blockSignals(False)  # 恢复信号。
        index = self.history_combo.findData(current) if current else -1  # 定位当前视频对应项。
        self.history_combo.setCurrentIndex(index if index >= 0 else 0)  # 命中则选中，否则回占位首项。
        self._selected_record = self._history_by_path.get(current) if index and index >= 0 else None  # 重建后同步选中记录（供仿实时）；回占位项则清空。

    def _format_history_label(self, rec):  # 把一条记录格式化成下拉摘要：MM-DD HH:MM 分X.XX 档 结果。
        ts = str(rec.get("timestamp") or "")  # ISO 起录时间戳。
        short = ts  # 默认原样，解析失败也不至于空。
        if "T" in ts:  # "2026-09-28T18:39:00" -> "09-28 18:39"。
            date_part, time_part = ts.split("T", 1)  # 拆日期与时间。
            short = f"{date_part[5:]} {time_part[:5]}"  # MM-DD HH:MM。
        try:  # 触发分保留两位小数。
            score_txt = f"{float(rec.get('score')):.2f}"
        except (TypeError, ValueError):  # 分数缺失或非法。
            score_txt = "--"
        tier = LIE_TIER_LABELS.get(str(rec.get("tier") or ""), str(rec.get("tier") or "--"))  # 精度档中文。
        outcome = str(rec.get("outcome") or "")  # 结束原因/结果（success/failure/solved/timeout/gone/abandoned/aborted）。
        result_txt = LIE_OUTCOME_LABELS.get(outcome, outcome or "--")  # 映射为中文结果，未知值原样显，空值显 --。
        return f"{short} 分{score_txt} {tier} {result_txt}"  # 拼接摘要：结果固定放在每条记录的最后。

    def _on_history_selected(self, index):  # 用户从历史下拉选中一条录像：设为当前视频，供「开始」复算验证。
        path = self.history_combo.itemData(index)  # 取该项的 mp4 路径。
        if not path:  # 占位项或无路径，忽略。
            return
        self.video_path = path  # 设为当前待验证视频。
        self._selected_record = self._history_by_path.get(path)  # 同步选中记录的完整边车（含 feed_trace），供仿实时复现。
        self.path_label.setText(os.path.basename(path))  # 显示文件名。
        self.append_log(f"history selected 已选择历史录像: {path}")  # 记日志。

    def _delete_selected_history(self):  # 删除历史下拉当前选中的录像（mp4+json）：弹确认框防误删，确认后删除并刷新下拉。
        path = self.history_combo.itemData(self.history_combo.currentIndex())  # 取当前选中项的 mp4 路径。
        if not path:  # 占位首项或无路径：无可删。
            self.append_log("no record selected 未选择要删除的历史记录")  # 记日志提示。
            return
        from qfluentwidgets import Dialog  # 延迟导入确认对话框（与 DashboardTaskPanel 一致）。
        dialog = Dialog("删除记录", f"确定删除该测谎录像记录？\n{os.path.basename(path)}", self.window())  # 构造确认框。
        dialog.yesButton.setText("删除")  # 确认按钮文案。
        dialog.cancelButton.setText("取消")  # 取消按钮文案。
        if not dialog.exec():  # 用户取消：不删。
            return
        removed = delete_record(path)  # 成对删除 mp4+json，返回是否删掉 mp4。
        if self.video_path == path:  # 删的是当前待验证视频：清空选择并复位路径标签。
            self.video_path = ""
            self.path_label.setText("未选择视频")
        self._reload_history()  # 重建历史下拉（删后该项消失）。
        self.append_log((f"deleted 已删除历史录像: {path}" if removed else f"delete failed 删除失败: {path}"))  # 记结果日志。

    def _refresh_precision_options(self):  # 按显卡可用性刷新精度下拉可选档与运算后端显示：GPU 五档齐全，CPU 移除高/极高/最强并回落中等。
        try:  # 探测异常（驱动问题等）按无显卡处理，不能拖垮验证页签加载。
            gpu = bool(gpu_backend_available())
        except Exception:  # 探测本身报错。
            gpu = False
        self.backend_label.setText("运算后端: GPU (torch CUDA)" if gpu else "运算后端: CPU (numpy)")  # 后端显示项，最佳努力（运行期真实降级以会话构造 clamp 为准）。
        current = self.precision_combo.currentData()  # 记录当前选中档 key，重建后尽量保持。
        tiers = LIE_PRECISION_TIERS if gpu else tuple(t for t in LIE_PRECISION_TIERS if t[0] not in LIE_PRECISION_GPU_ONLY)  # CPU 只保留低/中等。
        self.precision_combo.blockSignals(True)  # 重建期间不触发信号（也避免误触发持久化）。
        self.precision_combo.clear()  # 清空旧项：qfluentwidgets ComboBox 非 QComboBox 子类，逐项 disable 不可靠，改用重建规避。
        for key, label in tiers:  # 按可用档重新填充。
            self.precision_combo.addItem(label, None, key)
        self.precision_combo.blockSignals(False)  # 恢复信号。
        self._set_precision_value(current or PRECISION_TIER_DEFAULT)  # 还原原选中档；被移除（CPU 下的重载档）时回落中等。

    def _set_precision_value(self, key):  # 按 key 选中精度档；key 不在可选档（如 CPU 下的高/极高/最强）时回落中等，再不行落首项。
        index = self.precision_combo.findData(key)  # 按 userData 定位目标档。
        if index < 0:  # 目标档不可用。
            index = self.precision_combo.findData('medium')  # 回落中等。
        if index < 0:  # 连中等也不在（理论上不会）。
            index = 0  # 落首项兜底。
        self.precision_combo.setCurrentIndex(index)  # 选中目标档。

    def _load_lie_settings(self):  # 从 Dashboard.json 回填精度下拉与触发延迟，加载期置守卫避免自我触发回写。
        self._loading = True  # 进入加载：屏蔽 _persist_lie_settings。
        try:  # 配置读取异常不能拖垮页签构造/显示。
            data = load_dashboard_config()  # 读看板配置（缺失键由其内部补默认）。
            self._refresh_precision_options()  # 先按显卡可用性重建可选档。
            self._set_precision_value(str(data.get('Lie Detector Precision') or PRECISION_TIER_DEFAULT))  # 选中配置精度档（CPU 下重载档自动回落中等）。
            self.delay_spin.setValue(float(data.get('Lie Detector Trigger Delay', 5.0)))  # 回填触发延迟秒数。
        except Exception:  # 读取/回填异常：保持控件默认值。
            pass
        finally:
            self._loading = False  # 退出加载：恢复持久化。

    def _persist_lie_settings(self, *args):  # 用户改精度/延迟即合并写回 Dashboard.json 并通知测谎服务热更新（加载期由守卫跳过）。
        if getattr(self, '_loading', False):  # 程序回填控件值触发的信号，不回写。
            return
        try:  # 合并写：先读全量配置，只改精度与延迟两键，避免覆盖看板其它字段。
            data = load_dashboard_config()  # 读当前全量配置。
            data['Lie Detector Precision'] = self.precision_combo.currentData() or PRECISION_TIER_DEFAULT  # 精度档。
            data['Lie Detector Trigger Delay'] = round(float(self.delay_spin.value()), 1)  # 触发延迟（一位小数，与看板一致）。
            save_dashboard_config(data)  # 全量落盘（save 内部原子写）。
            lie_service = getattr(og.my_app, 'lie_service', None) if og.my_app is not None else None  # 取独立测谎监控服务（由 Globals 持有）。
            if lie_service is not None:  # 服务已就绪时通知它立即重读配置。
                lie_service.reload_config()  # 测谎参数热更新，验证页签与线上服务同档。
            self.append_log(f"settings saved 精度={data['Lie Detector Precision']} 触发延迟={data['Lie Detector Trigger Delay']}s 已写回看板配置")  # 记日志。
        except Exception as exc:  # 写回失败不影响验证，仅记日志。
            self.append_log(f"settings save failed 配置写回失败: {exc}")

    def _flow_group(self, *widgets):  # 把若干控件横向包进一个小容器，供流式布局当作整体摆放（标签与其控件不被拆行）。
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)  # 组内控件紧凑些。
        for widget in widgets:
            row.addWidget(widget)
        return box

    def _center_status_labels(self):  # 让只读标签与相邻控件等高并垂直居中：FlowLayout 按各项 sizeHint 高度贴行顶摆放，矮标签会顶对齐。
        try:  # 控件高度尚未就绪时保持默认，不影响页签可用。
            ref_h = max(self.pick_button.sizeHint().height(),
                        self.precision_combo.sizeHint().height(),
                        self.delay_spin.sizeHint().height())  # 取相邻控件的最大高度作基准。
        except Exception:  # 读取高度异常。
            return
        if ref_h <= 0:  # 未取得有效高度。
            return
        for label in (self.backend_label, self.path_label, self.fps_label, self.algorithm_label):  # 全部只读标签统一处理。
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # 文本左对齐 + 垂直居中。
            label.setMinimumHeight(ref_h)  # 抬高到控件高度，配合流式行高即在行内垂直居中。

    def pick_video(self):  # 弹出文件对话框选择录像。
        default_dir = LIE_RECORD_DIR if os.path.isdir(LIE_RECORD_DIR) else ""  # 默认打开录像目录（存在时），方便直接选历史录像。
        path, _ = QFileDialog.getOpenFileName(
            self, "选择谎言检测器录像", default_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:
            self.video_path = path
            self._selected_record = self._history_by_path.get(path)  # 手选视频：能对上已有边车则取（可仿实时），否则 None（仿实时会自动回落逐帧）。
            self.path_label.setText(os.path.basename(path))
            self.append_log(f"video selected 已选择视频: {path}")

    def start(self):  # 启动工作线程回放并验证。
        if not self.video_path or not Path(self.video_path).exists():
            self.append_log("please select a video first 请先选择视频")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.log_edit.clear()
        tier = self.precision_combo.currentData() or 'high'  # 用页签精度下拉当前档复算，与线上服务同档验证。
        delay = float(self.delay_spin.value())  # 触发延迟：验证时同样在定位到弹窗后等待该秒数再解题，与线上服务一致。
        mode = self.mode_combo.currentData() or 'ceiling'  # 复算模式：ceiling=算法上限逐帧；live=按边车 feed_trace 仿实时。
        sidecar = self._selected_record or {}  # 选中录像的完整边车（含 feed_trace/tier/fps）；未选/无则空，仿实时会自动回落逐帧。
        self.worker = LieDetectorWorker(self.video_path, tier, self, delay=delay, mode=mode, sidecar=sidecar)  # delay 走关键字，保持 (path, tier, parent) 位置参数不变。
        self.worker.log_message.connect(self.append_log)
        self.worker.algorithm_ready.connect(self.on_algorithm_ready)
        self.worker.finished_result.connect(self.on_finished)
        self.start_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if mode == 'live':  # 仿实时：不逐帧、不按原速，严格复现实时喂帧日程；fps 取边车 tier（与实时同档）。
            self.append_log(f"start verifying in LIVE mode 开始仿实时复现（边车精度 {sidecar.get('tier') or tier}，日程步数 {len(sidecar.get('feed_trace') or [])}；无日程自动回落逐帧）")
        else:
            self.append_log(f"start verifying at source fps 开始按原速逐帧验证（算法上限，精度 {tier}，触发延迟 {delay:.1f}s，画面右上角显示实测/源 FPS）")
        self.worker.start()

    def stop(self):  # 请求停止工作线程。
        if self.worker is not None and self.worker.isRunning():
            self.append_log("stopping... 正在停止")
            self.worker.stop()

    def on_algorithm_ready(self, label, summary):  # 算法就绪后更新徽标与日志：徽标按会话实际引擎显示。
        self.algorithm_label.setText(f"算法: {label}")
        self.append_log(f"algorithm ready 算法就绪: {summary}")

    def on_finished(self, ok, message):  # 工作线程结束：恢复按钮并输出结论。
        self.start_button.setEnabled(INFERENCE_AVAILABLE)
        self.pick_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        verdict = "PASSED 通过" if ok else "NOT PASSED 未通过"
        self.append_log(f"result 结果: {verdict} - {message}")
        self._reload_history()  # 复算结束后刷新历史下拉，让最新触发录像即时可选。

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
