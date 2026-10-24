# -*- coding: utf-8 -*-
"""独立测谎监控服务：后台守护线程值守【测谎触发】标注，命中即自动解题。

与脚本任务解耦：触发后暂停当前任务并释放按键 → 报警 → 起录（【测谎区域标注】原始视频）
→ d/lie 管线解题（鼠标直跟目标路径）→ 3 秒结算（成功点【测谎成功确定】/失败）→ 停录 → 恢复任务 → 冷却。

职责边界：
- 掉线重登不在本服务（拆至 src/autologin/monitor.py 独立线程）；
- UI 标注（测谎触发/成功/确定/坐标框）用 ok-script FeatureSet 模板匹配
  （best.pt 只有图形类别，YOLO 只负责解题图形跟踪）；
- 解题视觉算法全部在 src/liedetector/vision（D:\\lie 原样移植）。
"""
import os
import threading
import time

import cv2  # noqa: F401  确保环境具备 OpenCV（报警/绘制不直接用，但解题依赖）。
from ok import Logger, og

from src.dashboard_store import load_dashboard_config, save_dashboard_config
from src.keynames import normalize_key_name
from src.liedetector.capture import FramePump, make_independent_capture
from src.liedetector.recorder import LieRecorder
from src.liedetector.solver import LieSolver, SOLVE_FPS
from src.liedetector.vision.detector import Detector
from src.task_pause import pause_current_task, resume_task

logger = Logger.get_logger(__name__)

MONITOR_INTERVAL = 1.0 / 15  # 空闲监控节拍（约 15FPS）。
CONFIG_REFRESH_INTERVAL = 0.5  # 看板配置轮询刷新间隔秒数。
TRIGGER_CONFIRM_FRAMES = 3  # 连续命中【测谎触发】多少帧才认定为真触发（防抖）。
SOLVE_COOLDOWN = 2.0  # 一局结束后冷却秒数：等弹窗淡出、画面稳定。
POST_SOLVE_SLEEP = 0.5  # 解题结束后等弹窗关闭画面稳定的秒数。
TRIGGER_PROBE_INTERVAL = 2.0  # 未匹配到触发时探测实际最高分的限频间隔（诊断用）。

STATE_IDLE = "idle"  # 空闲监控态。
STATE_SOLVING = "solving"  # 正在解测谎态。


class LieDetectorService:
    """独立测谎监控服务。Globals 持有并随 app 启动；急停热键、报警、录像历史一并由本服务管理。"""

    def __init__(self, exit_event):
        self._exit_event = exit_event  # 进程退出事件。
        self._thread = None  # 后台守护线程句柄。
        self._config = {}  # 看板配置缓存。
        self._config_time = 0.0  # 上次读配置时间戳。
        self._last_status = None  # 上次状态文本（变化才打日志）。
        self._trigger_hits = 0  # 连续命中触发帧数。
        self._cooldown_until = 0.0  # 冷却截止时间戳。
        self._abort_event = threading.Event()  # 急停标志。
        self._abort_key = ''  # 规范化急停键名，空表示停用。
        self._abort_listener = None  # pynput 全局键盘监听器。
        self._abort_notice = False  # 急停一次性通知标志（看板轮询同步开关）。
        self._recorder = None  # 当前录像器（一局一个实例）。
        self._detector = None  # warm 解题后端（后台线程装配）。
        self._detector_error = None  # 后端装配失败原因（None=成功或仍在装配）。
        self._last_probe = 0.0  # 上次触发分数诊断探测时间戳。
        self._gpu = None  # 显卡模板匹配器（隐藏式启用，异常后降级 CPU）。
        self._gpu_off = False  # 显卡匹配运行期异常永久关闭标志。
        self.state = STATE_IDLE  # 当前服务状态（UI/诊断读取）。

    # ------------------------------------------------------------------ 生命周期

    def start(self):
        """启动后台守护线程；重复调用只启动一次。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="LieDetectorService", daemon=True)
        self._thread.start()
        threading.Thread(target=self._prepare_backend, name="LieBackendLoader", daemon=True).start()
        logger.info("Lie detector service started. 独立测谎监控服务已启动。")

    def reload_config(self):
        """通知服务立即重读看板配置（DashboardTab 保存时调用）。"""
        self._config_time = 0.0

    def backend_desc(self):
        """当前解题后端描述（看板只读标签用）。"""
        detector = self._detector
        if detector is not None:
            return detector.backend_label
        if self._detector_error is not None:
            return f"后端不可用: {self._detector_error}"
        return "后端装配中…"

    # 旧名兼容（DashboardTab 历史调用）。
    def _backend_desc(self):
        return self.backend_desc()

    # ------------------------------------------------------------------ 后端装配

    def _prepare_backend(self):
        """后台装配 warm 解题后端：缓存 TRT INT8 引擎毫秒级加载，缺失则建引擎（分钟级一次），
        失败沿显式降级链落到 torch CUDA / CPU。装配完成前触发会临时用 CPU 后端解题。"""
        try:
            calib_video = self._latest_record_video()  # 用最近录像作 INT8 校准源（无则缺省）。
            detector = Detector(device="gpu", calib_video=calib_video)
            detector.warmup()  # 首帧推理成本移到装配期。
            self._detector = detector
            logger.info(f"Lie backend ready: {detector.backend_label}. 解题后端就绪。")
        except Exception as e:
            self._detector_error = str(e)
            logger.warning(f"Lie backend init failed, solve will use temporary CPU detector: {e}. "
                           f"解题后端装配失败，触发时将临时用 CPU 后端。")

    @staticmethod
    def _latest_record_video():
        """取 lie_records 最新一个 mp4 作 TRT 校准源；无录像返回 None。"""
        try:
            from src.liedetector.recorder import LIE_RECORD_DIR
            if not os.path.isdir(LIE_RECORD_DIR):
                return None
            mp4s = [n for n in os.listdir(LIE_RECORD_DIR) if n.endswith(".mp4")]
            if not mp4s:
                return None
            latest = max(mp4s, key=lambda n: os.path.getmtime(os.path.join(LIE_RECORD_DIR, n)))
            return os.path.join(LIE_RECORD_DIR, latest)
        except Exception:
            return None

    def _detector_for_solve(self):
        """取解题后端：warm 后端就绪直接用；否则临时建 CPU 后端（触发绝不空转）。"""
        if self._detector is not None:
            return self._detector
        try:
            logger.warning("Lie backend not ready, solve on temporary CPU detector. 后端未就绪，本次以 CPU 后端解题。")
            return Detector(device="cpu")
        except Exception as e:
            logger.error(f"Lie solver unavailable: {e}. 解题后端无法装配，本局不解题。")
            return None

    # ------------------------------------------------------------------ 主循环

    def _run(self):
        """值守主循环：开关开启且标注就绪时逐帧匹配【测谎触发】，连续确认后处理一局。"""
        while not self._exit_event.is_set():
            try:
                self._sync_abort_key()
                auto_solve, trigger_name, region_name, threshold, _alarm = self._read_config()
                if not auto_solve:
                    self._set_status("auto solve off")
                    self._trigger_hits = 0
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                if not (trigger_name and region_name
                        and self._feature_ready(trigger_name) and self._feature_ready(region_name)):
                    self._set_status(f"annotation missing: {trigger_name or region_name}")
                    self._trigger_hits = 0
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                if time.time() < self._cooldown_until:
                    self._set_status("cooling down")
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                self._set_status("watching")
                frame = self._capture()
                if frame is None:
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                trigger_box = self.find_feature(frame, trigger_name, threshold)
                if trigger_box is None:
                    self._trigger_hits = 0
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                self._trigger_hits += 1
                if self._trigger_hits < TRIGGER_CONFIRM_FRAMES:
                    self._set_status(f"confirming {self._trigger_hits}/{TRIGGER_CONFIRM_FRAMES}")
                    self._idle_sleep(MONITOR_INTERVAL)
                    continue
                self._trigger_hits = 0
                self._handle_trigger(frame, trigger_box, trigger_name, region_name, threshold)
            except Exception as e:
                logger.warning(f"Lie service loop error: {e}. 测谎服务循环异常。")
                self._idle_sleep(MONITOR_INTERVAL)

    def _handle_trigger(self, frame, trigger_box, trigger_name, region_name, threshold):
        """处理一局：报警 → 暂停任务 → 起录 → 解题+结算 → 停录 → 恢复任务 → 冷却。"""
        # 每局开始复位急停标志：急停只作用于"当局"（用户重新打开自动解测谎后的新局必须干净启动），
        # 否则一次急停后所有后续局都会秒退成 aborted，表现为报警循环重响 + 一堆空录像。
        self._abort_event.clear()
        self.state = STATE_SOLVING
        self._set_status("solving")
        _auto, _trig, _reg, _thr, alarm_sound = self._read_config()
        self._play_alarm(alarm_sound)  # 报警先于一切，让用户第一时间知道触发。
        paused_task = pause_current_task(owner="lie detector")
        detector = self._detector_for_solve()
        if detector is None:
            self._finish_round(paused_task, "abandoned")
            return
        region_box = self.get_region(frame, region_name)
        crop = (region_box.x, region_box.y, region_box.width, region_box.height) if region_box is not None else None
        self._start_recorder(frame.shape, crop, detector)
        try:
            solver = LieSolver(self, self._exit_event, self._abort_event)
            outcome = solver.run(detector, frame, trigger_name, region_name, threshold)
        except Exception as e:
            logger.warning(f"Lie solve crashed: {e}. 解题异常，按失败收尾。")
            outcome = "timeout"
        finally:
            self._stop_recorder(outcome)
        self._finish_round(paused_task, outcome)

    def _finish_round(self, paused_task, outcome):
        """一局收尾：恢复任务、开冷却、回空闲态。"""
        resume_task(paused_task, owner="lie detector")
        self.state = STATE_IDLE
        self._set_status("watching")
        self._cooldown_until = time.time() + SOLVE_COOLDOWN
        self._idle_sleep(POST_SOLVE_SLEEP)
        logger.info(f"Lie round finished, outcome={outcome}, cooldown {SOLVE_COOLDOWN}s. 一局测谎结束，冷却后继续值守。")

    # ------------------------------------------------------------------ 录像协调

    def _start_recorder(self, frame_shape, crop, detector):
        """起录：独立采集克隆 + 【测谎区域标注】裁剪，异常吞掉绝不影响解题。"""
        try:
            capture, release = make_independent_capture(self._capture)
            recorder = LieRecorder()
            recorder.start(frame_shape, meta={
                "backend": detector.backend_label,
                "conf": getattr(detector, "conf", None),
            }, crop=crop, capture=capture, capture_release=release)
            self._recorder = recorder
        except Exception as e:
            logger.warning(f"Lie record start failed: {e}. 起录失败，本局不录。")
            self._recorder = None

    def _stop_recorder(self, outcome):
        """收尾录像（写边车+滚动保留），未起录时空转。"""
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        try:
            recorder.stop(outcome)
        except Exception as e:
            logger.warning(f"Lie record stop failed: {e}. 测谎录像收尾失败。")

    # ------------------------------------------------------------------ LieSolver host 接口（框架胶水）

    def find_feature(self, frame, feature_name, threshold):
        """全屏匹配指定标注，返回置信度最高的框或 None；显卡可用时优先走显卡匹配。"""
        if frame is None:
            return None
        handle = self._gpu_handle(frame, [feature_name])
        if handle is not None and self._gpu is not None and self._gpu.has(feature_name):
            try:
                box = self._gpu.best_box(handle, feature_name, threshold)
                if box is None:
                    self._probe_score(frame, feature_name, threshold)
                return box
            except Exception as e:
                self._disable_gpu(e)
        feature_set = self._feature_set()
        if feature_set is None:
            return None
        try:
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)
        except ValueError:  # 该分类名未在模板页标注。
            return None
        except Exception as e:
            logger.warning(f"Find feature {feature_name} failed: {e}. 匹配标注失败：{feature_name}。")
            return None
        if not boxes:
            self._probe_score(frame, feature_name, threshold)
            return None
        return max(boxes, key=lambda b: getattr(b, "confidence", 0))

    def get_region(self, frame, region_name):
        """直读【测谎坐标框】标注记录的坐标框（不做模板匹配）；未标注返回 None。"""
        feature_set = self._feature_set()
        if feature_set is None:
            return None
        try:
            return feature_set.get_box_by_name(frame, region_name)
        except Exception as e:
            logger.warning(f"Get region box failed: {e}. 读取坐标框失败。")
            return None

    def capture(self):
        """共享采集取一帧，失败返回 None。"""
        return self._capture()

    def start_pump(self):
        """为解题拉起独立采集泵；不可克隆时 (None, None) 回退共享采集。"""
        capture, release = make_independent_capture(self._capture)
        if capture is self._capture:
            return None, None
        try:
            pump = FramePump(capture, SOLVE_FPS)
            pump.start()
            return pump, release
        except Exception as e:
            logger.warning(f"Lie solve pump start failed: {e}. 解题采集泵启动失败，回退共享采集。")
            if release is not None:
                try:
                    release()
                except Exception:
                    pass
            return None, None

    def move_mouse(self, x, y):
        """移动鼠标到窗口内坐标 (x, y)。"""
        interaction = self._interaction()
        if interaction is None:
            return
        try:
            interaction.move(int(x), int(y))
        except Exception as e:
            logger.warning(f"Move mouse failed: {e}. 移动鼠标失败。")

    def click_window(self, x, y):
        """点击窗口内坐标：前台档先置顶（前台守卫会吞非前台点击）。"""
        if not self.is_background_mouse():
            self.ensure_in_front()
            time.sleep(0.05)
        interaction = self._interaction()
        if interaction is None:
            return
        try:
            interaction.move(int(x), int(y))
            interaction.click(int(x), int(y))
        except Exception as e:
            logger.warning(f"Click window failed at ({x},{y}): {e}. 窗口内点击失败。")

    def is_background_mouse(self):
        """是否后台鼠标档（PostMessage）：后台档不置顶、不碰真实光标。"""
        interaction = self._interaction()
        if interaction is None:
            return False
        try:
            from ok.device.interaction_methods.post_message import PostMessageInteraction
            return isinstance(interaction, PostMessageInteraction)
        except Exception:
            return False

    def ensure_in_front(self):
        """把游戏窗口置顶（前台鼠标档需要）。"""
        device_manager = getattr(og, "device_manager", None)
        hwnd = getattr(device_manager, "hwnd_window", None) if device_manager is not None else None
        if hwnd is None:
            return
        try:
            hwnd.bring_to_front()
        except Exception as e:
            logger.warning(f"Ensure in front failed: {e}. 窗口置顶失败。")

    def push_vision(self, frame):
        """把带标注画面推送给 UI 实时展示。"""
        app = getattr(og, "my_app", None)
        if app is not None and frame is not None:
            app.update_vision(frame)

    # ------------------------------------------------------------------ 显卡模板匹配加速

    def _gpu_handle(self, frame, names):
        """取本帧显卡匹配句柄：首次创建匹配器、模板或画面变化时重建；不可用返回 None。"""
        if self._gpu_off or frame is None or not names:
            return None
        try:
            from src.gpu_feature_match import GpuFeatureMatcher, gpu_match_available
            if not gpu_match_available():
                return None
            feature_set = self._feature_set()
            if feature_set is None:
                return None
            if self._gpu is None or self._gpu.feature_set is not feature_set:
                self._gpu = GpuFeatureMatcher(feature_set, gray=True, logger=logger)
                logger.info("Lie GPU template match enabled. 测谎服务显卡模板匹配加速已启用。")
            if not self._gpu.prepare(names, frame):
                return None
            return self._gpu.frame(frame)
        except Exception as e:
            self._disable_gpu(e)
            return None

    def _disable_gpu(self, error):
        """显卡异常处理：瞬时流冲突只记一次不降级，真故障降级 CPU。"""
        message = str(error)
        if "stream is capturing" in message or "StreamCapture" in message:
            logger.warning(f"Lie GPU match transient CUDA capture conflict, retry next frame: {error}. "
                           f"显卡匹配瞬时流冲突，下一帧自动重试。")
            return
        if not self._gpu_off:
            logger.warning(f"Lie GPU match failed, fallback to CPU: {error}. 显卡匹配失败，降级 CPU。")
        self._gpu_off = True
        self._gpu = None

    def _probe_score(self, frame, feature_name, threshold):
        """未匹配到时限频探测实际最高分并记日志，区分弹窗未出现/阈值偏高/模板不符。"""
        now = time.time()
        if now - self._last_probe < TRIGGER_PROBE_INTERVAL:
            return
        self._last_probe = now
        try:
            if self._gpu is not None:
                handle = self._gpu_handle(frame, [feature_name])
                score = self._gpu.best_score(handle, feature_name) if handle is not None else 0.0
            else:
                feature_set = self._feature_set()
                boxes = feature_set.find_feature(frame, feature_name, 1, 1, 0.01, True, limit=1) if feature_set else []
                score = max((getattr(b, "confidence", 0.0) for b in (boxes or [])), default=0.0)
            logger.info(f"Lie trigger not matched: best score={score:.3f} < threshold={threshold}. "
                        f"触发未达阈值：最高分={score:.3f}，阈值={threshold}。")
        except Exception:
            pass

    # ------------------------------------------------------------------ 急停热键

    def _sync_abort_key(self):
        """每轮主循环刷新急停键：配置变化时按需启停 pynput 全局监听。"""
        raw = str(self._get_config().get("Lie Detector Abort Key") or '').strip()
        normalized = normalize_key_name(raw)
        if normalized == self._abort_key:
            return
        self._abort_key = normalized
        if normalized:
            self._start_abort_listener()
            logger.info(f"Lie abort hotkey armed: {raw}. 急停按键已启用：{raw}。")
        else:
            self._stop_abort_listener()
            logger.info("Lie abort hotkey disabled. 急停按键已停用。")

    def _start_abort_listener(self):
        """启动 pynput 全局键盘监听；pynput 不可用时降级停用。"""
        self._stop_abort_listener()
        try:
            from pynput import keyboard
        except Exception as e:
            logger.warning(f"pynput unavailable, abort hotkey disabled: {e}. pynput 不可用，急停键无法启用。")
            self._abort_key = ''
            return
        try:
            listener = keyboard.Listener(on_press=self._on_abort_key_press)
            listener.daemon = True
            listener.start()
            self._abort_listener = listener
        except Exception as e:
            logger.warning(f"Abort listener start failed: {e}. 急停监听启动失败。")
            self._abort_listener = None

    def _stop_abort_listener(self):
        listener = self._abort_listener
        self._abort_listener = None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:
            pass

    def _on_abort_key_press(self, key):
        """全局按键回调（监听线程内，务必轻量吞异常）。"""
        if not self._abort_key:
            return
        try:
            name = getattr(key, 'char', None) or getattr(key, 'name', '') or ''
        except Exception:
            return
        if not name or normalize_key_name(name) != self._abort_key:
            return
        self._trigger_abort()

    def _trigger_abort(self):
        """急停：置中止标志、把自动解题开关写回为关、通知 UI 同步。"""
        if self._abort_event.is_set():
            return
        self._abort_event.set()
        self._abort_notice = True
        try:
            config = load_dashboard_config()
            config['Lie Detector Auto Solve'] = False
            save_dashboard_config(config)
        except Exception as e:
            logger.warning(f"Abort write config failed: {e}. 急停写回配置失败。")
        self.reload_config()
        logger.warning("Lie solve ABORTED by hotkey, auto-solve OFF. 急停生效，自动解测谎已关闭。")

    def consume_abort_notice(self):
        """看板 UI 轮询：取走一次性急停通知（取后即清）。"""
        if self._abort_notice:
            self._abort_notice = False
            return True
        return False

    # ------------------------------------------------------------------ 报警

    def _play_alarm(self, sound_path):
        """播放报警音频：异步不阻塞，失败只记日志。"""
        if not sound_path:
            return False
        path = sound_path if os.path.isabs(sound_path) else os.path.join(os.getcwd(), sound_path)
        if not os.path.exists(path):
            logger.warning(f"Lie alarm sound not found: {sound_path}. 报警音频不存在，跳过。")
            return False
        try:
            if path.lower().endswith('.wav'):
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                import ctypes
                winmm = ctypes.windll.winmm
                alias = "lie_alarm"
                winmm.mciSendStringW(f"close {alias}", None, 0, 0)
                if winmm.mciSendStringW(f'open "{path}" alias {alias}', None, 0, 0) != 0:
                    if winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0) != 0:
                        logger.warning(f"Lie alarm open failed: {sound_path}. 报警音频无法打开。")
                        return False
                winmm.mciSendStringW(f"play {alias}", None, 0, 0)
            logger.info(f"Lie alarm playing: {sound_path}. 报警音频已播放。")
            return True
        except Exception as e:
            logger.warning(f"Lie alarm play failed: {e}. 报警播放失败。")
            return False

    # ------------------------------------------------------------------ 配置与框架胶水

    def _read_config(self):
        """解析看板测谎配置：(自动解开关, 触发标注名, 区域标注名, 阈值, 报警音频)。"""
        config = self._get_config()
        auto_solve = bool(config.get("Lie Detector Auto Solve"))
        trigger_name = str(config.get("Lie Detector Trigger Feature") or '').strip()
        region_name = str(config.get("Lie Detector Region Feature") or '').strip()
        try:
            threshold = float(config.get("Lie Detector Threshold") or 0.75)
        except (TypeError, ValueError):
            threshold = 0.75
        alarm_sound = str(config.get("Lie Alarm Sound") or '').strip()
        return auto_solve, trigger_name, region_name, threshold, alarm_sound

    def _get_config(self):
        """取看板配置缓存，超过刷新间隔重读文件。"""
        now = time.time()
        if now - self._config_time >= CONFIG_REFRESH_INTERVAL:
            try:
                self._config = load_dashboard_config()
            except Exception as e:
                logger.warning(f"Load dashboard config failed: {e}. 读取看板配置失败，沿用旧配置。")
            self._config_time = now
        return self._config

    def _capture(self):
        """共享截图方法取一帧，失败返回 None。"""
        device_manager = getattr(og, "device_manager", None)
        method = getattr(device_manager, "capture_method", None) if device_manager is not None else None
        if method is None:
            return None
        try:
            return method.get_frame()
        except Exception:
            return None

    def _interaction(self):
        """取框架输入设备接口（鼠标/键盘），不可用返回 None。"""
        device_manager = getattr(og, "device_manager", None)
        return getattr(device_manager, "interaction", None) if device_manager is not None else None

    def _feature_set(self):
        """取执行器特征集（模板标注），不可用返回 None。"""
        executor = getattr(og, "executor", None)
        return getattr(executor, "feature_set", None) if executor is not None else None

    def _feature_ready(self, feature_name):
        """检查模板页是否存在指定分类名的标注。"""
        feature_set = self._feature_set()
        if feature_set is None:
            return False
        try:
            return feature_set.feature_exists(feature_name)
        except Exception as e:
            logger.warning(f"Load template annotations failed: {e}. 加载模板标注失败。")
            return False

    def _idle_sleep(self, seconds):
        """分片等待，每片检查退出事件。"""
        end = time.time() + seconds
        while not self._exit_event.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.1))

    def _set_status(self, status):
        """状态变化时打一次日志，避免刷屏。"""
        if status != self._last_status:
            self._last_status = status
            logger.info(f"Lie detector service: {status}. 测谎服务状态：{status}。")
