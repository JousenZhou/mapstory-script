# -*- coding: utf-8 -*-
"""掉线检测与自动重登监控：从测谎服务拆出的独立守护线程。

值守【掉线2】标注（连续 DISCONNECT_CONFIRM_FRAMES 帧确认），命中后：
暂停脚本任务 → 点击【掉线确定】关闭弹窗 → 等待游戏窗口退出 →
执行 AutoLoginFlow（连接→服务区→频道→开始游戏）→ 恢复脚本任务。

配置沿用看板 Auto Login 键组（configs/Dashboard.json，单一数据源）。
"""
import os
import threading
import time

from ok import Logger, og

from src.autologin.flow import AutoLoginFlow
from src.dashboard_store import load_dashboard_config
from src.task_pause import pause_current_task, resume_task

logger = Logger.get_logger(__name__)

DISCONNECT_TEMPLATE = "掉线2"  # 掉线确认标志模板：单独存在即触发重登流程。
DISCONNECT_DIALOG_TEMPLATE = "掉线"  # 掉线弹窗主体模板：与掉线2同时存在时需先点掉线确定。
DISCONNECT_OK_TEMPLATE = "掉线确定"  # 掉线弹窗确定按钮模板。
DISCONNECT_CONFIRM_FRAMES = 3  # 连续命中【掉线2】多少帧才认定为真掉线（防抖）。
DISCONNECT_OK_WAIT = 2.0  # 点击掉线确定后等待弹窗关闭的秒数。
DISCONNECT_GAME_EXIT_TIMEOUT = 15.0  # 点了【掉线确定】后等待游戏窗口关闭的超时秒数。
DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG = 3.0  # 无掉线弹窗（场景2）时的短等待秒数：窗口不会自己关。
MONITOR_INTERVAL = 1.0  # 值守节拍（秒）：掉线检测无需高频，降低与任务/测谎并发截图开销。
CONFIG_REFRESH_INTERVAL = 0.5  # 看板配置轮询刷新间隔秒数。


class DisconnectMonitor:
    """独立掉线监控线程：与测谎服务互不干扰，各自按配置开关值守。"""

    def __init__(self, exit_event):
        self._exit_event = exit_event  # 进程退出事件，置位后主循环立即结束。
        self._thread = None  # 后台守护线程句柄。
        self._config = {}  # 看板配置缓存。
        self._config_time = 0.0  # 上次读配置时间戳。
        self._hits = 0  # 连续命中【掉线2】的帧数。

    def start(self):
        """启动后台守护线程；重复调用只启动一次。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="DisconnectMonitor", daemon=True)
        self._thread.start()
        logger.info("Disconnect monitor started. 掉线重登监控已启动。")

    def reload_config(self):
        """通知监控立即重读看板配置（DashboardTab 保存时调用）。"""
        self._config_time = 0.0

    def _get_config(self):
        """取看板配置缓存，超过刷新间隔时重读文件。"""
        now = time.time()
        if now - self._config_time >= CONFIG_REFRESH_INTERVAL:
            try:
                self._config = load_dashboard_config()
            except Exception as e:
                logger.warning(f"Disconnect monitor load config failed: {e}. 读取看板配置失败，沿用旧配置。")
            self._config_time = now
        return self._config

    def _read_config(self):
        """解析自动登录配置：开关、匹配阈值、流程配置与原始配置（供 AutoLoginFlow 直读）。"""
        config = self._get_config()
        try:
            threshold = float(config.get("Auto Login Threshold") or 0.75)
        except (TypeError, ValueError):
            threshold = 0.75
        return {
            'enabled': bool(config.get("Auto Login Enabled")),
            'threshold': threshold,
            'raw': config,
        }

    def _run(self):
        """值守主循环：开关开启且掉线模板已标注时检测【掉线2】，命中即执行重登。"""
        while not self._exit_event.is_set():
            try:
                cfg = self._read_config()
                if not cfg['enabled']:
                    self._hits = 0
                    self._sleep(MONITOR_INTERVAL)
                    continue
                if not self._feature_ready(DISCONNECT_TEMPLATE):
                    self._hits = 0
                    self._sleep(MONITOR_INTERVAL)
                    continue
                frame = self._capture()
                if frame is None:
                    self._hits = 0
                    self._sleep(MONITOR_INTERVAL)
                    continue
                if self._find(frame, DISCONNECT_TEMPLATE, cfg['threshold']) is None:
                    self._hits = 0
                    self._sleep(MONITOR_INTERVAL)
                    continue
                self._hits += 1
                if self._hits < DISCONNECT_CONFIRM_FRAMES:
                    self._sleep(MONITOR_INTERVAL)
                    continue
                self._hits = 0
                self._handle_disconnect(frame, cfg)
            except Exception as e:
                logger.warning(f"Disconnect monitor loop error: {e}. 掉线监控循环异常。")
                self._sleep(MONITOR_INTERVAL)
        logger.info("Disconnect monitor stopped. 掉线重登监控已停止。")

    def _handle_disconnect(self, frame, cfg):
        """处理一次掉线：暂停任务 → 关弹窗 → 等窗口退出 → 重登 → 恢复任务。"""
        logger.info("Disconnect detected, starting auto relogin. 检测到掉线，开始自动重登流程。")
        paused_task = pause_current_task(owner="relogin")
        try:
            clicked_ok = self._click_disconnect_ok(frame, cfg['threshold'])
            exit_timeout = DISCONNECT_GAME_EXIT_TIMEOUT if clicked_ok else DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG
            self._wait_game_exit(cfg['threshold'], exit_timeout)
            flow = AutoLoginFlow(self._coco_json_path(), cfg['raw'], logger,
                                 game_frame_fn=self._capture, window_click_fn=self._click_in_window)
            success = flow.run(self._exit_event)
            if success:
                logger.info("Auto relogin completed successfully. 自动重登成功完成。")
            else:
                logger.warning("Auto relogin failed or aborted. 自动重登失败或被中止。")
        except Exception as e:
            logger.warning(f"Auto relogin error: {e}. 自动重登流程异常。")
        finally:
            resume_task(paused_task, owner="relogin")
            self._sleep(1.0)  # 等画面稳定再回到值守。

    def _click_disconnect_ok(self, frame, threshold):
        """Phase1：画面上同时存在【掉线】+【掉线确定】时点击确定关闭弹窗，返回是否点了确定。"""
        if not self._feature_ready(DISCONNECT_OK_TEMPLATE):
            return False
        dialog_box = self._find(frame, DISCONNECT_DIALOG_TEMPLATE, threshold)
        ok_box = self._find(frame, DISCONNECT_OK_TEMPLATE, threshold)
        if dialog_box is not None and ok_box is not None:
            cx = ok_box.x + ok_box.width // 2
            cy = ok_box.y + ok_box.height // 2
            logger.info(f"Clicking disconnect OK at ({cx},{cy}). 点击掉线确定按钮。")
            self._click_in_window(cx, cy)
            self._sleep(DISCONNECT_OK_WAIT)
            return True
        logger.info("Disconnect dialog not present (scenario 2), skip OK click. 无掉线弹窗，跳过确定点击。")
        return False

    def _wait_game_exit(self, threshold, timeout):
        """Phase2：等待游戏窗口关闭（掉线2消失或截图失败），超时后也继续重登。"""
        logger.info(f"Waiting for game window to close (timeout {timeout}s). 等待游戏窗口关闭（超时 {timeout}s）。")
        deadline = time.time() + timeout
        while not self._exit_event.is_set() and time.time() < deadline:
            frame = self._capture()
            if frame is None:
                logger.info("Game window closed (capture failed). 游戏窗口已关闭。")
                return
            if self._find(frame, DISCONNECT_TEMPLATE, threshold) is None:
                logger.info("Disconnect template gone from game window. 掉线模板已从游戏窗口消失。")
                return
            self._sleep(0.5)
        logger.warning(f"Game exit wait timeout ({timeout}s), proceeding anyway. 等待游戏退出超时，仍继续重登。")

    # ------------------------------------------------------------------ ok-script 框架胶水

    def _capture(self):
        """通过框架截图方法采集一帧画面，失败返回 None。"""
        device_manager = getattr(og, "device_manager", None)
        method = getattr(device_manager, "capture_method", None) if device_manager is not None else None
        if method is None:
            return None
        try:
            return method.get_frame()
        except Exception:
            return None

    def _feature_set(self):
        """取执行器的特征集（模板标注），不可用时返回 None。"""
        executor = getattr(og, "executor", None)
        return getattr(executor, "feature_set", None) if executor is not None else None

    def _feature_ready(self, feature_name):
        """检查模板页中是否存在指定分类名的标注。"""
        feature_set = self._feature_set()
        if feature_set is None:
            return False
        try:
            return feature_set.feature_exists(feature_name)
        except Exception as e:
            logger.warning(f"Load template annotations failed: {e}. 加载模板标注失败。")
            return False

    def _find(self, frame, feature_name, threshold):
        """全屏匹配指定标注，返回置信度最高的框或 None。"""
        feature_set = self._feature_set()
        if feature_set is None or frame is None:
            return None
        try:
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)
        except ValueError:  # 该分类名未在模板页标注。
            return None
        except Exception as e:
            logger.warning(f"Find feature {feature_name} failed: {e}. 匹配标注失败：{feature_name}。")
            return None
        if not boxes:
            return None
        return max(boxes, key=lambda b: getattr(b, "confidence", 0))

    def _coco_json_path(self):
        """取模板标注文件路径：优先框架 feature_set 已解析路径，回退项目默认。"""
        feature_set = self._feature_set()
        path = getattr(feature_set, "coco_json", None) if feature_set is not None else None
        if path:
            return path
        return os.path.join(os.getcwd(), 'ok_templates', 'coco_annotations.json')

    def _click_in_window(self, x, y, clicks=1):
        """通过框架输入接口点击游戏窗口内坐标；clicks>1 时同位置紧凑连点（双击）。"""
        self._ensure_in_front()  # interaction 的前台守卫在窗口非前台时会静默跳过点击。
        time.sleep(0.05)
        device_manager = getattr(og, "device_manager", None)
        interaction = getattr(device_manager, "interaction", None) if device_manager is not None else None
        if interaction is None:
            logger.warning("Interaction unavailable, cannot click. 输入接口不可用，无法点击。")
            return
        try:
            for index in range(max(1, int(clicks))):
                if index:
                    time.sleep(0.05)  # 连点间隔过长会被启动器当成两次单击。
                else:
                    interaction.move(x, y)
                interaction.click(x, y)
        except Exception as e:
            logger.warning(f"Click in window failed at ({x},{y}): {e}. 窗口内点击失败。")

    def _ensure_in_front(self):
        """把游戏/启动器窗口置前，保证前台点击守卫放行。"""
        device_manager = getattr(og, "device_manager", None)
        hwnd = getattr(device_manager, "hwnd_window", None) if device_manager is not None else None
        if hwnd is None:
            return
        try:
            hwnd.bring_to_front()
        except Exception as e:
            logger.warning(f"Ensure in front failed: {e}. 窗口置前失败。")

    def _sleep(self, seconds):
        """分片等待，每片检查退出事件，保证进程能及时停止。"""
        end = time.time() + seconds
        while not self._exit_event.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.1))
