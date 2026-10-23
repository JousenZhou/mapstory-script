# -*- coding: utf-8 -*-
"""解测谎会话：d/lie 视觉管线逐帧跟踪 + 鼠标直跟目标路径 + 3 秒结算协议。

解题循环（移植 d/lie 管线的实战消费端）：
- 独立采集泵取最新帧 → 读【测谎坐标框】区域 → 裁剪喂 Pipeline.process；
- 锁定目标有 follow_bbox（d/lie 内部已做 3 帧中值平滑）时，鼠标每帧直跟其中心，
  不做步长限速/对齐自检——保持与 d/lie 行为一致；
- 退出条件：【测谎触发】标注连续 TRIGGER_LOST_TOLERANCE 帧消失（gone）、
  状态机 GAME_OVER（solved）、急停（aborted）、墙钟兜底（timeout）。

结算协议（用户指定）：
【测谎触发标注】消失后 3 秒内出现【测谎成功】→ 点击【测谎成功确定】直至其消失 = success；
3 秒内无反应 = failure。
"""
import time

import cv2

from ok import Logger

from src.liedetector.capture import FramePump
from src.liedetector.vision.overlay_renderer import render
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.state_machine import GameState

logger = Logger.get_logger(__name__)

SOLVE_FPS = 30.0  # 管线帧率口径：状态机倒计时/超时按 30FPS 换算帧数（d/lie 设计值）。
SOLVE_MAX_SECONDS = 25.0  # 解题墙钟兜底：超时退回监控（防触发标注误匹配导致死循环）。
TRIGGER_LOST_TOLERANCE = 12  # 解题中允许连续丢失【测谎触发】的帧数（12 帧@30FPS≈0.4 秒）。
REGION_RESET_PIXELS = 4  # 【测谎坐标框】任一边位移超过该像素数视为新一局，重建管线会话。
SETTLE_WINDOW = 3.0  # 触发标注消失后等待【测谎成功】出现的窗口秒数（用户指定 3 秒）。
SETTLE_CONFIRM_TIMEOUT = 15.0  # 点击【测谎成功确定】后等待【测谎成功】消失的安全超时秒数。
SETTLE_CLICK_INTERVAL = 0.4  # 两次点击【测谎成功确定】的间隔秒数。
SETTLE_POLL_INTERVAL = 0.1  # 结算窗口内轮询画面间隔秒数。
VISION_PUSH_INTERVAL = 1.0 / 12  # 向 UI 推送叠加画面的最小间隔（预览无需跟满解题帧率）。

LIE_SUCCESS_TEMPLATE = "测谎成功"  # 结算成功标志模板。
LIE_SUCCESS_OK_TEMPLATE = "测谎成功确定"  # 成功弹窗确定按钮模板。


class LieSolver:
    """一局测谎的解题+结算会话。host 为服务端注入的框架胶水接口（duck-typed）：

    find_feature(frame, name, threshold) / get_region(frame, name) / capture() /
    start_pump() -> (pump, release) / move_mouse(x, y) / click_window(x, y) /
    is_background_mouse() / ensure_in_front() / push_vision(frame)
    """

    def __init__(self, host, exit_event, abort_event):
        self._host = host  # 框架胶水（由 LieDetectorService 实现）。
        self._exit_event = exit_event  # 进程退出事件。
        self._abort_event = abort_event  # 用户急停事件。

    # ------------------------------------------------------------------ 主入口

    def run(self, detector, first_frame, trigger_name, region_name, threshold):
        """解题 + 结算，返回最终 outcome：success/failure/timeout/aborted。"""
        outcome = self.solve(detector, first_frame, trigger_name, region_name, threshold)
        if outcome in ("gone", "solved"):  # 正常跑完（触发消失/状态机判结束）才结算。
            outcome = self.settle(threshold)
        return outcome

    # ------------------------------------------------------------------ 解题循环

    def solve(self, detector, first_frame, trigger_name, region_name, threshold):
        """跟踪阶段：鼠标直跟 follow_bbox 直到触发标注消失/状态机结束/急停/兜底超时。"""
        if not self._host.is_background_mouse():
            self._host.ensure_in_front()  # 前台档：鼠标跟随依赖前台窗口接收真实鼠标事件。
        pipeline = Pipeline(detector, SOLVE_FPS)  # 新会话：构造时自动复位 OC-SORT 轨迹簿。
        pipeline.sm.start(0)  # 进入 COUNTDOWN，状态机内部锚定倒计时。
        frame_idx = 0  # 已喂入管线的帧计数。
        region = None  # 当前有效区域 (x, y, w, h)。
        frame = first_frame
        lost_ticks = 0  # 连续未匹配到【测谎触发】的帧数。
        outcome = "timeout"  # 兜底退出原因。
        solve_start = time.perf_counter()
        last_vision_time = 0.0
        pump, pump_release = self._start_pump()
        try:
            while not self._exit_event.is_set():
                if self._abort_event.is_set():
                    logger.info("Lie solve aborted by hotkey. 急停按键中止解题。")
                    outcome = "aborted"
                    break
                if time.perf_counter() - solve_start > SOLVE_MAX_SECONDS:
                    logger.warning(f"Lie solve exceeded {SOLVE_MAX_SECONDS}s, resume. 解题超 {SOLVE_MAX_SECONDS} 秒未结束，退回监控。")
                    outcome = "timeout"
                    break
                found = self._host.find_feature(frame, trigger_name, threshold)
                if found is None:
                    lost_ticks += 1
                    if lost_ticks > TRIGGER_LOST_TOLERANCE:  # 连续多帧丢失：测谎已结束。
                        logger.info(f"Lie trigger gone after {lost_ticks} lost frames. 【测谎触发】已消失，测谎结束。")
                        outcome = "gone"
                        break
                else:
                    lost_ticks = 0
                region_box = self._host.get_region(frame, region_name)
                if region_box is None:
                    frame = self._next_frame(pump) or frame  # 未采到区域：等新帧重试。
                    continue
                new_region = (region_box.x, region_box.y, region_box.width, region_box.height)
                if region is None or self._region_reset(region, new_region):
                    if region is not None:  # 区域位移/缩放：重建会话（防上一局轨迹串扰）。
                        pipeline = Pipeline(detector, SOLVE_FPS)
                        pipeline.sm.start(0)
                        frame_idx = 0
                        logger.info("Lie region shifted, rebuild pipeline session. 区域位移，重建解题会话。")
                    region = new_region
                crop = frame[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
                result = pipeline.process(crop, frame_idx)
                frame_idx += 1
                if result.follow_bbox is not None:  # 鼠标直跟呈现框中心（3 帧中值平滑已在管线内做）。
                    x1, y1, x2, y2 = result.follow_bbox
                    self._host.move_mouse(region[0] + (x1 + x2) / 2.0, region[1] + (y1 + y2) / 2.0)
                now = time.perf_counter()
                if now - last_vision_time >= VISION_PUSH_INTERVAL:
                    last_vision_time = now
                    self._host.push_vision(self._draw(frame, region, result, found))
                if result.state == GameState.GAME_OVER.value:  # 状态机判定测谎结束。
                    logger.info("Lie pipeline reached GAME_OVER. 管线状态机判定测谎结束。")
                    outcome = "solved"
                    break
                frame = self._next_frame(pump) or frame
        finally:
            if pump is not None:
                pump.stop()
            if pump_release is not None:
                try:
                    pump_release()
                except Exception:
                    pass
        return outcome

    # ------------------------------------------------------------------ 结算

    def settle(self, threshold):
        """结算：触发标注消失后 SETTLE_WINDOW 秒内出现【测谎成功】→ 点确定收尾 = success；无反应 = failure。"""
        logger.info(f"Lie solve done, wait up to {SETTLE_WINDOW}s for success mark. 解题完成，等最多 {SETTLE_WINDOW} 秒看【测谎成功】。")
        deadline = time.time() + SETTLE_WINDOW
        while not self._exit_event.is_set():
            if self._abort_event.is_set():
                logger.info("Lie settle aborted by hotkey. 结算期收到急停。")
                return "aborted"
            frame = self._host.capture()
            if frame is not None:
                success_box = self._host.find_feature(frame, LIE_SUCCESS_TEMPLATE, threshold)
                if success_box is not None:
                    logger.info("Lie success mark detected, confirming. 检测到【测谎成功】，点击确定收尾。")
                    return self._confirm_success(threshold)
            if time.time() >= deadline:
                break
            self._sleep(SETTLE_POLL_INTERVAL)
        logger.warning(f"Lie success not shown within {SETTLE_WINDOW}s, mark FAILURE. {SETTLE_WINDOW} 秒内未出现【测谎成功】，判定失败。")
        return "failure"

    def _confirm_success(self, threshold):
        """出现【测谎成功】后：反复点击【测谎成功确定】直到【测谎成功】消失 = success。"""
        if not self._host.is_background_mouse():
            self._host.ensure_in_front()
        deadline = time.time() + SETTLE_CONFIRM_TIMEOUT
        while not self._exit_event.is_set():
            if self._abort_event.is_set():
                logger.info("Lie success confirm aborted by hotkey. 确认期收到急停。")
                return "aborted"
            frame = self._host.capture()
            if frame is None:
                self._sleep(SETTLE_POLL_INTERVAL)
                if time.time() >= deadline:
                    break
                continue
            if self._host.find_feature(frame, LIE_SUCCESS_TEMPLATE, threshold) is None:
                logger.info("Lie success mark gone, settle done (SUCCESS). 【测谎成功】已消失，测谎成功收尾。")
                return "success"
            ok_box = self._host.find_feature(frame, LIE_SUCCESS_OK_TEMPLATE, threshold)
            if ok_box is not None:
                cx = ok_box.x + ok_box.width // 2
                cy = ok_box.y + ok_box.height // 2
                logger.info(f"Clicking lie success OK at ({cx},{cy}). 点击【测谎成功确定】。")
                self._host.click_window(cx, cy)
            self._sleep(SETTLE_CLICK_INTERVAL)
            if time.time() >= deadline:
                break
        logger.warning("Lie success confirm timeout, still mark SUCCESS. 点击确定超时未消失，按成功收尾。")
        return "success"  # 已出现过成功标注，超时也归为成功。

    # ------------------------------------------------------------------ 工具

    def _start_pump(self):
        """拉起独立采集泵；不可克隆时 (None, None) 回退共享采集。"""
        try:
            return self._host.start_pump()
        except Exception as e:
            logger.warning(f"Lie solve pump start failed: {e}. 解题采集泵启动失败，回退共享采集。")
            return None, None

    def _next_frame(self, pump):
        """取解题用帧：泵就绪时等新帧；无泵时短等后走共享采集。"""
        if pump is not None:
            return pump.wait_new(1.0 / SOLVE_FPS)
        self._sleep(1.0 / SOLVE_FPS)
        return self._host.capture()

    @staticmethod
    def _region_reset(last, new):
        """区域任一边位移超过阈值即视为重置（新一局/窗口位移）。"""
        return any(abs(a - b) > REGION_RESET_PIXELS for a, b in zip(last, new))

    def _draw(self, frame, region, result, trigger_box):
        """叠加画面：触发标注红框 + 区域青框 + d/lie 轨迹渲染，供 UI 实时预览。"""
        try:
            canvas = frame.copy()
            if region is not None:
                x, y, w, h = region
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 2)
                overlay = render(result.frame, result.tracks, result.target_id, result.state,
                                 show_boxes=True, show_trails=True)
                canvas[y:y + h, x:x + w] = overlay  # 区域内替换为带轨迹渲染的画面（尺寸一致）。
            if trigger_box is not None:
                cv2.rectangle(canvas, (trigger_box.x, trigger_box.y),
                              (trigger_box.x + trigger_box.width, trigger_box.y + trigger_box.height),
                              (0, 0, 255), 2)
                cv2.putText(canvas, "LIE TRIGGER", (trigger_box.x, max(trigger_box.y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return canvas
        except Exception:
            return frame  # 绘制异常不影响解题。

    def _sleep(self, seconds):
        """分片等待，每片检查退出/急停事件。"""
        end = time.time() + seconds
        while not self._exit_event.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))
