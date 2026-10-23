# -*- coding: utf-8 -*-
"""测谎独立采集：克隆一路与共享截图方法互不抢帧的采集实例，并提供 latest-wins 取帧泵。

关键：解题循环与录像必须各自独享一路采集，不能与框架共享 capture_method 抢同一把锁。
以 WGC 为例其 get_frame 走「单请求-单响应」帧通道：共享时解题与录像会瓜分同一窗口的帧。
按当前采集方式克隆同类独立实例——WGC 建独立 frame pool/session（系统支持同窗口多会话），
BitBlt 复刻独立 lock/DC 的实例；其它方式（DXGI 等）回退共享采集。
"""
import threading
import time

from ok import Logger, og

logger = Logger.get_logger(__name__)


def make_independent_capture(fallback_capture):
    """克隆一路独立采集，返回 (capture_callable, release_callable)；不可克隆时回退 (fallback_capture, None)。"""
    try:
        device_manager = getattr(og, "device_manager", None)
        shared = getattr(device_manager, "capture_method", None) if device_manager is not None else None
        if shared is None:
            return fallback_capture, None
        hwnd_window = getattr(shared, "hwnd_window", None)
        if hwnd_window is None:
            return fallback_capture, None

        from ok.device.capture_methods.windows_graphics import WindowsGraphicsCaptureMethod
        if isinstance(shared, WindowsGraphicsCaptureMethod):  # WGC：建独立 frame pool/session。
            wgc_inst = WindowsGraphicsCaptureMethod(hwnd_window)
            if not wgc_inst.connected():  # 会话没起来：释放并回退，绝不半开坏会话。
                try:
                    wgc_inst.close()
                except Exception:
                    pass
                return fallback_capture, None

            def _grab_wgc():
                return wgc_inst.get_frame()

            def _release_wgc():
                try:
                    wgc_inst.close()
                except Exception as e:
                    logger.warning(f"Release independent WGC capture failed: {e}. 释放独立 WGC 采集会话失败。")

            return _grab_wgc, _release_wgc

        from ok.device.capture_methods.bitblt import BitBltCaptureMethod, ForegroundBitBltCaptureMethod
        if not isinstance(shared, (BitBltCaptureMethod, ForegroundBitBltCaptureMethod)):
            return fallback_capture, None  # DXGI 等其它方式：不克隆，回退共享。
        inst = type(shared)(hwnd_window)  # 复刻同类 BitBlt 实例：独立 lock/DC。
        inst.hwnd_window = hwnd_window
        inst.exit_event = getattr(shared, "exit_event", None)

        def _grab():
            return inst.get_frame()

        def _release():
            try:
                if isinstance(inst, ForegroundBitBltCaptureMethod):
                    inst.close()
                else:
                    from ok.device.capture_methods.bitblt_utils import clean_up_bitblt
                    for ctx in list(getattr(inst, "contexts", {}).values()):
                        clean_up_bitblt(ctx)
                    clean_up_bitblt(inst)
            except Exception as e:
                logger.warning(f"Release independent capture failed: {e}. 释放独立采集实例失败。")

        return _grab, _release
    except Exception as e:
        logger.warning(f"Independent capture setup failed: {e}. 独立采集构建失败，回退共享采集。")
        return fallback_capture, None


class FramePump:
    """latest-wins 单槽采集线程：后台按目标帧率抓最新帧入槽，消费方等待「新帧」即取。

    与录像器的采集线程各自独享一路采集实例，互不抢帧；处理慢时旧帧被直接覆盖（丢帧保实时）。
    """

    def __init__(self, capture, fps):
        self._capture = capture  # 采集回调（独立克隆实例的 get_frame），返回 BGR 帧或 None。
        self._interval = 1.0 / max(float(fps), 1e-3)  # 抓帧节拍间隔（秒）。
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame = None  # 单槽最新帧。
        self._seq = 0  # 已入槽帧序号，单调递增；消费方凭它识别「比上次交付更新」。
        self._delivered_seq = -1  # 消费方上次交付的帧序号。
        self._running = False
        self._thread = None

    def start(self):
        """拉起采集线程；重复调用只启动一次。"""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LieFramePump", daemon=True)
        self._thread.start()

    def _loop(self):
        """采集线程主体：deadline 调度抓帧入槽；落后于节拍则重置基准，避免恢复后疯狂补采。"""
        next_at = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now < next_at:  # 未到下一拍：睡到节拍点（分片以便及时停机）。
                time.sleep(min(next_at - now, 0.01))
                continue
            next_at = max(next_at + self._interval, now)  # 落后于节拍则重置基准（不补采）。
            try:
                frame = self._capture()
            except Exception:
                frame = None
            if frame is None:
                continue
            with self._new_frame:
                self._frame = frame
                self._seq += 1
                self._new_frame.notify_all()

    def wait_new(self, timeout):
        """阻塞到出现比「上次交付」更新的帧即返回最新帧；超时返回当前（可能为旧/空）帧。"""
        with self._new_frame:
            if self._seq <= self._delivered_seq:
                self._new_frame.wait(timeout)
            frame = self._frame
            self._delivered_seq = self._seq
            return frame

    def stop(self):
        """停机：清运行标志、唤醒可能卡在 wait 的消费线程、回收采集线程。"""
        self._running = False
        with self._new_frame:
            self._new_frame.notify_all()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
