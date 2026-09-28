# 自动重登流程状态机：游戏掉线后在全桌面依次匹配并点击
# 【连接】→【服务区】→【频道】→【开始游戏】，完成重登。
#
# 设计要点：
#   - 采集/点击分两种后端：【连接】用全桌面采集 + pynput 绝对点击（启动器在桌面、游戏窗口可能已关闭）；
#     【服务区/频道/开始游戏】用游戏窗口当前选择框采集 + 窗口内相对坐标点击（模板即窗口客户区尺度，
#     避免把客户区模板匹配到全桌面再点绝对坐标所产生的向上偏移）。
#   - 模板匹配用原生尺度匹配器（DesktopTemplateMatcher），绕开框架按帧宽缩放；全桌面帧很大，
#     匹配统一优先走显卡（CuPy FFT，隐藏式启用不设开关），无显卡/异常时自动降级 CPU。
#   - 每步点击后监控画面推进，卡住超过 STUCK_RETRY_SECONDS 秒自动重试点击；仍不推进才判失败。
#   - 频道/开始游戏等启动器按钮需双击才生效：步骤带连点次数，命中后在同一位置快速连点（默认双击）。
import time  # 导入 time，用于超时计时与步骤间等待。

from pynput.mouse import Button, Controller as MouseController  # 导入 pynput 鼠标控制器，发绝对坐标点击。

from src.autologin.desktop_capture import capture_desktop, virtual_screen_origin  # 导入全桌面截图工具与虚拟屏原点换算。
from src.autologin.desktop_matcher import DesktopTemplateMatcher  # 导入原生尺度模板匹配器。

# 固定模板名（用户在模板页标注时必须使用这些分类名）。
TEMPLATE_CONNECT = "连接"          # 启动器【连接】按钮。
TEMPLATE_START_GAME = "开始游戏"   # 启动器【开始游戏】按钮。

# 流程步骤定义：(模板名或None, 配置键, 步骤描述)。
# 模板名为 None 时表示从配置动态读取（服务区/频道由 GUI 下拉选择）。
STEP_TIMEOUT_DEFAULT = 30.0   # 每步默认超时秒数。
STEP_CLICK_WAIT = 1.5         # 点击后等待 UI 响应的秒数。
POLL_INTERVAL = 0.5           # 轮询匹配间隔秒数。
MAX_RETRY = 1                 # 每步超时后最大重试次数。
BACKEND_DESKTOP = "desktop"   # 桌面后端：全桌面采集 + pynput 绝对坐标点击（仅【连接】）。
BACKEND_WINDOW = "window"     # 窗口后端：游戏窗口选择框采集 + 窗口内相对坐标点击（服务区/频道/开始游戏）。
STUCK_RETRY_SECONDS = 5.0     # 点击后超过该秒数画面未推进视为卡住，自动重试点击。
MAX_RECLICK = 2               # 卡住后最多追加重试点击次数。
DOUBLE_CLICK_GAP = 0.05       # 连点（双击）两次点击之间的间隔秒数，需小于系统双击判定窗口（约0.5s）。


class AutoLoginFlow:
    """全桌面自动重登序列：连接 → 服务区 → 频道 → 开始游戏。"""

    def __init__(self, coco_json, config, logger=None, game_frame_fn=None, window_click_fn=None):
        """构造重登流程。

        Args:
            coco_json: 模板标注文件路径（ok_templates/coco_annotations.json），用于原生尺度匹配。
            config: 看板配置 dict，包含 Auto Login Server Feature / Channel Feature / Threshold / Step Timeout。
            logger: 日志器，None 时静默。
            game_frame_fn: 游戏窗口当前选择框采集回调（返回客户区 BGR 帧），窗口后端使用；None 时退回桌面采集。
            window_click_fn: 窗口内相对坐标点击回调 (x, y, clicks)；clicks 为需要在同一位置连点的次数。
                连点必须由回调一次完成（窗口置前只做一次），否则每次点击都重新置前会把两次按下拉开到
                250ms 以上，启动器只当成两次单击。None 时退回桌面绝对点击。
        """
        self._matcher = DesktopTemplateMatcher(coco_json, logger)  # 原生尺度匹配器：绕开框架按帧宽缩放，显卡加速隐藏式启用（无显卡/异常时自动降级 CPU）。
        self._config = config
        self._logger = logger
        self._mouse = MouseController()  # pynput 鼠标控制器，复用实例避免反复创建。
        self._game_frame_fn = game_frame_fn  # 游戏窗口采集回调（窗口后端）。
        self._window_click_fn = window_click_fn  # 窗口内相对坐标点击回调（窗口后端）。

    def _log(self, message):
        """输出日志（有日志器时）。"""
        if self._logger is not None:
            self._logger.info(message)

    def _warn(self, message):
        """输出警告日志。"""
        if self._logger is not None:
            self._logger.warning(message)

    def _log_diagnostics(self, steps):  # 记录桌面帧/游戏窗口帧尺寸与各步模板原生尺寸，确认“原生尺度匹配、后端坐标空间一致”。
        sizes = []  # 各步模板原生尺寸文本。
        for template_name, _desc, _backend, _reanchor, _clicks in steps:
            if not template_name:  # 未配置的步骤跳过。
                continue
            tsz = self._matcher.template_size(template_name)  # 取原生尺寸 (w, h)。
            sizes.append(f"{template_name}={tsz[0]}x{tsz[1]}" if tsz else f"{template_name}=无模板")
        dframe = capture_desktop(all_screens=True)  # 取一帧全虚拟屏桌面（连接步用桌面后端，覆盖所有显示器）。
        ox, oy = virtual_screen_origin()  # 全屏截图→pynput 坐标原点偏移（多屏时副屏在左为负）。
        dsize = f"{dframe.shape[1]}x{dframe.shape[0]}" if dframe is not None else "None"  # 桌面帧尺寸。
        wsize = "未注入"  # 游戏窗口帧尺寸（服务区/频道/开始游戏步用窗口后端）。
        if self._game_frame_fn is not None:  # 已注入窗口采集回调时取一帧看尺寸。
            try:
                wframe = self._game_frame_fn()
                wsize = f"{wframe.shape[1]}x{wframe.shape[0]}" if wframe is not None else "None"
            except Exception as e:  # 采集异常。
                wsize = f"采集异常:{e}"
        self._log(f"Desktop frame {dsize} (virtual, origin=({ox},{oy})), game window frame {wsize}, native templates: {', '.join(sizes)}. "
                  f"桌面帧 {dsize}（全虚拟屏，原点=({ox},{oy})），游戏窗口帧 {wsize}，原生模板尺寸：{', '.join(sizes)}。")

    def run(self, exit_event):
        """执行完整重登序列，返回是否成功。

        Args:
            exit_event: 进程退出事件（threading.Event），置位时立即中止流程。

        Returns:
            True=全部步骤完成，False=中途失败或被中止。
        """
        threshold = float(self._config.get("Auto Login Threshold") or 0.75)
        step_timeout = float(self._config.get("Auto Login Step Timeout") or STEP_TIMEOUT_DEFAULT)
        server_name = str(self._config.get("Auto Login Server Feature") or "").strip()
        channel_name = str(self._config.get("Auto Login Channel Feature") or "").strip()

        # 构建步骤序列：(模板名, 描述, 采集/点击后端, 重锚模板名, 连点次数)。
        # 连接在桌面启动器上→桌面后端；服务区/频道/开始游戏在游戏窗口选择框内→窗口后端。
        # 重锚模板名：该步轮询超时时先重点它一次再重试（频道面板需点服务区才展开，故频道步重锚=服务区）。
        # 连点次数：频道/开始游戏这类启动器按钮单击只选中不进入，需双击（连点2次）才生效；连接/服务区单击即可。
        steps = [
            (TEMPLATE_CONNECT, "连接 Connect", BACKEND_DESKTOP, None, 1),
            (server_name if server_name else None, "服务区 Server", BACKEND_WINDOW, None, 1),
            (channel_name if channel_name else None, "频道 Channel", BACKEND_WINDOW, server_name or None, 2),
            (TEMPLATE_START_GAME, "开始游戏 Start Game", BACKEND_WINDOW, None, 2),
        ]

        self._log(f"Auto login flow started, {len(steps)} steps, threshold={threshold}, timeout={step_timeout}s. "
                  f"自动重登流程开始，共 {len(steps)} 步。")
        self._log_diagnostics(steps)  # 记录桌面帧与各步模板原生尺寸，确认匹配在原生尺度进行（未被框架缩放）。

        for index, (template_name, description, backend, reanchor, clicks) in enumerate(steps, 1):
            if exit_event.is_set():
                self._warn("Auto login aborted: process exiting. 自动重登中止：进程退出。")
                return False
            if template_name is None:
                self._log(f"Step {index}/{len(steps)} [{description}]: skipped (not configured). "
                          f"步骤 {index}/{len(steps)} [{description}]：未配置，跳过。")
                continue
            # 推进判定用：下一个已配置步骤的模板名与其采集后端。
            next_template = next_backend = None
            for future in steps[index:]:
                if future[0]:
                    next_template, next_backend = future[0], future[2]
                    break
            success = self._execute_step(template_name, description, threshold, step_timeout, exit_event,
                                         index, len(steps), backend, reanchor, next_backend, next_template, clicks)
            if not success:
                return False

        self._log("Auto login flow completed successfully. 自动重登流程全部完成。")
        return True

    def _execute_step(self, template_name, description, threshold, timeout, exit_event, index, total,
                      backend, reanchor=None, next_backend=None, next_template=None, clicks=1):
        """执行单步：轮询匹配 → 连点（双击）→ 监控推进（卡住重试点击）。含超时重试与重锚。"""
        for attempt in range(1 + MAX_RETRY):
            if exit_event.is_set():
                return False
            box = self._poll_template(template_name, threshold, timeout, exit_event, backend)
            if box is None:
                if attempt < MAX_RETRY:
                    self._warn(f"Step {index}/{total} [{description}]: timeout after {timeout}s, retrying ({attempt + 1}/{MAX_RETRY}). "
                               f"步骤 {index}/{total} [{description}]：{timeout}s 超时，重试。")
                    if reanchor:  # 频道等需先展开面板的步骤：超时后重点击锚点（服务区）再重试。
                        self._reanchor(reanchor, threshold, exit_event)
                    continue
                self._warn(f"Step {index}/{total} [{description}]: failed after {1 + MAX_RETRY} attempts, aborting flow. "
                           f"步骤 {index}/{total} [{description}]：重试耗尽，流程中止。")
                return False
            # 命中：连点（双击）并监控推进；卡住超过 STUCK_RETRY_SECONDS 自动重试点击。
            for click_no in range(1 + MAX_RECLICK):
                x, y, w, h, confidence = box  # 整轮重试都用首次命中的这个框，坐标锁定不再刷新（原因见下）。
                cx = x + w // 2
                cy = y + h // 2
                self._log(f"Step {index}/{total} [{description}]: matched at ({cx},{cy}) conf={confidence:.3f}, clicking (#{click_no + 1}, {clicks}x). "
                          f"步骤 {index}/{total} [{description}]：匹配到 ({cx},{cy}) 置信度={confidence:.3f}，点击（第{click_no + 1}轮，连点{clicks}次）。")
                self._click_for(backend, cx, cy, clicks)  # 一次调用完成整串连点：窗口置前只做一次，两次按下紧凑相连。
                if self._advanced(backend, template_name, next_backend, next_template, threshold, exit_event, STUCK_RETRY_SECONDS):
                    return True
                self._warn(f"Step {index}/{total} [{description}]: stuck >{STUCK_RETRY_SECONDS}s after click, retry click (locked at ({cx},{cy})). "
                           f"步骤 {index}/{total} [{description}]：点击后 {STUCK_RETRY_SECONDS}s 画面未推进，在锁定坐标 ({cx},{cy}) 重试点击。")
                # 卡住后不重新匹配：频道被单击选中后会高亮、外观改变，重新匹配的最高分会漂到相邻频道上
                # （实测同一【频道3】模板先在 (722,379) 满分命中，点击后再匹配却落到 (815,410) 的另一个频道），
                # 后续点击就打在别的频道上永远进不去。_advanced 返回 False 已说明当前模板仍在画面上，
                # 所以直接在原坐标重试即可；真正的位置变化交给下一轮重锚 + 重新轮询处理。
            if attempt < MAX_RETRY:
                self._warn(f"Step {index}/{total} [{description}]: not advanced after clicks, retrying. "
                           f"步骤 {index}/{total} [{description}]：多次点击仍未推进，重试。")
                if reanchor:
                    self._reanchor(reanchor, threshold, exit_event)
                continue
            self._warn(f"Step {index}/{total} [{description}]: failed to advance, aborting flow. "
                       f"步骤 {index}/{total} [{description}]：点击后仍未推进，流程中止。")
            return False
        return False

    def _advanced(self, backend, template_name, next_backend, next_template, threshold, exit_event, seconds):
        """点击后监控画面是否推进：下一步模板出现 或 当前模板消失 即视为推进。"""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if exit_event.is_set():
                return True
            frame = self._capture_for(backend)
            if frame is not None:
                cur, cur_conf = self._match(frame, template_name)
                cur_gone = cur is None or cur_conf < threshold
                if next_template:
                    nframe = frame if next_backend == backend else self._capture_for(next_backend)
                    nxt, nxt_conf = self._match(nframe, next_template) if nframe is not None else (None, 0.0)
                    if (nxt is not None and nxt_conf >= threshold) or cur_gone:
                        return True
                elif cur_gone:
                    return True
            time.sleep(POLL_INTERVAL)
        return False

    def _reanchor(self, server_name, threshold, exit_event):
        """频道面板未展开时，重新点击服务区以（重新）展开频道列表，供下一轮轮询匹配。"""
        if not server_name or exit_event.is_set():
            return
        frame = self._capture_for(BACKEND_WINDOW)  # 服务区在游戏窗口选择框内。
        if frame is None:
            return
        box, conf = self._match(frame, server_name)
        if box is None or conf < threshold:
            self._warn(f"Re-anchor: server [{server_name}] not found (best={conf:.3f}), skip. "
                       f"重锚：未找到服务区 [{server_name}]（最高分={conf:.3f}），跳过。")
            return
        x, y, w, h, _conf = box
        cx, cy = x + w // 2, y + h // 2
        self._log(f"Re-anchor: re-click server [{server_name}] at ({cx},{cy}) to reopen channel panel. "
                  f"重锚：重点击服务区 [{server_name}] ({cx},{cy}) 以重新展开频道面板。")
        self._click_for(BACKEND_WINDOW, cx, cy)
        time.sleep(STEP_CLICK_WAIT)  # 等待面板展开动画。

    def _poll_template(self, template_name, threshold, timeout, exit_event, backend):
        """按后端轮询截图直到匹配到指定模板或超时。"""
        deadline = time.time() + timeout
        best_conf = 0.0  # 本轮询窗口内出现过的最高分，超时后写入日志便于判断“差一点”还是“不在画面”。
        while time.time() < deadline:
            if exit_event.is_set():
                return None
            frame = self._capture_for(backend)
            if frame is None:
                time.sleep(POLL_INTERVAL)
                continue
            box, conf = self._match(frame, template_name)
            if conf > best_conf:
                best_conf = conf
            if box is not None and conf >= threshold:
                return box
            time.sleep(POLL_INTERVAL)
        self._warn(f"Poll timeout for [{template_name}]: best score={best_conf:.3f} < threshold={threshold}. "
                   f"[{template_name}] 轮询超时：窗口内最高分={best_conf:.3f}，未达阈值={threshold}。")
        return None

    def _window_backend_active(self, backend):
        """窗口后端是否真正可用：仅当采集与点击回调都已注入时才启用。
        两者必须同时具备，才能保证“匹配坐标空间”与“点击坐标空间”一致（都在游戏窗口客户区）。"""
        return backend == BACKEND_WINDOW and self._game_frame_fn is not None and self._window_click_fn is not None

    def _capture_for(self, backend):
        """按后端取帧，与 _click_for 严格配对。
        窗口后端可用时用游戏窗口选择框采集（客户区尺度，坐标即窗口内相对坐标）；
        此时不回退桌面，避免“桌面帧匹配 + 窗口相对坐标点击”的空间错配（正是服务区点击向上偏移的根因）；
        采集不到帧时返回 None，由轮询等待重试。窗口后端不可用（未注入回调）时才用全桌面采集。"""
        if self._window_backend_active(backend):
            try:
                return self._game_frame_fn()
            except Exception as e:
                self._warn(f"Game window capture failed: {e}. 游戏窗口采集失败，本轮跳过。")
                return None
        return capture_desktop(all_screens=True)  # 截全部显示器：启动器可能在任一屏幕；坐标为虚拟屏图像坐标，点击时加原点偏移。

    def _click_for(self, backend, cx, cy, clicks=1):
        """按后端点击，与 _capture_for 严格配对：窗口后端用窗口内相对坐标点击，桌面后端用 pynput 绝对坐标点击。
        clicks>1 表示要在同一位置连点（双击）。窗口后端把整串连点交给回调一次完成，因为置前必须在连点前只做一次：
        每次点击都重新置前 + 等待，实测会把两次按下拉开到 251ms，启动器只当成两次单击（频道选中了却不进入）。"""
        total = max(1, int(clicks))  # 连点次数，非法值按单击处理。
        if self._window_backend_active(backend):
            try:
                self._window_click_fn(cx, cy, total)
                self._log(f"Window click sent at ({cx},{cy}), {total}x. 已在游戏窗口内 ({cx},{cy}) 发出点击（连点 {total} 次）。")
            except Exception as e:
                self._warn(f"Window click failed at ({cx},{cy}): {e}. 窗口内点击失败。")
            return
        for click_index in range(total):  # 桌面后端：pynput 绝对坐标连点，两次点击之间只留双击间隔。
            if click_index:
                time.sleep(DOUBLE_CLICK_GAP)
            self._click_screen(cx, cy)

    def _match(self, frame, template_name):
        """在给定帧（桌面或窗口客户区）上原生尺度匹配模板，返回 ((x,y,w,h,conf) 或 None, 最高分)。"""
        try:
            return self._matcher.find_best(frame, template_name)
        except Exception as e:
            self._warn(f"Match error for {template_name}: {e}. 模板匹配异常。")
            return None, 0.0

    def _click_screen(self, x, y):
        """使用 pynput 在屏幕绝对坐标处单击。(x, y) 为全屏截图(all_screens)图像坐标，
        需加虚拟屏原点偏移换算成 pynput/SetCursorPos 坐标，多显示器/动态分辨率下才落在正确屏幕。"""
        ox, oy = virtual_screen_origin()  # 全屏截图图像坐标 → pynput 坐标的偏移（副屏在左为负）。
        px, py = x + ox, y + oy  # 换算到 pynput/SetCursorPos 虚拟屏坐标。
        try:
            self._mouse.position = (px, py)  # 移动光标到目标坐标（虚拟屏物理像素）。
            time.sleep(0.05)  # 短暂停留，确保光标到位。
            actual = self._mouse.position  # 回读实际光标位置：验证坐标未被 DPI/多屏偏移。
            self._mouse.press(Button.left)  # 按下左键（显式 press/release，对部分启动器比一次性 click 更可靠）。
            time.sleep(0.05)  # 按下与释放之间留一点间隔，模拟真实点击。
            self._mouse.release(Button.left)  # 释放左键，完成单击。
            if abs(actual[0] - px) > 2 or abs(actual[1] - py) > 2:  # 回读与目标偏差过大：坐标空间不一致。
                self._warn(f"Desktop click coord mismatch: want ({px},{py}) actual ({actual[0]:.0f},{actual[1]:.0f}). "
                           f"点击坐标回读不一致（疑似 DPI/多屏偏移）：目标 ({px},{py})，实际 ({actual[0]:.0f},{actual[1]:.0f})。")
            else:  # 坐标一致：点击已发出。
                self._log(f"Desktop click sent at ({px},{py}) [img({x},{y})+origin({ox},{oy})], cursor confirmed. "
                          f"已在 ({px},{py}) 发出桌面点击（图像({x},{y})+原点({ox},{oy})），光标已到位。")
        except Exception as e:
            self._warn(f"Desktop click failed at ({px},{py}): {e}. 全桌面点击失败。")
