# 独立测谎监控服务：脱离脚本任务单独运行，值守看板配置的【测谎触发】标注。
# 命中触发即：暂停当前脚本任务 -> 自动解测谎（稠密光流+粒子滤波，有 N 卡走 torch CUDA、否则走 CPU，
# 与测谎检验页签同一套解法，无神经网络）-> 结束后恢复任务；
# 无脚本任务运行时也能独立监控与求解。
#
# 扩展：掉线自动重登——复用同一监控循环与截图/模板匹配资源，检测【掉线2】模板命中后
# 暂停任务、处理掉线弹窗、切换全桌面截图执行重登序列（连接→服务区→频道→开始游戏），完成后恢复任务。
#
# 设计要点：
#   - 服务是 app 级后台守护线程，由 Globals(og.my_app) 创建并启动，生命周期与进程一致。
#   - 截图与模板匹配复用框架 device_manager.capture_method 与 executor.feature_set
#     （两者均线程安全：get_frame 内部加锁、FeatureSet 有 self.lock，与任务的并发匹配 MatchBatch 同源）。
#   - 暂停/恢复：读取 executor.current_task，调用 task.pause()/unpause()；暂停后先等任务线程
#     阻塞到 sleep（PAUSE_SETTLE_SECONDS），再释放任务持有的按键（pop_held_keys），
#     避免解测谎期间任务残留按住方向键/攻击键。用户已手动暂停的任务不自动恢复，尊重用户意图。
#   - 配置热更新：每 CONFIG_REFRESH_INTERVAL 秒重读看板配置，DashboardTab 保存时也会显式通知。
#   - 触发防抖与冷却：弹窗淡入/淡出期模板匹配分数会在阈值附近抖动，单帧命中就解题会导致光流会话
#     刚建立（source=waiting、模板未学到）就被“触发消失”打断，表现为第一次能解、重播后反复秒退。
#     故要求连续 LIE_TRIGGER_CONFIRM_FRAMES 帧命中才真正解题；解题中允许连续 LIE_TRIGGER_LOST_TOLERANCE
#     帧丢失才判定结束；一局结束后初始化跨帧状态并进入 LIE_SOLVE_COOLDOWN 秒冷却，等画面稳定后重新等待下一次触发。
#   - 触发延迟：确认命中后不立即解题，先等看板配置的「触发延迟」秒数（默认 5 秒），给弹窗完全展开、
#     图形动画起势留出时间，避免光流会话在弹窗还在淡入/图形尚未移动时建模板。延迟期持续推送框选画面并倒计时，
#     若期间【测谎触发】连续丢失超过容忍帧数（弹窗已关）则放弃本局；延迟结束后重新取帧，不用陈旧画面开题。
#   - 急停按键：看板可配一个键盘按键（默认留空=不装全局钩子），配置后由 pynput 守护监听器全局监听；
#     敲击即置急停标志，解题与延迟等待循环每帧检查后立即退出（录像 outcome=aborted、恢复被暂停的任务），
#     同时把看板「自动解测谎」开关写回为关并通知 UI 同步，用于发现光流跟踪路径跑偏时人为及时接管。
#   - 显卡加速：【测谎触发】【掉线】【掉线2】【掉线确定】的全屏模板匹配可走 CuPy FFT（src/gpu_feature_match.py），
#     结果与框架 CPU 匹配（TM_CCOEFF_NORMED + 灰度 + limit=1）等价，但一帧只上传一次显存、只做一次帧变换即可
#     覆盖全部模板，空闲监控 15FPS 的匹配耗时从数十毫秒降到几毫秒。加速隐藏式启用（不设开关），
#     无 CuPy/无显卡/标注带 mask/运行期异常时自动降级 CPU 路径，行为保持一致。
import math  # 导入标准库 math，用于解测谎鼠标追踪的步长计算。
import os  # 导入标准库 os，用于测谎报警音频的路径解析与存在性检查。
import threading  # 导入标准库 threading，用于后台守护线程。
import time  # 导入标准库 time，用于监控节拍、解题计时与配置刷新计时。

import cv2  # 导入 OpenCV，用于实时画面标注绘制与 CPU 光流引擎可用性探测。
import numpy as np  # 导入 NumPy，用于光流轮廓点集的坐标偏移。

from ok import Logger, TriggerTask, og  # 导入日志器、触发任务类型（暂停时需排除）与全局对象（executor/device_manager/my_app）。

from src.dashboard_store import load_dashboard_config, save_dashboard_config  # 看板共享配置读写：测谎参数以看板为单一数据源，急停时也要回写总开关。
from src.liedetector.recorder import LieRecorder  # 测谎触发录像器：触发->解除全过程录原始帧 + 边车记录，供验证页签复算。
from src.autologin.flow import DOUBLE_CLICK_GAP, AutoLoginFlow  # 导入自动重登流程状态机与双击间隔常量（掉线触发后执行全桌面重登序列）。

try:  # 解测谎依赖可选：稠密光流 + 粒子滤波在线编排，缺失时服务照常运行，仅禁用自动解测谎。
    from src.liedetector.shape_session import (  # 与测谎检验页签同一套光流解法（无神经网络）。
        ShapeTrackParams, ShapeTrackSession, PRECISION_TIER_DEFAULT, PRECISION_TIER_KEYS,
        SOURCE_COLOR, SOURCE_BORDER, SOURCE_INTERPOLATED, SOURCE_PREDICTION, SOURCE_SCENE_ENDED)
    LIE_SOLVER_AVAILABLE = hasattr(cv2, "DISOpticalFlow_create")  # CPU 回退引擎依赖 OpenCV DIS；显卡引擎（torch Farneback）不需要，但降级路径必须可用。
except Exception:  # liedetector 模块缺失或依赖损坏。
    LIE_SOLVER_AVAILABLE = False  # 禁用自动解测谎，运行时日志提示。
    PRECISION_TIER_KEYS = ("low", "medium", "high", "ultra", "extreme")  # 兜底精度档名单：模块缺失时 _read_config 仍能校验配置值不报错。
    PRECISION_TIER_DEFAULT = "extreme"  # 兜底默认档，与 shape_session.PRECISION_TIER_DEFAULT 保持一致。

logger = Logger.get_logger(__name__)  # 服务日志器。

CAPTURE_FPS = 30  # 解测谎取帧固定帧率，与测谎检验页签一致。
CAPTURE_MIN_INTERVAL = 1.0 / CAPTURE_FPS  # 解测谎固定帧间隔秒数（约 0.0333）。
MONITOR_INTERVAL = 1.0 / 15  # 空闲监控节拍（约 15FPS），比解题态更省，降低与任务并发截图/匹配的开销。
LIE_REGION_SHIFT_PIXELS = 4  # 【测谎坐标框】移动/缩放超过该像素数视为新一局（或窗口位移），重置光流会话。
LIE_MOVE_MAX_STEP = 50  # 解测谎时每帧鼠标最多移动的像素数，分步追赶避免光标瞬移过大。
LIE_MAX_TICKS = 750  # 解测谎无结果兜底退出的帧数（约 25 秒 @30FPS），防止触发标注误匹配造成死循环。
LIE_TRIGGER_CONFIRM_FRAMES = 3  # 连续命中【测谎触发】多少帧才认定为真触发，滤掉弹窗淡入期的分数抖动与单帧误匹配。
LIE_TRIGGER_LOST_TOLERANCE = 8  # 解题中允许连续丢失【测谎触发】的帧数，超过才判定测谎结束（约 0.27 秒 @30FPS）。
LIE_SOLVE_COOLDOWN = 2.0  # 一局测谎结束后的冷却秒数：期间不响应新触发，等弹窗完全淡出、画面稳定。
LIE_TRIGGER_DELAY_DEFAULT = 5.0  # 触发延迟默认秒数（看板未配置或值非法时兜底），与 DASHBOARD_DEFAULTS 保持一致。
LIE_TRIGGER_DELAY_MAX = 60.0  # 触发延迟上限秒数，防止配置误填过大导致长时间卡在等待。
LIE_FRAME_WAIT = 0.05  # 解测谎中取不到坐标框时的重试等待秒数。
CONFIG_REFRESH_INTERVAL = 0.5  # 看板配置轮询刷新间隔秒数，保存时另有显式通知立即生效。
PAUSE_SETTLE_SECONDS = 0.2  # 暂停任务后等待其线程阻塞到 sleep 的秒数，再释放持有键避免按键残留。
POST_SOLVE_SLEEP = 0.5  # 解测谎结束后等待弹窗关闭、画面稳定的秒数。
TRIGGER_PROBE_INTERVAL = 2.0  # 未匹配到触发时探测实际最高匹配分数的限频间隔秒数（诊断用，帮助区分弹窗未出现/阈值偏高/模板不符）。

# —— 掉线自动重登常量 ——
DISCONNECT_TEMPLATE = "掉线2"  # 掉线确认标志模板：单独存在即触发重登流程。
DISCONNECT_DIALOG_TEMPLATE = "掉线"  # 掉线弹窗主体模板：与掉线2同时存在时需先点掉线确定。
DISCONNECT_OK_TEMPLATE = "掉线确定"  # 掉线弹窗确定按钮模板。
DISCONNECT_CONFIRM_FRAMES = 3  # 连续命中【掉线2】多少帧才认定为真掉线（防抖）。
DISCONNECT_OK_WAIT = 2.0  # 点击掉线确定后等待弹窗关闭的秒数。
DISCONNECT_GAME_EXIT_TIMEOUT = 15.0  # 点了【掉线确定】后等待游戏窗口关闭的超时秒数（弹窗关闭到窗口退出确实需要几秒）。
DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG = 3.0  # 跳过【掉线】弹窗（场景2）时的短等待秒数：没有确定可点，游戏窗口不会因此关闭，
# 实测这种情况掉线2 会一直挂在画面上，等满 15 秒纯属浪费，只需短等几秒确认窗口状态就进重登。

# —— 解测谎结算常量（仅用于游戏真实触发链路，测谎检验页签的复算验证不走结算）——
LIE_SUCCESS_TEMPLATE = "测谎成功"  # 解测谎结算成功标志模板：触发消失后出现即代表成功。
LIE_SUCCESS_OK_TEMPLATE = "测谎成功确定"  # 【测谎成功】弹窗的确定按钮模板：反复点击直到【测谎成功】消失。
LIE_SETTLE_WAIT = 5.0  # 触发标注消失后，等待【测谎成功】出现的窗口秒数；窗口内未出现即判定失败。
LIE_SETTLE_CONFIRM_TIMEOUT = 15.0  # 点击【测谎成功确定】后等待【测谎成功】消失的安全超时秒数，防死循环。
LIE_SETTLE_CLICK_INTERVAL = 0.4  # 两次点击【测谎成功确定】之间的间隔秒数。

STATE_IDLE = "idle"  # 服务空闲监控态。
STATE_SOLVING = "solving"  # 服务正在解测谎态。
STATE_RELOGGING = "relogging"  # 服务正在执行掉线重登态。

try:  # 复用框框 Pynput 交互的按键名映射（如 lshift -> shift_l），保证急停键与任务按键用同一套命名。
    from ok.device.interaction_methods.pynput import PynputInteraction as _PynputInteraction
    _KEY_NAME_MAP = _PynputInteraction.KEY_MAP  # ok-script 按键名 -> pynput 按键名（只收录需要改名的键）。
except Exception:  # 框框结构变化时降级为空映射，急停键仍按原名比较。
    _KEY_NAME_MAP = {}

# pynput 监听回调报出的键名 -> 规范名：pynput 不区分左右修饰键（Key.shift_l.name 也是 'shift'），
# 且命名风格与看板配置不同（page_up vs pageup），故两侧都归一到同一规范名再比较。
_KEY_NAME_ALIAS = {
    'shift_l': 'shift', 'shift_r': 'shift', 'ctrl_l': 'ctrl', 'ctrl_r': 'ctrl',
    'alt_l': 'alt', 'alt_r': 'alt', 'alt_gr': 'alt', 'cmd_l': 'cmd', 'cmd_r': 'cmd',
    'page_up': 'pageup', 'page_down': 'pagedown', 'caps_lock': 'capslock',
    'num_lock': 'numlock', 'scroll_lock': 'scrolllock', 'print_screen': 'printscreen',
    'return': 'enter',
}


def normalize_key_name(value):  # 把看板配置的按键名或 pynput 报出的按键名归一到同一规范名，供急停键匹配比较。
    name = str(value or '').strip().lower()  # 统一小写并去空格。
    if not name:  # 未配置。
        return ''
    name = _KEY_NAME_MAP.get(name, name)  # 先按框框映射转成 pynput 名（表里没有的键原样保留，如 f8）。
    return _KEY_NAME_ALIAS.get(name, name)  # 再收敛左右修饰键与命名风格差异。


class LieDetectorService:  # 独立测谎监控服务：后台守护线程值守测谎触发，命中即暂停任务并自动解测谎。

    def __init__(self, exit_event):  # 构造服务，exit_event 与 app 退出事件一致，用于优雅停止线程。
        self._exit_event = exit_event  # 进程退出事件，置位后主循环与所有分片等待立即结束。
        self._thread = None  # 后台守护线程句柄，start() 时创建。
        self._config = {}  # 最近一次读取的看板配置缓存。
        self._config_time = 0.0  # 上次读取配置的时间戳，用于按 CONFIG_REFRESH_INTERVAL 轮询刷新。
        self._last_status = None  # 上次记录的状态文本，仅在状态变化时打日志避免刷屏。
        self.state = STATE_IDLE  # 当前服务状态：空闲监控或正在解测谎，供 UI/诊断读取。
        self._last_trigger_probe = 0.0  # 上次探测触发匹配分数的时间戳，用于诊断限频。
        self._trigger_hits = 0  # 连续命中【测谎触发】的帧数，达到 LIE_TRIGGER_CONFIRM_FRAMES 才真正进入解题（防抖）。
        self._cooldown_until = 0.0  # 冷却截止时间戳：一局结束后这段时间内不响应新触发，避免弹窗残留画面重复触发。
        self._disconnect_hits = 0  # 连续命中【掉线2】的帧数，达到 DISCONNECT_CONFIRM_FRAMES 才触发重登（防抖）。
        self._gpu = None  # 显卡匹配器（GpuFeatureMatcher），首次需要时创建，看板开关关闭或显卡异常时为 None。
        self._gpu_off = False  # 显卡加速是否已被运行期异常永久关闭：置位后本进程内不再重试，避免每帧失败刷日志。
        self._watch_names = None  # 本轮值守要在同一帧上匹配的全部分类名，单点调用匹配时沿用同一组模板避免反复重建。
        self._recorder = None  # 当前测谎录像器（LieRecorder）：触发确认时起录、finally 收尾；非解题期为 None。
        self._abort_event = threading.Event()  # 急停标志：敲击看板配置的急停键后置位，解题与延迟等待循环每帧检查并立即退出。
        self._abort_key = ''  # 规范化后的急停按键名，空字符串表示未启用急停（也不装全局键盘钩子）。
        self._abort_listener = None  # pynput 全局键盘监听器，仅在配置了急停键时启动。
        self._abort_notice = False  # 急停已发生、等待看板 UI 把「自动解测谎」开关同步为关的一次性标志。
        self._logged_precision = None  # 上次打过日志的精度档：仅在档位变化时记一次，让日志能直接核对服务当前生效的档位（无需等触发解题）。

    def start(self):  # 启动后台守护线程；重复调用只启动一次。
        if self._thread is not None:  # 已启动过。
            return  # 幂等，直接返回。
        self._thread = threading.Thread(target=self._run, name="LieDetectorService", daemon=True)  # 守护线程随进程退出，不阻塞关闭。
        self._thread.start()  # 启动线程。
        logger.info("Lie detector service started. 独立测谎监控服务已启动。")  # 记录启动。

    def reload_config(self):  # 通知服务立即重读看板配置（DashboardTab 保存时调用）。
        self._config_time = 0.0  # 把刷新时间戳清零，主循环下一轮立即重读配置。

    def _read_config(self):  # 读取并解析看板测谎配置，返回 (自动解开关, 触发标注名, 坐标框标注名, 阈值, 触发延迟秒数, 报警音频, 精度档)。
        config = self._get_config()  # 取（必要时刷新）配置缓存。
        auto_solve = bool(config.get("Lie Detector Auto Solve"))  # 测谎总开关。
        trigger_name = str(config.get("Lie Detector Trigger Feature") or '').strip()  # 【测谎触发】标注分类名。
        region_name = str(config.get("Lie Detector Region Feature") or '').strip()  # 【测谎坐标框】标注分类名。
        try:  # 阈值可能被写成非法值，兜底默认 0.75。
            threshold = float(config.get("Lie Detector Threshold") or 0.75)  # 触发匹配阈值。
        except (TypeError, ValueError):  # 非数字阈值。
            threshold = 0.75  # 回退默认阈值。
        try:  # 延迟可能被写成非法值，兜底默认 5 秒。
            trigger_delay = float(config.get("Lie Detector Trigger Delay", LIE_TRIGGER_DELAY_DEFAULT))  # 触发后延迟多少秒才解题。
        except (TypeError, ValueError):  # 非数字延迟。
            trigger_delay = LIE_TRIGGER_DELAY_DEFAULT  # 回退默认延迟。
        trigger_delay = min(max(trigger_delay, 0.0), LIE_TRIGGER_DELAY_MAX)  # 夹到 [0, 上限]：负数按不延迟处理，过大值截断。
        alarm_sound = str(config.get("Lie Alarm Sound") or '').strip()  # 报警音频路径，留空表示不报警。
        precision = str(config.get("Lie Detector Precision") or PRECISION_TIER_DEFAULT).strip()  # 解测谎精度档 key，缺省按算法层默认档。
        if precision not in PRECISION_TIER_KEYS:  # 非法档位（配置被手改成未知值）。
            precision = PRECISION_TIER_DEFAULT  # 回退默认精度档，与 DASHBOARD_DEFAULTS 一致。
        if precision != self._logged_precision:  # 档位变化（含启动后首次读取）：记一次日志。
            self._logged_precision = precision  # 更新已记录档位。
            logger.info(f"Lie detector precision tier: {precision}. 解测谎精度档生效：{precision}（看板可改；无 N 卡时 high/ultra/extreme 会在装配时回落 medium）。")
        return auto_solve, trigger_name, region_name, threshold, trigger_delay, alarm_sound, precision  # 返回解析后的配置元组。

    def _get_config(self):  # 取看板配置缓存，超过刷新间隔时重读文件。
        now = time.time()  # 当前时间。
        if now - self._config_time >= CONFIG_REFRESH_INTERVAL:  # 到达刷新间隔（或 reload_config 清零后）。
            try:  # 配置文件损坏时沿用旧缓存，不能让服务崩溃。
                self._config = load_dashboard_config()  # 重读 configs/Dashboard.json。
            except Exception as e:  # 读取异常。
                logger.warning(f"Lie service load dashboard config failed: {e}. 测谎服务读取看板配置失败，沿用旧配置。")  # 记录并降级。
            self._config_time = now  # 记录本次刷新时间。
        return self._config  # 返回配置缓存。

    def _warmup_scoring_backend(self):  # 预热测谎后端：在守护线程里触发 torch CUDA 初始化、显存池分配与 cudnn 卷积算法调优，避免解测谎首帧冷启动卡顿（约 1~3 秒）。
        if not LIE_SOLVER_AVAILABLE:  # 光流解法不可用时不会有打分调用，无需预热。
            return  # 跳过。
        try:  # 预热失败不影响正常路径，正式求解时会再走一次后端装配与降级链路。
            from src.liedetector.gpu_shape_backend import warmup_shape_backend  # 延迟导入：无显卡环境不应影响服务启动，torch CUDA 初始化也较重。
            backend_name = warmup_shape_backend()  # 跑一次假数据打分 + 一次证据链，返回 "torch(cuda)/farneback" 或 "numpy/dis"。
            logger.info(f"Lie solve scoring backend ready: {backend_name}. 测谎后端预热完成：{backend_name}（有 N 卡走 torch CUDA + Farneback 光流，否则 NumPy + cv2 DIS）。")  # 记录实际生效的后端与光流引擎，便于核对走的是显卡还是 CPU。
        except Exception as e:  # 预热异常不能拖垮服务。
            logger.warning(f"Lie solve scoring backend warmup failed: {e}. 测谎打分后端预热失败，将在首次求解时重试。")  # 记录异常。

    def _run(self):  # 服务主循环：空闲监控掉线触发与测谎触发，命中则暂停任务并处理，直到进程退出。
        self._warmup_scoring_backend()  # 主循环前预热后端：已在守护线程内，1~3 秒的 torch CUDA 初始化不会阻塞 app 启动。
        while not self._exit_event.is_set():  # 主循环，退出事件置位时结束。
            try:  # 单轮异常不能拖垮服务，捕获后记录并继续下一轮。
                auto_solve, trigger_name, region_name, threshold, trigger_delay, alarm_sound, precision = self._read_config()  # 读取测谎配置。
                self._sync_abort_key()  # 刷新急停按键并按需启停全局键盘监听（配置改动 0.5 秒内生效）。
                auto_login_cfg = self._read_auto_login_config()  # 读取自动登录配置。
                watch_names = [DISCONNECT_TEMPLATE, DISCONNECT_DIALOG_TEMPLATE, DISCONNECT_OK_TEMPLATE]  # 本轮要在同一帧上匹配的全部分类名，一次帧变换全部复用。
                if trigger_name:  # 测谎触发标注已配置。
                    watch_names.append(trigger_name)  # 一并注册，掉线检测采集的这一帧也能直接用来检测测谎。
                self._watch_names = watch_names  # 记录下来：解题/延迟等待/等窗口关闭等单点匹配也注册同一组模板，避免反复重建显卡匹配器。
                frame = None  # 本轮已采集的画面，掉线检测与测谎检测共用同一帧，省掉一次截图。
                handle = None  # 本轮的显卡匹配句柄，None 表示走框架 CPU 匹配。
    
                # —— 掉线检测（优先级高于测谎：游戏都掉了，测谎无意义）——
                if auto_login_cfg['enabled'] and self._feature_ready(DISCONNECT_TEMPLATE):  # 自动登录开启且掉线模板已标注。
                    frame = self._capture()  # 采集一帧游戏窗口画面。
                    if frame is not None:  # 取到画面才做匹配。
                        handle = self._gpu_handle(frame, watch_names) if self._gpu_match_enabled() else None  # 开关开启时准备显卡句柄：一帧只上传一次显存、只做一次帧变换。
                        disconnect_box = self._find_trigger(frame, DISCONNECT_TEMPLATE, auto_login_cfg['threshold'], handle)  # 全屏匹配【掉线2】。
                        if disconnect_box is not None:  # 本帧命中掉线标志。
                            self._disconnect_hits += 1  # 累计连续命中帧数。
                            if self._disconnect_hits >= DISCONNECT_CONFIRM_FRAMES:  # 达到确认帧数，触发重登。
                                self._disconnect_hits = 0  # 清零计数。
                                self._handle_disconnect(frame, auto_login_cfg, handle)  # 执行掉线重登流程（阻塞直到完成），复用本帧显卡句柄匹配掉线弹窗。
                                continue  # 重登完成后跳过本轮测谎检测。
                        else:  # 本帧未命中。
                            self._disconnect_hits = 0  # 命中中断，确认计数清零重新累计。
    
                # —— 测谎检测（原有逻辑不变）——
                if not auto_solve:  # 自动解测谎关闭。
                    self._set_status("auto solve off")  # 记录状态。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待。
                    continue  # 下一轮。
                if not LIE_SOLVER_AVAILABLE:  # 推理模块不可用。
                    self._set_status("solver unavailable")  # 记录状态。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待。
                    continue  # 下一轮。
                if not (trigger_name and region_name and self._feature_ready(trigger_name) and self._feature_ready(region_name)):  # 两个测谎标注都必须在模板页标注过。
                    self._set_status(f"annotation missing: {trigger_name or region_name}")  # 记录缺失的标注名。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待。
                    continue  # 下一轮。
                if time.time() < self._cooldown_until:  # 冷却期内：上一局刚结束，弹窗可能仍在淡出，匹配分数会残留，先不检测触发。
                    self._set_status("cooling down")  # 记录冷却状态。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待到冷却结束。
                    continue  # 下一轮。
                self._set_status("watching")  # 标注就绪且不在冷却期，进入值守状态。
                if frame is None:  # 掉线检测未开启或没采到画面时才另行采集，本轮已采过就直接复用同一帧。
                    frame = self._capture()  # 采集一帧画面。
                    if frame is not None and self._gpu_match_enabled():  # 取到画面且显卡加速开启。
                        handle = self._gpu_handle(frame, watch_names)  # 准备本帧的显卡匹配句柄。
                if frame is None:  # 取不到画面（窗口未就绪等）。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待后重试。
                    continue  # 下一轮。
                trigger_box = self._find_trigger(frame, trigger_name, threshold, handle)  # 全屏匹配【测谎触发】标注。
                if trigger_box is None:  # 未触发测谎。
                    self._trigger_hits = 0  # 命中中断，确认计数清零重新累计。
                    if self._current_task() is None:  # 无脚本任务运行时，服务独立推送监视画面（框选坐标框）。
                        region_box = self._get_region_box(frame, region_name)  # 采集坐标框供画面框选。
                        self._update_vision(self.draw_lie_annotations(frame, None, region_box))  # 推送监视画面；有任务运行时让任务自己推送，避免争用画面通道。
                    self._idle_sleep(MONITOR_INTERVAL)  # 空闲等待。
                    continue  # 下一轮。
                self._trigger_hits += 1  # 本帧命中，累计连续命中帧数。
                if self._trigger_hits < LIE_TRIGGER_CONFIRM_FRAMES:  # 未达确认帧数：可能只是弹窗淡入期的分数抖动或单帧误匹配，继续观察不解题。
                    self._set_status(f"confirming {self._trigger_hits}/{LIE_TRIGGER_CONFIRM_FRAMES}")  # 记录确认进度，便于排查抖动触发。
                    self._update_vision(self.draw_lie_annotations(frame, trigger_box, self._get_region_box(frame, region_name)))  # 确认期也框选标注，直观看到疑似触发位置。
                    self._idle_sleep(MONITOR_INTERVAL)  # 短等后取下一帧继续确认。
                    continue  # 下一轮。
                self._trigger_hits = 0  # 确认通过，清零计数供下一局重新累计。
                self._handle_trigger(frame, trigger_box, trigger_name, region_name, threshold, trigger_delay, alarm_sound, precision)  # 命中触发：暂停任务、延迟等待并解测谎。
            except Exception as e:  # 主循环兜底异常处理。
                logger.warning(f"Lie detector service loop error: {e}. 测谎服务循环异常，已忽略并继续监控。")  # 记录异常并继续。
                self._idle_sleep(MONITOR_INTERVAL)  # 异常后短暂等待，避免热循环刷屏。
        self._stop_abort_listener()  # 主循环退出（进程结束）：停掉急停按键全局监听，避免残留键盘钩子线程。

    def _handle_trigger(self, frame, trigger_box, trigger_name, region_name, threshold, trigger_delay, alarm_sound, precision):  # 处理一次测谎触发：暂停任务 -> 报警 -> 推送标注 -> 延迟等待 -> 解测谎 -> 恢复任务。
        self._abort_event.clear()  # 清掉上一局可能残留的急停标志，本局重新接受急停键控制。
        self.state = STATE_SOLVING  # 切换到解题态。
        self._set_status("solving")  # 记录状态。
        paused_task = self._pause_current_task()  # 暂停当前脚本任务（用户已手动暂停的任务返回 None，解题后不自动恢复）。
        outcome = "abandoned"  # 录像结束原因：默认延迟期放弃；进入解题后由 _solve 返回值覆盖为 solved/timeout/gone。
        try:  # 无论解题成功与否，最终都要恢复被本服务暂停的任务。
            region_box = self._get_region_box(frame, region_name)  # 采集【测谎坐标框】坐标：既作录像裁剪区域，也供画面框选。
            self._start_recorder(frame, trigger_box, region_box, precision)  # 触发确认即起录，覆盖延迟等待与解题全程（触发->解除）。
            self._play_alarm(alarm_sound)  # 播放报警音频（未配置或文件不存在时自动跳过）；先于延迟播放，让用户立即知道测谎已触发。
            self._update_vision(self.draw_lie_annotations(frame, trigger_box, region_box))  # 触发瞬间立即把两个标注框选推送到实时画面。
            if not self._wait_trigger_delay(trigger_box, trigger_name, region_name, threshold, trigger_delay):  # 延迟等待期间弹窗已关或进程退出，本局无需再解。
                return  # 直接返回，finally 以 outcome=abandoned 收尾录像、恢复任务、重置状态并开启冷却。
            fresh = self._capture()  # 延迟结束后重新取一帧，避免拿几秒前的陈旧画面开题。
            if fresh is not None:  # 取到新帧才替换，取不到沿用触发帧。
                frame = fresh  # 更新为最新画面。
            self._set_status("solving")  # 延迟倒计时状态改回解题态，避免日志停在 delaying。
            outcome = self._solve(frame, trigger_name, region_name, threshold, precision)  # 进入解测谎子循环，返回 solved/timeout/gone。
            if outcome in ("gone", "solved") and self._feature_ready(LIE_SUCCESS_TEMPLATE):  # 正常解出（触发消失/场景结束）且【测谎成功】已标注才结算；timeout/aborted 保留原 outcome。
                outcome = self._settle_lie_result(threshold)  # 结算：5s 内看【测谎成功】判成功/失败，成功则点【测谎成功确定】直到成功标注消失。
        finally:  # 解题结束（正常/异常/退出）都要收尾录像、恢复任务与状态。
            self._stop_recorder(outcome)  # 收尾录像并写边车记录（未起录则空转）。
            if paused_task is not None:  # 本服务暂停了任务。
                self._resume_task(paused_task)  # 恢复它。
            self.state = STATE_IDLE  # 切回空闲监控态。
            self._reset_solve_state()  # 初始化跨帧状态并开启冷却，确保下一局从干净状态重新等触发。
            self._set_status("watching")  # 记录状态。
        self._idle_sleep(POST_SOLVE_SLEEP)  # 等弹窗关闭后画面稳定再继续监控（冷却期随后接管）。

    def _start_recorder(self, frame, trigger_box, region_box, precision):  # 触发确认即起录：录「触发->解除」全过程，只录【测谎区域标注】区域，异常吞掉绝不影响解题。
        try:  # 触发分取框上的置信度，写进录像名与边车。
            score = float(getattr(trigger_box, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):  # 分数非法。
            score = 0.0
        crop = None  # 录像裁剪区域 (x,y,w,h)；未采到坐标框时录整帧兜底。
        if region_box is not None:  # 采到【测谎坐标框】：按它裁剪，只录该区域。
            try:  # 框字段可能异常，宽松解析。
                crop = (int(region_box.x), int(region_box.y), int(region_box.width), int(region_box.height))
            except (TypeError, ValueError, AttributeError):  # 框字段非法。
                crop = None  # 退回录整帧。
        try:  # 录像为最佳努力能力，启动失败不得拖垮解测谎。
            recorder = LieRecorder()  # 一局一个录像器实例。
            recorder.start(frame.shape[:2], {"score": score, "tier": precision}, crop, capture=self._capture)  # 起流：分辨率取触发帧，元数据带触发分与精度档，裁剪区域取【测谎坐标框】；传 capture 让录像器起独立 30FPS 采集线程（与被光流拖慢的解题循环解耦，录像为真 30FPS）。
            self._recorder = recorder  # 记录：录像器自采集线程按 30FPS 拓帧覆盖触发->解除->结算全程，finally 收尾。
        except Exception as e:  # 录像启动异常。
            logger.warning(f"Lie record start failed: {e}. 测谎录像启动失败，本局不录，解题照常进行。")
            self._recorder = None

    def _stop_recorder(self, outcome):  # 收尾录像并写边车记录，异常吞掉；未起录时空转。
        recorder = self._recorder  # 取当前录像器。
        self._recorder = None  # 先摘引用，避免重复收尾或收尾后仍被写帧。
        if recorder is None:  # 本局未起录（录像不可用或启动失败）。
            return
        try:  # 收尾异常不影响主流程。
            recorder.stop(outcome)  # 排空写线程、关流、写边车、滚动保留。
        except Exception as e:  # 收尾异常。
            logger.warning(f"Lie record stop failed: {e}. 测谎录像收尾失败。")

    # ------------------------------------------------------------------ 掉线自动重登

    def _read_auto_login_config(self):  # 读取看板自动登录配置，返回 dict。
        config = self._get_config()  # 取（必要时刷新）配置缓存。
        enabled = bool(config.get("Auto Login Enabled"))  # 自动登录开关。
        try:  # 阈值可能被写成非法值，兜底默认 0.75。
            threshold = float(config.get("Auto Login Threshold") or 0.75)
        except (TypeError, ValueError):
            threshold = 0.75
        return {  # 返回解析后的配置字典。
            'enabled': enabled,
            'threshold': threshold,
            'server': str(config.get("Auto Login Server Feature") or '').strip(),
            'channel': str(config.get("Auto Login Channel Feature") or '').strip(),
            'step_timeout': float(config.get("Auto Login Step Timeout") or 30.0),
            'raw': config,  # 保留原始配置供 AutoLoginFlow 直接读取。
        }

    def _handle_disconnect(self, frame, auto_login_cfg, handle=None):  # 处理掉线触发：暂停任务 → 点击掉线确定 → 等待游戏退出 → 全桌面重登 → 恢复任务。handle 为触发帧的显卡匹配句柄。
        self.state = STATE_RELOGGING  # 切换到重登态，测谎检测自动挂起。
        self._set_status("relogging")  # 记录状态。
        logger.info("Disconnect detected, starting auto relogin. 检测到掉线，开始自动重登流程。")  # 记录触发。
        paused_task = self._pause_current_task()  # 暂停当前脚本任务并释放持有键。
        try:  # 无论重登成功与否，最终都要恢复被本服务暂停的任务。
            # Phase1：游戏窗口内处理掉线弹窗——如果【掉线】+【掉线确定】还在，点击确定关闭弹窗。
            clicked_ok = self._click_disconnect_ok(frame, auto_login_cfg['threshold'], handle)  # 复用触发帧与其显卡句柄，同一帧上连查【掉线】【掉线确定】；返回是否真的点了确定。
            # Phase2：等待游戏窗口关闭（掉线2消失或截图失败），超时后也继续执行重登。
            # 超时长短看上一阶段是否点了确定：点了确定才需要给弹窗关闭+窗口退出留足 15 秒；
            # 跳过【掉线】弹窗时根本没有可点的确定，窗口不会自己关，只短等 3 秒就进重登。
            exit_timeout = DISCONNECT_GAME_EXIT_TIMEOUT if clicked_ok else DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG
            self._wait_game_exit(auto_login_cfg['threshold'], exit_timeout)
            # Phase3：重登序列（连接→服务区→频道→开始游戏）。
            # 采集/点击分两种后端：连接用全桌面采集+pynput绝对点击（启动器在桌面、游戏窗口可能已关闭）；
            # 服务区/频道/开始游戏用游戏窗口当前选择框采集(self._capture)+窗口内相对坐标点击(self._click_in_window)，
            # 模板即窗口客户区尺度，匹配坐标与点击坐标同处客户区空间，避免桌面绝对点击的向上偏移。
            flow = AutoLoginFlow(self._coco_json_path(), auto_login_cfg['raw'], logger,
                                 game_frame_fn=self._capture, window_click_fn=self._click_in_window)  # 构建重登流程：注入窗口后端采集/点击回调。
            success = flow.run(self._exit_event)  # 执行重登序列（阻塞直到完成或失败）。
            if success:  # 重登成功。
                logger.info("Auto relogin completed successfully. 自动重登成功完成。")  # 记录成功。
            else:  # 重登失败。
                logger.warning("Auto relogin failed or aborted. 自动重登失败或被中止。")  # 记录失败。
        except Exception as e:  # 重登流程异常不能拖垮服务。
            logger.warning(f"Auto relogin error: {e}. 自动重登流程异常：{e}。")  # 记录异常。
        finally:  # 重登结束（正常/异常/退出）都要恢复任务与状态。
            if paused_task is not None:  # 本服务暂停了任务。
                self._resume_task(paused_task)  # 恢复它。
            self.state = STATE_IDLE  # 切回空闲监控态。
            self._set_status("watching")  # 记录状态。
        self._idle_sleep(POST_SOLVE_SLEEP)  # 重登完成后短暂等待画面稳定。

    def _click_disconnect_ok(self, frame, threshold, handle=None):  # Phase1：检测游戏窗口内是否还有【掉线】+【掉线确定】，有则点击确定关闭弹窗。返回是否点击了确定（供调用方决定等多久）。
        if not self._feature_ready(DISCONNECT_OK_TEMPLATE):  # 掉线确定模板未标注，跳过。
            return False  # 无法点击，直接进入下一阶段。
        dialog_box = self._find_trigger(frame, DISCONNECT_DIALOG_TEMPLATE, threshold, handle)  # 匹配【掉线】弹窗主体。
        ok_box = self._find_trigger(frame, DISCONNECT_OK_TEMPLATE, threshold, handle)  # 匹配【掉线确定】按钮。
        if dialog_box is not None and ok_box is not None:  # 两者同时存在（场景1）：点击确定关闭弹窗。
            cx = ok_box.x + ok_box.width // 2  # 计算确定按钮中心坐标。
            cy = ok_box.y + ok_box.height // 2
            logger.info(f"Clicking disconnect OK at ({cx},{cy}). 点击掉线确定按钮 ({cx},{cy})。")  # 记录点击。
            self._click_in_window(cx, cy)  # 通过框架 interaction 点击游戏窗口内坐标（确定按钮单击即可）。
            self._idle_sleep(DISCONNECT_OK_WAIT)  # 等待弹窗关闭动画。
            return True  # 已点击确定，调用方需给窗口退出留足超时。
        # 场景2：只有掉线2、没有掉线弹窗，无需也没有确定可点。
        logger.info("Disconnect dialog not present (scenario 2), skip OK click. 掉线弹窗不存在（场景2），跳过确定点击。")
        return False  # 未点击，调用方只短等即可。

    def _click_in_window(self, x, y, clicks=1):  # 通过框架输入设备接口点击游戏窗口内坐标（窗口相对坐标）；clicks>1 时在同一位置紧凑连点（双击）。
        self._ensure_in_front()  # 先把游戏/启动器窗口置前：interaction 的 clickable() 前台守卫在窗口非前台时会静默跳过点击（表现为“点击没反应”）。
        time.sleep(0.05)  # 等待窗口真正切到前台，再发点击。
        interaction = self._interaction()  # 取输入设备接口。
        if interaction is None:  # 接口不可用。
            logger.warning("Interaction unavailable, cannot click in window. 输入接口不可用，无法点击窗口内坐标。")
            return
        total = max(1, int(clicks))  # 连点次数，非法值按单击处理。
        try:  # 点击异常不能中断重登流程。
            for index in range(total):
                if index:  # 连点的第 2 次起：不再置前、不再重新移光标，只留极短间隔。
                    # 置前本身就要上百毫秒（bring_to_front 可能重新枚举窗口），每次点击都置前实测会把
                    # 两次按下拉开到 251ms，启动器的频道列表只当成两次单击：频道选中高亮了但不进入。
                    time.sleep(DOUBLE_CLICK_GAP)
                else:  # 首次点击前把光标移到目标位置。
                    interaction.move(x, y)
                interaction.click(x, y)  # 执行点击（框架内部会再定位一次并按下/释放）。
        except Exception as e:  # 点击失败。
            logger.warning(f"Click in window failed at ({x},{y}): {e}. 窗口内点击失败。")

    def _wait_game_exit(self, threshold, timeout=DISCONNECT_GAME_EXIT_TIMEOUT):  # Phase2：等待游戏窗口关闭（掉线2消失或截图失败），超时后也继续执行重登。
        logger.info(f"Waiting for game window to close (timeout {timeout}s). 等待游戏窗口关闭（超时 {timeout}s）。")
        deadline = time.time() + timeout  # 超时截止时间。
        while not self._exit_event.is_set() and time.time() < deadline:  # 循环直到超时或进程退出。
            frame = self._capture()  # 尝试采集游戏窗口画面。
            if frame is None:  # 截图失败：游戏窗口已关闭。
                logger.info("Game window closed (capture failed). 游戏窗口已关闭（截图失败）。")
                return  # 立即进入重登阶段。
            # 检查掉线2是否仍在游戏窗口内：消失说明窗口已关闭或场景已切换。
            still_there = self._find_trigger(frame, DISCONNECT_TEMPLATE, threshold)
            if still_there is None:  # 掉线2消失：游戏可能已退出。
                logger.info("Disconnect template gone from game window. 掉线模板已从游戏窗口消失。")
                return  # 进入重登阶段。
            self._idle_sleep(0.5)  # 每 0.5 秒检查一次。
        logger.warning(f"Game exit wait timeout ({timeout}s), proceeding with relogin anyway. 等待游戏退出超时（{timeout}s），仍然继续执行重登。")

    def _reset_solve_state(self):  # 一局测谎结束后初始化跨帧状态：清空触发确认计数与诊断限频时间戳，并开启冷却期。
        self._trigger_hits = 0  # 触发确认计数清零，下一局重新累计连续命中帧。
        self._last_trigger_probe = 0.0  # 诊断探测限频时间戳清零，下一局立即能打出分数便于排查。
        self._cooldown_until = time.time() + LIE_SOLVE_COOLDOWN  # 冷却截止时间：这段时间内主循环不检测触发，等弹窗完全淡出、画面稳定。
        logger.info(f"Lie detector state reset, cooldown {LIE_SOLVE_COOLDOWN}s. 测谎状态已初始化，冷却 {LIE_SOLVE_COOLDOWN} 秒后重新等待下一次触发。")  # 记录重置，便于核对冷却是否生效。

    # ------------------------------------------------------------------ 解测谎急停按键

    def _sync_abort_key(self):  # 每轮主循环刷新急停按键：配置变化时按需启停 pynput 全局监听（改动 0.5 秒内生效）。
        config = self._get_config()  # 取（必要时刷新）配置缓存。
        raw = str(config.get("Lie Detector Abort Key") or '').strip()  # 看板配置的急停按键原文，留空表示不启用。
        normalized = normalize_key_name(raw)  # 归一到规范名，与 pynput 报出的键名同一口径比较。
        if normalized == self._abort_key:  # 按键未变化。
            return  # 无需启停监听。
        self._abort_key = normalized  # 记录新的规范急停键（空串=停用）。
        if normalized:  # 配置了急停键：启动全局监听。
            self._start_abort_listener()  # 装 pynput 键盘钩子。
            logger.info(f"Lie abort hotkey armed: {raw}. 解测谎急停按键已启用：{raw}（敲击即中止本次解测谎并关闭自动解测谎）。")  # 记录启用。
        else:  # 清空了急停键：停用监听。
            self._stop_abort_listener()  # 卸 pynput 键盘钩子。
            logger.info("Lie abort hotkey disabled. 解测谎急停按键已停用（看板配置留空）。")  # 记录停用。

    def _start_abort_listener(self):  # 启动 pynput 全局键盘监听（守护线程）；先停旧监听避免重复挂钩，pynput 不可用时降级停用。
        self._stop_abort_listener()  # 停掉可能存在的旧监听器。
        try:  # pynput 为可选依赖，缺失时急停按键不可用，但服务其余功能照常。
            from pynput import keyboard  # 延迟导入：无 pynput 环境不应影响服务导入与启动。
        except Exception as e:  # pynput 不可用。
            logger.warning(f"pynput unavailable, abort hotkey disabled: {e}. pynput 不可用，解测谎急停按键无法启用。")  # 记录降级。
            self._abort_key = ''  # 标记停用，避免每轮重试导入。
            return  # 不装监听。
        try:  # 监听器启动失败（权限/平台限制）不能拖垮服务。
            listener = keyboard.Listener(on_press=self._on_abort_key_press)  # 全局按键回调。
            listener.daemon = True  # 守护线程，随进程退出不阻塞关闭。
            listener.start()  # 启动监听线程。
            self._abort_listener = listener  # 记录句柄供停用时回收。
        except Exception as e:  # 启动异常。
            logger.warning(f"Lie abort listener start failed: {e}. 解测谎急停按键监听启动失败。")  # 记录异常。
            self._abort_listener = None  # 无监听器。

    def _stop_abort_listener(self):  # 停掉当前的 pynput 全局键盘监听（若有），异常吞掉。
        listener = self._abort_listener  # 取当前监听器。
        self._abort_listener = None  # 先摘引用，避免重复停。
        if listener is None:  # 本就没有监听器。
            return  # 空转。
        try:  # 停止异常不影响主流程。
            listener.stop()  # 通知 pynput 结束监听线程。
        except Exception:  # 停止失败（已停等）。
            pass  # 忽略。

    def _on_abort_key_press(self, key):  # pynput 全局按键回调：命中配置的急停键则触发急停（运行在监听线程，务必轻量并吞异常）。
        if not self._abort_key:  # 未配置急停键（监听器本不应处于活动态）。
            return  # 忽略。
        try:  # 解析按键名：字符键取 char，功能/修饰键取 name（pynput 对二者用不同类型表示）。
            name = getattr(key, 'char', None) or getattr(key, 'name', '') or ''
        except Exception:  # 异常键对象。
            return  # 忽略。
        if not name:  # 无法识别的按键。
            return  # 忽略。
        if normalize_key_name(name) != self._abort_key:  # 非配置的急停键。
            return  # 忽略。
        self._trigger_abort()  # 命中：执行急停动作。

    def _trigger_abort(self):  # 急停动作：置中止标志让解题/延迟循环立即退出，并把「自动解测谎」开关写回为关、通知 UI 同步。
        if self._abort_event.is_set():  # 本局急停已生效（按键长按连发会重复回调），忽略后续重复触发。
            return  # 幂等。
        self._abort_event.set()  # 置急停标志：_solve 与 _wait_trigger_delay 每帧检查后立即退出。
        self._abort_notice = True  # 置一次性通知标志，供看板 refresh 把开关同步为关并提示。
        try:  # 关闭总开关落盘失败也要让本次中止生效（标志已置位），故异常仅记录。
            config = load_dashboard_config()  # 重新读盘，只改总开关，避免覆盖用户其它未保存改动。
            config['Lie Detector Auto Solve'] = False  # 「自动解测谎」同步为关。
            save_dashboard_config(config)  # 落盘 configs/Dashboard.json。
        except Exception as e:  # 写盘异常。
            logger.warning(f"Lie abort write config failed: {e}. 急停时关闭自动解测谎开关落盘失败。")  # 记录异常。
        self.reload_config()  # 让主循环下一轮立即重读配置（总开关已关），无需等 0.5 秒轮询。
        logger.warning("Lie solve ABORTED by hotkey, auto-solve switched OFF. 已按急停按键中止解测谎，「自动解测谎」开关已同步关闭。")  # 记录急停，便于核对是谁中止的。

    def consume_abort_notice(self):  # 供看板 UI 轮询：取走一次性急停通知（取后即清），返回是否需要把开关同步为关。
        if self._abort_notice:  # 有未消费的急停通知。
            self._abort_notice = False  # 清除，保证只同步一次。
            return True  # 通知 UI。
        return False  # 无通知。

    def _wait_trigger_delay(self, trigger_box, trigger_name, region_name, threshold, delay):  # 触发确认后的延迟等待：等弹窗完全展开、图形动画起势再解题，返回是否应继续解题。
        if delay <= 0:  # 未配置延迟（或配为 0）。
            return True  # 立即解题，与原有行为一致。
        logger.info(f"Lie trigger matched, wait {delay}s before solving. 已匹配到测谎触发，延迟 {delay} 秒后开始解测谎。")  # 记录延迟起点，便于核对实际等待时长。
        deadline = time.time() + delay  # 延迟结束时间点。
        lost_ticks = 0  # 连续未匹配到【测谎触发】的帧数，容忍值与解题态一致。
        while not self._exit_event.is_set():  # 循环直到延迟结束或进程退出。
            if self._abort_event.is_set():  # 用户在延迟等待期敲了急停键：本局直接放弃。
                logger.info("Lie solve aborted by hotkey during delay, skip solving. 延迟等待期间收到急停按键，已放弃本局解测谎。")
                return False  # 放弃解题，由 _handle_trigger 的 finally 收尾录像、恢复任务并开启冷却。
            remaining = deadline - time.time()  # 剩余等待秒数。
            if remaining <= 0:  # 延迟已到。
                return True  # 继续解题。
            self._set_status(f"delaying {math.ceil(remaining)}s")  # 倒计时按整秒变化，既能在日志看到进度又不会每帧刷屏。
            frame = self._capture()  # 延迟期也持续取帧，保证实时画面不冻结。
            if frame is not None:  # 取到画面才做复检与推送。
                # 延迟期录像由录像器自采集线程按真 30FPS 覆盖「触发->解除」全过程（见 _start_recorder 的 capture=self._capture），此处不再逐帧写。
                found = self._find_trigger(frame, trigger_name, threshold)  # 复检触发标注是否仍在画面上。
                if found is None:  # 本帧未命中：可能只是弹窗拖动或分数抖动造成的瞬时丢失，先容忍。
                    lost_ticks += 1  # 累计连续丢失帧数。
                    if lost_ticks > LIE_TRIGGER_LOST_TOLERANCE:  # 连续多帧丢失说明弹窗已关闭，本局无需再解。
                        logger.info(f"Lie trigger gone during {delay}s delay after {lost_ticks} lost frames, skip solving. 延迟等待期间【测谎触发】连续 {lost_ticks} 帧消失，弹窗已关闭，放弃本局解测谎。")
                        return False  # 放弃解题。
                else:  # 触发仍在。
                    lost_ticks = 0  # 丢失计数清零。
                    trigger_box = found  # 刷新触发框，瞬时丢失时沿用它继续绘制。
                self._update_vision(self.draw_lie_annotations(frame, trigger_box, self._get_region_box(frame, region_name)))  # 延迟期持续框选标注，直观看到等待中的画面。
            self._idle_sleep(min(CAPTURE_MIN_INTERVAL, remaining))  # 按解题节拍短等，不超过剩余延迟，同时保证退出可及时响应。
        return False  # 进程退出，不再解题。

    def _pause_current_task(self):  # 暂停当前脚本任务并释放其持有键，返回被暂停的任务（无需恢复时返回 None）。
        task = self._current_task()  # 取当前可暂停的脚本任务。
        if task is None:  # 无脚本任务运行。
            return None  # 独立解题，无需暂停/恢复。
        if getattr(task, "paused", False):  # 任务已被用户手动暂停。
            return None  # 解题后不自动恢复，尊重用户的暂停意图。
        try:  # 暂停可能因 current_task 变化等原因失败。
            task.pause()  # 暂停任务：置 _paused，任务线程会在下一次 sleep 处阻塞。
        except Exception as e:  # 暂停异常。
            logger.warning(f"Lie service pause task failed: {e}. 测谎服务暂停脚本任务失败，将独立解题。")  # 记录并降级为独立解题。
            return None  # 不恢复。
        self._idle_sleep(PAUSE_SETTLE_SECONDS)  # 等任务线程阻塞到 sleep，确保它不会再发按键。
        self._release_task_keys(task)  # 释放任务持有的方向键/攻击键，避免解题期间按键残留。
        logger.info(f"Task paused for lie detector: {getattr(task, 'name', task)}. 已暂停脚本任务进入解测谎。")  # 记录暂停。
        return task  # 返回被暂停的任务，解题后恢复。

    def _release_task_keys(self, task):  # 释放任务当前持有的按键：任务负责上报持有键，服务负责实际松键 IO。
        interaction = self._interaction()  # 取输入设备接口。
        if interaction is None:  # 输入接口不可用。
            return  # 无法松键。
        pop = getattr(task, "pop_held_keys", None)  # 任务的持有键上报接口（纯属性操作，线程安全）。
        if not callable(pop):  # 任务未实现该接口。
            return  # 无键可松。
        try:  # 上报接口异常不能影响解题。
            keys = pop() or []  # 取并清空任务持有键列表。
        except Exception:  # 上报异常。
            keys = []  # 视为无持有键。
        for key in keys:  # 逐个松开。
            try:  # 单个键松失败不影响其余键。
                interaction.send_key_up(key)  # 直接发松键事件。
            except Exception as e:  # 松键异常。
                logger.warning(f"Lie service release key {key} failed: {e}. 测谎服务松开按键失败：{key}。")  # 记录异常。

    def _resume_task(self, task):  # 恢复被本服务暂停的脚本任务。
        try:  # 恢复可能因任务已停止等原因失败。
            if getattr(task, "_enabled", False):  # 任务仍处于启用状态才恢复，已停止的任务不动它。
                task.unpause()  # 恢复任务：清 _paused 并唤醒执行器线程。
                logger.info(f"Task resumed after lie detector: {getattr(task, 'name', task)}. 解测谎结束，已恢复脚本任务。")  # 记录恢复。
        except Exception as e:  # 恢复异常。
            logger.warning(f"Lie service resume task failed: {e}. 测谎服务恢复脚本任务失败。")  # 记录异常。

    def _current_task(self):  # 取当前可被本服务暂停的脚本任务（排除触发任务与未启用任务），无则返回 None。
        executor = getattr(og, "executor", None)  # 取执行器。
        if executor is None:  # 执行器尚未就绪。
            return None  # 无任务。
        task = getattr(executor, "current_task", None)  # 当前正在运行的一次性任务。
        if task is None:  # 无任务运行。
            return None  # 返回空。
        if isinstance(task, TriggerTask):  # 触发任务的暂停语义不同（全局暂停），不由本服务管理。
            return None  # 排除。
        if not getattr(task, "_enabled", False):  # 任务已停用。
            return None  # 排除。
        return task  # 返回可暂停的脚本任务。

    def _solve(self, first_frame, trigger_name, region_name, threshold, precision_tier=PRECISION_TIER_DEFAULT):  # 解测谎子循环：稠密光流+粒子滤波跟踪透明图形并移动光标，直到【测谎触发】标注消失或场景结束。
        self._ensure_in_front()  # 游戏窗口置顶：鼠标追踪依赖前台窗口接收鼠标事件。
        session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=precision_tier), logger=None)  # 光流粒子滤波在线会话，与测谎检验页签同一套算法（无神经网络）；按看板精度档装配。
        fps = float(CAPTURE_FPS)  # 采集帧率，喂给会话换算时间阈值（场景结束/淡出判定）。
        region = None  # 当前有效的【测谎坐标框】区域 (x, y, w, h)，未采集到前保持 None。
        mouse_pos = None  # 上一帧鼠标目标点（画面坐标），用于分步平滑追赶。
        frame = first_frame  # 从触发帧开始求解。
        tick = 0  # 已处理帧数，用于兜底超时与诊断限频。
        last_diag_time = 0.0  # 上次诊断日志时间。
        trigger_box = None  # 最近一次命中的【测谎触发】框，丢失容忍期内沿用它继续绘制。
        lost_ticks = 0  # 连续未匹配到【测谎触发】的帧数，超过容忍值才判定测谎真正结束。
        outcome = "timeout"  # 录像结束原因，默认兜底超时（含进程退出中断）；命中三个 break 出口时分别改写为 timeout/gone/solved。
        while not self._exit_event.is_set():  # 循环直到触发标注持续消失、场景结束、兜底超时、用户急停或进程退出。
            if self._abort_event.is_set():  # 用户敲了急停键：立即中止解题（跟踪路径不对时人为及时接管）。
                logger.info(f"Lie solve aborted by hotkey at tick {tick}, resume. 第 {tick} 帧收到急停按键，已立即中止解测谎并退回监控。")
                outcome = "aborted"  # 录像标记为用户急停中止。
                self._set_status("aborted by hotkey")  # 状态立即可见，便于核对是谁中止的。
                break  # 退出子循环，finally 会恢复被暂停的任务并开启冷却。
            tick += 1  # 帧计数累加。
            if tick > LIE_MAX_TICKS:  # 长时间未结束，可能触发标注误匹配，退回监控。
                logger.warning(f"Lie solve exceeded {LIE_MAX_TICKS} frames, resume. 解测谎超过 {LIE_MAX_TICKS} 帧未结束，退回监控。")  # 记录兜底退出。
                outcome = "timeout"  # 录像标记为兜底超时。
                break  # 退出子循环。
            found = self._find_trigger(frame, trigger_name, threshold)  # 本帧匹配【测谎触发】标注。
            if found is None:  # 本帧未匹配到：可能只是弹窗淡出期或分数抖动造成的瞬时丢失，先容忍。
                lost_ticks += 1  # 累计连续丢失帧数。
                if lost_ticks > LIE_TRIGGER_LOST_TOLERANCE:  # 连续多帧都丢失才认定测谎结束，避免抖动导致光流会话刚建立就被打断。
                    logger.info(f"Lie detector finished after {lost_ticks} lost frames, resume. 【测谎触发】标注连续 {lost_ticks} 帧消失，测谎已结束，解除测谎状态。")  # 记录退出原因与丢失帧数。
                    outcome = "gone"  # 录像标记为触发消失（正常解除）。
                    break  # 退出子循环。
            else:  # 本帧命中，触发仍在页面上。
                lost_ticks = 0  # 丢失计数清零。
                trigger_box = found  # 记录最新触发框供画面绘制。
            region_box = self._get_region_box(frame, region_name)  # 直接采集【测谎坐标框】标注记录的坐标框信息，不做模板匹配。
            new_region = (region_box.x, region_box.y, region_box.width, region_box.height) if region_box is not None else None  # 转为四元组，未采集到时为 None。
            if new_region is None:  # 本帧未采集到坐标框：沿用旧区域，从未有过则跳过求解。
                if region is None:  # 从未采集到坐标框。
                    self._idle_sleep(LIE_FRAME_WAIT)  # 短等后重试。
                    new_frame = self._capture()  # 取新帧。
                    if new_frame is not None:  # 取到新帧才替换；numpy 数组不能用 or 判真值，必须显式判 None。
                        frame = new_frame  # 更新当前帧，取不到沿用旧帧。
                    continue  # 进入下一轮检查。
            elif region is None or any(abs(new_region[i] - region[i]) > LIE_REGION_SHIFT_PIXELS for i in range(4)):  # 区域首次出现或位置/尺寸明显变化（任一边超阈值）。
                region = new_region  # 更新区域。
                session.reset(region[2], region[3])  # 重置光流会话：清空光流历史、模板与粒子跟踪器，避免上一局污染。
                mouse_pos = None  # 鼠标目标点重新校准。
            else:  # 区域位置稳定。
                region = new_region  # 刷新区域（尺寸可能微调）。
            # 录像由录像器的独立采集线程按真 30FPS 抓帧（见 _start_recorder 的 capture=self._capture），解题循环不再逐帧写，避免与自采集重复喂帧。
            crop = frame[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]  # 裁出谎言检测图形区域。
            result = session.update(crop, fps)  # 喂入光流粒子滤波会话，返回单帧跟踪结果（区域局部坐标）。
            if result.tracker_alive and result.center is not None:  # 有有效跟踪输出时才移动光标。
                abs_center = (region[0] + float(result.center[0]), region[1] + float(result.center[1]))  # 目标中心由区域局部坐标换算为画面绝对坐标。
                mouse_pos = self._move_mouse_toward(mouse_pos, abs_center)  # 分步向目标中心移动鼠标，模拟人眼平滑追踪。
            if time.time() - last_diag_time >= 1.0:  # 每秒限频输出一次诊断日志。
                last_diag_time = time.time()  # 记录本次诊断时间。
                center = result.center  # 目标中心（区域局部坐标）。
                center_desc = f"({center[0]:.1f},{center[1]:.1f})" if center is not None else "-"  # 中心描述。
                logger.info(f"Lie solve diag: source={result.source} conf={result.confidence:.2f} center={center_desc} cursor={mouse_pos} whites={result.white_candidates} snr={result.border_snr:.2f} tick={tick} 解测谎诊断：光流来源/置信度/目标中心/光标/白色候选数。")  # 供排查跟踪为空等问题。
            self._update_vision(self.draw_shape_overlay(frame, region, result, trigger_box))  # 把光流轮廓、目标中心与触发标注框选推送给 UI。
            if result.source == SOURCE_SCENE_ENDED:  # 场景结束（切场景/结算文字）：测谎已解，退出。
                logger.info("Lie detector scene ended, resume. 光流判定场景结束，测谎已解，解除测谎状态。")  # 记录退出原因。
                outcome = "solved"  # 录像标记为已解出。
                break  # 退出子循环。
            self._idle_sleep(CAPTURE_MIN_INTERVAL)  # 按固定 30FPS 节拍取帧。
            new_frame = self._capture()  # 取最新一帧画面。
            if new_frame is not None:  # 取到新帧才替换，取不到沿用上一帧继续求解。
                frame = new_frame  # 更新当前帧。
        return outcome  # 返回结束原因（solved/timeout/gone），供录像边车记录。

    def _settle_lie_result(self, threshold):  # 解测谎结算：触发标注消失后开 LIE_SETTLE_WAIT 秒窗口，出现【测谎成功】则成功并点确定收尾，窗口内未出现则失败。
        self._set_status("settling")  # 结算态，供日志/诊断区分于解题态。
        logger.info(f"Lie solve done, wait up to {LIE_SETTLE_WAIT}s for success mark. 解测谎完成，等待最多 {LIE_SETTLE_WAIT} 秒看【测谎成功】是否出现。")  # 记录结算起点。
        deadline = time.time() + LIE_SETTLE_WAIT  # 成功标注出现的截止时刻。
        while not self._exit_event.is_set():  # 循环直到出现成功标注/超时/急停/进程退出。
            if self._abort_event.is_set():  # 结算期收到急停按键。
                logger.info("Lie settle aborted by hotkey. 结算期收到急停按键，已中止结算。")
                return "aborted"  # 录像标记为急停。
            frame = self._capture()  # 取一帧画面。
            if frame is not None:  # 取到画面才做匹配与推送。
                # 结算期录像同样由录像器自采集线程按真 30FPS 覆盖“触发->解除->结算”全过程，此处不再逐帧写。
                success_box = self._find_trigger(frame, LIE_SUCCESS_TEMPLATE, threshold)  # 全屏匹配【测谎成功】。
                self._update_vision(self.draw_lie_annotations(frame, success_box, None))  # 把成功标注框选推送到实时画面（未命中不画）。
                if success_box is not None:  # 5s 内出现【测谎成功】：判定成功，进点击确定收尾。
                    logger.info("Lie success mark detected, confirming. 已检测到【测谎成功】，开始点击【测谎成功确定】收尾。")
                    return self._confirm_lie_success(threshold)  # 返回 success/aborted。
            remaining = deadline - time.time()  # 剩余等待秒数。
            if remaining <= 0:  # 窗口已到仍未出现成功标注。
                break  # 退出循环，下方判失败。
            self._idle_sleep(min(MONITOR_INTERVAL, remaining))  # 按监控节拍短等，不超过剩余窗口，保证退出可及时响应。
        logger.warning(f"Lie success not shown within {LIE_SETTLE_WAIT}s, mark FAILURE. {LIE_SETTLE_WAIT} 秒内未出现【测谎成功】，判定测谎失败。")  # 记录失败。
        return "failure"  # 录像标记为失败。

    def _confirm_lie_success(self, threshold):  # 出现【测谎成功】后：反复点击【测谎成功确定】直到【测谎成功】消失，代表结算完成（成功）。
        self._ensure_in_front()  # 游戏窗口置顶，点击才能落到前台窗口。
        deadline = time.time() + LIE_SETTLE_CONFIRM_TIMEOUT  # 安全超时截止时刻，防止成功标注始终不消失时无限点击。
        while not self._exit_event.is_set():  # 循环直到成功标注消失/急停/超时/进程退出。
            if self._abort_event.is_set():  # 确认期收到急停按键。
                logger.info("Lie success confirm aborted by hotkey. 确认期收到急停按键，已中止结算。")
                return "aborted"  # 录像标记为急停。
            frame = self._capture()  # 取一帧画面。
            if frame is None:  # 取不到画面：短等重试。
                self._idle_sleep(MONITOR_INTERVAL)
                if time.time() >= deadline:  # 超时兜底。
                    break
                continue  # 下一拍。
            # 确认期录像同样由录像器自采集线程按真 30FPS 覆盖全过程，此处不再逐帧写。
            success_box = self._find_trigger(frame, LIE_SUCCESS_TEMPLATE, threshold)  # 重新匹配【测谎成功】。
            if success_box is None:  # 【测谎成功】已消失：测谎流程结束（成功）。
                logger.info("Lie success mark gone, settle done (SUCCESS). 【测谎成功】已消失，测谎流程结束（成功）。")
                return "success"  # 录像标记为成功。
            ok_box = self._find_trigger(frame, LIE_SUCCESS_OK_TEMPLATE, threshold)  # 匹配【测谎成功确定】按钮。
            self._update_vision(self.draw_lie_annotations(frame, success_box, ok_box))  # 把成功框与确定框推送到实时画面。
            if ok_box is not None:  # 确定按钮在画面上：点击它（点后成功标注还在则下一拍继续点，即“重试”）。
                cx = ok_box.x + ok_box.width // 2  # 确定按钮中心横坐标。
                cy = ok_box.y + ok_box.height // 2  # 确定按钮中心纵坐标。
                logger.info(f"Clicking lie success OK at ({cx},{cy}). 点击【测谎成功确定】({cx},{cy})。")  # 记录点击。
                self._click_in_window(cx, cy)  # 通过框架输入接口点击窗口内坐标。
            self._idle_sleep(LIE_SETTLE_CLICK_INTERVAL)  # 两次点击间隔，避免疯狂连点。
            if time.time() >= deadline:  # 超时仍未消失。
                break  # 退出循环，下方兜底按成功收尾。
        logger.warning("Lie success confirm timeout, still mark SUCCESS. 点击【测谎成功确定】超时仍未消失，按成功收尾。")  # 记录超时兜底。
        return "success"  # 已出现过成功标注，超时也归为成功。

    def _move_mouse_toward(self, current, target):  # 每帧向预测光标位置分步移动鼠标，返回移动后的位置。
        if current is None:  # 首帧无参照点，直接跳到预测点（游戏内光标也会瞬间到位）。
            position = target  # 目标位置即预测光标点。
        else:  # 已有参照点，按最大步长追赶。
            dx = target[0] - current[0]  # 到预测点的横向位移。
            dy = target[1] - current[1]  # 到预测点的纵向位移。
            distance = math.hypot(dx, dy)  # 直线距离。
            if distance <= LIE_MOVE_MAX_STEP:  # 距离在一个步长内，一步到位。
                position = target  # 直接到达预测点。
            else:  # 距离超过单步上限，按最大步长截断方向向量。
                position = (current[0] + dx / distance * LIE_MOVE_MAX_STEP, current[1] + dy / distance * LIE_MOVE_MAX_STEP)  # 沿目标方向移动一步。
        self._move(int(position[0]), int(position[1]))  # 画面坐标转为鼠标移动事件发给游戏窗口。
        return position  # 返回当前位置，供下一帧作参照。

    def _move(self, x, y):  # 通过输入设备接口移动鼠标到画面坐标 (x, y)。
        interaction = self._interaction()  # 取输入设备接口。
        if interaction is None:  # 接口不可用。
            return  # 无法移动。
        try:  # 移动异常不能中断解题。
            interaction.move(x, y)  # 发鼠标移动事件。
        except Exception as e:  # 移动异常。
            logger.warning(f"Lie service move failed: {e}. 测谎服务移动鼠标失败。")  # 记录异常。

    def _ensure_in_front(self):  # 把游戏窗口置顶，鼠标追踪依赖前台窗口接收鼠标事件。
        device_manager = getattr(og, "device_manager", None)  # 取设备管理器。
        hwnd = getattr(device_manager, "hwnd_window", None) if device_manager is not None else None  # 取目标窗口句柄。
        if hwnd is None:  # 无窗口句柄（未选择窗口等）。
            return  # 跳过置顶。
        try:  # 置顶失败不影响解题。
            hwnd.bring_to_front()  # 请求系统把窗口切到前台。
        except Exception as e:  # 置顶异常。
            logger.warning(f"Lie service ensure_in_front failed: {e}. 测谎服务窗口置顶失败。")  # 记录异常。

    def _play_alarm(self, sound_path):  # 播放测谎报警音频：异步播放不阻塞解测谎，失败只记日志，返回是否已播放。
        if not sound_path:  # 未配置音频：不报警。
            return False  # 静默跳过。
        path = sound_path if os.path.isabs(sound_path) else os.path.join(os.getcwd(), sound_path)  # 相对路径按项目根目录解析。
        if not os.path.exists(path):  # 文件不存在：不报警并提示。
            logger.warning(f"Lie alarm sound not found, skip: {sound_path}. 测谎报警音频不存在，跳过报警：{sound_path}。")  # 提醒检查路径。
            return False  # 未播放。
        try:  # 播放失败也不能拖垮解测谎。
            if path.lower().endswith('.wav'):  # WAV 用系统声音接口直接异步播放。
                import winsound  # Windows 系统声音模块。
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)  # 异步播放，不阻塞服务线程。
            else:  # mp3/wma 等格式走 Windows MCI 解码播放。
                import ctypes  # 调用系统 winmm.dll。
                winmm = ctypes.windll.winmm  # Windows 多媒体 API。
                alias = "lie_alarm"  # 固定设备别名，重复触发时先关旧实例再重开。
                winmm.mciSendStringW(f"close {alias}", None, 0, 0)  # 关闭上一次播放残留，避免占用。
                if winmm.mciSendStringW(f'open "{path}" alias {alias}', None, 0, 0) != 0:  # 自动识别格式打开失败。
                    if winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0) != 0:  # 再按 mpeg 音频显式打开。
                        logger.warning(f"Lie alarm sound open failed: {sound_path}. 测谎报警音频无法打开（格式可能不支持）：{sound_path}。")  # 两种打开方式都失败。
                        return False  # 未播放。
                winmm.mciSendStringW(f"play {alias}", None, 0, 0)  # 异步播放，不等待播完。
            logger.info(f"Lie alarm playing: {sound_path}. 测谎报警音频已播放：{sound_path}。")  # 记录报警供排查。
            return True  # 已播放。
        except Exception as e:  # 播放异常不中断解测谎。
            logger.warning(f"Lie alarm play failed: {e}. 测谎报警播放失败：{e}。")  # 记录异常原因。
            return False  # 未播放。

    def _capture(self):  # 通过框架截图方法采集一帧画面，失败返回 None。
        device_manager = getattr(og, "device_manager", None)  # 取设备管理器。
        method = getattr(device_manager, "capture_method", None) if device_manager is not None else None  # 取截图方法（WGC/BitBlt 等）。
        if method is None:  # 截图方法不可用（未选择窗口等）。
            return None  # 无法截图。
        try:  # 截图异常（窗口失效等）不能让服务崩溃。
            return method.get_frame()  # 取一帧画面（内部加锁，线程安全，WGC 返回副本）。
        except Exception:  # 截图异常。
            return None  # 按取不到画面处理。

    def _interaction(self):  # 取框架输入设备接口（鼠标/键盘），不可用时返回 None。
        device_manager = getattr(og, "device_manager", None)  # 取设备管理器。
        return getattr(device_manager, "interaction", None) if device_manager is not None else None  # 返回输入接口或 None。

    def _feature_set(self):  # 取执行器的特征集（模板标注），不可用时返回 None。
        executor = getattr(og, "executor", None)  # 取执行器。
        return getattr(executor, "feature_set", None) if executor is not None else None  # 返回特征集或 None。

    def _coco_json_path(self):  # 取模板标注文件路径：优先用框架 feature_set 已解析的绝对路径，回退到项目默认路径。
        feature_set = self._feature_set()  # 取特征集。
        path = getattr(feature_set, "coco_json", None) if feature_set is not None else None  # 框架已解析为绝对路径。
        if path:  # 拿到了直接用。
            return path
        return os.path.join(os.getcwd(), 'ok_templates', 'coco_annotations.json')  # 回退：项目根目录默认路径（与 config.template_matching 一致）。

    def _find_trigger(self, frame, feature_name, threshold, handle=None):  # 全屏匹配指定标注，返回置信度最高的框或 None；看板开关开启时优先走显卡匹配。
        if handle is None and self._gpu_match_enabled():  # 调用方未备句柄（解题/延迟等待/等窗口关闭等单点调用），此处自建。
            handle = self._gpu_handle(frame, self._watch_names or [feature_name])  # 沿用本轮值守的同一组模板，避免反复重建。
        if handle is not None and self._gpu is not None and self._gpu.has(feature_name):  # 该分类已注册到显卡匹配器，走显卡路径。
            try:  # 显卡异常（显存不足/驱动重置等）不能中断值守。
                box = self._gpu.best_box(handle, feature_name, threshold)  # 显卡上取最高分框，低于阈值返回 None。
                if box is None and self._probe_due():  # 未达阈值且到了探测间隔：报一次实际最高分供排查。
                    self._log_probe_score(feature_name, threshold, self._gpu.best_score(handle, feature_name))
            except Exception as e:  # 显卡匹配失败。
                self._disable_gpu(e)  # 关闭加速并落到下面的 CPU 路径，本帧仍能得出正确结果。
            else:  # 显卡路径正常完成（命中或未命中）。
                return box  # 直接返回，不再跑 CPU 匹配。
        feature_set = self._feature_set()  # 取特征集。
        if feature_set is None:  # 特征集不可用。
            return None  # 按未匹配处理。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)  # variance=1 表示全屏搜索，测谎弹窗位置不固定；灰度匹配与任务默认一致。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。
        except Exception as e:  # 匹配异常。
            logger.warning(f"Lie service find trigger failed: {e}. 测谎服务匹配触发标注失败。")  # 记录异常。
            return None  # 按未匹配处理。
        if not boxes:  # 无达标匹配框。
            self._probe_trigger_score(feature_set, frame, feature_name, threshold)  # 限频探测实际最高分，供排查弹窗未出现/阈值偏高/模板不符。
            return None  # 未触发。
        return max(boxes, key=lambda b: getattr(b, "confidence", 0))  # 取置信度最高的框，与 find_one 语义一致。

    def _probe_trigger_score(self, feature_set, frame, feature_name, threshold):  # 诊断（CPU 路径）：限频用极低阈值探测实际最高匹配分数并记日志。
        if not self._probe_due():  # 未到探测间隔。
            return  # 跳过，避免每帧多一次匹配拖慢监控。
        try:  # 探测异常不能影响主流程。
            probe = feature_set.find_feature(frame, feature_name, 1, 1, 0.01, True, limit=1)  # 阈值 0.01（非 0，避免框架回退默认 0.95）拿最高分框。
            score = max((getattr(b, "confidence", 0.0) for b in (probe or [])), default=0.0)  # 取实际最高匹配分数。
        except Exception:  # 探测失败静默跳过。
            return  # 不记录。
        self._log_probe_score(feature_name, threshold, score)  # 与显卡路径共用同一条诊断日志文案。

    def _probe_due(self):  # 诊断探测限频判定：到达间隔才返回 True 并刷新时间戳（CPU/GPU 两条路径共用）。
        now = time.time()  # 当前时间。
        if now - self._last_trigger_probe < TRIGGER_PROBE_INTERVAL:  # 限频，避免每帧多一次匹配拖慢监控。
            return False  # 未到探测间隔。
        self._last_trigger_probe = now  # 记录本次探测时间。
        return True  # 本轮应当探测。

    def _log_probe_score(self, feature_name, threshold, score):  # 输出未达阈值的诊断日志：区分弹窗未出现/阈值偏高/模板不符。
        logger.info(f"Lie trigger not matched: {feature_name} best score={score:.3f} < threshold={threshold}. 测谎触发未达阈值：最高分={score:.3f}，阈值={threshold}。分数接近阈值=弹窗在画面但相似度不足；分数很低=弹窗未出现或模板不符。")

    # ------------------------------------------------------------------ 显卡模板匹配加速

    def _gpu_match_enabled(self):  # 显卡加速隐藏式启用（不设开关）：默认优先走显卡，运行期显卡异常后本进程内降级 CPU 不再重试。
        return not self._gpu_off  # 仅受运行期异常降级标志控制，无显卡环境由 _gpu_handle 探测后静默走 CPU。

    def _gpu_handle(self, frame, names):  # 取本帧的显卡匹配句柄：首次创建匹配器、模板或画面尺寸变化时自动重建，不可用时返回 None。
        if frame is None or not names:  # 无画面或无待匹配分类。
            return None  # 走 CPU。
        try:  # 显卡初始化/上传异常不能拖垮服务。
            from src.gpu_feature_match import GpuFeatureMatcher, gpu_match_available  # 延迟导入：CuPy 初始化较重，且无显卡环境不应影响模块导入。
            if not gpu_match_available():  # 未装 CuPy 或无可用显卡。
                return None  # 走 CPU，不记日志（无显卡是常见环境）。
            feature_set = self._feature_set()  # 取框架特征集。
            if feature_set is None:  # 特征集不可用。
                return None  # 走 CPU（CPU 路径也会因同样原因返回未匹配）。
            if self._gpu is None or self._gpu.feature_set is not feature_set:  # 首次使用或框架重建了特征集（标注重载）。
                self._gpu = GpuFeatureMatcher(feature_set, gray=True, logger=logger)  # 新建匹配器，灰度匹配与 CPU 路径一致。
                logger.info("Lie service GPU template match enabled. 测谎服务已启用显卡模板匹配加速（测谎触发/掉线检测）。")  # 记录启用，便于核对实际走的是哪条路径。
            if not self._gpu.prepare(names, frame):  # 没有可用模板（全未标注或全带 mask）。
                return None  # 整体回退 CPU，由 _find_trigger 按 has() 逐个判定。
            return self._gpu.frame(frame)  # 上传本帧并做帧变换，返回可被多个模板复用的句柄。
        except Exception as e:  # 显卡异常。
            self._disable_gpu(e)  # 关闭加速并回退 CPU。
            return None  # 本帧走 CPU。

    def _disable_gpu(self, error):  # 运行期显卡异常：本进程内关闭加速并回退 CPU，避免每帧重复失败刷日志。
        if not self._gpu_off:  # 首次失败才记日志。
            logger.warning(f"Lie service GPU match failed, fallback to CPU: {error}. 测谎服务显卡匹配失败，已自动降级 CPU 模板匹配。")
        self._gpu_off = True  # 置位关闭标志。
        self._gpu = None  # 释放匹配器持有的显存。

    def _get_region_box(self, frame, region_name):  # 直接采集【测谎坐标框】标注记录的坐标框信息作为解测谎输入区域，不做模板匹配；未标注返回 None。
        feature_set = self._feature_set()  # 取特征集。
        if feature_set is None:  # 特征集不可用。
            return None  # 按坐标框缺失处理。
        try:  # 框架按标注名读取记录坐标，自动按当前画面分辨率缩放。
            return feature_set.get_box_by_name(frame, region_name)  # 找不到标注时返回 None。
        except Exception as e:  # 读取坐标框异常。
            logger.warning(f"Lie service get region box failed: {e}. 测谎服务读取坐标框失败。")  # 记录异常。
            return None  # 按坐标框缺失处理。

    def _feature_ready(self, feature_name):  # 检查模板页中是否存在指定分类名的标注。
        feature_set = self._feature_set()  # 取特征集。
        if feature_set is None:  # 特征集不可用。
            return False  # 标注不可用。
        try:  # 加载标注文件失败时视为未就绪。
            return feature_set.feature_exists(feature_name)  # 查询 FeatureSet 中是否加载到该标注。
        except Exception as e:  # 加载标注文件失败。
            logger.warning(f"Lie service failed to load template annotations: {e}. 测谎服务加载模板标注失败。")  # 记录加载失败日志。
            return False  # 标注不可用。

    def _update_vision(self, frame):  # 把带标注画面推送给 UI 实时展示（og.my_app 即持有本服务的 Globals）。
        app = getattr(og, "my_app", None)  # 取 app 级全局对象。
        if app is not None:  # 启动早期 my_app 可能尚未赋值，跳过即可。
            app.update_vision(frame)  # 线程安全地写入最新画面。

    def _idle_sleep(self, seconds):  # 分片等待指定秒数，每片检查退出事件，保证进程能及时停止服务。
        end = time.time() + seconds  # 计算结束时间点。
        while not self._exit_event.is_set():  # 未退出时循环等待。
            remaining = end - time.time()  # 剩余等待秒数。
            if remaining <= 0:  # 等待时间已到。
                return  # 返回。
            time.sleep(min(remaining, 0.1))  # 最多睡 0.1 秒一片，便于及时响应退出。

    def _set_status(self, status):  # 状态变化时打一次日志，避免每轮刷屏。
        if status != self._last_status:  # 状态发生变化。
            self._last_status = status  # 记录最新状态。
            logger.info(f"Lie detector service: {status}. 测谎服务状态：{status}。")  # 输出状态日志。

    def draw_lie_annotations(self, frame, trigger_box, region_box):  # 在画面上框选测谎标注：【测谎触发】红色、【测谎坐标框】青色，未命中的不画。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if trigger_box is not None:  # 【测谎触发】标注命中时用红色框选。
            cv2.rectangle(canvas, (trigger_box.x, trigger_box.y), (trigger_box.x + trigger_box.width, trigger_box.y + trigger_box.height), (0, 0, 255), 2)  # 红色框标出触发标注位置。
            self.draw_text(canvas, "LIE TRIGGER", (trigger_box.x, max(trigger_box.y - 6, 14)), (0, 0, 255))  # 触发标注标签。
        if region_box is not None:  # 【测谎坐标框】标注命中时用青色框选。
            cv2.rectangle(canvas, (region_box.x, region_box.y), (region_box.x + region_box.width, region_box.y + region_box.height), (255, 255, 0), 2)  # 青色框标出谎言检测区域。
            self.draw_text(canvas, "LIE REGION", (region_box.x, max(region_box.y - 6, 14)), (255, 255, 0))  # 坐标框标签。
        return canvas  # 返回绘制完成的画面。

    def draw_shape_overlay(self, frame, region, result, trigger_box=None):  # 绘制光流解测谎叠加画面：区域框、目标轮廓/中心（按 source 配色）、触发标注框与状态文本，风格与测谎检验页签一致。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if trigger_box is not None:  # 【测谎触发】标注在页面上时用红色框选，方便确认触发匹配位置。
            cv2.rectangle(canvas, (trigger_box.x, trigger_box.y), (trigger_box.x + trigger_box.width, trigger_box.y + trigger_box.height), (0, 0, 255), 2)  # 红色框标出触发标注。
            self.draw_text(canvas, "LIE TRIGGER", (trigger_box.x, max(trigger_box.y - 6, 14)), (0, 0, 255))  # 触发标注标签。
        if region is not None:  # 已确定图形区域时叠加区域框与光流跟踪结果。
            rx, ry, rw, rh = region  # 谎言检测区域。
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色框标出测谎坐标框边界。
            self.draw_text(canvas, "LIE REGION", (rx, max(ry - 6, 14)), (255, 255, 0))  # 区域标签。
            if result is not None:  # 有光流结果时绘制轮廓与中心。
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
                if result.contour is not None:  # 有有效轮廓时画轮廓（区域局部坐标偏移到画面坐标）。
                    pts = result.contour + np.array([[[rx, ry]]], dtype=np.int32)  # 加区域偏移。
                    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)  # 2 像素粗线。
                if result.center is not None:  # 有有效中心时画中心点与光标十字。
                    cx, cy = int(result.center[0] + rx), int(result.center[1] + ry)  # 换算到画面坐标。
                    cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)  # 半径 5 实心圆。
                    cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 2)  # 光标横向十字。
                    cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 2)  # 光标纵向十字。
                if result.tracker_alive:  # 跟踪输出有效时显示详细参数。
                    text = f"SHAPE {source} conf={result.confidence:.2f} snr={result.border_snr:.2f}"  # 来源/置信度/信噪比。
                elif source == SOURCE_SCENE_ENDED:  # 场景已结束。
                    text = "SHAPE: scene ended"  # 结束文本。
                else:  # 未学习/等待。
                    text = "SHAPE: learning"  # 学习中文本。
                self.draw_text(canvas, text, (8, 22), color)  # 左上角状态横幅。
        return canvas  # 返回绘制完成的画面。

    def draw_text(self, canvas, text, position, color):  # 在画面上绘制带黑色描边的文字，保证可读性。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # 先画黑色描边。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  # 再画彩色文字。
