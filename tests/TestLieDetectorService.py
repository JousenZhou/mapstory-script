# 独立测谎监控服务回归测试：验证配置解析、鼠标追踪、报警守卫、画面标注、触发匹配、
# 当前任务识别，以及“命中触发 -> 暂停脚本任务 -> 解测谎 -> 恢复任务”的协调逻辑
# （服务与脚本任务解耦、无任务时也能独立求解，是本次改造的核心行为）。
# 另覆盖触发防抖与冷却：瞬时丢失不能打断光流会话，一局结束后要初始化状态并进入冷却期。
# 还覆盖触发延迟：匹配到触发后先等配置的秒数才解题，延迟期弹窗已关则放弃本局，延迟结束后用新帧开题。
# 以及显卡模板匹配加速：看板开关控制走显卡还是 CPU，显卡未注册的分类与显卡异常都要自动回退 CPU 路径。
import threading  # 用永不置位的退出事件构造服务，测试只直接调方法不启动线程。
import time  # 验证冷却截止时间戳。
import unittest
from types import SimpleNamespace  # 构造假配置/假匹配框/假执行器。
from unittest.mock import MagicMock, patch  # 隔离框架 IO（截图、输入、特征集、og）。

import numpy as np  # 构造合成画面。

from ok import TriggerTask  # 触发任务类型：_current_task 需要把它排除在可暂停任务之外。

import src.liedetector.service as service_module  # patch 模块级 og / ShapeTrackSession 用它。
from src.liedetector.service import (  # 被测服务与关键常量/状态。
    LieDetectorService,
    LIE_MOVE_MAX_STEP,
    STATE_IDLE,
    STATE_SOLVING,
)


class TestLieDetectorService(unittest.TestCase):

    def setUp(self):  # 每个用例构造一个未启动线程的服务实例。
        self.service = LieDetectorService(threading.Event())  # 退出事件永不置位，模拟进程运行中。

    # ------------------------------------------------------------------ 配置解析

    def test_read_config_parses_and_trims(self):
        # 从看板配置读出六元组：开关、触发名、坐标框名、阈值、触发延迟、报警音频，字符串两端空格要清掉。
        fake = {
            "Lie Detector Auto Solve": True,
            "Lie Detector Trigger Feature": "  测谎触发  ",
            "Lie Detector Region Feature": "测谎坐标框",
            "Lie Detector Threshold": "0.8",  # 字符串阈值应转成浮点。
            "Lie Detector Trigger Delay": "3.5",  # 字符串延迟同样应转成浮点。
            "Lie Alarm Sound": "  alarm.mp3 ",
        }
        with patch.object(self.service, "_get_config", return_value=fake):
            auto, trigger, region, threshold, trigger_delay, alarm = self.service._read_config()
        self.assertTrue(auto)
        self.assertEqual("测谎触发", trigger)
        self.assertEqual("测谎坐标框", region)
        self.assertEqual(0.8, threshold)
        self.assertEqual(3.5, trigger_delay)
        self.assertEqual("alarm.mp3", alarm)

    def test_read_config_bad_threshold_falls_back(self):
        # 阈值非法时回退默认 0.75，不能让服务崩溃。
        with patch.object(self.service, "_get_config", return_value={"Lie Detector Threshold": "abc"}):
            _, _, _, threshold, _, _ = self.service._read_config()
        self.assertEqual(0.75, threshold)

    def test_read_config_delay_defaults_and_clamps(self):
        # 触发延迟：未配置或非法时兜底默认 5 秒，负数按 0 处理，过大值截断到上限。
        default_delay = service_module.LIE_TRIGGER_DELAY_DEFAULT
        with patch.object(self.service, "_get_config", return_value={}):
            _, _, _, _, missing, _ = self.service._read_config()
        self.assertEqual(default_delay, missing)  # 看板未配该键时用默认值。
        with patch.object(self.service, "_get_config", return_value={"Lie Detector Trigger Delay": "abc"}):
            _, _, _, _, bad, _ = self.service._read_config()
        self.assertEqual(default_delay, bad)  # 非数字延迟回退默认值。
        with patch.object(self.service, "_get_config", return_value={"Lie Detector Trigger Delay": -2}):
            _, _, _, _, negative, _ = self.service._read_config()
        self.assertEqual(0.0, negative)  # 负数按不延迟处理。
        with patch.object(self.service, "_get_config",
                          return_value={"Lie Detector Trigger Delay": service_module.LIE_TRIGGER_DELAY_MAX + 100}):
            _, _, _, _, huge, _ = self.service._read_config()
        self.assertEqual(service_module.LIE_TRIGGER_DELAY_MAX, huge)  # 误填过大值被截断，不会长时间卡在等待。

    def test_reload_config_forces_refresh(self):
        # reload_config 把刷新时间戳清零，主循环下一轮立即重读看板配置（看板保存时调用）。
        self.service._config_time = 123.0
        self.service.reload_config()
        self.assertEqual(0.0, self.service._config_time)

    # ------------------------------------------------------------------ 鼠标追踪

    def test_move_mouse_toward(self):
        # 首帧直接跳到预测点，之后按最大步长分步追赶，距离在上限内一步到位。
        moves = []  # 记录发给游戏窗口的鼠标坐标。
        with patch.object(self.service, "_move", side_effect=lambda x, y: moves.append((x, y))):
            pos = self.service._move_mouse_toward(None, (200, 100))  # 首帧无参照点。
            self.assertEqual((200, 100), pos)
            self.assertEqual([(200, 100)], moves)
            pos = self.service._move_mouse_toward((200, 100), (400, 100))  # 距离 200 超上限。
            self.assertEqual((200 + LIE_MOVE_MAX_STEP, 100), pos)  # 按最大步长截断。
            pos = self.service._move_mouse_toward(pos, (pos[0] + 10, 100))  # 距离在上限内。
            self.assertEqual((210 + LIE_MOVE_MAX_STEP, 100), pos)  # 一步到位。
        self.assertEqual(3, len(moves))  # 三次移动全部发出。

    # ------------------------------------------------------------------ 报警守卫

    def test_play_alarm_guards(self):
        # 未配置或文件不存在时不播放且不抛异常。
        self.assertFalse(self.service._play_alarm(""))  # 留空不报警。
        self.assertFalse(self.service._play_alarm(None))  # None 同样不报警。
        self.assertFalse(self.service._play_alarm("assets/not_exist_alarm.mp3"))  # 文件不存在不报警。

    # ------------------------------------------------------------------ 画面标注

    def test_draw_lie_annotations(self):
        # 触发标注红色框、坐标框青色框，都未命中时不绘制。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        canvas = self.service.draw_lie_annotations(frame, None, None)  # 都未命中。
        self.assertTrue((canvas == frame).all())  # 画面保持原样。
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)  # 假触发标注框。
        region = SimpleNamespace(x=100, y=50, width=200, height=150)  # 假坐标框。
        canvas2 = self.service.draw_lie_annotations(frame, trigger, region)  # 两个标注都命中。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。
        self.assertEqual(0, canvas2[10, 10][0])  # 触发框红色：蓝通道为 0。
        self.assertEqual(255, canvas2[10, 10][2])  # 触发框红色：红通道拉满。
        self.assertEqual(255, canvas2[50, 100][0])  # 坐标框青色：蓝通道拉满。
        self.assertEqual(255, canvas2[50, 100][1])  # 坐标框青色：绿色通道拉满。

    def test_draw_shape_overlay(self):
        # 光流解测谎叠加画面：空结果/无区域不报错，有效跟踪结果与触发框会改动画面。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        region = (50, 40, 300, 200)  # 图形区域四元组。
        waiting = SimpleNamespace(source="waiting", contour=None, center=None, confidence=0.0,
                                  border_snr=0.0, tracker_alive=False)  # 未学到模板的空结果。
        canvas = self.service.draw_shape_overlay(frame, region, waiting, None)  # 空结果。
        self.assertEqual(frame.shape, canvas.shape)  # 画面尺寸不变。
        contour = np.array([[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]], dtype=np.int32)  # 假轮廓 (N,1,2)。
        tracked = SimpleNamespace(source="color", contour=contour, center=np.array([25.0, 25.0]),
                                  confidence=0.9, border_snr=1.5, tracker_alive=True)  # 有效跟踪结果。
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)  # 假触发标注框。
        canvas2 = self.service.draw_shape_overlay(frame, region, tracked, trigger)  # 全要素。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。
        canvas3 = self.service.draw_shape_overlay(frame, None, None, None)  # 无区域无结果。
        self.assertEqual(frame.shape, canvas3.shape)  # 不报错且尺寸不变。

    # ------------------------------------------------------------------ 触发匹配

    def test_find_trigger_value_error_returns_none(self):
        # 标注未在模板页标注时框架抛 ValueError，服务按未匹配处理不中断。
        fs = MagicMock()
        fs.find_feature.side_effect = ValueError("no annotation")
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIsNone(self.service._find_trigger(None, "测谎触发", 0.7))

    def test_find_trigger_returns_best_box(self):
        # 多个达标框时取置信度最高者，与 find_one 语义一致。
        fs = MagicMock()
        low = SimpleNamespace(confidence=0.6)
        high = SimpleNamespace(confidence=0.9)
        fs.find_feature.return_value = [low, high]
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIs(high, self.service._find_trigger(None, "测谎触发", 0.5))

    def test_find_trigger_no_feature_set(self):
        # 特征集不可用（未选窗口等）时按未匹配处理。
        with patch.object(self.service, "_feature_set", return_value=None):
            self.assertIsNone(self.service._find_trigger(None, "测谎触发", 0.5))

    def test_get_region_box_returns_none_on_error(self):
        # 读取坐标框异常时按缺失处理，不中断解题。
        fs = MagicMock()
        fs.get_box_by_name.side_effect = RuntimeError("boom")
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIsNone(self.service._get_region_box(None, "测谎坐标框"))

    def test_feature_ready(self):
        # 标注就绪检查：特征集存在该分类返回 True，特征集不可用返回 False。
        fs = MagicMock()
        fs.feature_exists.return_value = True
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertTrue(self.service._feature_ready("测谎触发"))
        with patch.object(self.service, "_feature_set", return_value=None):
            self.assertFalse(self.service._feature_ready("测谎触发"))

    # ------------------------------------------------------------------ 当前任务识别

    def test_current_task_excludes_trigger_task(self):
        # 触发任务的暂停语义是全局暂停，不由本服务管理，必须排除。
        trigger_task = MagicMock(spec=TriggerTask)
        executor = SimpleNamespace(current_task=trigger_task)
        with patch.object(service_module, "og", SimpleNamespace(executor=executor)):
            self.assertIsNone(self.service._current_task())

    def test_current_task_excludes_disabled(self):
        # 已停用的任务不暂停也不恢复。
        task = MagicMock()
        task._enabled = False
        executor = SimpleNamespace(current_task=task)
        with patch.object(service_module, "og", SimpleNamespace(executor=executor)):
            self.assertIsNone(self.service._current_task())

    def test_current_task_returns_enabled_non_trigger(self):
        # 启用中的一次性脚本任务才是可暂停对象。
        task = MagicMock()
        task._enabled = True
        executor = SimpleNamespace(current_task=task)
        with patch.object(service_module, "og", SimpleNamespace(executor=executor)):
            self.assertIs(task, self.service._current_task())

    def test_current_task_no_executor(self):
        # 执行器尚未就绪时返回 None（启动早期）。
        with patch.object(service_module, "og", SimpleNamespace()):
            self.assertIsNone(self.service._current_task())

    # ------------------------------------------------------------------ 暂停/恢复协调

    def test_pause_current_task_no_task(self):
        # 无脚本任务运行时独立解题，无需暂停/恢复。
        with patch.object(self.service, "_current_task", return_value=None):
            self.assertIsNone(self.service._pause_current_task())

    def test_pause_current_task_skips_user_paused(self):
        # 用户已手动暂停的任务不再暂停，解题后也不自动恢复，尊重用户意图。
        task = MagicMock()
        task.paused = True
        with patch.object(self.service, "_current_task", return_value=task):
            self.assertIsNone(self.service._pause_current_task())
        task.pause.assert_not_called()

    def test_pause_current_task_pauses_and_releases_keys(self):
        # 暂停任务 -> 等其阻塞到 sleep -> 释放任务持有键，返回被暂停的任务供解题后恢复。
        task = MagicMock()
        task.paused = False
        task.pop_held_keys.return_value = ["left", "a"]  # 任务上报持有移动键与攻击键。
        interaction = MagicMock()
        with patch.object(self.service, "_current_task", return_value=task), \
                patch.object(self.service, "_interaction", return_value=interaction), \
                patch.object(self.service, "_idle_sleep") as idle:
            paused = self.service._pause_current_task()
        self.assertIs(task, paused)  # 返回被暂停的任务。
        task.pause.assert_called_once()  # 已暂停。
        task.pop_held_keys.assert_called_once()  # 已取持有键。
        interaction.send_key_up.assert_any_call("left")  # 松开移动键。
        interaction.send_key_up.assert_any_call("a")  # 松开攻击键。
        self.assertEqual(2, interaction.send_key_up.call_count)  # 两个键都松开。
        idle.assert_called_once()  # 暂停后 settle 等待，确保任务已阻塞再松键。

    def test_release_task_keys_no_interaction(self):
        # 输入接口不可用时不松键也不报错。
        task = MagicMock()
        with patch.object(self.service, "_interaction", return_value=None):
            self.service._release_task_keys(task)
        task.pop_held_keys.assert_not_called()

    def test_resume_task_only_when_enabled(self):
        # 只恢复仍启用的任务，已停止的任务不动它。
        task = MagicMock()
        task._enabled = True
        self.service._resume_task(task)
        task.unpause.assert_called_once()
        stopped = MagicMock()
        stopped._enabled = False
        self.service._resume_task(stopped)
        stopped.unpause.assert_not_called()

    # ------------------------------------------------------------------ 触发处理全流程

    def test_handle_trigger_pauses_solves_resumes(self):
        # 命中触发：暂停任务 -> 报警 -> 推送标注 -> 解题 -> 恢复任务 -> 回到空闲态。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        task = MagicMock()
        seen = {}

        def fake_solve(*args):  # 解题期间服务应处于 SOLVING 态。
            seen["state"] = self.service.state

        with patch.object(self.service, "_pause_current_task", return_value=task), \
                patch.object(self.service, "_resume_task") as resume, \
                patch.object(self.service, "_play_alarm") as alarm, \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_solve", side_effect=fake_solve) as solve, \
                patch.object(self.service, "_idle_sleep"):
            self.service._handle_trigger(frame, trigger, "测谎触发", "测谎坐标框", 0.7, 0.0, "alarm.mp3")
        alarm.assert_called_once_with("alarm.mp3")  # 报警按配置播放。
        solve.assert_called_once()  # 解题被调用。
        self.assertEqual(STATE_SOLVING, seen["state"])  # 解题期间处于 SOLVING 态。
        resume.assert_called_once_with(task)  # 解题后恢复被暂停的任务。
        self.assertEqual(STATE_IDLE, self.service.state)  # 结束回到空闲态。

    def test_handle_trigger_without_task_solves_independently(self):
        # 无脚本任务运行时也独立解题，且不去恢复任何任务。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        with patch.object(self.service, "_pause_current_task", return_value=None), \
                patch.object(self.service, "_resume_task") as resume, \
                patch.object(self.service, "_play_alarm"), \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_solve") as solve, \
                patch.object(self.service, "_idle_sleep"):
            self.service._handle_trigger(frame, trigger, "测谎触发", "测谎坐标框", 0.7, 0.0, "")
        solve.assert_called_once()  # 无任务也独立解题。
        resume.assert_not_called()  # 没有暂停任务就不恢复。
        self.assertEqual(STATE_IDLE, self.service.state)

    def test_handle_trigger_resumes_even_if_solve_raises(self):
        # 解题异常也要在 finally 里恢复任务，绝不能把脚本任务永久卡在暂停态。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        task = MagicMock()
        with patch.object(self.service, "_pause_current_task", return_value=task), \
                patch.object(self.service, "_resume_task") as resume, \
                patch.object(self.service, "_play_alarm"), \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_solve", side_effect=RuntimeError("boom")), \
                patch.object(self.service, "_idle_sleep"):
            with self.assertRaises(RuntimeError):
                self.service._handle_trigger(frame, trigger, "测谎触发", "测谎坐标框", 0.7, 0.0, "")
        resume.assert_called_once_with(task)  # finally 保证恢复。
        self.assertEqual(STATE_IDLE, self.service.state)

    # ------------------------------------------------------------------ 触发延迟

    def test_wait_trigger_delay_zero_returns_immediately(self):
        # 延迟配为 0 时不等待、不取帧，直接进解题，与加参数前的行为一致。
        with patch.object(self.service, "_capture") as capture, \
                patch.object(self.service, "_idle_sleep") as sleep:
            proceed = self.service._wait_trigger_delay(None, "测谎触发", "测谎坐标框", 0.7, 0.0)
        self.assertTrue(proceed)
        capture.assert_not_called()  # 不延迟就不做多余取帧。
        sleep.assert_not_called()

    def test_wait_trigger_delay_waits_then_proceeds(self):
        # 触发标注一直在：等满延迟秒数后返回 True，期间持续推送框选画面（不会看起来像卡死）。
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        vision = MagicMock()
        start = time.time()
        with patch.object(self.service, "_capture", return_value=frame), \
                patch.object(self.service, "_find_trigger", return_value=trigger), \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision", vision):
            proceed = self.service._wait_trigger_delay(trigger, "测谎触发", "测谎坐标框", 0.7, 0.2)
        self.assertTrue(proceed)
        self.assertGreaterEqual(time.time() - start, 0.2)  # 确实等满了配置的延迟秒数。
        self.assertTrue(vision.called)  # 延迟期一直在推送画面。

    def test_wait_trigger_delay_aborts_when_trigger_gone(self):
        # 延迟等待期间弹窗已关（触发连续丢失超容忍帧数）：放弃本局，不再白等剩余延迟。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        with patch.object(self.service, "_capture", return_value=frame), \
                patch.object(self.service, "_find_trigger", return_value=None), \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_idle_sleep"):  # 不真睡（时间不推进），靠丢失计数收敛退出。
            proceed = self.service._wait_trigger_delay(None, "测谎触发", "测谎坐标框", 0.7, 30.0)
        self.assertFalse(proceed)

    def test_handle_trigger_skips_solve_when_delay_aborts(self):
        # 延迟判定弹窗已关时不能进解题，但报警已播、任务仍要恢复并回到空闲态。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        task = MagicMock()
        with patch.object(self.service, "_pause_current_task", return_value=task), \
                patch.object(self.service, "_resume_task") as resume, \
                patch.object(self.service, "_play_alarm") as alarm, \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_wait_trigger_delay", return_value=False) as wait, \
                patch.object(self.service, "_solve") as solve, \
                patch.object(self.service, "_idle_sleep"):
            self.service._handle_trigger(frame, trigger, "测谎触发", "测谎坐标框", 0.7, 5.0, "alarm.mp3")
        wait.assert_called_once()  # 延迟确实被评估过。
        alarm.assert_called_once_with("alarm.mp3")  # 报警先于延迟播放，用户立即收到提示。
        solve.assert_not_called()  # 弹窗已关，不进解题。
        resume.assert_called_once_with(task)  # finally 仍恢复任务。
        self.assertEqual(STATE_IDLE, self.service.state)

    def test_handle_trigger_recaptures_frame_after_delay(self):
        # 延迟结束后必须用新帧开题，不能拿几秒前的陈旧画面喂给光流会话。
        stale = np.full((300, 400, 3), 20, dtype=np.uint8)
        fresh = np.full((300, 400, 3), 90, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        seen = {}
        with patch.object(self.service, "_pause_current_task", return_value=None), \
                patch.object(self.service, "_resume_task"), \
                patch.object(self.service, "_play_alarm"), \
                patch.object(self.service, "_get_region_box", return_value=None), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_wait_trigger_delay", return_value=True), \
                patch.object(self.service, "_capture", return_value=fresh), \
                patch.object(self.service, "_solve", side_effect=lambda f, *a: seen.setdefault("frame", f)), \
                patch.object(self.service, "_idle_sleep"):
            self.service._handle_trigger(stale, trigger, "测谎触发", "测谎坐标框", 0.7, 5.0, "")
        self.assertIs(fresh, seen["frame"])  # 解题拿到的是延迟后重新采集的帧。

    # ------------------------------------------------------------------ 解题子循环

    @unittest.skipUnless(service_module.LIE_SOLVER_AVAILABLE, "liedetector optical-flow solver unavailable")
    def test_solve_finishes_when_trigger_gone(self):
        # 解测谎子循环：触发标注持续消失超过容忍帧数才退出，期间持续把光流叠加画面推送给 UI。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        region = SimpleNamespace(x=100, y=50, width=200, height=150)
        state = {"tick": 0}  # 触发标注检查次数计数。

        def fake_find_trigger(frame_arg, name, threshold):  # 首轮命中，之后持续消失代表测谎真结束。
            state["tick"] += 1
            return trigger if state["tick"] == 1 else None

        fake_result = SimpleNamespace(source="waiting", center=None, contour=None, confidence=0.0,
                                      border_snr=0.0, tracker_alive=False, white_candidates=0,
                                      flow_residual=None)  # 空光流结果：不移鼠标、不提前结束。
        fake_session = SimpleNamespace(reset=lambda w, h: None, update=lambda crop, fps: fake_result)  # 假光流会话，不实际算光流。
        with patch.object(self.service, "_ensure_in_front"), \
                patch.object(service_module, "ShapeTrackSession", return_value=fake_session), \
                patch.object(self.service, "_find_trigger", side_effect=fake_find_trigger), \
                patch.object(self.service, "_get_region_box", return_value=region), \
                patch.object(self.service, "_update_vision") as vision, \
                patch.object(self.service, "_idle_sleep"), \
                patch.object(self.service, "_capture", return_value=None):  # 隔离窗口/光流/匹配/UI/取帧。
            self.service._solve(frame, "测谎触发", "测谎坐标框", 0.7)
        # 首帧命中后需再连续丢失 LIE_TRIGGER_LOST_TOLERANCE+1 帧才判定结束，避免弹窗淡出抖动造成秒退。
        self.assertEqual(service_module.LIE_TRIGGER_LOST_TOLERANCE + 2, state["tick"])
        self.assertTrue(vision.called)  # 解测谎画面已推送给 UI。

    @unittest.skipUnless(service_module.LIE_SOLVER_AVAILABLE, "liedetector optical-flow solver unavailable")
    def test_solve_survives_transient_trigger_loss(self):
        # 触发标注瞬时丢失（弹窗淡出期分数抖动）不能打断解题：容忍帧数内恢复命中，光流会话不被重建。
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)
        region = SimpleNamespace(x=100, y=50, width=200, height=150)
        state = {"tick": 0, "updates": 0, "resets": 0}

        def fake_find_trigger(frame_arg, name, threshold):  # 首帧命中，第 2~4 帧瞬时丢失（在容忍值内），之后恢复命中。
            state["tick"] += 1
            return None if 2 <= state["tick"] <= 4 else trigger

        waiting = SimpleNamespace(source="waiting", center=None, contour=None, confidence=0.0,
                                  border_snr=0.0, tracker_alive=False, white_candidates=0,
                                  flow_residual=None)  # 学习中的空结果，不终止子循环。
        ended = SimpleNamespace(source=service_module.SOURCE_SCENE_ENDED, center=None, contour=None,
                                confidence=0.0, border_snr=0.0, tracker_alive=False, white_candidates=0,
                                flow_residual=None)  # 场景结束，用于终止子循环。

        def fake_update(crop, fps):  # 第 5 次喂帧时判定场景结束。
            state["updates"] += 1
            return ended if state["updates"] >= 5 else waiting

        def fake_reset(w, h):  # 记录会话重置次数：容忍期内不应重新 reset。
            state["resets"] += 1

        fake_session = SimpleNamespace(reset=fake_reset, update=fake_update)
        with patch.object(self.service, "_ensure_in_front"), \
                patch.object(service_module, "ShapeTrackSession", return_value=fake_session), \
                patch.object(self.service, "_find_trigger", side_effect=fake_find_trigger), \
                patch.object(self.service, "_get_region_box", return_value=region), \
                patch.object(self.service, "_update_vision"), \
                patch.object(self.service, "_idle_sleep"), \
                patch.object(self.service, "_capture", return_value=None):
            self.service._solve(frame, "测谎触发", "测谎坐标框", 0.7)
        self.assertEqual(5, state["tick"])  # 瞬时丢失的 3 帧没有导致提前退出，一直撑到场景结束。
        self.assertEqual(1, state["resets"])  # 光流会话只在区域首次出现时建了一次，未被抖动打断重建。

    # ------------------------------------------------------------------ 结束后初始化与冷却

    def test_reset_solve_state_starts_cooldown(self):
        # 一局结束后初始化跨帧状态：确认计数与诊断限频清零，并开启冷却期避免残留画面立即重复触发。
        self.service._trigger_hits = 2  # 残留的确认计数。
        self.service._last_trigger_probe = 123.0  # 残留的诊断限频时间戳。
        self.service._cooldown_until = 0.0  # 初始无冷却。
        before = time.time()  # 调重置前的时间点。
        self.service._reset_solve_state()
        self.assertEqual(0, self.service._trigger_hits)  # 确认计数已清零。
        self.assertEqual(0.0, self.service._last_trigger_probe)  # 诊断限频已清零，下一局能立即打出分数。
        self.assertLessEqual(before + service_module.LIE_SOLVE_COOLDOWN, self.service._cooldown_until)  # 冷却截止时间已推到未来。

    # ------------------------------------------------------------------ 显卡模板匹配加速

    def test_gpu_match_enabled_reads_dashboard_switch(self):
        # 看板开关控制是否走显卡：未配该键（旧配置文件）默认开启，显式关闭则走 CPU。
        with patch.object(self.service, "_get_config", return_value={}):
            self.assertTrue(self.service._gpu_match_enabled())  # 缺键时默认开启。
        with patch.object(self.service, "_get_config", return_value={'Lie Detector GPU Match': True}):
            self.assertTrue(self.service._gpu_match_enabled())
        with patch.object(self.service, "_get_config", return_value={'Lie Detector GPU Match': False}):
            self.assertFalse(self.service._gpu_match_enabled())  # 用户关掉开关即回退 CPU。
        self.service._gpu_off = True  # 模拟运行期显卡异常已关闭加速。
        with patch.object(self.service, "_get_config", return_value={'Lie Detector GPU Match': True}):
            self.assertFalse(self.service._gpu_match_enabled())  # 本进程内不再重试显卡。
        self.service._gpu_off = False
        with patch.object(self.service, "_get_config", side_effect=RuntimeError("boom")):
            self.assertFalse(self.service._gpu_match_enabled())  # 配置读取异常时保守回退 CPU。

    def test_gpu_handle_guards(self):
        # 无画面或无待匹配分类时直接返回 None，不去碰显卡。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        with patch.object(self.service, "_feature_set") as fs:
            self.assertIsNone(self.service._gpu_handle(None, ["测谎触发"]))
            self.assertIsNone(self.service._gpu_handle(frame, []))
        fs.assert_not_called()  # 两个前置守卫都在取特征集之前。

    def test_gpu_handle_returns_none_when_gpu_unavailable(self):
        # 未装 CuPy 或无显卡时返回 None，不创建匹配器也不记错。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        import src.gpu_feature_match as gpu_module
        with patch.object(gpu_module, "gpu_match_available", return_value=False), \
                patch.object(self.service, "_feature_set", return_value=MagicMock()):
            self.assertIsNone(self.service._gpu_handle(frame, ["测谎触发"]))
        self.assertIsNone(self.service._gpu)  # 没有可用显卡就不持有匹配器。
        self.assertFalse(self.service._gpu_off)  # 无显卡是常见环境，不当作故障关闭。

    def test_gpu_handle_creates_matcher_once_and_reuses_it(self):
        # 首次调用创建匹配器并准备本帧句柄；特征集未变时后续帧复用同一个匹配器。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        feature_set = MagicMock()
        handle = object()  # 假帧句柄，只需能被原样返回。
        import src.gpu_feature_match as gpu_module
        gpu = MagicMock()
        gpu.feature_set = feature_set  # 真实匹配器会把特征集存在同名属性上，供服务判定是否需要重建。
        gpu.prepare.return_value = True
        gpu.frame.return_value = handle
        with patch.object(gpu_module, "gpu_match_available", return_value=True), \
                patch.object(gpu_module, "GpuFeatureMatcher", return_value=gpu) as ctor, \
                patch.object(self.service, "_feature_set", return_value=feature_set):
            self.assertIs(handle, self.service._gpu_handle(frame, ["测谎触发"]))
            self.assertIs(handle, self.service._gpu_handle(frame, ["测谎触发"]))
        self.assertEqual(1, ctor.call_count)  # 特征集没变就不重建匹配器，核 FFT 缓存得以保留。
        self.assertIs(gpu, self.service._gpu)
        self.assertEqual(2, gpu.prepare.call_count)  # 每帧仍需 prepare（内部按键判定是否真要重建模板）。
        gpu.frame.assert_any_call(frame)

    def test_gpu_handle_returns_none_when_no_template(self):
        # 全部标注都带 mask 或未标注时 prepare 失败，整体回退 CPU。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        import src.gpu_feature_match as gpu_module
        gpu = MagicMock()
        gpu.prepare.return_value = False
        with patch.object(gpu_module, "gpu_match_available", return_value=True), \
                patch.object(gpu_module, "GpuFeatureMatcher", return_value=gpu), \
                patch.object(self.service, "_feature_set", return_value=MagicMock()):
            self.assertIsNone(self.service._gpu_handle(frame, ["测谎触发"]))
        gpu.frame.assert_not_called()  # 没模板就不必上传画面。

    def test_gpu_handle_disables_gpu_on_exception(self):
        # 显卡初始化/上传抛异常时关闭加速并返回 None，不能把服务线程带崩。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        import src.gpu_feature_match as gpu_module
        gpu = MagicMock()
        gpu.prepare.side_effect = RuntimeError("cuda out of memory")
        with patch.object(gpu_module, "gpu_match_available", return_value=True), \
                patch.object(gpu_module, "GpuFeatureMatcher", return_value=gpu), \
                patch.object(self.service, "_feature_set", return_value=MagicMock()):
            self.assertIsNone(self.service._gpu_handle(frame, ["测谎触发"]))
        self.assertTrue(self.service._gpu_off)  # 已关闭加速。
        self.assertIsNone(self.service._gpu)  # 匹配器已释放。

    def test_find_trigger_uses_gpu_box_and_skips_cpu(self):
        # 显卡已注册该分类时直接用显卡结果，不再跑一次 CPU 匹配（这正是提速的关键）。
        gpu_box = SimpleNamespace(confidence=0.9)
        gpu = MagicMock()
        gpu.has.return_value = True
        gpu.best_box.return_value = gpu_box
        self.service._gpu = gpu
        handle = object()
        fs = MagicMock()
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIs(gpu_box, self.service._find_trigger(None, "测谎触发", 0.7, handle))
        gpu.best_box.assert_called_once_with(handle, "测谎触发", 0.7)
        fs.find_feature.assert_not_called()  # CPU 匹配未被调用。

    def test_find_trigger_falls_back_to_cpu_for_unregistered_feature(self):
        # 显卡未注册的分类（未标注/带 mask）仍走框架 CPU 匹配，行为与开关关闭时一致。
        cpu_box = SimpleNamespace(confidence=0.8)
        gpu = MagicMock()
        gpu.has.return_value = False
        self.service._gpu = gpu
        fs = MagicMock()
        fs.find_feature.return_value = [cpu_box]
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIs(cpu_box, self.service._find_trigger(None, "测谎触发", 0.7, object()))
        fs.find_feature.assert_called_once()  # 已回退 CPU。
        gpu.best_box.assert_not_called()

    def test_find_trigger_falls_back_to_cpu_on_gpu_error(self):
        # 显卡匹配抛异常：关闭加速并用 CPU 兼底，本帧仍能得到正确结果。
        cpu_box = SimpleNamespace(confidence=0.8)
        gpu = MagicMock()
        gpu.has.return_value = True
        gpu.best_box.side_effect = RuntimeError("device lost")
        self.service._gpu = gpu
        fs = MagicMock()
        fs.find_feature.return_value = [cpu_box]
        with patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIs(cpu_box, self.service._find_trigger(None, "测谎触发", 0.7, object()))
        self.assertTrue(self.service._gpu_off)  # 已关闭加速。
        self.assertIsNone(self.service._gpu)  # 匹配器已释放。
        fs.find_feature.assert_called_once()  # 同一帧内已回退到 CPU 并拿到结果。

    def test_find_trigger_builds_handle_when_not_given(self):
        # 单点调用（解题/延迟等待/等窗口关闭）未传句柄时自建，并沿用本轮值守的模板名单。
        gpu = MagicMock()
        gpu.has.return_value = True
        gpu.best_box.return_value = SimpleNamespace(confidence=0.9)
        handle = object()
        self.service._gpu = gpu
        self.service._watch_names = ["掉线2", "测谎触发"]
        with patch.object(self.service, "_gpu_match_enabled", return_value=True), \
                patch.object(self.service, "_gpu_handle", return_value=handle) as build:
            self.service._find_trigger("frame", "测谎触发", 0.7)
        build.assert_called_once_with("frame", ["掉线2", "测谎触发"])  # 模板名单与值守轮一致，不会反复重建。

    def test_find_trigger_skips_gpu_when_switch_off(self):
        # 看板开关关闭时完全不碰显卡，路径与改造前一致（只调一次 CPU 匹配）。
        cpu_box = SimpleNamespace(confidence=0.8)
        fs = MagicMock()
        fs.find_feature.return_value = [cpu_box]
        with patch.object(self.service, "_gpu_match_enabled", return_value=False), \
                patch.object(self.service, "_gpu_handle") as build, \
                patch.object(self.service, "_feature_set", return_value=fs):
            self.assertIs(cpu_box, self.service._find_trigger(None, "测谎触发", 0.7))
        build.assert_not_called()  # 未尝试创建显卡句柄。
        self.assertEqual(1, fs.find_feature.call_count)  # 没有额外的诊断匹配。

    def test_find_trigger_gpu_miss_probes_score(self):
        # 显卡路径未命中时限频报一次实际最高分，诊断能力与 CPU 路径保持一致。
        gpu = MagicMock()
        gpu.has.return_value = True
        gpu.best_box.return_value = None  # 未达阈值。
        gpu.best_score.return_value = 0.42
        self.service._gpu = gpu
        self.service._last_trigger_probe = 0.0  # 本轮应当探测。
        with patch.object(self.service, "_log_probe_score") as log, \
                patch.object(self.service, "_feature_set", return_value=MagicMock()):
            self.assertIsNone(self.service._find_trigger(None, "测谎触发", 0.75, object()))
        log.assert_called_once_with("测谎触发", 0.75, 0.42)  # 分数与阈值都进了日志。
        self.service._last_trigger_probe = time.time()  # 刚刚探测过。
        with patch.object(self.service, "_log_probe_score") as log2:
            self.assertIsNone(self.service._find_trigger(None, "测谎触发", 0.75, object()))
        log2.assert_not_called()  # 限频生效，不会每帧都多算一次分数。

    def test_probe_due_rate_limits(self):
        # 探测限频判定：首次到达返回 True 并刷新时间戳，间隔内返回 False。
        self.service._last_trigger_probe = 0.0
        self.assertTrue(self.service._probe_due())
        self.assertFalse(self.service._probe_due())  # 刚刷新过，间隔未到。
        self.service._last_trigger_probe = time.time() - service_module.TRIGGER_PROBE_INTERVAL - 1
        self.assertTrue(self.service._probe_due())  # 超过间隔后再次探测。

    def test_click_disconnect_ok_reuses_handle(self):
        # 处理掉线弹窗时【掉线】与【掉线确定】复用同一个显卡句柄（同一帧只变换一次）。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        dialog = SimpleNamespace(x=0, y=0, width=10, height=10)
        ok_box = SimpleNamespace(x=20, y=10, width=10, height=10)  # 中心 (25, 15)。
        handle = object()
        calls = []

        def fake_find(f, name, threshold, h=None):
            calls.append((name, h))
            return dialog if name == service_module.DISCONNECT_DIALOG_TEMPLATE else ok_box

        with patch.object(self.service, "_feature_ready", return_value=True), \
                patch.object(self.service, "_find_trigger", side_effect=fake_find), \
                patch.object(self.service, "_click_in_window") as click, \
                patch.object(self.service, "_idle_sleep"):
            self.service._click_disconnect_ok(frame, 0.75, handle)
        self.assertEqual([(service_module.DISCONNECT_DIALOG_TEMPLATE, handle),
                          (service_module.DISCONNECT_OK_TEMPLATE, handle)], calls)  # 两次匹配都带着同一句柄。
        click.assert_called_once_with(25, 15)  # 点击确定按钮中心（确定按钮单击即可，不连点）。

    def test_click_in_window_double_click_fronts_once(self):
        # 双击必须「只置前一次 + 紧凑连点」：旧实现每次点击都重新置前（bring_to_front 实测上百毫秒），
        # 两次按下被拉开到 251ms，启动器的频道列表只当成两次单击（频道选中高亮了却不进入）。
        order = []  # 按发生顺序记录置前/等待/移光标/点击，用来卡住“第二次点击前不再置前”。
        interaction = MagicMock()
        interaction.move.side_effect = lambda *a: order.append("move")
        interaction.click.side_effect = lambda *a: order.append("click")
        with patch.object(self.service, "_ensure_in_front", side_effect=lambda: order.append("front")), \
                patch.object(self.service, "_interaction", return_value=interaction), \
                patch("time.sleep", side_effect=lambda seconds: order.append("sleep")):  # 不真等，只卡顺序。
            self.service._click_in_window(722, 379, 2)
        self.assertEqual(["front", "sleep", "move", "click", "sleep", "click"], order)  # 置前只在首次点击前做一次。
        self.assertEqual(2, interaction.click.call_count)  # 两次点击都发出去了。
        self.assertEqual(1, interaction.move.call_count)  # 光标只在首次点击前移动一次，不重复定位。
        interaction.click.assert_called_with(722, 379)  # 两次点击坐标完全相同，才会被系统判成双击。

    def test_click_in_window_single_click_by_default(self):
        # 不传 clicks 时仍是单击（掉线确定按钮走这条路径）。
        interaction = MagicMock()
        with patch.object(self.service, "_ensure_in_front"), \
                patch.object(self.service, "_interaction", return_value=interaction), \
                patch("time.sleep"):
            self.service._click_in_window(25, 15)
        self.assertEqual(1, interaction.click.call_count)
        self.assertEqual(1, interaction.move.call_count)

    def test_click_disconnect_ok_returns_whether_clicked(self):
        # 返回值决定下一阶段等多久：点了确定为 True，场景2（只有掉线2）与未标注都为 False。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        box = SimpleNamespace(x=0, y=0, width=10, height=10)
        with patch.object(self.service, "_feature_ready", return_value=True), \
                patch.object(self.service, "_find_trigger", return_value=box), \
                patch.object(self.service, "_click_in_window"), \
                patch.object(self.service, "_idle_sleep"):
            self.assertTrue(self.service._click_disconnect_ok(frame, 0.75))  # 弹窗与确定都在：点了确定。
        with patch.object(self.service, "_feature_ready", return_value=True), \
                patch.object(self.service, "_find_trigger", return_value=None), \
                patch.object(self.service, "_click_in_window") as click:
            self.assertFalse(self.service._click_disconnect_ok(frame, 0.75))  # 场景2：没弹窗，跳过点击。
        click.assert_not_called()
        with patch.object(self.service, "_feature_ready", return_value=False), \
                patch.object(self.service, "_find_trigger") as find:
            self.assertFalse(self.service._click_disconnect_ok(frame, 0.75))  # 掉线确定未标注：连匹配都不做。
        find.assert_not_called()

    def test_handle_disconnect_waits_short_when_dialog_skipped(self):
        # 跳过【掉线】弹窗时没有确定可点、窗口也不会自己关，等窗口关闭只给 3 秒（原来白等 15 秒）。
        self.assertEqual(3.0, service_module.DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG)  # 钉住用户要求的短等待秒数。
        self.assert_exit_timeout(clicked_ok=False, expected=service_module.DISCONNECT_GAME_EXIT_TIMEOUT_NO_DIALOG)

    def test_handle_disconnect_waits_long_when_ok_clicked(self):
        # 点了【掉线确定】后要留给弹窗关闭与窗口退出足够时间，仍是 15 秒。
        self.assert_exit_timeout(clicked_ok=True, expected=service_module.DISCONNECT_GAME_EXIT_TIMEOUT)

    def assert_exit_timeout(self, clicked_ok, expected):  # 跑一次 _handle_disconnect，校对传给 _wait_game_exit 的超时。
        cfg = {'enabled': True, 'threshold': 0.75, 'server': '', 'channel': '', 'step_timeout': 30.0, 'raw': {}}
        with patch.object(self.service, "_pause_current_task", return_value=None), \
                patch.object(self.service, "_click_disconnect_ok", return_value=clicked_ok), \
                patch.object(self.service, "_wait_game_exit") as wait, \
                patch.object(self.service, "_coco_json_path", return_value="coco.json"), \
                patch.object(service_module, "AutoLoginFlow") as flow_cls, \
                patch.object(self.service, "_idle_sleep"):
            flow_cls.return_value.run.return_value = True  # 重登序列直接成功，不跑真实流程。
            self.service._handle_disconnect(np.full((30, 40, 3), 20, dtype=np.uint8), cfg)
        wait.assert_called_once_with(0.75, expected)  # 阈值与超时都按上一阶段的结果传下去了。

    def test_run_shares_one_frame_and_handle_between_disconnect_and_lie_check(self):
        # 主循环一轮内只截图一次、只准备一次显卡句柄，掉线检测与测谎检测共用。
        frame = np.full((30, 40, 3), 20, dtype=np.uint8)
        handle = object()
        watch = [service_module.DISCONNECT_TEMPLATE, service_module.DISCONNECT_DIALOG_TEMPLATE,
                 service_module.DISCONNECT_OK_TEMPLATE, "测谎触发"]  # 预期的值守模板名单。
        exit_event = MagicMock()
        exit_event.is_set.side_effect = [False, True]  # 第一轮进入循环，轮末退出。
        service = LieDetectorService(exit_event)
        find_calls = []

        def fake_find(f, name, threshold, h=None):
            find_calls.append((name, h))
            return None  # 两个检测都未命中，本轮只做值守。

        with patch.object(service_module, "LIE_SOLVER_AVAILABLE", True), \
                patch.object(service, "_read_config", return_value=(True, "测谎触发", "测谎坐标框", 0.75, 0.0, "")), \
                patch.object(service, "_read_auto_login_config", return_value={'enabled': True, 'threshold': 0.75,
                                                                              'server': '', 'channel': '',
                                                                              'step_timeout': 30.0, 'raw': {}}), \
                patch.object(service, "_feature_ready", return_value=True), \
                patch.object(service, "_capture", return_value=frame) as capture, \
                patch.object(service, "_gpu_match_enabled", return_value=True), \
                patch.object(service, "_gpu_handle", return_value=handle) as build, \
                patch.object(service, "_find_trigger", side_effect=fake_find), \
                patch.object(service, "_current_task", return_value=MagicMock()), \
                patch.object(service, "_get_region_box", return_value=None), \
                patch.object(service, "_update_vision"), \
                patch.object(service, "_idle_sleep"):
            service._run()
        self.assertEqual(1, capture.call_count)  # 一轮只截图一次，不再为两个检测各截一次。
        build.assert_called_once_with(frame, watch)  # 显卡句柄只建一次，且一次注册全部待匹配分类。
        self.assertEqual([(service_module.DISCONNECT_TEMPLATE, handle), ("测谎触发", handle)], find_calls)  # 两次匹配共用同一句柄。
        self.assertEqual(watch, service._watch_names)  # 名单已记录，供单点匹配沿用。


if __name__ == '__main__':
    unittest.main()
