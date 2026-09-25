# 自动重登流程（AutoLoginFlow）点击行为回归测试。
#
# 重点是「频道必须真的被双击」与「重试不能漂移」这两件事，都来自线上实测：
#   - 旧实现每次点击都重新置前窗口（bring_to_front 实测上百毫秒），两次按下被拉开到 251ms，
#     启动器的频道列表只当成两次单击——频道选中高亮了却不进入，流程在频道步反复卡住直到超时。
#     现在连点整串交给窗口点击回调一次完成，置前只做一次，两次按下紧凑相连。
#   - 旧实现卡住后重新匹配再点，而频道被选中后外观会变，最高分会漂到相邻频道
#     （实测同一【频道3】模板先在 (722,379) 满分命中，点击后再匹配落到 (815,410)），
#     后续点击就打在别的频道上。现在同一轮重试锁定首次命中的坐标。
import threading  # 构造永不置位的退出事件，测试只直接调方法不启线程。
import unittest
from unittest.mock import patch  # 隔离模板匹配、推进判定与真实鼠标 IO。

import numpy as np  # 构造合成的游戏窗口帧。

from src.autologin.flow import (  # 被测流程与关键常量。
    AutoLoginFlow,
    BACKEND_DESKTOP,
    BACKEND_WINDOW,
    MAX_RECLICK,
)

FRAME = np.zeros((100, 120, 3), dtype=np.uint8)  # 合成窗口帧，只用来让采集回调有返回值。
FIRST_BOX = (700, 370, 44, 18, 1.000)   # 首次满分命中的频道框 (x, y, w, h, conf)，中心 (722,379)，与线上日志一致。
DRIFT_BOX = (793, 401, 44, 18, 0.954)   # 频道高亮后重新匹配漂移到的相邻频道框，中心 (815,410)。
ROUNDS = 1 + MAX_RECLICK         # 每一轮尝试内的点击次数（首次 + 卡住重试）。


class TestAutoLoginFlowClicks(unittest.TestCase):

    def make_flow(self, clicks):  # 造一个注入窗口后端回调的流程，点击只记录不发真实 IO。
        return AutoLoginFlow(  # 标注文件路径不会被真正读取：_match 全部打桩。
            'coco.json', {}, None,
            game_frame_fn=lambda: FRAME,
            window_click_fn=lambda x, y, n: clicks.append((x, y, n)))

    def test_window_double_click_is_sent_as_one_call(self):
        # 频道步（clicks=2）只向窗口回调发一次调用，由回调内部完成「置前一次 + 紧凑连点」。
        clicks = []
        flow = self.make_flow(clicks)
        with patch.object(flow, "_match", return_value=(FIRST_BOX, FIRST_BOX[4])), \
                patch.object(flow, "_advanced", return_value=True):
            ok = flow._execute_step("频道3", "频道 Channel", 0.75, 0.2, threading.Event(),
                                    3, 4, BACKEND_WINDOW, None, BACKEND_WINDOW, "开始游戏", 2)
        self.assertTrue(ok)  # 点击后画面推进即视为本步成功。
        self.assertEqual([(722, 379, 2)], clicks)  # 一次调用、中心坐标正确、连点 2 次。

    def test_single_click_step_still_sends_one_click(self):
        # 服务区步（clicks=1）行为不变：仍是一次调用、连点 1 次。
        clicks = []
        flow = self.make_flow(clicks)
        with patch.object(flow, "_match", return_value=(FIRST_BOX, FIRST_BOX[4])), \
                patch.object(flow, "_advanced", return_value=True):
            self.assertTrue(flow._execute_step("蓝蜗牛", "服务区 Server", 0.75, 0.2, threading.Event(),
                                               2, 4, BACKEND_WINDOW, None, BACKEND_WINDOW, "频道3", 1))
        self.assertEqual([(722, 379, 1)], clicks)

    def test_stuck_retry_locks_first_hit_coordinates(self):
        # 卡住重试必须锁定首次命中的坐标：同一轮 3 次点击全打在 (722,379)，不跟着漂移的匹配结果跑。
        clicks = []
        flow = self.make_flow(clicks)
        with patch.object(flow, "_match", side_effect=[(FIRST_BOX, FIRST_BOX[4]), (DRIFT_BOX, DRIFT_BOX[4])]), \
                patch.object(flow, "_advanced", return_value=False), \
                patch.object(flow, "_reanchor") as reanchor:
            ok = flow._execute_step("频道3", "频道 Channel", 0.75, 0.2, threading.Event(),
                                    3, 4, BACKEND_WINDOW, "蓝蜗牛", BACKEND_WINDOW, "开始游戏", 2)
        self.assertFalse(ok)  # 两轮尝试都没推进，本步判失败。
        self.assertEqual([(722, 379, 2)] * ROUNDS + [(815, 410, 2)] * ROUNDS, clicks)  # 每轮内部坐标不变，换轮才重新匹配。
        self.assertEqual(1, reanchor.call_count)  # 整轮点完仍未推进才重锚一次（重新展开频道面板）。

    def test_desktop_backend_bursts_clicks_at_same_position(self):
        # 桌面后端没有窗口置前开销，连点由流程自己按 DOUBLE_CLICK_GAP 紧凑发出，坐标完全相同。
        sent = []
        flow = AutoLoginFlow('coco.json', {}, None)  # 不注入窗口回调 → 全部走桌面后端。
        with patch.object(flow, "_click_screen", side_effect=lambda x, y: sent.append((x, y))), \
                patch("time.sleep"):  # 不真等，只验证连点次数与坐标。
            flow._click_for(BACKEND_DESKTOP, 3204, 481, 2)
            flow._click_for(BACKEND_DESKTOP, 3204, 481, 1)
        self.assertEqual([(3204, 481)] * 3, sent)  # 连点 2 次 + 单击 1 次。

    def test_window_click_callback_exception_does_not_break_step(self):
        # 窗口点击回调抛异常时只记警告，流程继续走推进判定，不能把整条重登序列打断。
        flow = AutoLoginFlow('coco.json', {}, None,
                             game_frame_fn=lambda: FRAME,
                             window_click_fn=lambda x, y, n: (_ for _ in ()).throw(RuntimeError("点击失败")))
        with patch.object(flow, "_match", return_value=(FIRST_BOX, FIRST_BOX[4])), \
                patch.object(flow, "_advanced", return_value=True), \
                patch.object(flow, "_warn") as warn:
            self.assertTrue(flow._execute_step("频道3", "频道 Channel", 0.75, 0.2, threading.Event(),
                                               3, 4, BACKEND_WINDOW, None, BACKEND_WINDOW, "开始游戏", 2))
        self.assertTrue(any("Window click failed" in str(call) for call in warn.call_args_list))  # 记了降级警告。


if __name__ == '__main__':
    unittest.main()
