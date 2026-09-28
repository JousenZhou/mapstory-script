# MapleSingleSpotTask 回归测试：验证配置裁剪、校验、归位记录与画面标注逻辑。
import time
import unittest
from unittest.mock import patch, call

import numpy as np
import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MapleSingleSpotTask import MapleSingleSpotTask  # 导入单点挂机任务类。
from src.dashboard_store import DASHBOARD_DEFAULTS  # 看板共享配置默认值，验证共享键已从任务页裁剪。


class TestMapleSingleSpotSmoke(TaskTestCase):
    task_class = MapleSingleSpotTask

    config = config

    def test_config_trimmed(self):
        # 巡逻边界与总开关应被裁剪，小地图/黄点配置应保留，看板共享配置应全部从任务页移除。
        for removed in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent", "Stuck Seconds", "Resume Wait Seconds"):
            self.assertNotIn(removed, self.task.default_config)
            self.assertNotIn(removed, self.task.config_description)
        for shared_key in DASHBOARD_DEFAULTS:  # 看板三栏共享键（含测谎六键与朝向模板）不再出现在任务页。
            self.assertNotIn(shared_key, self.task.default_config)
            self.assertNotIn(shared_key, self.task.config_description)
        for kept in ("Minimap Feature", "Minimap Threshold", "Map Rect", "Dot Hue Min", "Dot Hue Max", "Dot Min Pixels", "Del Key Interval Variance"):
            self.assertIn(kept, self.task.default_config)
        self.assertEqual(5.0, self.task.default_config["Return Offset Max Percent"])  # 碰撞偏移归位max 默认 5%。
        self.assertIn("Return Offset Max Percent", self.task.config_description)  # 新配置项有帮助文本。

    def test_validate_config(self):
        self.assertIsNone(self.task.validate_config("Return Offset Max Percent", 5.0))  # 合法百分比。
        self.assertIsNone(self.task.validate_config("Return Offset Max Percent", "3.5"))  # 数字字符串也合法。
        self.assertIsNotNone(self.task.validate_config("Return Offset Max Percent", 120))  # 越界报错。
        self.assertIsNotNone(self.task.validate_config("Return Offset Max Percent", -1))  # 负数报错。
        self.assertIsNotNone(self.task.validate_config("Return Offset Max Percent", "abc"))  # 非数字报错。
        self.assertIsNone(self.task.validate_config("Map Rect", "5,20,90,75"))  # 父类校验的合法地图区域仍生效。
        self.assertIsNotNone(self.task.validate_config("Map Rect", "5,20,90"))  # 父类校验的缺项报错仍生效。

    def test_wait_home_percent(self):
        # 小地图与黄点就绪时应返回黄点相对地图区域宽度的横向百分比。
        frame = np.full((200, 500, 3), 20, dtype=np.uint8)
        with patch.object(self.task, "next_frame", return_value=frame), \
                patch.object(self.task, "find_minimap", return_value=object()), \
                patch.object(self.task, "map_rect", return_value=(50, 20, 400, 160)), \
                patch.object(self.task, "detect_dot", return_value=(250, 100, 12)):
            percent = self.task.wait_home_percent("完整小地图", 0.8, 18, 38, 4)
        self.assertAlmostEqual(50.0, percent)  # (250-50)/400*100 = 50%。

    def test_wait_home_percent_timeout(self):
        # 一直检测不到黄点时应超时返回 None，而不是无限等待。
        with patch.object(self.task, "next_frame", return_value=None), \
                patch.object(self.task, "sleep", side_effect=lambda s: time.sleep(0.001)):
            start = time.time()
            percent = self.task.wait_home_percent("完整小地图", 0.8, 18, 38, 4)
        self.assertIsNone(percent)  # 超时未记录到初始坐标。
        self.assertLess(time.time() - start, 30)  # 等待时间受 HOME_WAIT_SECONDS 限制。

    def test_release_all_keys(self):
        # 松开全部持有按键：攻击键松开后记录时间戳，移动键清空。
        self.task._held_attack_key = "a"
        self.task._held_move_key = "right"
        self.task._attack_released_at = 0.0
        with patch.object(self.task, "send_key_up") as mock_up:
            self.task.release_all_keys()
        self.assertEqual([call("a"), call("right")], mock_up.call_args_list)  # 两个键都被松开。
        self.assertIsNone(self.task._held_attack_key)  # 攻击键状态清空。
        self.assertIsNone(self.task._held_move_key)  # 移动键状态清空。
        self.assertGreater(self.task._attack_released_at, 0.0)  # 松键时间戳已记录，走动前会等待后摇结束。

    def test_draw_overlay(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)  # 构造纯色合成画面。
        canvas = self.task.draw_overlay(frame, None, None, None, 50.0, 5.0, None, [], None, None)  # 全部未检测到时不报错。
        self.assertEqual(frame.shape, canvas.shape)  # 画面尺寸不变。
        canvas2 = self.task.draw_overlay(frame, None, (50, 40, 300, 200), (200, 140, 12), 50.0, 5.0, None, [], None, None)  # 有地图区域与黄点时绘制归位线与容差带。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。


if __name__ == '__main__':
    unittest.main()
