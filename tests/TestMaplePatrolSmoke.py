# MaplePatrolTask 回归测试：验证配置裁剪、校验、黄点检测与画面标注逻辑。
import unittest

import numpy as np
import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MaplePatrolTask import MaplePatrolTask


class TestMaplePatrolSmoke(TaskTestCase):
    task_class = MaplePatrolTask

    config = config

    def test_config_trimmed(self):
        # 定时位移与转身策略配置应被裁剪，攻击与巡逻配置应保留。
        for removed in ("Move Interval", "Move Away Seconds", "Move Back Seconds", "Turn Interval"):
            self.assertNotIn(removed, self.task.default_config)
            self.assertNotIn(removed, self.task.config_description)
        for kept in ("Attack Key Left", "Attack Key Right", "Monster Features", "GPU Match", "Patrol Left Percent", "Patrol Right Percent", "Minimap Feature"):
            self.assertIn(kept, self.task.default_config)

    def test_validate_config(self):
        self.assertIsNone(self.task.validate_config("Attack Key Left", "a"))  # 合法按键复用父类校验。
        self.assertIsNotNone(self.task.validate_config("Attack Key Right", "not_a_key"))  # 非法按键报错。
        self.assertIsNone(self.task.validate_config("Patrol Left Percent", 10.0))  # 合法占比。
        self.assertIsNotNone(self.task.validate_config("Patrol Left Percent", 120))  # 越界占比报错。
        self.assertIsNotNone(self.task.validate_config("Dot Hue Min", 200))  # 色相越界报错。
        self.assertIsNone(self.task.validate_config("Map Rect", "5,20,90,75"))  # 合法地图区域。
        self.assertIsNotNone(self.task.validate_config("Map Rect", "5,20,90"))  # 缺一个数报错。

    def test_parse_map_rect(self):
        self.assertEqual((5.0, 20.0, 90.0, 75.0), MaplePatrolTask.parse_map_rect("5,20,90,75"))
        self.assertEqual((10.0, 30.0, 60.0, 50.0), MaplePatrolTask.parse_map_rect("10%,30%,60%,50%"))  # 带百分号也能解析。
        with self.assertRaises(ValueError):
            MaplePatrolTask.parse_map_rect("abc")

    def test_detect_dot(self):
        # 合成画面：两个黄点，无上一帧时取最大块，有上一帧时优先最近块。
        frame = np.full((200, 500, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (150, 80), 4, (0, 255, 255), -1)
        cv2.circle(frame, (420, 60), 6, (0, 255, 255), -1)
        rect = (50, 20, 400, 160)
        largest = self.task.detect_dot(frame, rect, None, 18, 38, 4)
        self.assertEqual((420, 60), (largest[0], largest[1]))
        nearest = self.task.detect_dot(frame, rect, (148, 79), 18, 38, 4)
        self.assertEqual((150, 80), (nearest[0], nearest[1]))
        self.assertIsNone(self.task.detect_dot(frame, rect, None, 90, 120, 4))  # 色相窗口不含黄色时无结果。

    def test_update_stuck_anchor(self):
        anchor_x, anchor_time = self.task.update_stuck_anchor(100, None, 0.0, 400)  # 首次建立锚点。
        self.assertEqual(100, anchor_x)
        same_x, same_time = self.task.update_stuck_anchor(101, anchor_x, anchor_time, 400)  # 变化小于阈值不重置。
        self.assertEqual(anchor_time, same_time)
        moved_x, _ = self.task.update_stuck_anchor(120, anchor_x, anchor_time, 400)  # 变化超过阈值重置锚点。
        self.assertEqual(120, moved_x)

    def test_draw_overlay(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)
        canvas = self.task.draw_overlay(frame, None, None, None, 10.0, 90.0, None, [], None, None)  # 全部未检测到时不报错。
        self.assertEqual(frame.shape, canvas.shape)
        canvas2 = self.task.draw_overlay(frame, None, (50, 40, 300, 200), (200, 140, 12), 10.0, 90.0, None, [], None, None)  # 有地图区域与黄点时绘制边界红线。
        self.assertFalse((canvas2 == frame).all())


if __name__ == '__main__':
    unittest.main()
