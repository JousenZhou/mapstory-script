# MaplePatrolTask 回归测试：验证配置裁剪、校验、黄点检测与画面标注逻辑。
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MaplePatrolTask import MaplePatrolTask  # 导入巡逻任务类。
from src.dashboard_store import DASHBOARD_DEFAULTS  # 看板共享配置默认值，验证共享键已从任务页裁剪。


class TestMaplePatrolSmoke(TaskTestCase):
    task_class = MaplePatrolTask

    config = config

    def test_config_trimmed(self):
        # 定时位移与转身策略配置应被裁剪，巡逻专属配置应保留，看板共享配置应全部从任务页移除。
        for removed in ("Move Interval", "Move Away Seconds", "Move Back Seconds", "Turn Interval"):
            self.assertNotIn(removed, self.task.default_config)
            self.assertNotIn(removed, self.task.config_description)
        for shared_key in DASHBOARD_DEFAULTS:  # 看板三栏共享键（含测谎六键与朝向模板）不再出现在任务页。
            self.assertNotIn(shared_key, self.task.default_config)
            self.assertNotIn(shared_key, self.task.config_description)
        for kept in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent", "Minimap Feature", "Del Key Interval Variance"):
            self.assertIn(kept, self.task.default_config)

    def test_validate_config(self):
        self.assertIsNone(self.task.validate_config("Attack Key", "a"))  # 按键校验已搬到看板，任务侧不再拦截。
        self.assertIsNone(self.task.validate_config("Melee Attack Key", "not_a_key"))  # 同上，非法按键也在看板保存时拦截。
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
        # 合成画面：两个黄色块，直接采信面积最大块，不做跨帧追踪。
        frame = np.full((200, 500, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (150, 80), 4, (0, 255, 255), -1)
        cv2.circle(frame, (420, 60), 6, (0, 255, 255), -1)
        rect = (50, 20, 400, 160)
        largest = self.task.detect_dot(frame, rect, 18, 38, 4)
        self.assertEqual((420, 60), (largest[0], largest[1]))
        self.assertIsNone(self.task.detect_dot(frame, rect, 90, 120, 4))  # 色相窗口不含黄色时无结果。

    def test_detect_template_facing(self):
        # 用左右朝向模板判定朝向：双命中取置信度高者，都未命中返回 None。
        frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        with patch.object(self.task, "find_one_raw", return_value=None):
            self.assertIsNone(self.task.detect_template_facing(frame, "左", "右", 0.8))  # 都未命中时朝向未知。
        with patch.object(self.task, "find_one_raw", side_effect=[SimpleNamespace(confidence=0.9), None]):
            self.assertEqual(-1, self.task.detect_template_facing(frame, "左", "右", 0.8))  # 仅左朝向命中。
        with patch.object(self.task, "find_one_raw", side_effect=[None, SimpleNamespace(confidence=0.9)]):
            self.assertEqual(1, self.task.detect_template_facing(frame, "左", "右", 0.8))  # 仅右朝向命中。
        with patch.object(self.task, "find_one_raw", side_effect=[SimpleNamespace(confidence=0.7), SimpleNamespace(confidence=0.95)]):
            self.assertEqual(1, self.task.detect_template_facing(frame, "左", "右", 0.8))  # 双命中时采信置信度更高的一侧。
        with patch.object(self.task, "find_one_raw", side_effect=[SimpleNamespace(confidence=0.95), SimpleNamespace(confidence=0.7)]):
            self.assertEqual(-1, self.task.detect_template_facing(frame, "左", "右", 0.8))  # 反向双命中取左侧。

    def test_update_stuck_anchor(self):
        anchor_x, anchor_time = self.task.update_stuck_anchor(100, None, 0.0, 400)  # 首次建立锚点。
        self.assertEqual(100, anchor_x)
        same_x, same_time = self.task.update_stuck_anchor(101, anchor_x, anchor_time, 400)  # 变化小于阈值不重置。
        self.assertEqual(anchor_time, same_time)
        moved_x, _ = self.task.update_stuck_anchor(120, anchor_x, anchor_time, 400)  # 变化超过阈值重置锚点。
        self.assertEqual(120, moved_x)

    def test_draw_overlay(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)  # 构造纯色合成画面。
        canvas = self.task.draw_overlay(frame, None, None, None, 10.0, 90.0, None, [], None, None)  # 全部未检测到时不报错。
        self.assertEqual(frame.shape, canvas.shape)  # 画面尺寸不变。
        canvas2 = self.task.draw_overlay(frame, None, (50, 40, 300, 200), (200, 140, 12), 10.0, 90.0, None, [], None, None)  # 有地图区域与黄点时绘制边界红线。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。

    def test_lie_detector_config(self):
        # 测谎配置已搬到看板（单一数据源），任务页不再展示；看板默认值保持原有约定。
        self.assertTrue(self.task.default_config["Patrol Enabled"])  # 巡逻打怪总开关默认开启。
        for lie_key in ("Lie Detector Auto Solve", "Lie Detector Trigger Feature", "Lie Detector Region Feature", "Lie Detector Threshold", "Lie Detector Trigger Delay", "Lie Alarm Sound"):
            self.assertNotIn(lie_key, self.task.default_config)  # 测谎六键已搬到看板。
        self.assertTrue(DASHBOARD_DEFAULTS["Lie Detector Auto Solve"])  # 看板侧默认开启，全部任务默认支持。
        self.assertEqual("测谎触发", DASHBOARD_DEFAULTS["Lie Detector Trigger Feature"])  # 触发标注默认名。
        self.assertEqual("测谎坐标框", DASHBOARD_DEFAULTS["Lie Detector Region Feature"])  # 坐标框标注默认名。
        self.assertEqual(0.75, DASHBOARD_DEFAULTS["Lie Detector Threshold"])  # 触发匹配阈值默认 0.75。
        self.assertEqual(5.0, DASHBOARD_DEFAULTS["Lie Detector Trigger Delay"])  # 触发延迟默认 5 秒。


if __name__ == '__main__':
    unittest.main()
