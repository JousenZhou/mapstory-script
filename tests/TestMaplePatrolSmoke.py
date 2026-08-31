# MaplePatrolTask 回归测试：验证配置裁剪、校验、黄点检测与画面标注逻辑。
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

import src.tasks.MaplePatrolTask as patrol_module  # 导入巡逻任务模块，供 patch 模块级 og 用。
from src.tasks.MaplePatrolTask import MaplePatrolTask, LIE_MOVE_MAX_STEP  # 导入任务类与解测谎鼠标步长常量。


class TestMaplePatrolSmoke(TaskTestCase):
    task_class = MaplePatrolTask

    config = config

    def test_config_trimmed(self):
        # 定时位移与转身策略配置应被裁剪，攻击与巡逻配置应保留。
        for removed in ("Move Interval", "Move Away Seconds", "Move Back Seconds", "Turn Interval"):
            self.assertNotIn(removed, self.task.default_config)
            self.assertNotIn(removed, self.task.config_description)
        for kept in ("Attack Key", "Melee Attack Key", "Melee Distance", "Monster Features", "GPU Match", "Patrol Left Percent", "Patrol Right Percent", "Minimap Feature", "Character Facing Left Feature", "Character Facing Right Feature"):
            self.assertIn(kept, self.task.default_config)

    def test_validate_config(self):
        self.assertIsNone(self.task.validate_config("Attack Key", "a"))  # 合法按键复用父类校验。
        self.assertIsNotNone(self.task.validate_config("Melee Attack Key", "not_a_key"))  # 非法按键报错。
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
        # 解测谎配置：默认开启，标注分类名默认与模板页标注一致。
        self.assertTrue(self.task.default_config["Patrol Enabled"])  # 巡逻打怪总开关默认开启。
        self.assertTrue(self.task.default_config["Lie Detector Auto Solve"])  # 默认开启。
        self.assertEqual("测谎触发", self.task.default_config["Lie Detector Trigger Feature"])  # 触发标注默认名。
        self.assertEqual("测谎坐标框", self.task.default_config["Lie Detector Region Feature"])  # 坐标框标注默认名。
        self.assertEqual("", self.task.default_config["Lie Alarm Sound"])  # 报警音频默认留空（未上传不报警）。
        self.assertIsNone(self.task.validate_config("Lie Detector Threshold", 0.7))  # 合法阈值。
        self.assertIsNotNone(self.task.validate_config("Lie Detector Threshold", "abc"))  # 非数字报错。
        self.assertIsNotNone(self.task.validate_config("Lie Detector Threshold", 0))  # 0 不在开区间内报错。
        self.assertIsNotNone(self.task.validate_config("Lie Detector Threshold", 1.5))  # 越界报错。

    def test_play_lie_alarm_guards(self):
        # 报警守卫：未配置或文件不存在时不播放且不抛异常。
        self.assertFalse(self.task.play_lie_alarm(""))  # 留空不报警。
        self.assertFalse(self.task.play_lie_alarm(None))  # None 同样不报警。
        self.assertFalse(self.task.play_lie_alarm("assets/not_exist_alarm.mp3"))  # 文件不存在不报警。
        self.assertIn("Lie Alarm Sound", self.task.config_description)  # 帮助文本已配置。

    def test_find_lie_box(self):
        with patch.object(self.task, "find_one", side_effect=ValueError("no annotation")):  # 标注未标注时框架抛 ValueError。
            self.assertIsNone(self.task.find_lie_box("测谎触发", None, 0.7))  # 按未匹配处理不中断。
        box = SimpleNamespace(x=1, y=2, width=3, height=4)  # 假匹配框。
        with patch.object(self.task, "find_one", return_value=box):  # 匹配命中。
            self.assertIs(box, self.task.find_lie_box("测谎触发", None, 0.7))  # 原样返回匹配框。

    def test_get_lie_region_box(self):
        with patch.object(self.task, "get_box_by_name", side_effect=ValueError("no annotation")):  # 标注未标注时框架抛 ValueError。
            self.assertIsNone(self.task.get_lie_region_box("测谎坐标框"))  # 按坐标框缺失处理不中断。
        box = SimpleNamespace(x=1, y=2, width=3, height=4)  # 标注记录的坐标框。
        with patch.object(self.task, "get_box_by_name", return_value=box):  # 标注存在。
            self.assertIs(box, self.task.get_lie_region_box("测谎坐标框"))  # 直接返回记录坐标，不做模板匹配。
        with patch.object(self.task, "get_box_by_name", return_value=None):  # 无画面帧时框架返回 None。
            self.assertIsNone(self.task.get_lie_region_box("测谎坐标框"))  # 按坐标框缺失处理。

    def test_move_mouse_toward(self):
        moves = []  # 记录发给游戏窗口的鼠标坐标。
        with patch.object(self.task, "move", side_effect=lambda x, y: moves.append((x, y))):  # 拦截鼠标移动。
            pos = self.task.move_mouse_toward(None, (200, 100))  # 首帧无参照点直接跳到预测点。
            self.assertEqual((200, 100), pos)  # 到达预测点。
            self.assertEqual([(200, 100)], moves)  # 发出一次移动。
            pos = self.task.move_mouse_toward((200, 100), (400, 100))  # 距离 200 超出单步上限。
            self.assertEqual((200 + LIE_MOVE_MAX_STEP, 100), pos)  # 按最大步长截断。
            pos = self.task.move_mouse_toward(pos, (pos[0] + 10, 100))  # 距离在上限内。
            self.assertEqual((210 + LIE_MOVE_MAX_STEP, 100), pos)  # 一步到位。
        self.assertEqual(3, len(moves))  # 三次移动全部发出。

    def test_draw_lie_overlay(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)  # 构造纯色合成画面。
        canvas = self.task.draw_lie_overlay(frame, (50, 40, 300, 200), [], [], {}, None, None, (0.0, 0.0))  # 空数据不报错。
        self.assertEqual(frame.shape, canvas.shape)  # 画面尺寸不变。
        track = SimpleNamespace(track_id=1, rect=(20, 20, 30, 30), kalman_velocity=(1.5, -0.5))  # 假轨迹：框+速度。
        target = SimpleNamespace(track_id=1, rect=(20, 20, 30, 30))  # 假目标轨迹。
        trails = {}  # 空尾迹表，绘制时自动建立。
        canvas2 = self.task.draw_lie_overlay(frame, (50, 40, 300, 200), [((10.0, 10.0, 40.0, 40.0), 0.9)], [track], trails, target, (100, 90), (0.6, 0.8))  # 全要素绘制。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。
        self.assertIn(1, trails)  # 轨迹尾迹已建立。
        self.task.draw_lie_overlay(frame, (50, 40, 300, 200), [], [], trails, None, None, (0.0, 0.0))  # 轨迹消亡后尾迹被清理。
        self.assertNotIn(1, trails)  # 死轨迹尾迹已删除。

    def test_draw_lie_annotations(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)  # 构造纯色合成画面。
        canvas = self.task.draw_lie_annotations(frame, None, None)  # 两个标注都未命中时不报错不绘制。
        self.assertTrue((canvas == frame).all())  # 画面保持原样。
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)  # 假触发标注框。
        region = SimpleNamespace(x=100, y=50, width=200, height=150)  # 假坐标框。
        canvas2 = self.task.draw_lie_annotations(frame, trigger, region)  # 两个标注都命中时框选。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。
        self.assertEqual(0, canvas2[10, 10][0])  # 触发框红色：蓝通道为 0。
        self.assertEqual(255, canvas2[10, 10][2])  # 触发框红色：红通道拉满。
        self.assertEqual(255, canvas2[50, 100][0])  # 坐标框青色：蓝通道拉满。
        self.assertEqual(255, canvas2[50, 100][1])  # 坐标框青色：绿色通道拉满。

    def test_solve_lie_detector_finishes_when_trigger_gone(self):
        frame = np.full((300, 400, 3), 20, dtype=np.uint8)  # 构造合成画面。
        trigger = SimpleNamespace(x=10, y=10, width=50, height=20)  # 假触发标注框。
        region = SimpleNamespace(x=100, y=50, width=200, height=150)  # 假坐标框。
        state = {"tick": 0}  # 触发标注检查次数计数。
        def fake_find_lie_box(name, frame_arg, threshold):  # 模拟触发标注匹配：首轮在，次轮消失代表测谎结束。
            if name == "测谎触发":  # 触发标注。
                state["tick"] += 1  # 计数累加。
                return trigger if state["tick"] == 1 else None  # 首轮命中，次轮消失代表测谎结束。
            return None  # 触发以外的标注不走模板匹配。
        fake_detector = SimpleNamespace(detect=lambda crop: [])  # 检测不到任何图形。
        with patch.object(patrol_module, "og") as mock_og, \
                patch.object(self.task, "find_lie_box", side_effect=fake_find_lie_box), \
                patch.object(self.task, "get_lie_region_box", return_value=region), \
                patch.object(self.task, "get_lie_detector", return_value=fake_detector), \
                patch.object(self.task, "ensure_in_front"), \
                patch.object(self.task, "move"):  # 隔离 UI 推送、标注匹配、检测器与窗口/鼠标。
            self.task.solve_lie_detector(frame, "测谎触发", "测谎坐标框", 0.7)  # 运行子循环直到触发消失。
        self.assertEqual(2, state["tick"])  # 第二轮触发消失后退出。
        self.assertTrue(mock_og.my_app.update_vision.called)  # 解测谎画面已推送给 UI。


if __name__ == '__main__':
    unittest.main()
