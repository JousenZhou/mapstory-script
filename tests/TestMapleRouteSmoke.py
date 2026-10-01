# MapleRouteTask 回归冒烟测试：验证配置裁剪/新增、定位得分校验、apply_cmd 的按住-换键-松键时序、
# 看门狗卡住脱困、pop_held_keys 追加上下梯键。均通过 patch 执行器按键方法在无设备/无画面下断言，不触发真实输入。
import time
from unittest.mock import patch

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MapleRouteTask import MapleRouteTask  # 导入被测路线任务类。


class TestMapleRouteSmoke(TaskTestCase):
    task_class = MapleRouteTask

    config = config

    def setUp(self):  # task 为类级共享实例，每个用例前复位可变按键/计时状态，避免用例间串扰。
        self.task._held_move_key = None  # 未按住左右移动键。
        self.task._held_ud_key = None  # 未按住上下梯键。
        self.task._held_attack_key = None  # 未按住攻击键。
        self.task._t_last_jump = 0.0  # 上次跳跃时间复位。
        self.task._t_last_tp = 0.0  # 上次瞬移时间复位。

    def test_config_trimmed_and_route_keys_added(self):  # 巡逻往返键应被裁剪，路线跟随键应补齐。
        for removed in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent", "Stuck Seconds", "Resume Wait Seconds"):
            self.assertNotIn(removed, self.task.default_config)  # 由路线跟随+看门狗取代。
            self.assertNotIn(removed, self.task.config_description)
        for kept in ("Map Name", "Jump Key", "Teleport Key", "Locate Score Max", "Route Attack Enabled", "Teleport To Edge", "Watchdog Seconds"):
            self.assertIn(kept, self.task.default_config)  # 路线专属配置存在。
        self.assertIn("Minimap Feature", self.task.default_config)  # 复用父类小地图配置。

    def test_validate_config_locate_score(self):  # 定位得分上限须在 0-1，非数字或越界报错。
        self.assertIsNone(self.task.validate_config("Locate Score Max", 0.5))  # 合法。
        self.assertIsNotNone(self.task.validate_config("Locate Score Max", 1.5))  # 越界。
        self.assertIsNotNone(self.task.validate_config("Locate Score Max", "abc"))  # 非数字。

    def test_apply_cmd_movement_hold_release(self):  # 左右指令：按下新键、换向先松旧、none 时松键，全程只保留一个移动键。
        self.task._held_move_key = None  # 初始未按住。
        with patch.object(self.task, "send_key_down") as down, patch.object(self.task, "send_key_up") as up:
            self.task.apply_cmd('right', 'none', 'none', 'space', '', 1.0)  # 向右。
            self.assertEqual('right', self.task._held_move_key)  # 记录按住右。
            down.assert_called_once_with('right')  # 按下右键一次。
            up.assert_not_called()  # 无旧键可松。
            self.task.apply_cmd('left', 'none', 'none', 'space', '', 1.0)  # 换向左。
            self.assertEqual('left', self.task._held_move_key)  # 换键成功。
            up.assert_called_with('right')  # 先松旧右键。
            self.task.apply_cmd('none', 'none', 'none', 'space', '', 1.0)  # 无指令。
            self.assertIsNone(self.task._held_move_key)  # 松开并保持静止。
            up.assert_called_with('left')  # 最后松开左键。

    def test_apply_cmd_ladder_and_jump(self):  # 上下指令按住梯键；jump 指令在过间隔后点按跳跃键一次。
        self.task._held_ud_key = None  # 初始未按住上下。
        self.task._t_last_jump = 0.0  # 距上次跳跃足够久。
        with patch.object(self.task, "send_key_down") as down, patch.object(self.task, "send_key") as tap:
            self.task.apply_cmd('none', 'up', 'jump', 'space', '', 1.0)  # 爬梯并跳跃。
            self.assertEqual('up', self.task._held_ud_key)  # 按住上键。
            down.assert_any_call('up')  # 按下上键。
            tap.assert_any_call('space', down_time=0.05)  # 触发一次跳跃点按。
            tapped_after = tap.call_count  # 记录当前点按次数。
            self.task.apply_cmd('none', 'up', 'jump', 'space', '', 1.0)  # 立刻再次同样指令。
            self.assertEqual(tapped_after, tap.call_count)  # 未过最小间隔，不重复乱跳。

    def test_apply_cmd_teleport_downgrade_to_jump(self):  # 主循环在无瞬移键时把 teleport 降级为 jump，这里仅验证有瞬移键时点按瞬移键。
        self.task._t_last_tp = 0.0  # 距上次瞬移足够久。
        with patch.object(self.task, "send_key") as tap:
            self.task.apply_cmd('none', 'none', 'teleport', 'space', 'e', 1.0)  # 配置了瞬移键 e。
            tap.assert_any_call('e', down_time=0.05)  # 点按瞬移键。

    def test_update_watchdog_stuck_triggers_escape(self):  # 按住移动但全局位置长期停滞时，看门狗松键并反向短移脱困。
        self.task._held_move_key = 'right'  # 正在向右按住移动。
        with patch.object(self.task, "send_key_up") as up, patch.object(self.task, "send_key") as tap:
            stale_time = time.time() - 100  # 很久以前的锚点。
            anchor, _ = self.task.update_watchdog((100, 100), (100, 100), stale_time, wd_range=5.0, wd_seconds=8.0)  # 停滞超时。
            self.assertIsNone(anchor)  # 触发脱困后清空锚点。
            self.assertIsNone(self.task._held_move_key)  # 移动键被清空。
            up.assert_called_with('right')  # 松开原右键。
            tap.assert_any_call('left', down_time=0.5)  # 反向短移。
        # 正常移动时不触发脱困：位移超过阈值应重置锚点为当前位置。
        self.task._held_move_key = 'right'
        anchor2, _ = self.task.update_watchdog((150, 100), (100, 100), time.time(), wd_range=5.0, wd_seconds=8.0)
        self.assertEqual((150, 100), anchor2)  # 明显移动，锚点更新到新位置。

    def test_pop_held_keys_includes_up_down(self):  # 重写后需把上下梯键一并上报并清空。
        self.task._held_ud_key = 'up'  # 正按住上键。
        with patch.object(self.task, "send_key_up"):  # 巡逻松键走的是属性清空，这里兜底 patch 实际按键。
            keys = self.task.pop_held_keys()  # 上报持有键。
        self.assertIn('up', keys)  # 上下键被包含。
        self.assertIsNone(self.task._held_ud_key)  # 已清空。


if __name__ == '__main__':
    import unittest
    unittest.main()
