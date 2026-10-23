# MaplePlatformTask 回归测试：验证配置裁剪与补充、像素换算、区间/滞回/折返纯函数决策、校验、地台局域匹配与画面标注逻辑。
import unittest
from types import SimpleNamespace
from unittest.mock import patch, call

import numpy as np

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MaplePlatformTask import MaplePlatformTask  # 导入地台定点挂机任务类。
from src.dashboard_store import DASHBOARD_DEFAULTS, SUPER_PLATFORM  # 看板共享配置默认值与地台类别名。


def make_box(x, y, w, h, conf=0.9, name="box"):  # 构造匹配框桩对象，只提供被测代码用到的属性。
    return SimpleNamespace(x=x, y=y, width=w, height=h, confidence=conf, name=name, flipped=False)


class TestMaplePlatformSmoke(TaskTestCase):
    task_class = MaplePlatformTask

    config = config

    def test_config_trimmed_and_added(self):
        # 小地图巡逻体系配置应被整体裁剪，地台挂机配置应完整补充（GUI 少键是已知坑，逐键核对）。
        for removed in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent",
                        "Minimap Feature", "Minimap Threshold", "Map Rect",
                        "Dot Hue Min", "Dot Hue Max", "Dot Min Pixels"):
            self.assertNotIn(removed, self.task.default_config)
            self.assertNotIn(removed, self.task.config_description)
        for shared_key in DASHBOARD_DEFAULTS:  # 看板共享键（角色/怪物/攻击/测谎）不落在任务页。
            self.assertNotIn(shared_key, self.task.default_config)
            self.assertNotIn(shared_key, self.task.config_description)
        for kept in ("Stuck Seconds", "Resume Wait Seconds"):  # 卡住脱困沿用巡逻基类的配置键。
            self.assertIn(kept, self.task.default_config)
        for added in ("Platform Feature", "Spot Percent", "Patrol Left Pixels", "Patrol Right Pixels",
                      "Platform Threshold", "Band Exit Margin Pixels", "Home Tolerance Pixels"):
            self.assertIn(added, self.task.default_config)
            self.assertNotIn("Home Tolerance Percent", self.task.default_config)  # 旧的百分比容差键已废弃。
            self.assertIn(added, self.task.config_description)  # 每个新键都要有双语帮助文本。
        self.assertEqual('', self.task.default_config["Platform Feature"])  # 地台标注默认为空待选。
        self.assertEqual(50.0, self.task.default_config["Spot Percent"])  # 定点默认 50%。
        self.assertEqual(100, self.task.default_config["Patrol Left Pixels"])  # 巡逻左偏移默认 100px。
        self.assertEqual(100, self.task.default_config["Patrol Right Pixels"])  # 巡逻右偏移默认 100px。
        self.assertIsInstance(self.task.default_config["Patrol Left Pixels"], int)  # 像素键必须是 int，看板任务卡才会渲染成整数 SpinBox。
        self.assertIsInstance(self.task.default_config["Patrol Right Pixels"], int)
        self.assertEqual(0.7, self.task.default_config["Platform Threshold"])  # 地台阈值默认 0.7。
        self.assertEqual(100, self.task.default_config["Band Exit Margin Pixels"])  # 脱区触发余量默认 100px：走到边界只折返，越过边界+100 才归位。
        self.assertEqual(10, self.task.default_config["Home Tolerance Pixels"])  # 归位容差默认 10px（具体像素值，不再占地台宽度百分比）。
        self.assertIsInstance(self.task.default_config["Band Exit Margin Pixels"], int)  # 像素键必须是 int，看板任务卡才会渲染成整数 SpinBox。
        self.assertIsInstance(self.task.default_config["Home Tolerance Pixels"], int)
        self.assertEqual(SUPER_PLATFORM, self.task.config_type["Platform Feature"]["category"])  # 看板任务卡按「地台」类别渲染下拉框。

    def test_attack_to_move_wait_shared(self):
        # 攻击后摇等待已提升为看板角色配置全局共享键：任务页不落地，运行时由 apply_shared_config 注入，读取助手容错回退。
        self.assertIn("Attack To Move Wait Seconds", DASHBOARD_DEFAULTS)  # 看板共享默认值存在（0.5）。
        self.assertNotIn("Attack To Move Wait Seconds", self.task.default_config)  # 任务私有配置不再携带该键。
        self.task.config.pop("Attack To Move Wait Seconds", None)  # 模拟看板读取失败：键缺失。
        self.assertEqual(0.5, self.task.attack_to_move_wait())  # 回退默认保护值。
        self.task.config.update({"Attack To Move Wait Seconds": 0})  # 0 合法：不等待直接走。
        self.assertEqual(0.0, self.task.attack_to_move_wait())
        self.task.config.update({"Attack To Move Wait Seconds": "0.3"})  # 数字字符串可用。
        self.assertAlmostEqual(0.3, self.task.attack_to_move_wait())
        self.task.config.update({"Attack To Move Wait Seconds": -1})  # 负数夹到 0。
        self.assertEqual(0.0, self.task.attack_to_move_wait())
        self.task.config.update({"Attack To Move Wait Seconds": "abc"})  # 非法残留回退默认值不抛异常。
        self.assertEqual(0.5, self.task.attack_to_move_wait())

    def test_target_loss_buffer_shared(self):
        # 目标丢失缓冲已提升为看板角色配置共享键：默认 0.3，读取助手容错回退，地台任务运行时经 apply_shared_config 采集。
        self.assertIn("Target Loss Buffer Seconds", DASHBOARD_DEFAULTS)  # 看板共享默认值存在（0.3）。
        self.assertAlmostEqual(0.3, DASHBOARD_DEFAULTS["Target Loss Buffer Seconds"])
        self.assertNotIn("Target Loss Buffer Seconds", self.task.default_config)  # 任务私有配置不携带该键（走看板单一数据源）。
        self.task.config.pop("Target Loss Buffer Seconds", None)  # 模拟看板读取失败：键缺失。
        self.assertAlmostEqual(0.3, self.task.target_loss_buffer())  # 回退默认缓冲值。
        self.task.config.update({"Target Loss Buffer Seconds": 0})  # 0 合法：不缓冲立即松键。
        self.assertEqual(0.0, self.task.target_loss_buffer())
        self.task.config.update({"Target Loss Buffer Seconds": "1.5"})  # 数字字符串可用。
        self.assertAlmostEqual(1.5, self.task.target_loss_buffer())
        self.task.config.update({"Target Loss Buffer Seconds": -1})  # 负数夹到 0。
        self.assertEqual(0.0, self.task.target_loss_buffer())
        self.task.config.update({"Target Loss Buffer Seconds": "abc"})  # 非法残留回退默认值不抛异常。
        self.assertAlmostEqual(0.3, self.task.target_loss_buffer())

    def test_band_pixels(self):
        # 表驱动：区间 = 定点画面 x − 左偏移 ~ + 右偏移，不夹范围：用户显式指定像素，超出地台部分同样直观可见。
        cases = [  # (定点x, 左偏移, 右偏移, 期望左, 期望右)
            (300.0, 100.0, 100.0, 200.0, 400.0),
            (300.0, 150.0, 50.0, 150.0, 350.0),  # 左右不对称。
            (100.0, 150.0, 50.0, -50.0, 150.0),  # 左端越出画面也不夹。
            (300.0, 0.0, 0.0, 300.0, 300.0),  # 偏移为 0 退化为一个点。
            (300.0, -10.0, -10.0, 300.0, 300.0),  # 负偏移按 0 处理。
        ]
        for spot_x, left_px, right_px, want_lo, want_hi in cases:
            lo, hi = MaplePlatformTask.band_pixels(spot_x, left_px, right_px)
            self.assertEqual((want_lo, want_hi), (lo, hi), f"spot_x={spot_x} left={left_px} right={right_px}")

    def test_percent_to_x(self):
        # 表驱动：百分比按地台当前画面框换算成画面 x，越界百分比夹到框两端。
        cases = [  # (地台x, 地台w, 百分比, 期望x)
            (100.0, 400.0, 0.0, 100.0),
            (100.0, 400.0, 50.0, 300.0),
            (100.0, 400.0, 70.0, 380.0),
            (100.0, 400.0, 100.0, 500.0),
            (100.0, 400.0, -10.0, 100.0),  # 负数夹到左端。
            (100.0, 400.0, 110.0, 500.0),  # 超 100 夹到右端。
            (0.0, 1280.0, 25.0, 320.0),
        ]
        for px, pw, pct, want in cases:
            self.assertAlmostEqual(want, MaplePlatformTask.percent_to_x(px, pw, pct), places=6)

    def test_out_of_band(self):
        # 表驱动：脱区判定带外扩余量——贴边界与余量内的惯性冲过头都不算脱区（归巡逻折返管），越过余量才算被怪撞飞。
        cases = [  # (角色x, lo, hi, 余量, 期望)
            (300.0, 200.0, 400.0, 0.0, False),
            (200.0, 200.0, 400.0, 0.0, False),  # 恰在左边界。
            (400.0, 200.0, 400.0, 0.0, False),  # 恰在右边界。
            (199.0, 200.0, 400.0, 0.0, True),   # 余量为 0 时退化为旧语义。
            (401.0, 200.0, 400.0, 0.0, True),
            (410.0, 200.0, 400.0, 15.0, False),  # 越过右边界 10px 在余量内：折返惯性，不归位。
            (415.0, 200.0, 400.0, 15.0, False),  # 恰在余量边缘算未脱区。
            (416.0, 200.0, 400.0, 15.0, True),   # 刚超余量才算真脱区。
            (190.0, 200.0, 400.0, 15.0, False),  # 越过左边界 10px 在余量内。
            (184.0, 200.0, 400.0, 15.0, True),   # 左向越余量真脱区。
        ]
        for x, lo, hi, margin, want in cases:
            self.assertEqual(want, MaplePlatformTask.out_of_band(x, lo, hi, margin), f"x={x} margin={margin}")

    def test_home_reached(self):
        # 表驱动：滞回退出归位的容差判定，含等号边界（12px 容差恰好回到位）。
        cases = [  # (角色x, 定点x, 容差px, 期望)
            (300.0, 300.0, 12.0, True),
            (311.0, 300.0, 12.0, True),
            (289.0, 300.0, 12.0, True),
            (312.0, 300.0, 12.0, True),   # 恰在容差边缘算回到位。
            (313.0, 300.0, 12.0, False),  # 刚超容差继续归位。
            (210.0, 300.0, 12.0, False),  # 回到区间内但还没到定点，仍在归位。
        ]
        for x, spot_x, tol_px, want in cases:
            self.assertEqual(want, MaplePlatformTask.home_reached(x, spot_x, tol_px), f"x={x}")

    def test_patrol_turn(self):
        # 表驱动：只在沿当前方向走到对侧边界时折返；向内走不折返；全部按画面像素。
        cases = [  # (direction, x, lo, hi, 期望新方向或 None)
            (1, 399.0, 200.0, 400.0, None),
            (1, 400.0, 200.0, 400.0, -1),   # 向右达右边界折返向左。
            (-1, 200.0, 200.0, 400.0, 1),   # 向左达左边界折返向右。
            (-1, 201.0, 200.0, 400.0, None),
            (1, 200.0, 200.0, 400.0, None),  # 向右贴左边界不折返。
            (-1, 400.0, 200.0, 400.0, None),
        ]
        for direction, x, lo, hi, want in cases:
            self.assertEqual(want, MaplePlatformTask.patrol_turn(direction, x, lo, hi),
                             f"dir={direction} x={x}")

    def test_validate_config(self):
        for key in ("Spot Percent",):
            self.assertIsNone(self.task.validate_config(key, 50.0))  # 合法百分比。
            self.assertIsNone(self.task.validate_config(key, "3.5"))  # 数字字符串也合法。
            self.assertIsNotNone(self.task.validate_config(key, 120))  # 越界报错。
            self.assertIsNotNone(self.task.validate_config(key, -1))  # 负数报错。
            self.assertIsNotNone(self.task.validate_config(key, "abc"))  # 非数字报错。
        for key in ("Patrol Left Pixels", "Patrol Right Pixels", "Band Exit Margin Pixels", "Home Tolerance Pixels"):
            self.assertIsNone(self.task.validate_config(key, 100))  # 合法像素偏移。
            self.assertIsNone(self.task.validate_config(key, 0))  # 0 合法（另一侧偏移非 0 即可）。
            self.assertIsNone(self.task.validate_config(key, "250"))  # 数字字符串也合法。
            self.assertIsNotNone(self.task.validate_config(key, -1))  # 负偏移报错。
            self.assertIsNotNone(self.task.validate_config(key, "abc"))  # 非数字报错。
        self.assertIsNone(self.task.validate_config("Platform Threshold", 0.7))  # 合法阈值。
        self.assertIsNotNone(self.task.validate_config("Platform Threshold", 1.5))  # 超 1 报错。
        self.assertIsNotNone(self.task.validate_config("Platform Threshold", 0))  # 下限 0.01 以下报错。
        self.assertIsNotNone(self.task.validate_config("Platform Threshold", "x"))  # 非数字报错。

    def test_find_platform_local_hit_skips_fullscreen(self):
        # 局域命中时直接返回，不再全屏搜索，抑制重复贴图远处错配。
        hit = make_box(120, 200, 300, 40)
        with patch.object(self.task, "find_one", return_value=hit) as mock_local, \
                patch.object(self.task, "find_one_raw") as mock_full:
            self.task._expect_first = False
            result = self.task.find_platform("地台1", np.zeros((100, 100, 3), dtype=np.uint8), 0.7, (100, 190, 300, 40))
        self.assertIs(hit, result)  # 返回局域命中框。
        mock_full.assert_not_called()  # 未走全屏回退。
        box = mock_local.call_args.kwargs["box"]  # 局域搜索框。
        self.assertLess(box.x, 100)  # 向左外扩。
        self.assertGreater(box.x + box.width, 400)  # 向右外扩。

    def test_find_platform_fallback_fullscreen(self):
        # 局域未命中时回退全屏搜索。
        with patch.object(self.task, "find_one", return_value=None), \
                patch.object(self.task, "find_one_raw", return_value=None) as mock_full:
            result = self.task.find_platform("地台1", np.zeros((100, 100, 3), dtype=np.uint8), 0.7, (100, 190, 300, 40))
        self.assertIsNone(result)  # 全屏也没命中。
        mock_full.assert_called_once()  # 已回退全屏。

    def test_find_platform_no_rect_goes_fullscreen(self):
        # 无参考区域（首帧无标注框）时直接全屏搜索，不做局域。
        with patch.object(self.task, "find_one") as mock_local, \
                patch.object(self.task, "find_one_raw", return_value=None) as mock_full:
            self.task.find_platform("地台1", np.zeros((100, 100, 3), dtype=np.uint8), 0.7, None)
        mock_local.assert_not_called()  # 无参考框不局域。
        mock_full.assert_called_once()

    def test_find_platform_missing_annotation_returns_none(self):
        # 分类名未在模板页标注（ValueError）时返回 None，不回退全屏反复抛错。
        with patch.object(self.task, "find_one", side_effect=ValueError("not found")), \
                patch.object(self.task, "find_one_raw") as mock_full:
            result = self.task.find_platform("已删除的标注", np.zeros((100, 100, 3), dtype=np.uint8), 0.7, (10, 10, 50, 20))
        self.assertIsNone(result)
        mock_full.assert_not_called()

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
        canvas = self.task.draw_overlay(frame, None, None, 50.0, 100, 100, 100, False, None, [], None, None)  # 全部未检测到时不报错。
        self.assertEqual(frame.shape, canvas.shape)  # 画面尺寸不变。
        self.assertTrue((canvas == frame).all())  # 无定位结果时不绘制任何内容。
        self.task.config.update({"Attack Range X Min": -150, "Attack Range X Max": 150,
                                 "Attack Range Y Min": -40, "Attack Range Y Max": 20,
                                 "Melee Distance": 30})  # 攻击判定共享参数（运行时由看板覆盖，测试直接注入）。
        character = make_box(190, 120, 40, 60, name="角色")  # 角色框中心 x=210。
        monster = make_box(260, 140, 30, 30, name="怪1")
        plat_rect = (60, 180, 300, 30)  # 地台框：x 60-360，定点 50% 对应 x=210，区间 210±100 → 110~310。
        canvas2 = self.task.draw_overlay(frame, plat_rect, 50.0, 50.0, 100, 100, 100, False,
                                         character, [monster], monster, monster)  # 定位齐全时绘制地台框、区间色框与边界线、脱区余量线框、定点色框与攻击标注。
        self.assertFalse((canvas2 == frame).all())  # 画面确实被绘制过。
        canvas3 = self.task.draw_overlay(frame, plat_rect, None, 50.0, 100, 100, 0, True,
                                         None, [], None, None)  # 归位中、余量为 0 且不匹配角色：定点色框用兜底尺寸且不强绘余量带，不报错。
        self.assertFalse((canvas3 == frame).all())


if __name__ == '__main__':
    unittest.main()
