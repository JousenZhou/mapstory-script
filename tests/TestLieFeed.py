# 测谎「逐帧喂入层」回归测试（纯计算，不依赖 Qt/torch/录像）：
#   - compute_effective_fps：帧间隔 dt → 有效帧率的夹取与兜底（实时与回放共用的唯一 fps 口径）。
#   - should_reset：区域首帧/位移阈值的重置判定（替换实时与回放的两份手写逻辑，杜绝漂移）。
#   - encode_trace / decode_trace：喂帧日程的边车紧凑序列化往返（含 rect=None 与脏行容错）。
#   - iter_replay_plan：按日程生成回放执行计划——游标按 dt×src_fps 跳帧（复现实时掉帧）、fps 随 dt、reset 透传。
import unittest

from src.liedetector.feed import (  # 被测纯函数与数据类。
    EFFECTIVE_FPS_MAX,
    EFFECTIVE_FPS_MIN,
    FeedStep,
    compute_effective_fps,
    decode_trace,
    encode_trace,
    iter_replay_plan,
    should_reset,
)


class TestComputeEffectiveFps(unittest.TestCase):
    """帧间隔 → 有效帧率换算：唯一 fps 口径来源，实时不再写死 30、回放不再写死 src_fps。"""

    def test_normal_interval_is_inverse_dt(self):
        # 常规间隔：dt=0.02s → 1/dt=50fps，落在 [5,60] 内原样返回。
        self.assertAlmostEqual(50.0, compute_effective_fps(0.02), places=6)
        # dt=1/30≈0.0333s → 30fps（算法设计基准）。
        self.assertAlmostEqual(30.0, compute_effective_fps(1.0 / 30.0), places=6)

    def test_clamped_to_bounds(self):
        # 过快连写（dt 极小）→ 上限夹到 60，不放过阈值被压没。
        self.assertEqual(EFFECTIVE_FPS_MAX, compute_effective_fps(0.001))  # 1/0.001=1000 → 夹 60。
        # 过慢/卡顿（dt 很大）→ 下限夹到 5，防止阈值被拉爆。
        self.assertEqual(EFFECTIVE_FPS_MIN, compute_effective_fps(1.0))  # 1/1.0=1 → 夹 5。

    def test_invalid_dt_falls_back_to_default(self):
        # dt<=1e-6（含 0、负数）视为非法间隔，回退兜底帧率 30。
        self.assertEqual(30.0, compute_effective_fps(0.0))
        self.assertEqual(30.0, compute_effective_fps(-0.5))
        self.assertEqual(30.0, compute_effective_fps(1e-9))
        # 可显式指定兜底值。
        self.assertEqual(24.0, compute_effective_fps(0.0, default_fps=24.0))
        # 非数值（TypeError 路径）也安全回退兜底，绝不抛。
        self.assertEqual(30.0, compute_effective_fps(None))


class TestShouldReset(unittest.TestCase):
    """区域重置判定：首帧、任一边位移/尺寸超阈值才重置；本帧无区域不重置。"""

    def test_first_region_resets(self):
        # last 为空（首帧）+ new 有区域 → 必须重置以新尺寸建会话。
        self.assertTrue(should_reset(None, (10, 10, 100, 100)))

    def test_no_new_region_does_not_reset(self):
        # 本帧没采到区域（None）：不因此重置，沿用旧区域。
        self.assertFalse(should_reset((10, 10, 100, 100), None))
        self.assertFalse(should_reset(None, None))  # 首帧也没采到：同样不重置。

    def test_small_shift_keeps_region(self):
        # 四边都变化 <= 阈值（默认 4）：视为抖动，不重置。
        self.assertFalse(should_reset((10, 10, 100, 100), (13, 12, 102, 98)))

    def test_boundary_exactly_threshold_not_reset(self):
        # 严格大于才重置：恰好等于阈值（差=4）不重置。
        self.assertFalse(should_reset((10, 10, 100, 100), (14, 10, 100, 100)))

    def test_any_side_over_threshold_resets(self):
        # 任一边变化 > 阈值即重置：x/y/w/h 各自超阈值都要触发。
        self.assertTrue(should_reset((10, 10, 100, 100), (15, 10, 100, 100)))  # x 位移 5。
        self.assertTrue(should_reset((10, 10, 100, 100), (10, 10, 100, 106)))  # h 变化 6。
        # 自定义阈值更小：位移 3 超过 threshold=2 触发重置。
        self.assertTrue(should_reset((10, 10, 100, 100), (13, 10, 100, 100), threshold=2))


class TestTraceCodec(unittest.TestCase):
    """喂帧日程 encode/decode 往返：紧凑数组序列化、rect=None 存 null、脏行容错。"""

    def test_roundtrip_preserves_steps(self):
        trace = [
            FeedStep(dt_ms=33, rect=(0, 0, 200, 150), reset=True, source="color", conf=0.81234),
            FeedStep(dt_ms=50, rect=(5, -2, 190, 140), reset=False, source="border", conf=0.42),
            FeedStep(dt_ms=40, rect=None, reset=False, source="waiting", conf=0.0),  # 沿用旧区域：rect=None。
        ]
        rows = encode_trace(trace)
        self.assertEqual(3, len(rows), "每步编码为一行")
        # 行格式：[dt_ms, x, y, w, h, reset, source, conf]；rect=None 时坐标写 null。
        self.assertEqual([33, 0, 0, 200, 150, True, "color", 0.8123], rows[0])  # conf 四舍五入到 4 位。
        self.assertEqual([None, None, None, None], rows[2][1:5], "rect=None 应编码为四个 null")
        back = decode_trace(rows)
        self.assertEqual(len(trace), len(back), "往返步数一致")
        for original, decoded in zip(trace, back):
            self.assertEqual(original.dt_ms, decoded.dt_ms)
            self.assertEqual(original.rect, decoded.rect, "rect 往返保持元组或 None")
            self.assertEqual(original.reset, decoded.reset)
            self.assertEqual(original.source, decoded.source)
            self.assertAlmostEqual(original.conf, decoded.conf, places=4, msg="conf 精度按编码 4 位对齐")

    def test_encode_empty_and_decode_empty(self):
        # 空日程编码得空列表；空/None 解码回空列表（旧记录无字段的兼容路径）。
        self.assertEqual([], encode_trace([]))
        self.assertEqual([], decode_trace([]))
        self.assertEqual([], decode_trace(None))

    def test_decode_skips_dirty_rows(self):
        # 单行结构损坏（长度不足/类型非法）应跳过，不影响其余好行。
        rows = [
            [33, 0, 0, 200, 150, True, "color", 0.5],
            ["bad", "row"],  # 脏行：dt 非整数。
            [40, 1, 2, 3, 4, False, "border", 0.3],
        ]
        back = decode_trace(rows)
        self.assertEqual(2, len(back), "脏行应被跳过")
        self.assertEqual(33, back[0].dt_ms)
        self.assertEqual(40, back[1].dt_ms)
        self.assertEqual((1, 2, 3, 4), back[1].rect)


class TestIterReplayPlan(unittest.TestCase):
    """回放执行计划：游标按 dt×src_fps 跳帧复现掉帧、fps 由 dt 换算、rect/reset 透传。"""

    def test_cursor_advances_by_dt(self):
        # src_fps=30，每步 dt=100ms → 每步应前进 round(0.1*30)=3 帧，复现「实时每 100ms 喂一帧、丢弃中间 2 帧」。
        trace = [
            FeedStep(dt_ms=100, rect=None, reset=True, source="color", conf=0.5),
            FeedStep(dt_ms=100, rect=None, reset=False, source="color", conf=0.5),
            FeedStep(dt_ms=100, rect=None, reset=False, source="border", conf=0.4),
        ]
        plan = list(iter_replay_plan(trace, src_fps=30.0))
        self.assertEqual([0, 3, 6], [s.frame_index for s in plan], "游标每步前进 3 帧")
        # 每步 fps 由 dt=0.1s 换算 → 10fps（在 [5,60] 内原样）。
        for s in plan:
            self.assertAlmostEqual(10.0, s.fps, places=6)

    def test_fast_steps_advance_one_frame_min(self):
        # dt 极小（快于一个源帧）时游标至少前进 1 帧，不能倒退或原地。
        trace = [
            FeedStep(dt_ms=1, rect=None, reset=True, source="color", conf=0.5),
            FeedStep(dt_ms=1, rect=None, reset=False, source="color", conf=0.5),
        ]
        plan = list(iter_replay_plan(trace, src_fps=30.0))
        self.assertEqual([0, 1], [s.frame_index for s in plan], "round(0.001*30)=0 → 兜底前进 1 帧")
        self.assertEqual(EFFECTIVE_FPS_MAX, plan[0].fps, "dt 极小 → fps 夹到上限 60")

    def test_first_step_zero_dt(self):
        # 首步 dt_ms=0（无真实间隔）：游标从 0 起、fps 回退兜底 30、下一步仍至少前进 1 帧。
        trace = [
            FeedStep(dt_ms=0, rect=(0, 0, 100, 100), reset=True, source="waiting", conf=0.0),
            FeedStep(dt_ms=0, rect=None, reset=False, source="color", conf=0.6),
        ]
        plan = list(iter_replay_plan(trace, src_fps=30.0))
        self.assertEqual(0, plan[0].frame_index)
        self.assertEqual(30.0, plan[0].fps, "dt 非法回退默认 30")
        self.assertEqual(1, plan[1].frame_index, "dt=0 兜底前进 1 帧")
        self.assertEqual((0, 0, 100, 100), plan[0].rect, "rect 原样透传")
        self.assertTrue(plan[0].reset, "reset 原样透传")
        self.assertIsNone(plan[1].rect, "rect=None 透传（沿用旧区域）")

    def test_rect_and_reset_passthrough(self):
        # 混合日程：跳帧数按各自 dt、reset 标志逐步透传，供回放精确复现。
        trace = [
            FeedStep(dt_ms=200, rect=(10, 20, 30, 40), reset=True, source="color", conf=0.7),
            FeedStep(dt_ms=300, rect=(0, 0, 30, 40), reset=False, source="border", conf=0.3),
        ]
        plan = list(iter_replay_plan(trace, src_fps=30.0))
        # 第一步 idx=0；dt=0.2s×30=6 → 第二步 idx=6；dt=0.3×30=9 → 若继续则 idx=15。
        self.assertEqual([0, 6], [s.frame_index for s in plan])
        self.assertEqual(True, plan[0].reset)
        self.assertEqual(False, plan[1].reset)
        self.assertEqual((10, 20, 30, 40), plan[0].rect)
        self.assertAlmostEqual(5.0, plan[1].fps, places=6, msg="dt=0.3→3.33fps→夹下限 5")

    def test_invalid_src_fps_falls_back(self):
        # src_fps 非法（0/过大）→ 回退 30 作为跳帧基准，游标仍按 dt×30 前进。
        trace = [
            FeedStep(dt_ms=100, rect=None, reset=True, source="color", conf=0.5),
            FeedStep(dt_ms=100, rect=None, reset=False, source="color", conf=0.5),
        ]
        plan = list(iter_replay_plan(trace, src_fps=0.0))
        self.assertEqual([0, 3], [s.frame_index for s in plan], "非法 src_fps 回退 30 → 每步前进 3 帧")

    def test_start_frame_offsets_cursor(self):
        # 起点对齐：录像在触发确认即起录、解题晚若干秒才开题，首喂帧对应 mp4 第 K 帧；游标必须从 K 起跳，
        # 否则全程错位、回放喂的不是实时看过的帧序列（失败局跑不出轨迹的主因）。
        trace = [
            FeedStep(dt_ms=100, rect=(0, 0, 50, 50), reset=True, source="color", conf=0.5),
            FeedStep(dt_ms=100, rect=None, reset=False, source="color", conf=0.5),
        ]
        plan = list(iter_replay_plan(trace, src_fps=30.0, start_frame=105))
        self.assertEqual([105, 108], [s.frame_index for s in plan], "游标从 start_frame 起按 dt 步进")

    def test_start_frame_default_and_illegal_values(self):
        # 旧记录无偏移字段：缺省 0 保持原行为；非法/负数偏移安全归零，不报错不倒退。
        trace = [FeedStep(dt_ms=100, rect=None, reset=True, source="color", conf=0.5)] * 2
        self.assertEqual([0, 3], [s.frame_index for s in iter_replay_plan(trace, 30.0)], "缺省起点为 0")
        self.assertEqual([0, 3], [s.frame_index for s in iter_replay_plan(trace, 30.0, -9)], "负偏移归零")
        self.assertEqual([0, 3], [s.frame_index for s in iter_replay_plan(trace, 30.0, "bad")], "非法偏移回退 0")


if __name__ == "__main__":
    unittest.main()
