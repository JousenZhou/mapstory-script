# TestCursorFootprintMask：验证测谎「准星足迹掩膜」（9s 缺陷修复）的三条契约。
#
# 缺陷本体：测谎回放画面里烧进了上一轮操作留下的绿色准星十字，它自己会动，是透明相位里最强的
#   时序残差源。实测第 9s（t452）起 border 弱测量把跟踪中心钉在准星上 160 帧不分离
#   （距准星中位 16.7px、重合 99.4%）——用户看到的「丢失后平滑跟着鼠标走」。
# 修复口径（A/B 三档 all/border/witness 定案，只遮弱测量）：
#   ① border 弱测量拿到「准星足迹内置零」的证据图；
#   ② 旋转见证器必须继续拿到**原始**证据图——遮了它会让答案残差簇提前 ≈25 帧出生，
#      `windowed_deg`（累计转角×参考窗/年龄）被年龄稀释从 78° 跌到 44°、过不了
#      witness_min_cum_deg=60 的门槛，锚点永不升格 → 缰绳失效 → 尾部 t853 从 7.4px 回退到 100.5px；
#   ③ 绝不就地写 `temporal.normalized`（它是 CUDA Graph 的静态输出缓冲，且白相/后续消费方都读原图）。
#
# 运行：按仓库规范逐模块起进程：.venv\Scripts\python.exe -m tests.TestCursorFootprintMask
import unittest

import cv2  # 画准星与目标
import numpy as np

from src.liedetector.shape_session import (  # 被测会话与参数。
    ShapeTrackParams,
    ShapeTrackSession,
)
from src.liedetector.shape_tracking import _to_numpy_array  # 后端数组统一取回 NumPy。

SIDE = 200  # 合成帧边长（处理尺度=1.0，故与全尺度同坐标）。
CROSS = (60, 140)  # 准星中心 (x,y)：放在画面左下，与目标轨迹错开。
CROSS_ARM = 9  # 准星十字臂长（像素）。


def make_frame(target_xy, tick, target_fg=70, cross=True):  # 造一帧：暗底 + 目标方块 + 绿色准星十字。
    frame = np.full((SIDE, SIDE, 3), 40, np.uint8)  # 深灰底（水面近似静止）。
    tx, ty = target_xy  # 目标中心
    cv2.rectangle(frame, (tx - 14, ty - 14), (tx + 14, ty + 14), (target_fg, target_fg, target_fg), -1)  # 灰目标（透明相位）
    if cross:  # 烧录准星：高对比绿十字，逐帧闪现/消失（静止十字会被光流补偿掉、留不下时序残差）
        cv2.line(frame, (CROSS[0] - CROSS_ARM, CROSS[1]), (CROSS[0] + CROSS_ARM, CROSS[1]), (40, 235, 40), 3)  # 横臂
        cv2.line(frame, (CROSS[0], CROSS[1] - CROSS_ARM), (CROSS[0], CROSS[1] + CROSS_ARM), (40, 235, 40), 3)  # 竖臂
    return frame


def white_phase_session(**overrides):  # 建会话：处理尺度 1.0、短选窗，便于快速跑到透明相位。
    params = ShapeTrackParams(
        process_scale=1.0, particle_count=60, global_proposals=80, control_count=40,
        coarse_top=8, refine_per_top=4, template_learn_window=0.15, **overrides)
    session = ShapeTrackSession(params=params, logger=None)  # 真实会话，光流/证据全走真实实现。
    session.reset(SIDE, SIDE)  # 建局
    return session


class TestFootprintContract(unittest.TestCase):
    """`_mask_cursor_footprint` 的本体契约：遮里面、留外面、不动原图、五条退路。"""

    def setUp(self):
        self.session = white_phase_session()  # 只用来取参数与调用被测方法。

    def _map_and_frame(self, value=8.0):  # 一张全常数证据图 + 一帧带准星画面。
        frame = make_frame((150, 60), tick=0)  # 准星在 CROSS（本体用例不靠闪现，直接面在即可）
        evidence = np.full((SIDE, SIDE), value, np.float32)  # 证据图（与处理帧同尺寸）
        return evidence, frame

    def test_footprint_zeroed_outside_untouched(self):
        """足迹内置零、足迹外逐比特不变：掩膜只该拿走准星自己那几列残差。"""

        evidence, frame = self._map_and_frame()  # 常数 8.0 证据
        out = self.session._mask_cursor_footprint(evidence, frame)  # 被测
        self.assertIsNot(out, evidence, "应产出新数组而不是改原图")
        kept = out > 0  # 仍保留残差的区域（足迹外）
        dug = ~kept  # 被遮掉的区域（足迹内）
        self.assertTrue(kept.any(), "足迹外必须原样保留（不能整图清零）")
        self.assertTrue(dug.any(), "足迹内必须被遮（否则掩膜没生效）")
        self.assertTrue(np.all(out[kept] == 8.0), "保留区数值必须逐一等于原值")
        self.assertTrue(np.all(out[dug] == 0.0), "遮罩区必须为零")
        ys, xs = np.where(dug)  # 被遮区域的像素坐标
        self.assertLess(abs(float(xs.mean()) - CROSS[0]), 12.0, "被遮区域应贴着准星（不能误挖目标）")
        self.assertLess(abs(float(ys.mean()) - CROSS[1]), 12.0, "被遮区域应贴着准星（不能误挖目标）")

    def test_original_evidence_not_mutated(self):
        """原图不可写：见证器与后续消费方读的是同一张 normalized，就地遮会污染它们。"""

        evidence, frame = self._map_and_frame()  # 常数图
        snapshot = evidence.copy()  # 留底
        self.session._mask_cursor_footprint(evidence, frame)  # 被测
        self.assertTrue(np.array_equal(evidence, snapshot), "入参证据图必须逐比特不变")

    def test_no_cursor_returns_same_object(self):
        """无准星帧（绿色像素不足）原样返回同一对象：行为退回无掩膜口径。"""

        evidence = np.full((SIDE, SIDE), 5.0, np.float32)  # 证据图
        frame = make_frame((150, 60), tick=0, cross=False)  # 画面里没画准星
        self.assertIs(self.session._mask_cursor_footprint(evidence, frame), evidence, "检不出准星不应产生新图")

    def test_switch_off_and_shape_mismatch_are_noops(self):
        """两条退路：开关关闭、足迹与证据尺寸不一致，都必须原样返回。"""

        evidence, frame = self._map_and_frame()  # 正常可遮的一对照
        off = white_phase_session(cursor_mask_enabled=False)  # 关开关
        self.assertIs(off._mask_cursor_footprint(evidence, frame), evidence, "开关关闭不应遮")
        small = np.full((SIDE // 2, SIDE // 2), 8.0, np.float32)  # 证据尺寸与处理帧不同步
        self.assertIs(self.session._mask_cursor_footprint(small, frame), small, "尺寸不符宁可不遮")

    def test_large_green_blob_is_not_masked(self):
        """面积上限护栏：被绿判据误收的大面绿块（地形/技能光效）不该遮 —— 遮掉等于自断一成证据。

        实测 11 个素材里真准星足迹稳定在 0.04%~2.6%，但有俩素材的个别帧最大绿块飘到 9.3%/11.1%，
        护栏就是为它们存在的（放行全部真准星帧、挡住大面绿块）。
        """

        frame = np.full((SIDE, SIDE, 3), 40, np.uint8)  # 暗底
        cv2.rectangle(frame, (20, 20), (20 + 49, 20 + 49), (40, 235, 40), -1)  # 50×50 绿块 = 面积的 6.25% > 上限 5%
        evidence = np.full((SIDE, SIDE), 8.0, np.float32)  # 常数证据
        self.assertIs(self.session._mask_cursor_footprint(evidence, frame), evidence, "超上限的大绿块不应遮")
        loose = white_phase_session(cursor_mask_max_frac=0.2)  # 把上限放宽到 20%
        masked = loose._mask_cursor_footprint(evidence, frame)  # 同一帧再遮一次
        self.assertIsNot(masked, evidence, "放宽上限后同帧应可遮：证明上句是护栏生效而非检不出绿色")
        self.assertFalse(bool((masked > 0).all()), "放宽后确实挖掉了绿块区域")

    def test_torch_backend_masked_on_device(self):
        """torch 路径（CPU 张量也走同一分支）：产出同设备同类型的新张量，足迹外不动。"""

        try:
            import torch  # 只为构造一个张量证据；无 torch 时跳过本用例。
        except Exception:
            self.skipTest("本机无 torch，跳过张量分支")
        evidence, frame = self._map_and_frame()  # numpy 对照图
        tensor = torch.from_numpy(evidence.copy())  # 同数据的 CPU 张量
        out = self.session._mask_cursor_footprint(tensor, frame)  # 被测
        self.assertTrue(torch.is_tensor(out), "应返回张量而不是下载成 numpy")
        self.assertEqual(tuple(out.shape), tuple(tensor.shape), "形状不变")
        self.assertEqual(out.dtype, tensor.dtype, "dtype 不变")
        masked = out.numpy()  # 取回比较
        self.assertFalse(np.array_equal(masked, evidence), "张量分支确实遮了")
        self.assertTrue(np.all(masked[masked > 0] == 8.0), "足迹外原值保留")


class TestConsumerSplit(unittest.TestCase):
    """消费方分档（修复的灵魂）：border 弱测量吃遮过的图，旋转见证器吃原图。"""

    def _drive(self, **overrides):  # 跑到「透明相位 + 检查点」，返回三个消费方各自看到的证据图。
        session = white_phase_session(**overrides)  # 真实会话
        seen = {"aligner": {}, "border": {}, "witness": {}}  # tick → 证据图
        origin_update = session.aligner.update  # 对齐器原实现

        def spy_aligner(frame):  # 记出厂原图
            ev = origin_update(frame)  # 真实证据
            if ev is not None:
                seen["aligner"][session.frame_index] = _to_numpy_array(ev.normalized).copy()
            return ev

        session.aligner.update = spy_aligner  # 实例级装桩

        def spy_witness(evidence_np, gray_np, dt=None):  # 记见证器看到的图（继续调原实现）
            seen["witness"][session.frame_index] = np.asarray(evidence_np).copy()
            return origin_witness_update(evidence_np, gray_np, dt)

        for tick in range(12):  # 白相：学模板（亮白目标）
            session.update(make_frame((150, 60), tick, target_fg=255), 30)
        self.assertIsNotNone(session.tracker, "白相应学到模板并建跟踪器")
        origin_observe_border = session.tracker.observe_border  # 跟踪器原实现
        origin_witness_update = session.witness.update  # 见证器原实现
        session.tracker.observe_border = lambda m: (  # 装桩需在跟踪器建立之后
            seen["border"].__setitem__(session.frame_index, _to_numpy_array(m).copy()),
            origin_observe_border(m))[1]
        session.witness.update = spy_witness  # 装桩

        for tick in range(12, 12 + 2 * session.params.witness_check_interval):  # 透明相位：目标淡出并右移，准星隔帧闪现
            session.update(make_frame((150 + (tick - 12) * 2, 60), tick, target_fg=70, cross=(tick % 2 == 0)), 30)
        session.tracker.observe_border = origin_observe_border  # 卸桩（会话是局部对象，仅为对称）
        session.witness.update = origin_witness_update
        return seen

    def test_border_sees_masked_map(self):
        """border 弱测量：准星足迹内的残差被遮为 0、足迹外逐比特不动。

        掩盖只在准星本帧确有残差时才减总量（本场景里十字隔帧闪现，熄灭帧本就无残差可遮），
        所以逐帧扫一遍、取「确实咬掉东西」的那一帧做清零断言，其余帧只要求不得逆向变大。
        """

        seen = self._drive()  # 跑一局
        self.assertTrue(seen["border"], "透明相位应至少发生一次弱测量")
        zone = 14  # 准星外接框半宽（臂长 9 + 线宽 + 抖动 3，留余量）
        cy0, cy1 = CROSS[1] - zone, CROSS[1] + zone + 1  # 行区间
        cx0, cx1 = CROSS[0] - zone, CROSS[0] + zone + 1  # 列区间
        bitten = None  # 实际发生遮罩的那一帧
        for tick, got in seen["border"].items():  # 逐帧核对
            pristine = seen["aligner"].get(tick)  # 同帧出厂原图
            if pristine is None:
                continue
            outside = np.ones_like(got, dtype=bool)  # 足迹外区域掩膜
            outside[cy0:cy1, cx0:cx1] = False  # 括开准星区域
            self.assertTrue(np.array_equal(got[outside], pristine[outside]), f"t{tick} 足迹外不得被动过")
            self.assertTrue(np.all(got[~outside] <= pristine[~outside] + 1e-6), f"t{tick} 足迹内只能变小")
            if got.sum() < pristine.sum() - 1e-6 and bitten is None:
                bitten = tick  # 本帧准星处确有残差且被遮掉
        self.assertIsNotNone(bitten, "至少有一帧弱测量应看到被遮掉的准星残差")
        got = seen["border"][bitten]  # 取那一帧
        pristine = seen["aligner"][bitten]  # 同帧原图
        self.assertGreater(float(pristine[CROSS[1] - 3:CROSS[1] + 4, CROSS[0] - 3:CROSS[0] + 4].sum()),
                           0.0, "准星十字中心在原图里应有可见残差")
        self.assertEqual(float(got[CROSS[1], CROSS[0]]), 0.0, "准星中心处的残差必须被遮为 0")

    def test_witness_sees_pristine_map(self):
        """旋转见证器：必须吃到与出厂原图逐比特一致的证据（锚点年龄口径不得被掩膜改动）。"""

        seen = self._drive()  # 同一局
        self.assertTrue(seen["witness"], "检查点应喂过见证器")
        for tick, got in seen["witness"].items():  # 逐个检查点核对
            pristine = seen["aligner"].get(tick)  # 同帧原图
            if pristine is None:
                continue
            self.assertTrue(np.array_equal(got, pristine), f"见证器在 t{tick} 应看到原始证据图")

    def test_disabling_mask_restores_pristine_border(self):
        """回滚开关：cursor_mask_enabled=False 时弱测量也拿到原图（行为与改前完全一致）。"""

        seen = self._drive(cursor_mask_enabled=False)  # 关掩膜跑同一局
        tick = max(seen["border"])  # 最后一次弱测量
        self.assertTrue(np.array_equal(seen["border"][tick], seen["aligner"][tick]), "关开关应逐比特等于原图")


if __name__ == "__main__":
    unittest.main(verbosity=2)
