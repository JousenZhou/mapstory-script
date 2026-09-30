# TestGpuShapeBackend：验证测谎「找目标」打分内核的双后端（NumPy/torch CUDA）数值一致性、
# 后端选择与运行期永久降级逻辑。全部用合成证据图与合成模板驱动，不依赖录像文件。
#
# 设计要点：
#   - 等价性用例直接调用 *_core 函数并分别注入 numpy 门面 / torch 门面，绕过薄壳的
#     「显卡异常静默回落 NumPy」兜底。这样一旦 torch 后端真的跑不通会直接抛错让测试失败，
#     而不是被降级掩盖成「NumPy 对比 NumPy」的假通过。
#   - 无 torch / 无 N 卡的机器上，显卡对照用例自动 skipTest，NumPy 路径与降级逻辑仍然全量验证。
import unittest

import cv2
import numpy as np

from src.liedetector import gpu_shape_backend as gsb  # 双后端打分内核与后端管理。
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # 在线编排：验证后端装配与精度档参数。
from src.liedetector.shape_tracking import (  # 算法内核薄壳与模板构建。
    build_shape_template,
    sample_map,
    score_rotated_borders,
    score_shape_contours,
    transform_template_points,
)
from src.liedetector.torch_array import numpy_api  # CPU 数组门面：内核注入用。

NX = numpy_api()  # CPU 门面单例（创建开销极小，可在模块层直接取）。


def _circle_contour(radius=30.0, count=180):
    """合成一个圆形轮廓点集 (count,1,2)，用于构建通用轮廓模型模板（symmetry=30、非矩形）。"""

    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)  # 圆周角度均匀采样。
    pts = np.stack([60.0 + radius * np.cos(theta), 60.0 + radius * np.sin(theta)], axis=1)  # 极坐标转直角坐标。
    return pts.astype(np.float32).reshape(-1, 1, 2)  # 整理成轮廓点集形状。


def _square_contour(side=60.0, x0=30.0, y0=30.0, per_side=40):
    """合成一个正方形轮廓点集 (4*per_side,1,2)，用于触发规则矩形边框评分模型（symmetry=90、矩形）。

    必须沿周长顺序采点（上→右→下→左），否则轮廓会自交，contourArea/minAreaRect 算出的
    矩形度过低，无法触发矩形模型。
    """

    pts = []  # 收集四条边上的点，严格沿周长顺序。
    for i in range(per_side):  # 上边：左→右。
        pts.append((x0 + side * i / per_side, y0))  # 上边点。
    for i in range(per_side):  # 右边：上→下。
        pts.append((x0 + side, y0 + side * i / per_side))  # 右边点。
    for i in range(per_side):  # 下边：右→左。
        pts.append((x0 + side - side * i / per_side, y0 + side))  # 下边点。
    for i in range(per_side):  # 左边：下→上。
        pts.append((x0, y0 + side - side * i / per_side))  # 左边点。
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)  # 整理成轮廓点集形状。


def _make_evidence(size=128, seed=7):
    """合成一张单通道 float32 证据图：低幅噪声底 + 中央亮正方形边界 + 内部弱残差，模拟目标时序残差。"""

    rng = np.random.default_rng(seed)  # 固定随机源保证可复现。
    ev = (rng.random((size, size), dtype=np.float32) * 0.4).astype(np.float32)  # 低幅噪声底。
    lo, hi = 40, 88  # 中央亮边界框范围。
    ev[lo, lo:hi] = 5.0  # 上边。
    ev[hi - 1, lo:hi] = 5.0  # 下边。
    ev[lo:hi, lo] = 5.0  # 左边。
    ev[lo:hi, hi - 1] = 5.0  # 右边。
    ev[lo + 10:hi - 10, lo + 10:hi - 10] = 1.8  # 内部弱残差（半透明目标内部也有痕迹）。
    return ev


def _make_states(count, center=(64.0, 64.0), seed=3):
    """合成一批位姿假设 (count,6)：中心小幅扰动 + 随机速度/角度/角速度。"""

    rng = np.random.default_rng(seed)  # 固定随机源。
    states = np.zeros((count, 6), dtype=np.float32)  # 状态矩阵。
    states[:, 0] = center[0] + rng.normal(0.0, 6.0, count)  # x 围绕中心扰动。
    states[:, 1] = center[1] + rng.normal(0.0, 6.0, count)  # y 围绕中心扰动。
    states[:, 2] = rng.normal(0.0, 1.0, count)  # vx。
    states[:, 3] = rng.normal(0.0, 1.0, count)  # vy。
    states[:, 4] = rng.uniform(0.0, 90.0, count)  # 角度铺满一个象限。
    states[:, 5] = rng.normal(0.0, 0.5, count)  # 角速度。
    return states


class TestGpuShapeBackend(unittest.TestCase):
    """双后端打分内核测试：后端管理、核心函数 torch/NumPy 数值一致性、薄壳注入与在线会话装配。"""

    # --------------------------------------------------------------- 后端管理

    def _gpu_or_skip(self):  # 取 torch 数组门面；无 torch / 无 N 卡时跳过显卡对照用例。
        if not gsb.gpu_backend_available():  # 探测不可用。
            self.skipTest("torch CUDA/NVIDIA GPU 不可用，跳过显卡对照用例")  # 跳过而非失败：无显卡是常见环境。
        return gsb.torch_module()  # 返回 torch 门面供核心函数注入。

    @staticmethod
    def _np(value):  # 把门面结果统一下载回 NumPy 供断言。
        if isinstance(value, np.ndarray):  # 已是 NumPy。
            return value
        return value.detach().cpu().numpy()  # torch 张量 D2H。

    def test_numpy_backend_singleton(self):
        """NumPy 后端是单例、名字为 numpy、is_gpu 为 False。"""

        first = gsb.numpy_backend()  # 第一次取。
        second = gsb.numpy_backend()  # 第二次取。
        self.assertIs(first, second, "NumPy 后端应为单例")  # 单例。
        self.assertEqual(first.name, "numpy", "后端名应为 numpy")  # 名字。
        self.assertFalse(first.is_gpu, "NumPy 后端 is_gpu 应为 False")  # 非显卡。

    def test_require_backend_name_valid(self):
        """require_gpu_backend 返回的后端名只能是 torch 或 numpy（有 N 卡走 torch，没有则 numpy）。"""

        backend = gsb.require_gpu_backend()  # 请求显卡后端。
        self.assertIn(backend.name, ("torch", "numpy"), "后端名应为 torch 或 numpy")  # 合法名字。
        if gsb.gpu_backend_available():  # 本机有显卡。
            self.assertEqual(backend.name, "torch", "有 N 卡时应返回 torch 后端")  # 应走显卡。
            self.assertTrue(backend.is_gpu, "torch 后端 is_gpu 应为 True")  # 是显卡后端。

    def test_warmup_returns_backend_name(self):
        """预热返回「后端(设备)/光流引擎」描述串，且不抛异常（预热失败也不能影响正常路径）。"""

        name = gsb.warmup_shape_backend()  # 预热。
        self.assertIn(name, ("torch(cuda)/farneback", "numpy/dis"), "预热应返回合法的后端+引擎描述串")  # 合法描述串。
        if gsb.gpu_backend_available():  # 本机有显卡：打分走 torch CUDA、光流走 Farneback。
            self.assertEqual(name, "torch(cuda)/farneback", "有 N 卡时预热应返回 torch(cuda)/farneback")  # 显卡链路。

    def test_permanent_degradation_after_failures(self):
        """连续运行期异常达到上限后 require_gpu_backend 永久回退 NumPy，且清零计数也不复活。"""

        saved = (gsb._gpu_failed, gsb._gpu_backend, gsb._fail_count)  # 保存全局状态，测试后恢复避免污染其他用例。
        try:  # 隔离全局状态。
            gsb._fail_count = 0  # 从零开始累计。
            gsb._gpu_failed = False  # 复位永久禁用标记。
            gsb._gpu_backend = None  # 清空单例。
            for _ in range(gsb._FAIL_LIMIT):  # 连续触发上限次运行期异常。
                gsb.mark_gpu_failure()  # 记录一次失败。
            self.assertTrue(gsb._gpu_failed, "达到失败上限后应永久禁用显卡后端")  # 已永久禁用。
            self.assertIsNone(gsb._gpu_backend, "降级后应丢弃显卡后端单例")  # 单例被丢弃。
            self.assertEqual(gsb.require_gpu_backend().name, "numpy", "降级后请求显卡后端应返回 NumPy")  # 回退 NumPy。
            gsb.clear_gpu_failure()  # 清零连续失败计数。
            self.assertEqual(gsb.require_gpu_backend().name, "numpy", "永久禁用后清零计数也不应复活显卡后端")  # 仍为 NumPy。
        finally:  # 无论断言成功与否都恢复全局状态。
            gsb._gpu_failed, gsb._gpu_backend, gsb._fail_count = saved  # 还原。

    # --------------------------------------------------------------- 核心函数双后端数值一致性

    def test_sample_map_core_equivalence(self):
        """采样内核：torch 与 NumPy 在相同坐标上取值一致（含越界置 0）。"""

        tx = self._gpu_or_skip()  # 取 torch 门面。
        ev = _make_evidence()  # 证据图。
        rng = np.random.default_rng(11)  # 随机坐标源。
        x = rng.uniform(-5.0, ev.shape[1] + 5.0, (3, 50)).astype(np.float32)  # 含越界的 x 坐标。
        y = rng.uniform(-5.0, ev.shape[0] + 5.0, (3, 50)).astype(np.float32)  # 含越界的 y 坐标。
        np_out = self._np(gsb.sample_map_core(NX, ev, x, y))  # NumPy 门面核心。
        cp_out = self._np(gsb.sample_map_core(tx, ev, x, y))  # torch 门面核心，下载回主存。
        np.testing.assert_allclose(np_out, cp_out, rtol=1e-5, atol=1e-5, err_msg="sample_map 双后端结果应一致")  # 数值一致。

    def test_transform_template_points_core_equivalence(self):
        """模板点变换内核：标量尺度与数组尺度两种路径，torch 与 NumPy 的点集/法向都一致。"""

        tx = self._gpu_or_skip()  # 取 torch 门面。
        template = build_shape_template(_circle_contour())  # 通用轮廓模板。
        states = _make_states(48)  # 位姿假设。
        for scales in (1.0, np.linspace(0.9, 1.1, 48).astype(np.float32)):  # 标量尺度与逐假设数组尺度各测一次。
            np_pts, np_nrm = gsb.transform_template_points_core(NX, template, states, scales)  # NumPy 门面核心。
            cp_pts, cp_nrm = gsb.transform_template_points_core(tx, template, states, scales)  # torch 门面核心。
            np.testing.assert_allclose(np_pts, self._np(cp_pts), rtol=1e-4, atol=1e-3, err_msg="模板点集双后端结果应一致")  # 点集一致。
            np.testing.assert_allclose(np_nrm, self._np(cp_nrm), rtol=1e-4, atol=1e-3, err_msg="模板法向双后端结果应一致")  # 法向一致。

    def test_score_shape_contours_core_equivalence(self):
        """通用轮廓打分内核：torch 与 NumPy 的得分与覆盖率一致（标量尺度 + 数组尺度）。"""

        tx = self._gpu_or_skip()  # 取 torch 门面。
        template = build_shape_template(_circle_contour())  # 通用轮廓模板（走 9 组法向偏移 + 12 扇区归约）。
        ev = _make_evidence()  # 证据图。
        states = _make_states(64)  # 位姿假设。
        play_height = ev.shape[0]  # 有效高度。
        for scales in (1.0, np.linspace(0.9, 1.1, 64).astype(np.float32)):  # 标量与数组尺度各测一次。
            np_res = gsb.score_shape_contours_core(NX, ev, states, template, scales, play_height)  # NumPy 门面核心。
            cp_res = gsb.score_shape_contours_core(tx, ev, states, template, scales, play_height)  # torch 门面核心。
            self.assertTrue(np.all(np.isfinite(self._np(cp_res.scores))), "torch 得分不应出现 NaN/Inf")  # 显卡结果数值健康。
            np.testing.assert_allclose(np_res.scores, self._np(cp_res.scores), rtol=1e-3, atol=1e-3, err_msg="轮廓得分双后端结果应一致")  # 得分一致。
            np.testing.assert_allclose(np_res.coverage, self._np(cp_res.coverage), rtol=1e-3, atol=1e-3, err_msg="轮廓覆盖率双后端结果应一致")  # 覆盖率一致。

    def test_score_rotated_borders_core_equivalence(self):
        """矩形四边打分内核：torch 与 NumPy 的得分与覆盖率一致。"""

        tx = self._gpu_or_skip()  # 取 torch 门面。
        ev = _make_evidence()  # 证据图。
        states = _make_states(48)  # 位姿假设。
        play_height = ev.shape[0]  # 有效高度。
        side = 48.0  # 矩形边长。
        np_res = gsb.score_rotated_borders_core(NX, ev, states, side, play_height)  # NumPy 门面核心。
        cp_res = gsb.score_rotated_borders_core(tx, ev, states, side, play_height)  # torch 门面核心。
        self.assertTrue(np.all(np.isfinite(self._np(cp_res.scores))), "torch 四边得分不应出现 NaN/Inf")  # 显卡结果数值健康。
        np.testing.assert_allclose(np_res.scores, self._np(cp_res.scores), rtol=1e-3, atol=1e-3, err_msg="四边得分双后端结果应一致")  # 得分一致。
        np.testing.assert_allclose(np_res.coverage, self._np(cp_res.coverage), rtol=1e-3, atol=1e-3, err_msg="四边覆盖率双后端结果应一致")  # 覆盖率一致。

    def test_rectangle_model_routes_through_borders(self):
        """矩形模板 + 标量尺度时，通用打分内核应转发到四边评分，torch 与 NumPy 结果一致。"""

        tx = self._gpu_or_skip()  # 取 torch 门面。
        template = build_shape_template(_square_contour())  # 正方形模板。
        self.assertTrue(template.use_rectangle_model, "正方形应启用矩形模型")  # 确认走矩形分支。
        ev = _make_evidence()  # 证据图。
        states = _make_states(40)  # 位姿假设。
        play_height = ev.shape[0]  # 有效高度。
        np_res = gsb.score_shape_contours_core(NX, ev, states, template, 1.0, play_height)  # NumPy 门面核心（标量尺度触发矩形分支）。
        cp_res = gsb.score_shape_contours_core(tx, ev, states, template, 1.0, play_height)  # torch 门面核心。
        np.testing.assert_allclose(np_res.scores, self._np(cp_res.scores), rtol=1e-3, atol=1e-3, err_msg="矩形模型得分双后端结果应一致")  # 得分一致。
        np.testing.assert_allclose(np_res.coverage, self._np(cp_res.coverage), rtol=1e-3, atol=1e-3, err_msg="矩形模型覆盖率双后端结果应一致")  # 覆盖率一致。

    # --------------------------------------------------------------- 薄壳与在线会话

    def test_thin_shell_returns_numpy_and_matches_core(self):
        """薄壳函数接受显式 backend 注入，返回 NumPy 数组且与直接核心调用一致。"""

        template = build_shape_template(_circle_contour())  # 通用轮廓模板。
        ev = _make_evidence()  # 证据图。
        states = _make_states(32)  # 位姿假设。
        play_height = ev.shape[0]  # 有效高度。
        shell = score_shape_contours(ev, states, template, 1.0, play_height, backend=gsb.numpy_backend())  # 薄壳（NumPy 后端）。
        core = gsb.score_shape_contours_core(NX, ev, states, template, 1.0, play_height)  # 直接核心（NumPy 门面）。
        self.assertIsInstance(shell.scores, np.ndarray, "薄壳应返回 NumPy 数组")  # 返回类型。
        self.assertIsInstance(shell.coverage, np.ndarray, "薄壳覆盖率应为 NumPy 数组")  # 返回类型。
        np.testing.assert_allclose(shell.scores, core.scores, rtol=1e-5, atol=1e-5, err_msg="薄壳结果应与核心一致")  # 数值一致。
        # 其余薄壳同样验证返回 NumPy。
        self.assertIsInstance(sample_map(ev, states[:, 0], states[:, 1], backend=gsb.numpy_backend()), np.ndarray, "sample_map 薄壳应返回 NumPy")  # 采样薄壳。
        pts, nrm = transform_template_points(template, states, 1.0, backend=gsb.numpy_backend())  # 变换薄壳。
        self.assertIsInstance(pts, np.ndarray, "transform_template_points 应返回 NumPy 点集")  # 点集类型。
        self.assertIsInstance(nrm, np.ndarray, "transform_template_points 应返回 NumPy 法向")  # 法向类型。
        border = score_rotated_borders(ev, states, 48.0, play_height, backend=gsb.numpy_backend())  # 四边薄壳。
        self.assertIsInstance(border.scores, np.ndarray, "score_rotated_borders 应返回 NumPy 得分")  # 得分类型。

    def test_thin_shell_gpu_backend_equivalence(self):
        """薄壳在显卡后端注入下与 NumPy 后端结果一致（端到端验证降级兜底之外的正常显卡路径）。"""

        cp = self._gpu_or_skip()  # 无显卡则跳过。
        gpu_backend = gsb.require_gpu_backend()  # 请求显卡后端。
        if not gpu_backend.is_gpu:  # 已被降级。
            self.skipTest("显卡后端已降级为 NumPy，跳过对照")  # 跳过。
        template = build_shape_template(_circle_contour())  # 通用轮廓模板。
        ev = _make_evidence()  # 证据图。
        states = _make_states(64)  # 位姿假设。
        play_height = ev.shape[0]  # 有效高度。
        np_out = score_shape_contours(ev, states, template, 1.0, play_height, backend=gsb.numpy_backend())  # NumPy 后端薄壳。
        gpu_out = score_shape_contours(ev, states, template, 1.0, play_height, backend=gpu_backend)  # 显卡后端薄壳。
        self.assertIsInstance(gpu_out.scores, np.ndarray, "显卡薄壳应把结果下载回 NumPy")  # 返回类型已是 NumPy。
        np.testing.assert_allclose(np_out.scores, gpu_out.scores, rtol=1e-3, atol=1e-3, err_msg="薄壳双后端得分应一致")  # 得分一致。
        np.testing.assert_allclose(np_out.coverage, gpu_out.coverage, rtol=1e-3, atol=1e-3, err_msg="薄壳双后端覆盖率应一致")  # 覆盖率一致。

    def test_session_resolves_backend_and_precision_tier(self):
        """精度档装配：GPU 下 high/ultra/extreme 各应用自身参数；关闭打分（NumPy 后端）时三个重载档门控回落 medium。"""

        # 关闭打分开关（gpu_scoring=False）：无论本机有无 N 卡，后端恒为 NumPy，high/ultra/extreme 必被门控回落 medium。
        for requested in ("high", "ultra", "extreme"):  # 三个受 CPU 门控约束的重载档。
            off_params = ShapeTrackParams(gpu_scoring=False, precision_tier=requested)  # 关闭显卡打分并请求重载档。
            off_session = ShapeTrackSession(params=off_params, logger=None)  # 建会话触发后端与档位装配。
            self.assertFalse(off_session.backend.is_gpu, "关闭开关时后端应为 NumPy")  # NumPy 后端。
            self.assertEqual(off_session.backend.name, "numpy", "关闭开关时后端名应为 numpy")  # 名字。
            self.assertEqual(off_params.precision_tier, "medium", f"NumPy 后端下 {requested} 应门控回落到 medium")  # 权威 clamp。
            self.assertEqual(off_params.particle_count, 320, "回落 medium 应恢复实时档粒子数 320")  # 粒子回实时档。
            self.assertEqual(off_params.control_count, 140, "回落 medium 应恢复实时档对照组 140")  # 对照组回实时档。
            self.assertEqual(off_params.global_proposals, 600, "回落 medium 应恢复实时档粗扫 600")  # 粗扫回实时档。
            self.assertEqual(off_params.dis_preset, cv2.DISOPTICAL_FLOW_PRESET_FAST, "回落 medium 时 DIS 应为 FAST")  # DIS 回实时档。
            self.assertEqual(off_params.temporal_lags, (1, 2), "回落 medium 时应保持实时档 lag(1,2)")  # 时间基线回实时档。
            self.assertEqual((off_params.flow_levels, off_params.flow_iterations, off_params.flow_winsize), (3, 2, 11), "回落 medium 时光流参数应回实时档 3/2/11")  # 光流参数回实时档。
            # extreme 独有高分辨率（0.55/384）不得泄漏到 CPU 路径：回落 medium 后必须停在 0.5/320。
            self.assertEqual(off_params.process_scale, 0.5, f"NumPy 后端下 {requested} 回落后处理尺度应停在 0.5")  # 处理尺度不升。
            self.assertEqual(off_params.max_process_side, 320.0, f"NumPy 后端下 {requested} 回落后分辨率上限应停在 320")  # 分辨率不升。
            self.assertEqual((off_params.coarse_top, off_params.refine_per_top), (30, 8), f"NumPy 后端下 {requested} 回落后重定位漏斗应为 30/8")  # 漏斗不拉宽。
            self.assertEqual(off_session.aligner.engine_name, "cv2-dis", "NumPy 后端的光流引擎应为 cv2 DIS")  # 引擎名。
            self.assertEqual(off_session.aligner.flow_engine.flow.getFinestScale(), 2, "对齐器 DIS 实例应为 FAST（finestScale=2）")  # DIS 确实传入对齐器。
            self.assertEqual(off_session.aligner.history.maxlen, 3, "对齐器历史长度应按 lag(1,2) 建为 3")  # lag 确实传入对齐器。

        if not gsb.gpu_backend_available():  # 无 N 卡：high/ultra 都会被门控，无法验证真正生效，跳过显卡对照。
            self.skipTest("torch CUDA/NVIDIA GPU 不可用，跳过显卡精度档对照")  # 跳过而非失败：无显卡是常见环境。

        # GPU 后端 + high 档：应用 MEDIUM + lag(1,2,4) + 600 粒子/300 对照/1500 粗扫，并首次加宽漏斗。
        high_session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="high"), logger=None)  # 默认 gpu_scoring=True + high 档。
        self.assertTrue(high_session.backend.is_gpu, "有 N 卡时 high 档后端应为显卡")  # 显卡后端。
        self.assertEqual(high_session.params.precision_tier, "high", "GPU 下 high 档不应被门控")  # 档位保留。
        self.assertEqual(high_session.params.particle_count, 600, "high 档应应用 600 粒子")  # 粒子升档。
        self.assertEqual(high_session.params.control_count, 300, "high 档应应用 300 对照组")  # 对照组升档。
        self.assertEqual(high_session.params.global_proposals, 1500, "high 档应应用 1500 粗扫")  # 粗扫升档。
        self.assertEqual((high_session.params.coarse_top, high_session.params.refine_per_top), (36, 10), "high 档应把重定位漏斗加宽到 36/10")  # 漏斗升档。
        self.assertEqual(high_session.params.max_process_side, 320.0, "high 档仍不升分辨率")  # 分辨率保持 320。
        self.assertEqual(high_session.params.dis_preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, "high 档应把 DIS 升到 MEDIUM")  # DIS 升档。
        self.assertEqual(high_session.params.temporal_lags, (1, 2, 4), "high 档应恢复第三个时间基线 lag(1,2,4)")  # 时间基线升档。
        self.assertEqual((high_session.params.flow_levels, high_session.params.flow_iterations, high_session.params.flow_winsize), (3, 3, 15), "high 档光流参数应为 levels=3/iterations=3/winsize=15")  # 光流参数升档。
        self.assertEqual(high_session.aligner.engine_name, "torch-farneback", "显卡后端的光流引擎应为 torch Farneback")  # 引擎名。
        self.assertEqual(
            (high_session.aligner.flow_engine.levels, high_session.aligner.flow_engine.iterations, high_session.aligner.flow_engine.winsize),
            (3, 3, 15),
            "high 档的光流参数应真的传到引擎里",
        )  # 参数确实传入引擎。
        self.assertEqual(high_session.aligner.history.maxlen, 5, "对齐器历史长度应按 lag(1,2,4) 建为 5")  # lag 确实传入对齐器。

        # GPU 后端 + ultra 档：在 high 基础上加大粒子/对照/粗扫到 1000/500/2400，DIS/lag 与 high 相同。
        ultra_session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="ultra"), logger=None)  # ultra 档。
        self.assertTrue(ultra_session.backend.is_gpu, "有 N 卡时 ultra 档后端应为显卡")  # 显卡后端。
        self.assertEqual(ultra_session.params.precision_tier, "ultra", "GPU 下 ultra 档不应被门控")  # 档位保留。
        self.assertEqual(ultra_session.params.particle_count, 1000, "ultra 档应应用 1000 粒子")  # 粒子再升档。
        self.assertEqual(ultra_session.params.control_count, 500, "ultra 档应应用 500 对照组")  # 对照组再升档。
        self.assertEqual(ultra_session.params.global_proposals, 2400, "ultra 档应应用 2400 粗扫")  # 粗扫再升档。
        self.assertEqual((ultra_session.params.coarse_top, ultra_session.params.refine_per_top), (44, 10), "ultra 档漏斗应再加宽到 44/10")  # 漏斗再升档。
        self.assertEqual(ultra_session.params.max_process_side, 320.0, "ultra 档仍不升分辨率")  # 分辨率保持 320。
        self.assertEqual(ultra_session.params.dis_preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, "ultra 档 DIS 仍为 MEDIUM")  # DIS 与 high 同。
        self.assertEqual(ultra_session.params.temporal_lags, (1, 2, 4), "ultra 档时间基线仍为 lag(1,2,4)")  # 时间基线与 high 同。
        self.assertEqual((ultra_session.params.flow_levels, ultra_session.params.flow_iterations, ultra_session.params.flow_winsize), (4, 4, 15), "ultra 档光流参数应为 levels=4/iterations=4/winsize=15")  # 光流参数再升档。
        self.assertEqual(ultra_session.aligner.engine_name, "torch-farneback", "ultra 档同样应走 torch Farneback")  # 引擎名。
        self.assertEqual(
            (ultra_session.aligner.flow_engine.levels, ultra_session.aligner.flow_engine.iterations, ultra_session.aligner.flow_engine.winsize),
            (4, 4, 15),
            "ultra 档的光流参数应真的传到引擎里",
        )  # 参数确实传入引擎。
        self.assertEqual(ultra_session.aligner.history.maxlen, 5, "ultra 档历史长度同样按 lag(1,2,4) 建为 5")  # lag 与 high 同。

        # GPU 后端 + extreme 档（最强）：粒子/对照/漏斗拉满，且是唯一升分辨率的档。
        extreme_session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # extreme 档。
        self.assertTrue(extreme_session.backend.is_gpu, "有 N 卡时 extreme 档后端应为显卡")  # 显卡后端。
        self.assertEqual(extreme_session.params.precision_tier, "extreme", "GPU 下 extreme 档不应被门控")  # 档位保留。
        self.assertEqual(extreme_session.params.particle_count, 1400, "extreme 档应应用 1400 粒子")  # 粒子拉满。
        self.assertEqual(extreme_session.params.control_count, 720, "extreme 档应应用 720 对照组")  # 对照组拉满。
        self.assertEqual(extreme_session.params.global_proposals, 3600, "extreme 档应应用 3600 粗扫")  # 粗扫拉满。
        self.assertEqual((extreme_session.params.coarse_top, extreme_session.params.refine_per_top), (56, 12), "extreme 档漏斗应拉满到 56/12")  # 漏斗拉满。
        self.assertEqual(extreme_session.params.temporal_lags, (1, 2, 4), "extreme 档时间基线仍为 lag(1,2,4)，不额外加 lag")  # 时间基线与 ultra 同。
        self.assertEqual((extreme_session.params.flow_levels, extreme_session.params.flow_iterations, extreme_session.params.flow_winsize), (5, 4, 15), "extreme 档光流参数应为 levels=5/iterations=4/winsize=15")  # 光流金字塔再多加一层。
        self.assertEqual(
            (extreme_session.aligner.flow_engine.levels, extreme_session.aligner.flow_engine.iterations, extreme_session.aligner.flow_engine.winsize),
            (5, 4, 15),
            "extreme 档的光流参数应真的传到引擎里",
        )  # 参数确实传入引擎。
        # 分辨率是 extreme 的核心杠杆，但 effective_scale = min(process_scale, max_process_side / 区域长边)：
        # 只抬 max_process_side 会被 process_scale=0.5 卡死在 364，必须两者一起抬才能真到 384。
        # 下面用线上实测的测谎区域 728x486 做回归护栏。
        self.assertEqual(extreme_session.params.process_scale, 0.55, "extreme 档应把 process_scale 抬到 0.55，否则 384 会被 0.5 卡成 364")  # 两个上限必须同时抬。
        self.assertEqual(extreme_session.params.max_process_side, 384.0, "extreme 档分辨率上限应为 384")  # 分辨率升档。
        extreme_session.reset(728, 486)  # 用真实测谎区域尺寸算实际生效尺度。
        self.assertAlmostEqual(728 * extreme_session.effective_scale, 384.0, delta=0.5, msg="728x486 区域下 extreme 档处理宽度应真正达到 384")  # 分辨率确实生效。
        ultra_session.reset(728, 486)  # 同区域下对照 ultra 档。
        self.assertAlmostEqual(728 * ultra_session.effective_scale, 320.0, delta=0.5, msg="728x486 区域下 ultra 档处理宽度应仍为 320")  # ultra 不升分辨率。
        self.assertGreater(extreme_session.effective_scale, ultra_session.effective_scale, "extreme 档的实际处理尺度应严格高于 ultra 档")  # 两档确实拉开了。

    def test_session_respects_custom_particle_count(self):
        """会话装配只覆盖「仍停在默认值」的参数：调用方自定义的粒子数不被档位覆盖，其余字段照常升到 high 档。"""

        if not gsb.gpu_backend_available():  # 无显卡时 high 档会被门控回落 medium，无法验证「升到 high 档」。
            self.skipTest("torch CUDA/NVIDIA GPU 不可用，跳过显卡档参数对照")  # 跳过。
        params = ShapeTrackParams(particle_count=150, precision_tier="high")  # 显式自定义粒子数 + 请求 high 档。
        session = ShapeTrackSession(params=params, logger=None)  # 建会话。
        self.assertTrue(session.backend.is_gpu, "有 N 卡时后端应为显卡")  # 显卡后端。
        self.assertEqual(params.precision_tier, "high", "GPU 下 high 档不应被门控")  # 档位保留。
        self.assertEqual(params.particle_count, 150, "自定义粒子数不应被 high 档覆盖")  # 保持自定义值。
        self.assertEqual(params.control_count, 300, "未自定义的对照组仍应升到 high 档 300")  # 对照组升档。
        self.assertEqual(params.global_proposals, 1500, "未自定义的粗扫仍应升到 high 档 1500")  # 粗扫升档。


if __name__ == "__main__":
    unittest.main()
