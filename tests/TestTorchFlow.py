# TestTorchFlow：验证显卡稠密光流引擎（Farneback 的 torch 张量等价实现）、双后端证据链
# 与 CUDA Graph 整图捕获的正确性。全部用合成序列驱动，不依赖录像文件。
#
# 设计要点：
#   - 光流的「对不对」以 cv2.calcOpticalFlowFarneback 为参照：两者是同一个算法的两份实现，
#     参数逐项对齐后差异只应来自浮点次序与边界处理，因此既有对**已知平移真值**的绝对 EPE 门限，
#     也有对 cv2 参照的相对门限（绝对门限防止两边一起跑偏，相对门限防止移植本身出错）。
#   - CUDA Graph 与 eager 跑的是**同一份** _chain，唯一区别是输入换成固定地址的静态缓冲，
#     因此两者输出必须逐位相等（不是「相近」）。这条用例是整图捕获不改变数值的硬保证。
#   - 捕获失败必须永久回退 eager 而**不是**降级到 CPU：这只是省不掉 kernel launch 开销，
#     算子本身在显卡上仍是好的。
#   - 等价性用例直接构造 TorchFarnebackFlow / TensorTemporalAligner，绕过薄壳的
#     「显卡异常静默回落 NumPy」兜底，一旦显卡链路真的跑不通会直接抛错让测试失败。
#   - 无 torch / 无 N 卡的机器上显卡用例自动 skipTest，门面与 CPU 引擎用例仍然全量验证。
import unittest

import cv2
import numpy as np

from src.liedetector import gpu_shape_backend as gsb  # 后端装配与可用性探测。
from src.liedetector.tensor_evidence import TensorTemporalAligner  # 双后端时序证据链。
from src.liedetector.torch_array import numpy_api  # CPU 数组门面。
from src.liedetector.torch_flow import (  # 三个接口一致的光流引擎。
    Cv2DisFlow,
    Cv2FarnebackFlow,
    TorchFarnebackFlow,
)

LAGS = (1, 2, 4)  # 与 high/ultra 档一致的时间基线。
FLOW_PARAMS = dict(levels=3, iterations=3, winsize=15, poly_n=5, poly_sigma=1.2, pyr_scale=0.5)  # high 档参数。
HEIGHT, WIDTH = 120, 160  # 合成帧尺寸（比生产的 214×320 小，控制单测耗时）。
MARGIN = 14  # EPE 统计时裁掉的最外圈像素：两实现的边界填充方式不同，边缘误差必然偏大。
EPE_GT_LIMIT = 0.35  # 整数平移下对真值的平均端点误差上限（像素）。
EPE_GT_LIMIT_SUBPIXEL = 0.5  # 半像素平移下的上限（Farneback 本身在亚像素上有偏置）。
EPE_REF_LIMIT = 0.2  # 与 cv2 参照实现之间的平均端点误差上限（像素）。
EPE_REF_LIMIT_SUBPIXEL = 0.3  # 半像素平移下与 cv2 参照的上限。
PEARSON_LIMIT = 0.95  # 证据链跨设备一致性下限。


def _smooth_canvas(height, width, seed, block=8):
    """造一张平滑随机纹理底图：低频斑块为主 + 少量高频细节。

    刻意不用纯白噪声：逐像素独立的噪声在金字塔降采样后几乎被抹平，光流在那里本来就无法求解，
    量出来的 EPE 反映的是素材而不是实现质量。低频斑块接近真实游戏画面，是可解的。
    """

    rng = np.random.default_rng(seed)  # 固定随机源保证可复现。
    rows = max(2, height // block) + 2  # 低频网格行数（多留 2 行避免 resize 边界外推）。
    columns = max(2, width // block) + 2  # 低频网格列数。
    coarse = rng.random((rows, columns), dtype=np.float32)  # 低频控制点。
    canvas = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)  # 三次插值放大成平滑斑块。
    canvas = canvas + rng.random((height, width), dtype=np.float32) * 0.08  # 叠加弱高频，避免完全无纹理的平坦区。
    return np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8)  # 量化到 uint8（光流接口只收 CV_8U）。


def _integer_sequence(count, step=1, height=HEIGHT, width=WIDTH):
    """整数平移序列：帧 i = 底图上向右下移动 i*step 像素后裁出的窗口。

    用「大底图 + 移动裁剪窗口」而不是 np.roll：后者会把画面绕回来，在边缘造出真实不存在的
    剧烈运动，光流在那里必然出错，会污染 EPE 统计。

    返回 ``(frames, 真值位移)``。窗口随 i 增大向右下移动，等价于画面内容向左上移动，
    按 ``previous(p + flow(p)) == current(p)`` 的约定，current 相对 previous（早 k 帧）的
    真值位移是 ``+k*step``。
    """

    pad = step * count + 2 * MARGIN + 2  # 底图四周多留出的余量，保证最后一帧仍在底图内。
    canvas = _smooth_canvas(height + pad, width + pad, seed=11)  # 一次造好底图。
    frames = []
    for index in range(count):
        top = MARGIN + index * step  # 本帧窗口左上角行。
        left = MARGIN + index * step  # 本帧窗口左上角列。
        frames.append(np.ascontiguousarray(canvas[top:top + height, left:left + width]))  # 裁剪并保证连续。
    return frames, float(step)  # 每帧位移步长，真值 = k * 步长。


def _subpixel_sequence(count, height=HEIGHT, width=WIDTH):
    """半像素平移序列：底图取 2 倍分辨率，窗口每帧只移动 1 个高分辨率像素。

    ``INTER_AREA`` 在正好 2 倍降采样时是 2×2 盒式平均，窗口移动 1 个高分辨率像素
    等价于输出图移动 0.5 个像素，且每一帧用的是同一套滤波权重，因此位移是干净的 0.5px。
    """

    pad = count + 2 * MARGIN + 4  # 高分辨率下需要的余量。
    canvas = _smooth_canvas((height + pad) * 2, (width + pad) * 2, seed=13, block=16)  # 2 倍分辨率底图。
    frames = []
    for index in range(count):
        top = MARGIN * 2 + index  # 高分辨率窗口左上角，每帧 +1 个高分辨率像素。
        window = canvas[top:top + height * 2, top:top + width * 2]  # 裁 2 倍尺寸的窗口。
        downsampled = cv2.resize(window, (width, height), interpolation=cv2.INTER_AREA)  # 2×2 盒式降采样。
        frames.append(np.ascontiguousarray(downsampled))
    return frames, 0.5  # 每帧位移 0.5 像素。


def _color_sequence(count, height=HEIGHT, width=WIDTH):
    """彩色 BGR 序列：平滑纹理底 + 一个逐帧移动的亮方块，供证据链用例使用。

    证据链要的是「运动边界处响应强」，因此必须有真实的物体位移，只用纹理平移不够。
    """

    gray, step = _integer_sequence(count, 1, height, width)  # 复用整数平移序列当背景。
    frames = []
    for index, plane in enumerate(gray):
        frame = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)  # 灰度扩成三通道。
        top = 16 + index * 2  # 方块逐帧下移 2 像素。
        left = 20 + index * 3  # 同时右移 3 像素。
        frame[top:top + 30, left:left + 30] = (235, 235, 235)  # 白色方块，制造清晰的运动边界。
        frames.append(frame)
    return frames, step


def _mean_epe(flow, expected_dx, expected_dy):
    """内部区域的平均端点误差（像素）；``flow`` 为 (H,W,2)。"""

    data = np.asarray(flow, dtype=np.float64)[MARGIN:-MARGIN, MARGIN:-MARGIN]  # 裁掉不可靠的最外圈。
    return float(np.mean(np.hypot(data[..., 0] - expected_dx, data[..., 1] - expected_dy)))


def _to_numpy(value):
    """把后端数组拉回 numpy（torch 张量 detach + D2H）。"""

    detach = getattr(value, "detach", None)  # torch 张量才有。
    return detach().cpu().numpy() if callable(detach) else np.asarray(value)


class TestArrayFacadeGraphOps(unittest.TestCase):
    """门面为 CUDA Graph 新增的两个算子：``select``（取共享存储的视图）与 ``copy_into``（地址不变的就地拷贝）。"""

    def test_select_drops_axis_and_shares_storage(self):
        """select 必须去掉被索引的那一维，语义与 a[i] 一致（两套门面都要满足）。"""

        xp = numpy_api()  # CPU 门面（无显卡也能验证）。
        data = np.arange(24, dtype=np.float32).reshape(4, 3, 2)  # (4,3,2) 测试数组。
        picked = xp.select(data, 0, 2)  # 取第 0 轴下标 2。
        self.assertEqual(picked.shape, (3, 2), "select 应去掉被索引的轴")  # 轴确实被去掉。
        np.testing.assert_array_equal(picked, data[2], err_msg="select 的值应与 a[i] 完全一致")  # 值一致。
        column = xp.select(data, 1, 1)  # 换一根轴再验一次。
        self.assertEqual(column.shape, (4, 2), "select 应支持任意轴")  # 形状正确。
        np.testing.assert_array_equal(column, data[:, 1], err_msg="select 在 axis=1 上也应与 a[:,1] 一致")

    def test_copy_into_keeps_destination_address(self):
        """copy_into 必须是就地写：目标对象身份不变（CUDA Graph 靠这一点保住固定地址）。"""

        xp = numpy_api()
        destination = np.zeros((2, 3), dtype=np.float32)  # 目标缓冲。
        snapshot = destination  # 记住对象身份。
        returned = xp.copy_into(destination, np.full((3,), 7.0, dtype=np.float32))  # 广播写入一行。
        self.assertIs(returned, destination, "copy_into 应返回目标本身")  # 返回的就是目标。
        self.assertIs(snapshot, destination, "目标对象身份不能被换掉")  # 身份未变。
        np.testing.assert_array_equal(destination, np.full((2, 3), 7.0, dtype=np.float32), err_msg="广播写入结果不对")

    @unittest.skipUnless(gsb.gpu_backend_available(), "no torch CUDA available")
    def test_torch_facade_matches_numpy_semantics(self):
        """torch 门面的 select/copy_into 与 numpy 门面语义一致，且 select 返回的是共享存储的视图。"""

        xp = gsb.require_gpu_backend().xp  # 显卡门面。
        data = np.arange(24, dtype=np.float32).reshape(4, 3, 2)  # 与 CPU 用例同一份数据。
        uploaded = xp.asarray(data)  # 上传到显存。
        picked = xp.select(uploaded, 0, 2)  # 取视图。
        np.testing.assert_array_equal(_to_numpy(picked), data[2], err_msg="torch select 的值应与 a[i] 一致")
        self.assertEqual(picked.data_ptr(), xp.select(uploaded, 0, 2).data_ptr(), "select 应返回共享存储的视图")
        buffer = xp.zeros((4, 3, 2), dtype=xp.float32)  # 固定地址缓冲。
        address = buffer.data_ptr()  # 记住地址。
        xp.copy_into(xp.narrow(buffer, 0, 1, 1), data[1])  # 就地写入一个切片。
        self.assertEqual(buffer.data_ptr(), address, "copy_into 不能改变缓冲地址")  # 地址未变。
        np.testing.assert_array_equal(_to_numpy(buffer)[1], data[1], err_msg="copy_into 写入的值不对")


class TestTorchFarnebackFlow(unittest.TestCase):
    """显卡光流引擎：以 cv2.calcOpticalFlowFarneback 为参照、以已知平移为真值。"""

    def setUp(self):
        if not gsb.gpu_backend_available():  # 无 torch / 无 N 卡：整组跳过。
            self.skipTest("no torch CUDA available")
        self.xp = gsb.require_gpu_backend().xp  # 显卡门面。

    def _reference(self, current, previous):
        """用同参数的 CPU Farneback 算一份参照位移。"""

        return Cv2FarnebackFlow(**FLOW_PARAMS).calc_batch(current, previous)

    def _check(self, frames, per_frame_step, lags, gt_limit, ref_limit, label):
        """对每个 lag 同时量「对真值的 EPE」与「对 cv2 参照的 EPE」。"""

        engine = TorchFarnebackFlow(self.xp, **FLOW_PARAMS)  # 被测引擎。
        current = frames[-1]  # 当前帧取序列最后一帧。
        stacked = np.stack([frames[-1 - lag] for lag in lags], axis=0)  # 各 lag 的历史帧合成 batch。
        produced = engine.calc_batch(self.xp.asarray(current), self.xp.asarray(stacked))  # 一次算完全部 lag。
        reference = self._reference(current, stacked)  # CPU 参照。
        produced_np = _to_numpy(produced)  # 拉回主存。
        self.assertEqual(produced_np.shape, (len(lags), HEIGHT, WIDTH, 2), f"{label}: calc_batch 输出形状不对")
        self.assertTrue(np.all(np.isfinite(produced_np)), f"{label}: 位移场里出现了 inf/nan")
        worst_gt = 0.0
        worst_ref = 0.0
        for index, lag in enumerate(lags):
            expected = lag * per_frame_step  # 真值位移：两个轴同值。
            worst_gt = max(worst_gt, _mean_epe(produced_np[index], expected, expected))
            difference = produced_np[index] - reference[index]  # 与 cv2 参照逐像素作差。
            worst_ref = max(worst_ref, float(np.mean(np.hypot(difference[..., 0], difference[..., 1]))))
        self.assertLessEqual(worst_gt, gt_limit, f"{label}: 对真值的平均 EPE={worst_gt:.4f}px 超过 {gt_limit}")
        self.assertLessEqual(worst_ref, ref_limit, f"{label}: 与 cv2 参照的平均 EPE={worst_ref:.4f}px 超过 {ref_limit}")
        return worst_gt, worst_ref

    def test_integer_shift_matches_cv2_and_ground_truth(self):
        """整数平移：真值 EPE ≤ 0.35px，且与 cv2 参照的差 ≤ 0.2px。"""

        frames, step = _integer_sequence(max(LAGS) + 2)  # 造够最大 lag + 当前帧。
        self._check(frames, step, LAGS, EPE_GT_LIMIT, EPE_REF_LIMIT, "整数平移")

    def test_subpixel_shift_matches_cv2_and_ground_truth(self):
        """半像素平移：真值 EPE ≤ 0.5px，且与 cv2 参照的差 ≤ 0.3px（亚像素是移植最容易出错的地方）。"""

        frames, step = _subpixel_sequence(max(LAGS) + 2)  # 每帧 0.5px。
        self._check(frames, step, LAGS, EPE_GT_LIMIT_SUBPIXEL, EPE_REF_LIMIT_SUBPIXEL, "半像素平移")

    def test_single_lag_and_batch_agree(self):
        """单 lag 传 (H,W) 与合批传 (1,H,W) 必须给出同一份位移（calc_batch 内部会补 batch 维）。"""

        frames, _ = _integer_sequence(3)  # 只需 2 帧。
        engine = TorchFarnebackFlow(self.xp, **FLOW_PARAMS)
        single = _to_numpy(engine.calc_batch(self.xp.asarray(frames[-1]), self.xp.asarray(frames[0])))
        batched = _to_numpy(
            engine.calc_batch(self.xp.asarray(frames[-1]), self.xp.asarray(np.stack([frames[0], frames[0]])))
        )
        self.assertEqual(single.shape, (1, HEIGHT, WIDTH, 2), "单 lag 也应返回带 batch 维的形状")
        np.testing.assert_array_equal(single[0], batched[0], err_msg="单 lag 与合批的第一份位移应完全一致")


class TestFlowEngineNames(unittest.TestCase):
    """引擎名常量：日志/UI 徽标与降级判据都靠它，不依赖显卡，无 N 卡也要验。"""

    def test_engine_names_are_stable(self):
        self.assertEqual(TorchFarnebackFlow.engine_name, "torch-farneback")  # 显卡 Farneback。
        self.assertEqual(Cv2DisFlow.engine_name, "cv2-dis")  # CPU 生产路径。
        self.assertEqual(Cv2FarnebackFlow.engine_name, "cv2-farneback")  # CPU 参照，只供单测与 A/B 诊断。


class TestTensorEvidenceChain(unittest.TestCase):
    """证据链：跨设备一致性、CUDA Graph 与 eager 逐位等价、捕获失败的永久回退。"""

    def setUp(self):
        if not gsb.gpu_backend_available():  # 无 torch / 无 N 卡：整组跳过。
            self.skipTest("no torch CUDA available")
        self.backend = gsb.require_gpu_backend()  # 显卡后端。
        self.frames, _ = _color_sequence(max(LAGS) + 8)  # 彩色序列，够跑满稳态若干帧。

    def _gpu_aligner(self, disable_graph=False):
        """建一个显卡证据链对齐器；``disable_graph`` 用于强制走 eager 当参照。"""

        aligner = TensorTemporalAligner(
            LAGS, 1.0, TorchFarnebackFlow(self.backend.xp, **FLOW_PARAMS), self.backend
        )
        aligner._graph_disabled = disable_graph  # 直接置位即可跳过捕获（_try_capture 会立刻返回）。
        return aligner

    def test_cross_device_pearson(self):
        """显卡链路 vs CPU 同算法参照（cv2 Farneback + numpy 门面）：Pearson ≥ 0.95。"""

        gpu = self._gpu_aligner(disable_graph=True)  # 参照侧用 eager，隔离出「设备」这一个变量。
        cpu = TensorTemporalAligner(
            LAGS, 1.0, Cv2FarnebackFlow(**FLOW_PARAMS), gsb.numpy_backend()
        )
        flat_gpu, flat_cpu, pairs = [], [], 0
        for frame in self.frames:
            left = gpu.update(frame)  # 显卡证据。
            right = cpu.update(frame)  # CPU 证据。
            self.assertEqual(left is None, right is None, "两侧「可用 lag」的判定必须一致")
            if left is None:
                continue
            pairs += 1
            flat_gpu.append(_to_numpy(left.normalized).ravel())  # 展平累积。
            flat_cpu.append(_to_numpy(right.normalized).ravel())
            self.assertAlmostEqual(left.raw_mean, right.raw_mean, delta=1.0, msg="残差均值跨设备差得太多")
        self.assertGreater(pairs, 0, "一帧有效证据都没算出来")
        pearson = float(np.corrcoef(np.concatenate(flat_gpu), np.concatenate(flat_cpu))[0, 1])
        self.assertGreaterEqual(pearson, PEARSON_LIMIT, f"证据链跨设备 Pearson={pearson:.6f} 低于 {PEARSON_LIMIT}")

    def test_cuda_graph_matches_eager_bitwise(self):
        """CUDA Graph 与 eager 跑同一份 _chain，输出必须逐位相等（不是相近）。"""

        eager = self._gpu_aligner(disable_graph=True)  # 纯 eager 参照。
        graphed = self._gpu_aligner()  # 允许捕获。
        captured_at = None
        for index, frame in enumerate(self.frames):
            left = eager.update(frame)
            right = graphed.update(frame)
            if captured_at is None and graphed._graph is not None:
                captured_at = index  # 记下捕获发生在第几帧（应为第一个稳态帧）。
            self.assertEqual(left is None, right is None, "两侧「可用 lag」的判定必须一致")
            if left is None:
                continue
            # 图的输出是静态缓冲，必须当场拷走，否则下一帧 replay 会覆盖掉。
            x = _to_numpy(left.normalized).astype(np.float64).copy()
            y = _to_numpy(right.normalized).astype(np.float64).copy()
            self.assertEqual(float(np.abs(x - y).max()), 0.0, f"第 {index} 帧：CUDA Graph 与 eager 的证据图不逐位相等")
            self.assertEqual(left.raw_mean, right.raw_mean, f"第 {index} 帧：残差均值不逐位相等")
        self.assertIsNotNone(captured_at, "稳态帧之后应该已经捕获出 CUDA Graph")
        self.assertEqual(captured_at, max(LAGS), "捕获应发生在历史队列刚好填满的那一帧")
        self.assertIsNotNone(graphed._graph, "序列跑完后整图应仍然有效")

    def test_reset_releases_and_recaptures_graph(self):
        """reset() 必须丢弃整图与静态缓冲（区域尺寸可能变了），并在再次进入稳态时重新捕获。"""

        aligner = self._gpu_aligner()
        for frame in self.frames:  # 先跑到稳态，触发捕获。
            aligner.update(frame)
        self.assertIsNotNone(aligner._graph, "稳态后应已捕获整图")
        aligner.reset()  # 跨局重置。
        self.assertIsNone(aligner._graph, "reset 后整图必须被丢弃")
        self.assertIsNone(aligner._static_current_color, "reset 后静态缓冲必须被释放")
        self.assertEqual(len(aligner.history), 0, "reset 后历史队列应为空")
        first = aligner.update(self.frames[0])  # 重置后第一帧历史不足。
        self.assertIsNone(first, "重置后历史不足，应返回 None")
        for frame in self.frames[1:]:  # 再跑一遍到稳态。
            aligner.update(frame)
        self.assertIsNotNone(aligner._graph, "重新进入稳态后应再次捕获整图")
        self.assertFalse(aligner._graph_disabled, "正常重新捕获不应把图永久禁掉")

    def test_capture_failure_falls_back_to_eager_permanently(self):
        """捕获失败要永久回退 eager（而非降级到 CPU），且结果与纯 eager 逐位一致。"""

        aligner = self._gpu_aligner()  # 允许捕获。
        reference = self._gpu_aligner(disable_graph=True)  # 纯 eager 参照。
        torch = self.backend.xp.torch  # 裸 torch 模块。
        original = torch.cuda.CUDAGraph  # 记住原类，用完必须还原。

        class _Unsupported:  # 替身：一构造就抛错，模拟驱动/显存不支持捕获。
            def __init__(self, *args, **kwargs):
                raise RuntimeError("cuda graph capture unsupported")

        torch.cuda.CUDAGraph = _Unsupported  # 注入失败。
        try:
            for index, frame in enumerate(self.frames):
                left = aligner.update(frame)  # 捕获会失败，但本帧仍应给出 eager 结果。
                right = reference.update(frame)
                self.assertEqual(left is None, right is None, f"第 {index} 帧：可用 lag 判定不一致")
                if left is None:
                    continue
                self.assertEqual(
                    float(np.abs(_to_numpy(left.normalized) - _to_numpy(right.normalized)).max()),
                    0.0,
                    f"第 {index} 帧：捕获失败后应给出与纯 eager 逐位一致的结果",
                )
        finally:
            torch.cuda.CUDAGraph = original  # 还原全局，避免污染其他用例。
        self.assertTrue(aligner._graph_disabled, "捕获失败后应永久禁用整图")
        self.assertIsNone(aligner._graph, "捕获失败后不应留下半张图")
        self.assertIsNone(aligner._static_current_color, "捕获失败后静态缓冲应被释放")
        self.assertEqual(aligner.engine_name, "torch-farneback", "捕获失败不应把光流引擎降级成 CPU DIS")

    def test_cpu_backend_never_captures(self):
        """CPU 门面上不应尝试捕获整图（没有 launch 开销问题，也没有 cuda.CUDAGraph）。"""

        cpu = TensorTemporalAligner(LAGS, 1.0, Cv2DisFlow(2), gsb.numpy_backend())
        frames, _ = _color_sequence(max(LAGS) + 4)
        for frame in frames:
            cpu.update(frame)
        self.assertIsNone(cpu._graph, "CPU 门面不应持有整图")
        self.assertTrue(cpu._graph_disabled, "CPU 门面应直接把整图标记为禁用")
        self.assertEqual(cpu.engine_name, "cv2-dis", "CPU 链路的光流引擎应为 cv2 DIS")


if __name__ == "__main__":
    unittest.main()
