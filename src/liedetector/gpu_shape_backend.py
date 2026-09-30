"""测谎「找目标」打分内核的双后端实现：同一份算法代码，分别跑在 NumPy（CPU）与 torch CUDA（NVIDIA 显卡）上。

设计要点（对应光流 + 粒子滤波 GPU 加速改造的 Phase 1）：
- 双编译结构：打分函数全部写成后端无关的数组代码，数组门面（torch_array.NumpyArrayApi
  或 TorchArrayApi）由 ShapeScoreBackend.xp 注入；有 N 卡走 torch CUDA，没有则自动回落 NumPy，
  行为与结果等价。内核只允许用门面上那套显式最小 API，不得出现 numpy 专有的方法写法
  （``.astype`` / ``.max(axis=)`` / ``.copy()`` / ``.get()`` / ``[::-1]`` 均已被门面方法取代）。
- 职责切分：打分与时序证据链上 GPU；全部 RNG 仍留在 CPU NumPy（``random_seed`` 的可复现契约不变）。
- 降级策略：显卡运行期异常（显存不足/驱动重置等）由 shape_tracking 的薄壳捕获，
  自动改走 NumPy 重算本帧并计数；连续失败达到 _FAIL_LIMIT 次后本进程永久回退 NumPy，
  与 gpu_feature_match 的「隐藏式启用 + 永久降级」工程模式一致。
- 首次调用会初始化 CUDA 上下文与 torch 显存池（约 1~3 秒），服务启动时可用
  warmup_shape_backend() 预热（同时跑一次证据链，把光流核缓存与帧缓冲一并建好），避免解题首帧冷启动。
- CuPy 已从测谎链路移除；``src/gpu_match.py``、``src/gpu_feature_match.py``（任务模板匹配、
  自动登录）仍用 CuPy，与本模块无关。

打分数据流（每次调用）：证据图/状态/尺度上传后端（320px 证据图约 230KB，开销可忽略）
→ 后端上做旋转投影、9 组法向偏移采样、扇区/内部归约 → 得分与覆盖率转回 NumPy float32。
"""

from __future__ import annotations

import math  # 数学函数：与 shape_tracking 打分逻辑共用。
import threading  # 后端单例的线程安全初始化。
from dataclasses import dataclass  # 打分结果数据类（从 shape_tracking 迁移，避免循环导入）。

import numpy as np  # NumPy：CPU 侧标量判定与预热假数据。

from src.liedetector.torch_array import (  # 数组门面：一套 API，numpy 与 torch CUDA 两份实现。
    numpy_api,  # CPU 门面单例。
    torch_gpu_available,  # torch CUDA 可用性探测（含无驱动异常兜底）。
    torch_module,  # 显卡门面单例，同时显式创建 CUDA 上下文（不碰任何全局性能开关，详见门面 docstring）。
)


@dataclass
class ShapeEvidence:
    """一批粒子假设在证据图上的轮廓打分结果。"""

    scores: np.ndarray  # 每个粒子的综合得分 (count,)。
    coverage: np.ndarray  # 每个粒子的边界覆盖率 (count,)，衡量有多少采样点命中。


@dataclass
class BorderEvidence:
    """矩形模型专用的四边打分结果。"""

    scores: np.ndarray  # 每个粒子的四边综合得分 (count,)。
    coverage: np.ndarray  # 每个粒子取最强两边后的覆盖率 (count,)。


_FAIL_LIMIT = 3  # 连续运行期异常达到该次数后本进程永久回退 NumPy，避免每帧都失败刷屏。

_lock = threading.Lock()  # 保护后端单例与降级计数（服务线程与页签线程可能并发求解）。
_gpu_backend: "ShapeScoreBackend | None" = None  # torch 后端单例，None 表示尚未创建或已被降级丢弃。
_gpu_failed = False  # 显卡后端是否已被永久禁用（探测失败或运行期异常达到上限）。
_gpu_requests = 0  # 显卡后端的累计请求次数，用于区分「从未请求」与「请求后降级」。
_fail_count = 0  # 连续运行期异常计数，成功打分后由调用方清零。
_numpy_backend: "ShapeScoreBackend | None" = None  # NumPy 后端单例。
_gpu_available: bool | None = None  # torch CUDA 可用性探测缓存：首次探测要初始化 CUDA 驱动，进程内只探一次。


def gpu_backend_available() -> bool:  # torch CUDA 打分后端是否可用（结果缓存）。
    global _gpu_available  # 写探测缓存。
    if _gpu_available is None:  # 尚未探测过。
        try:  # 探测本身也可能因驱动异常抛错。
            _gpu_available = bool(torch_gpu_available())  # torch 已装、CUDA 可用且至少一块显卡。
        except Exception:  # 探测异常按不可用处理。
            _gpu_available = False  # 永久回落 NumPy。
    return _gpu_available  # 返回缓存结果。


def numpy_backend() -> ShapeScoreBackend:  # 取 NumPy 后端单例（CPU 路径）。
    global _numpy_backend  # 写单例缓存。
    if _numpy_backend is None:  # 尚未创建。
        _numpy_backend = ShapeScoreBackend(numpy_api(), "numpy")  # CPU 门面后端。
    return _numpy_backend  # 返回单例。


def require_gpu_backend() -> ShapeScoreBackend:  # 请求显卡后端：可用返回 torch 单例，不可用/已降级返回 NumPy 后端。
    global _gpu_backend, _gpu_failed, _gpu_requests  # 写单例与降级标记。
    with _lock:  # 服务线程与页签线程可能并发请求。
        _gpu_requests += 1  # 记录请求，供 backend_in_use 判断。
        if _gpu_failed:  # 已被永久禁用。
            return numpy_backend()  # 直接走 CPU。
        if not gpu_backend_available():  # 无 torch 或无 N 卡。
            _gpu_failed = True  # 标记禁用，后续请求不再探测。
            return numpy_backend()  # 走 CPU。
        if _gpu_backend is None:  # 首次创建。
            try:  # 构造门面会导入 torch 并初始化 CUDA，驱动异常时在这里抛。
                _gpu_backend = ShapeScoreBackend(torch_module(), "torch")  # torch CUDA 后端。
            except Exception:  # torch 导入/初始化失败。
                _gpu_failed = True  # 永久回退 NumPy。
                _gpu_backend = None  # 清掉半成品。
                return numpy_backend()  # 走 CPU。
        return _gpu_backend  # 返回显卡后端单例。


def backend_in_use() -> ShapeScoreBackend:  # 取当前实际生效的后端（供会话装配与诊断日志）。
    if _gpu_backend is not None and _gpu_requests > 0:  # 显卡后端已创建且被请求过。
        return _gpu_backend  # 生效的是显卡后端。
    return numpy_backend()  # 其余情况一律 NumPy。


def mark_gpu_failure() -> None:  # 记录一次显卡运行期异常；连续失败达到上限后永久回退 NumPy。
    global _fail_count, _gpu_backend, _gpu_failed  # 写失败计数与降级标记。
    with _lock:  # 与请求路径共用锁，避免降级竞态。
        _fail_count += 1  # 连续失败计数 +1。
        if _fail_count >= _FAIL_LIMIT:  # 达到上限。
            _gpu_failed = True  # 永久禁用显卡后端。
            _gpu_backend = None  # 丢弃单例，释放引用。


def clear_gpu_failure() -> None:  # 打分成功后清零连续失败计数（偶发异常不触发永久降级）。
    global _fail_count  # 写失败计数。
    with _lock:  # 与降级路径共用锁。
        _fail_count = 0  # 清零。


def warmup_shape_backend() -> str:  # 预热当前后端：返回「后端/光流引擎」描述串，异常静默（预热失败不影响正常路径）。
    """跑一次打分内核 + 一次完整时序证据链，把冷启动开销全部提到服务启动阶段。

    两条链路各自要预热的东西不同：

    - 打分内核：触发 CUDA 上下文创建（torch 首次约 1~3 秒）与显存池分配；
    - 证据链：触发 Farneback 的卷积核/衰减掩码/采样网格缓存，以及帧环形缓冲的显存分配。

    预热用的是**默认精度档**的光流参数与 lag（默认档即 high），因为实际求解绝大多数时候跑在这一档。
    预热尺寸与真实处理尺寸不必一致：本模块刻意不开 ``cudnn.benchmark``（详见 torch_array 的说明），
    所以换形状不会触发昂贵的算法搜索，预热只需把上下文、显存池与各类缓存建起来即可。

    返回 ``"torch(cuda)/farneback"`` 或 ``"numpy/dis"``，供服务日志核对实际生效的链路。
    """

    backend = require_gpu_backend()  # 触发探测与单例创建。
    try:  # 用一批假数据跑通打分链路，触发显存池分配与内部缓存。
        rng = np.random.default_rng(0)  # 预热专用随机源，与业务 RNG 无关。
        evidence = rng.random((64, 64), dtype=np.float32)  # 小证据图。
        states = np.zeros((8, 6), dtype=np.float32)  # 8 个假位姿。
        states[:, 0] = 32.0  # 居中 x。
        states[:, 1] = 32.0  # 居中 y。
        points = np.stack(  # 120 点单位圆轮廓（局部坐标）。
            [np.cos(np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)),  # x 分量。
             np.sin(np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False))],  # y 分量。
            axis=1,
        ).astype(np.float32) * 8.0  # 半径 8 像素。
        normals = points / 8.0  # 圆的法向即径向单位向量。
        template = _WarmupTemplate(points, normals)  # 轻量模板替身，只带打分用到的字段。
        score_shape_contours_core(backend.xp, evidence, states, template, 1.0, 64)  # 跑一次完整打分。
    except Exception:  # 预热失败不影响可用性判定，正式调用时会再走降级链路。
        pass
    _warmup_temporal_chain(backend)  # 再跑一次证据链（内部同样异常静默）。
    if backend.is_gpu:  # 显卡：torch 后端 + Farneback 光流。
        return f"{backend.name}(cuda)/farneback"  # 描述串供服务日志直接打印。
    return f"{backend.name}/dis"  # CPU：NumPy 打分 + cv2 DIS 光流。


def _warmup_temporal_chain(backend: ShapeScoreBackend) -> None:  # 证据链预热：合成几帧跑满全部 lag，把光流核与帧缓冲的缓存建起来。
    try:  # 延迟导入：避免 gpu_shape_backend → shape_session 的模块级循环引用，也避免无显卡环境白担 cv2/torch_flow 的导入成本。
        from src.liedetector.shape_session import PRECISION_TIERS, PRECISION_TIER_DEFAULT  # 默认精度档的参数。
        from src.liedetector.tensor_evidence import TensorTemporalAligner  # 双后端证据链。
        from src.liedetector.torch_flow import Cv2DisFlow, TorchFarnebackFlow  # 两套光流引擎。

        tier = PRECISION_TIERS[PRECISION_TIER_DEFAULT]  # 默认档（high）。
        lags = tuple(tier["temporal_lags"])  # 该档的时间基线。
        params = {name[len("flow_") :]: value for name, value in tier.items() if name.startswith("flow_")}  # 剔掉 flow_ 前缀即引擎构造参数。
        engine = TorchFarnebackFlow(backend.xp, **params) if backend.is_gpu else Cv2DisFlow(tier["dis_preset"])  # 按后端选引擎。
        aligner = TensorTemporalAligner(lags, 1.0, engine, backend)  # 预热专用对齐器，用完就丢。
        rng = np.random.default_rng(0)  # 预热专用随机源。
        frame = rng.integers(0, 255, size=(214, 320, 3), dtype=np.uint8)  # 实际处理尺度的合成帧。
        for step in range(max(lags) + 1):  # 喂满最大 lag + 1 帧，最后一帧会一次算完全部 lag 的合批光流。
            shifted = np.roll(frame, step, axis=1)  # 每帧水平平移 1 像素，造出真实位移供光流求解。
            aligner.update(shifted)  # 跑完整证据链。
    except Exception:  # 预热失败不影响可用性判定，正式求解时会重新装配并走降级链路。
        pass


class _WarmupTemplate:  # 预热专用的模板替身：只提供打分函数访问的字段，避免依赖完整 ShapeTemplate。

    def __init__(self, points, normals):  # 构造替身。
        self.points = points  # 轮廓采样点（局部坐标）。
        self.normals = normals  # 对应法向。
        self.interior_points = points[::8] * 0.5  # 稀疏内部采样点，取轮廓点的子集缩小一半。
        self.nominal_size = 12.0  # 标称尺寸。
        self.use_rectangle_model = False  # 预热只走通用轮廓模型。
        self.rectangle_angle = 0.0  # 矩形偏置角（未用）。
        self.rectangle_side = 16.0  # 矩形边长（未用）。


class ShapeScoreBackend:  # 打分后端：持有数组门面（numpy 门面或 torch 门面）与后端名。

    def __init__(self, xp, name: str):  # 构造后端。
        self.xp = xp  # 数组门面：全部打分代码通过它执行。
        self.name = name  # 后端名："numpy" 或 "torch"，供日志与诊断。

    @property
    def is_gpu(self) -> bool:  # 是否显卡后端。
        return self.name != "numpy"  # 除 CPU 门面外都算显卡。

    def to_backend(self, array):  # 把 NumPy 数组搬到后端设备（CPU 后端近似零拷贝）。
        return self.xp.asarray(array)  # torch 门面会做 H2D 上传；numpy 门面同设备直接复用。

    def to_numpy(self, array) -> np.ndarray:  # 把后端数组搬回 NumPy（显卡后端做 D2H 下载）。
        return self.xp.to_numpy(array)  # 门面内部区分 torch 张量与 numpy 数组。


def _on(xp, array):  # 确保数组落在后端设备上：模板点集等常驻 CPU 的小数组在这里上传（每次仅几 KB）。
    return xp.asarray(array)  # 两份门面的 asarray 都先判定是否已在目标设备/同类型，不在才搬运。


# ---------------------------------------------------------------------------
# 打分内核：与原 shape_tracking 的 NumPy 实现逐行等价，np 换成后端数组模块 xp。
# ---------------------------------------------------------------------------


def sample_map_core(xp, image, x, y):  # 在单通道证据图上做最近邻批量采样，越界位置返回 0。
    image = _on(xp, image)  # 证据图确保在后端设备上（CPU 传入时在这里上传）。
    x = _on(xp, x)  # 采样坐标同样归位。
    y = _on(xp, y)  # 采样坐标同样归位。
    height, width = image.shape  # 证据图尺寸。
    xi = xp.astype(xp.rint(x), xp.int32)  # x 坐标整数化。
    yi = xp.astype(xp.rint(y), xp.int32)  # y 坐标整数化。
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)  # 标记在图内的采样点。
    xi = xp.clip(xi, 0, width - 1)  # 钳制索引，防止花式索引越界报错。
    yi = xp.clip(yi, 0, height - 1)  # 钳制 y 索引。
    values = image[yi, xi]  # 批量取值。
    return xp.where(valid, values, 0.0)  # 越界点置 0，等价于「此处没有证据」。


def transform_template_points_core(xp, template, states, scales):  # 把模板轮廓点与法向批量旋转缩放到每个假设姿态。
    states = _on(xp, states)  # 状态矩阵确保在后端设备上。
    points_local = _on(xp, template.points)  # 模板轮廓点上传后端（常驻 CPU，每次调用上传几 KB）。
    normals_local = _on(xp, template.normals)  # 模板法向上传后端。
    angles = xp.deg2rad(states[:, 4])[:, None, None]  # 各假设角度转弧度并扩维广播。
    cosine = xp.cos(angles)  # 余弦。
    sine = xp.sin(angles)  # 正弦。
    if np.isscalar(scales):  # 标量尺度：所有假设共用同一尺度（np.isscalar 只看 Python 侧类型，与后端数组库无关）。
        scale_values = xp.full((len(states), 1, 1), float(scales), dtype=xp.float32)  # 广播成 (count,1,1)。
    else:  # 数组尺度：每个假设一个尺度。
        scale_values = xp.astype(_on(xp, scales), xp.float32)[:, None, None]  # 上传并扩维成 (count,1,1)。
    local_x = points_local[None, :, 0:1]  # 模板局部 x，形状 (1,N,1)。
    local_y = points_local[None, :, 1:2]  # 模板局部 y。
    point_x = states[:, 0, None, None] + scale_values * (  # 旋转缩放后平移到假设中心 x。
        local_x * cosine - local_y * sine  # 二维旋转的 x 分量。
    )
    point_y = states[:, 1, None, None] + scale_values * (  # 旋转缩放后平移到假设中心 y。
        local_x * sine + local_y * cosine  # 二维旋转的 y 分量。
    )
    normal_x = normals_local[None, :, 0:1] * cosine - normals_local[None, :, 1:2] * sine  # 法向只旋转不平移不缩放。
    normal_y = normals_local[None, :, 0:1] * sine + normals_local[None, :, 1:2] * cosine  # 法向 y 分量。
    return xp.concatenate([point_x, point_y], axis=2), xp.concatenate([normal_x, normal_y], axis=2)  # 返回 (count,N,2) 点集与法向。


def score_rotated_borders_core(xp, evidence, states, side, play_height) -> BorderEvidence:  # 矩形模型四边打分（后端版）。
    evidence = _on(xp, evidence)  # 证据图确保在后端设备上。
    states = _on(xp, states)  # 状态矩阵确保在后端设备上。
    count = len(states)  # 假设数量（用于返回数组形状）。
    samples_per_side = 22  # 每条边的采样点数。
    half = side * 0.5  # 半边长。
    along = xp.linspace(-0.82 * half, 0.82 * half, samples_per_side, dtype=xp.float32)  # 边内采样位置，只取中间 82% 避免角点歧义。

    local_x = xp.stack(  # 四条边上采样点的局部 x 坐标（上、右、下、左）。
        [along, xp.full_like(along, half), xp.flip(along, axis=0), xp.full_like(along, -half)]  # 下边反向，保证四条边环绕方向一致。
    )
    local_y = xp.stack(  # 四条边上采样点的局部 y 坐标。
        [xp.full_like(along, -half), along, xp.full_like(along, half), xp.flip(along, axis=0)]  # 与 local_x 配对构成闭合正方形。
    )
    unit = xp.full_like(along, 1.0)  # 单位向量（门面不提供 ones_like，用 full_like 等价构造）。
    zero = xp.zeros_like(along)  # 零向量。
    normal_x = xp.stack(  # 四条边的外法向 x 分量。
        [zero, unit, zero, -unit]  # 上/右/下/左。
    )
    normal_y = xp.stack(  # 四条边的外法向 y 分量。
        [-unit, zero, unit, zero]  # 与 normal_x 配对。
    )

    radians = xp.deg2rad(states[:, 4])[:, None, None]  # 每个假设的角度转弧度，扩维便于广播。
    cosine = xp.cos(radians)  # 余弦。
    sine = xp.sin(radians)  # 正弦。
    base_x = (  # 旋转平移后的采样点 x 坐标，形状 (count, 4, samples)。
        states[:, 0, None, None]  # 假设中心 x。
        + local_x[None, :, :] * cosine  # 局部 x 旋转分量。
        - local_y[None, :, :] * sine  # 局部 y 旋转分量。
    )
    base_y = (  # 旋转平移后的采样点 y 坐标。
        states[:, 1, None, None]  # 假设中心 y。
        + local_x[None, :, :] * sine  # 局部 x 旋转分量。
        + local_y[None, :, :] * cosine  # 局部 y 旋转分量。
    )
    rotated_normal_x = normal_x[None, :, :] * cosine - normal_y[None, :, :] * sine  # 法向 x 分量随角度旋转。
    rotated_normal_y = normal_x[None, :, :] * sine + normal_y[None, :, :] * cosine  # 法向 y 分量随角度旋转。

    border_samples = []  # 收集边界上的多偏移采样。
    for offset in (-3.0, -1.5, 0.0, 1.5, 3.0):  # 沿法向取 5 个偏移，容忍边界定位误差。
        border_samples.append(  # 追加一个偏移下的采样结果。
            sample_map_core(  # 在证据图上采样。
                xp,  # 后端数组模块。
                evidence,  # 证据图。
                base_x + offset * rotated_normal_x,  # 偏移后的 x。
                base_y + offset * rotated_normal_y,  # 偏移后的 y。
            )
        )
    border = xp.max(xp.stack(border_samples, axis=0), axis=0)  # 5 个偏移取最大，得到边界响应。

    context_samples = []  # 收集上下文采样。
    context_distance = max(7.0, side * 0.085)  # 上下文距离随边长自适应，至少 7 像素。
    for offset in (-1.35 * context_distance, -context_distance, context_distance, 1.35 * context_distance):  # 边界两侧各取 2 个上下文点。
        context_samples.append(  # 追加一个上下文偏移的采样。
            sample_map_core(  # 在证据图上采样。
                xp,  # 后端数组模块。
                evidence,  # 证据图。
                base_x + offset * rotated_normal_x,  # 偏移后的 x。
                base_y + offset * rotated_normal_y,  # 偏移后的 y。
            )
        )
    context = xp.mean(xp.stack(context_samples, axis=0), axis=0)  # 4 个上下文点取均值，代表局部背景水平。

    contrast = border - context  # 边界相对背景的对比度，是真正的形状证据。
    side_scores = xp.mean(contrast, axis=2)  # 每条边的平均对比度 (count, 4)。
    # 片段后期可能只剩一个角（相邻两条边）还看得见。因此取最强两边，
    # 但要对「单条边一枝独秀」施加惩罚，避免仅凭一条岩石边缘就取胜。
    sorted_sides = xp.sort(side_scores, axis=1)  # 四条边得分升序排列。
    strongest_two = xp.mean(sorted_sides[:, -2:], axis=1)  # 最强两边的均值。
    side_imbalance = sorted_sides[:, -1] - sorted_sides[:, -2]  # 最强边与次强边的差距，衡量失衡程度。
    mean_score = xp.mean(side_scores, axis=1)  # 四条边的整体均值。
    coverage_by_side = xp.mean(border > context + 0.70, axis=2)  # 每条边上「显著高于背景」的采样点比例。
    sorted_coverage = xp.sort(coverage_by_side, axis=1)  # 覆盖率升序。
    coverage = xp.mean(sorted_coverage[:, -2:], axis=1)  # 取最强两边的覆盖率。
    scores = (  # 综合得分。
        0.25 * mean_score  # 整体对比度。
        + 0.65 * strongest_two  # 最强两边为主。
        + 0.10 * sorted_sides[:, -1]  # 最强边额外加成。
        - 0.12 * xp.maximum(side_imbalance - 2.0, 0.0)  # 失衡惩罚，超过 2.0 才开始扣。
        + 0.80 * coverage  # 覆盖率加成。
    )
    # 惩罚那些把视频/UI 强边界当成自己一条边的假设。
    # 用软惩罚而非硬剔除，真实目标靠近面板时仍然允许被找回。
    valid = (  # 采样点是否落在可靠区域内。
        (base_x >= 12.0)  # 左边留 12 像素。
        & (base_x <= evidence.shape[1] - 13.0)  # 右边留 13 像素。
        & (base_y >= 6.0)  # 上边留 6 像素。
        & (base_y <= play_height - 7.0)  # 下边以有效高度为准留 7 像素。
    )
    valid_fraction = xp.mean(valid, axis=(1, 2))  # 每个假设的有效采样点比例。
    scores -= 8.0 * (1.0 - valid_fraction)  # 越界惩罚。
    return BorderEvidence(xp.astype(scores, xp.float32), xp.astype(coverage, xp.float32))  # 返回打分与覆盖率（后端数组）。


def score_shape_contours_core(xp, evidence, states, template, scales, play_height) -> ShapeEvidence:  # 通用轮廓打分（后端版）。
    evidence = _on(xp, evidence)  # 证据图确保在后端设备上（整帧只上传一次，后续采样全在显存）。
    states = _on(xp, states)  # 状态矩阵确保在后端设备上。
    if template.use_rectangle_model and np.isscalar(scales):  # 矩形模型且标量尺度时走规则四边评分（更抗白色蒙版毛边）。
        rectangle_states = xp.clone(states)  # 复制状态，避免污染调用方的角度。
        rectangle_states[:, 4] = (  # 角度叠加矩形偏置并折回 90 度对称域。
            rectangle_states[:, 4] + template.rectangle_angle
        ) % 90.0
        rectangle_evidence = score_rotated_borders_core(  # 转发到矩形四边评分。
            xp,  # 后端数组模块。
            evidence,  # 证据图。
            rectangle_states,  # 调整角度后的状态。
            template.rectangle_side * float(scales),  # 按尺度缩放的边长。
            play_height,  # 有效高度。
        )
        return ShapeEvidence(  # 统一封装成 ShapeEvidence 返回。
            rectangle_evidence.scores,  # 得分。
            rectangle_evidence.coverage,  # 覆盖率。
        )
    points, normals = transform_template_points_core(xp, template, states, scales)  # 通用模型：变换学习到的 120 点轮廓。
    point_x, point_y = points[:, :, 0], points[:, :, 1]  # 拆出采样点坐标。
    normal_x, normal_y = normals[:, :, 0], normals[:, :, 1]  # 拆出法向。
    # 边界 5 个法向偏移 + 上下文 4 个偏移合并成一次采样调用，减少重复裁剪/钳位开销。
    offsets = (-3.0, -1.5, 0.0, 1.5, 3.0)  # 边界法向偏移。
    context_distance = max(7.0, template.nominal_size * 0.08)  # 上下文距离随形状尺寸自适应。
    context_offsets = (  # 边界两侧各 2 个上下文偏移。
        -1.35 * context_distance,  # 外侧远点。
        -context_distance,  # 外侧近点。
        context_distance,  # 内侧近点。
        1.35 * context_distance,  # 内侧远点。
    )
    sample_x = xp.concatenate(  # 9 组采样坐标纵向拼接。
        [point_x + offset * normal_x for offset in offsets + context_offsets]  # 边界 + 上下文。
    )
    sample_y = xp.concatenate(
        [point_y + offset * normal_y for offset in offsets + context_offsets]
    )
    sampled = sample_map_core(xp, evidence, sample_x, sample_y)  # 一次性采样。
    grouped = sampled.reshape(9, *point_x.shape)  # 拆回 9 组，前 5 组边界、后 4 组上下文。
    border = xp.max(grouped[:5], axis=0)  # 边界 5 组取最大得边界响应。
    context = xp.mean(grouped[5:], axis=0)  # 上下文 4 组取均值得局部背景。
    contrast = border - context  # 边界相对背景的对比度。
    sector_count = 12  # 把轮廓分成 12 个扇区分别统计。
    sector_length = len(template.points) // sector_count  # 每个扇区的采样点数。
    usable = sector_length * sector_count  # 实际可用点数（丢弃除不尽的尾巴）。
    sector_scores = xp.mean(  # 每个扇区的平均对比度 (count, 12)。
        contrast[:, :usable].reshape(len(states), sector_count, sector_length), axis=2  # 按扇区聚合。
    )
    point_coverage = border > context + 0.70  # 单点是否「显著高于背景」。
    sector_coverage = xp.mean(  # 每个扇区的覆盖率。
        point_coverage[:, :usable].reshape(len(states), sector_count, sector_length),  # 按扇区聚合。
        axis=2,  # 求均值。
    )
    sorted_scores = xp.sort(sector_scores, axis=1)  # 扇区得分升序。
    sorted_coverage = xp.sort(sector_coverage, axis=1)  # 扇区覆盖率升序。
    top_half = xp.mean(sorted_scores[:, -6:], axis=1)  # 最强 6 个扇区的均值。
    top_quarter = xp.mean(sorted_scores[:, -3:], axis=1)  # 最强 3 个扇区额外加成。
    imbalance = sorted_scores[:, -1] - sorted_scores[:, -3]  # 最强与第三强的差距，衡量是否只靠单侧取胜。
    coverage = xp.mean(sorted_coverage[:, -6:], axis=1)  # 最强 6 个扇区的覆盖率均值。
    scores = (  # 边界综合得分。
        0.28 * xp.mean(sector_scores, axis=1)  # 全部扇区均值。
        + 0.54 * top_half  # 最强一半扇区为主。
        + 0.18 * top_quarter  # 最强四分之一额外加成。
        - 0.10 * xp.maximum(imbalance - 2.2, 0.0)  # 失衡惩罚，超过 2.2 才开始扣。
        + 0.75 * coverage  # 覆盖率加成。
    )
    radians = xp.deg2rad(states[:, 4])[:, None]  # 角度转弧度，用于变换内部点。
    cosine = xp.cos(radians)  # 余弦。
    sine = xp.sin(radians)  # 正弦。
    if np.isscalar(scales):  # 标量尺度。
        scale_values = xp.full((len(states), 1), float(scales), dtype=xp.float32)  # 广播成 (count,1)。
    else:  # 数组尺度。
        scale_values = xp.astype(_on(xp, scales), xp.float32)[:, None]  # 上传并扩维成 (count,1)。
    interior_local = _on(xp, template.interior_points)  # 内部采样点上传后端。
    interior_x = states[:, 0, None] + scale_values * (  # 内部采样点变换后的 x。
        interior_local[None, :, 0] * cosine  # 局部 x 旋转分量。
        - interior_local[None, :, 1] * sine  # 局部 y 旋转分量。
    )
    interior_y = states[:, 1, None] + scale_values * (  # 内部采样点变换后的 y。
        interior_local[None, :, 0] * sine  # 局部 x 旋转分量。
        + interior_local[None, :, 1] * cosine  # 局部 y 旋转分量。
    )
    interior = sample_map_core(xp, evidence, interior_x, interior_y)  # 采样内部残差。
    interior_mean = xp.mean(xp.clip(interior, 0.0, 8.0), axis=1)  # 内部残差均值，截断 8.0 抑制异常亮区。
    interior_coverage = xp.mean(interior > 1.40, axis=1)  # 内部有显著残差的点比例。
    scores += 0.28 * interior_mean + 0.55 * interior_coverage  # 内部证据叠加到总分（半透明目标内部也会有残差）。
    coverage = 0.78 * coverage + 0.22 * interior_coverage  # 覆盖率以边界为主、内部为辅混合。
    valid = (  # 轮廓采样点是否落在可靠区域内。
        (point_x >= 10.0)  # 左边留 10 像素。
        & (point_x <= evidence.shape[1] - 11.0)  # 右边留 11 像素。
        & (point_y >= 6.0)  # 上边留 6 像素。
        & (point_y <= play_height - 7.0)  # 下边以有效高度为准留 7 像素。
    )
    scores -= 7.0 * (1.0 - xp.mean(valid, axis=1))  # 越界惩罚：有效比例越低扣分越多。
    return ShapeEvidence(xp.astype(scores, xp.float32), xp.astype(coverage, xp.float32))  # 返回打分与覆盖率（后端数组）。
