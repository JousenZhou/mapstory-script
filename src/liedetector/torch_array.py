"""测谎 GPU 化改造的数组门面层：同一份算法代码，numpy 与 torch CUDA 两套实现。

本模块只做「把 numpy 风格的数组 API 映射到 torch」这一件事，不含任何测谎业务逻辑。
打分内核（gpu_shape_backend）、时序证据链（tensor_evidence）与粒子滤波（shape_tracking）
全部通过注入的 xp 门面执行，因此：

- 有 N 卡 + 装了 torch：xp = TorchArrayApi，全链路（光流/滤波/粒子）跑 CUDA；
- 其余情况：xp = NumpyArrayApi，与改造前的 CPU 行为一致。

两套实现必须逐条对齐的关键语义（写内核时依赖这些约定）：
1. ``median`` 偶数长度取中间两个数的均值，与 ``np.median`` 一致（torch.median 只给下中位数，故用两次 kthvalue 手工实现）；
2. ``rint`` 为四舍六入五成双，两边一致（torch 侧用 ``torch.round``，它也是 round-half-to-even）；
3. ``warp`` 是唯一的设备专有算子：numpy 走 ``cv2.remap(BORDER_REFLECT)``，torch 走
   ``grid_sample(padding_mode='reflection')``。两者只在画面最外圈 1~2 像素的填充方式上不同，
   而证据链随后会把 ``max(8, W*0.014)`` 宽的边缘带整体清零，差异不可见；
4. 布尔数组求均值前一律先转 float32（torch 对 bool 求 mean 会报 dtype 错）；
5. ``asarray`` 对已在目标设备上的数组直接复用，不做无谓拷贝；
6. ``clip`` 对整型数组保持整型（torch.clamp 遇到浮点边界会把结果提升成 float，
   而采样内核钳制完索引后还要拿它当花式索引下标用，dtype 必须仍是整型）；
7. 不支持 numpy 的负步长切片 ``a[::-1]``（torch 直接报 ``step must be greater than zero``），
   反向一律走 ``flip(a, axis)``；
8. Python 标量当「弱类型」看待（``asarray(1.0)`` 得到 float32 而不是 float64），
   与 numpy 在 NEP 50 下对 ``float``/``int`` 字面量的处理一致，避免 ``maximum(1.0, float32)``
   在显卡上把整条链提升成 float64；
9. ``warp`` 与 ``to_gray`` 是两个设备专有算子：CPU 侧分别走 ``cv2.remap`` 与 ``cv2.cvtColor``
   （与改造前逐位一致，打包版行为零变化），显卡侧走 ``grid_sample`` 与定点灰度公式。
   两者的跨设备差异都只在最后一位量级（灰度 ≤1 个灰阶、约 0.15% 像素；重映射只在画面
   最外圈 1~2 像素的填充方式上不同），而证据链随后会把 ``max(8, W*0.014)`` 宽的边缘带整体清零。
10. ``select`` 返回的是**共享存储的视图**（``torch.select`` / ``np.take`` 标量下标），
    ``copy_into`` 是**地址不变**的就地拷贝。CUDA Graph 要求图内读到的张量地址固定，
    证据链靠这两个算子把「固定缓冲 + 每帧刷新」表达出来，因此两套门面的语义必须一致。

刻意不做的事：不在这里初始化 CUDA、不缓存后端单例（那是 gpu_shape_backend 的职责），
本模块只暴露纯函数式门面，便于单测直接注入。
"""

from __future__ import annotations

import contextlib  # numpy 门面的空上下文（与 torch.no_grad 对齐调用方式）。

import numpy as np  # CPU 门面的底层实现。


def _download(array):  # 显卡张量降级下载：CPU 门面收到留在显存的 torch 张量时先拉回主存。
    detach = getattr(array, "detach", None)  # torch 张量的去求导方法，numpy 数组没有。
    if callable(detach):  # 是 torch 张量。
        return detach().cpu().numpy()  # D2H 后转 numpy（已在主存的张量上近似零开销）。
    return array  # 其余情况原样交给 numpy。


class NumpyArrayApi:
    """CPU 门面：绝大多数方法直接转发 numpy，只补齐 numpy 没有的几个名字。"""

    name = "numpy"  # 门面名，供日志与诊断。
    is_gpu = False  # 非显卡门面。

    float32 = np.float32  # dtype 常量：与 torch 门面同名，内核代码无需区分。
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8  # 灰度帧的存储精度（cv2 的光流接口只收 CV_8U）。

    # ------------------------------------------------------------------ 数组搬运

    def asarray(self, array, dtype=None):  # 转成本门面数组；已是 numpy 时近似零拷贝。
        data = _download(array)  # 打分内核显卡降级时，证据图可能仍留在显存。
        return np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)

    def astype(self, array, dtype):  # 显式类型转换（替代 numpy 专有的 .astype 方法）。
        return _download(array).astype(dtype)

    def clone(self, array):  # 独立拷贝（替代 .copy() 方法，torch 侧叫 .clone()）。
        return np.array(_download(array), copy=True)

    def to_numpy(self, array):  # 统一取回 numpy（CPU 门面下就是原数组）。
        return np.asarray(_download(array))

    def fill(self, array, value):  # 就地填满同一个值。
        array.fill(value)
        return array

    def copy_into(self, destination, source):  # 就地把 source 写进 destination（地址不变，可广播）。
        destination[...] = _download(source)
        return destination

    @contextlib.contextmanager
    def inference(self):  # 推理上下文：CPU 无需关闭自动求导，给一个空上下文保持调用方式一致。
        yield

    # ------------------------------------------------------------------ 视图与索引

    def narrow(self, array, axis, start, length=None):  # 取某轴上 [start, start+length) 的视图。
        data = _download(array)
        shape = data.shape
        stop = shape[axis] if length is None else start + length
        index = [slice(None)] * len(shape)
        index[axis] = slice(start, stop)
        return data[tuple(index)]

    def take(self, array, indices, axis=0):  # 按索引数组取行/列。
        return np.take(_download(array), np.asarray(_download(indices)), axis=axis)

    def select(self, array, axis, index):  # 取某轴上的单个下标并去掉该轴（替代 a[i]，两套门面语义一致）。
        return np.take(_download(array), int(index), axis=axis)

    def ravel(self, array):  # 展平成一维（替代 numpy 的 .ravel() / .reshape(-1)）。
        return np.ravel(_download(array))

    def column(self, array, index):  # 取第 index 列并降成一维（替代 a[:, index]）。
        return self.ravel(self.narrow(array, 1, index, 1))

    def put_rows(self, array, indices, values):  # 按行索引就地覆写（替代 a[idx] = v，两套门面语义一致）。
        data = _download(array)
        data[np.asarray(_download(indices))] = _download(values)
        return data

    def flip(self, array, axis=0):  # 沿某轴反转（替代 numpy 的 a[::-1] 负步长切片）。
        return np.flip(np.asarray(array), axis=axis)

    def warp(self, image, flow):  # 按光流重映射（CPU 走 cv2.remap，与改造前逐位一致）。
        import cv2  # 延迟导入：本门面被 torch 路径使用时不应牵出 cv2 依赖判断。

        image = np.asarray(image)  # (H,W,C) 或 (B,H,W,C)。
        height, width = image.shape[-3], image.shape[-2]
        base_x, base_y = self._meshgrid(height, width)
        return cv2.remap(
            image,
            base_x + np.asarray(flow)[..., 0],
            base_y + np.asarray(flow)[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def to_gray(self, color):  # BGR → 灰度：CPU 直接转给 cv2，与改造前的 cvtColor 逐位一致。
        import cv2  # 延迟导入，理由同 warp。

        return cv2.cvtColor(np.asarray(color), cv2.COLOR_BGR2GRAY)

    def _meshgrid(self, height, width):  # 采样网格坐标缓存：同一尺寸只建一次。
        cache = getattr(self, "_grid_cache", None)
        if cache is None:
            cache = {}
            self._grid_cache = cache
        key = (height, width)
        grids = cache.get(key)
        if grids is None:
            grids = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            cache[key] = grids
        return grids

    # ------------------------------------------------------------------ 逐元素运算

    def rint(self, array):
        return np.rint(array)

    def clip(self, array, low, high):
        return np.clip(array, low, high)

    def where(self, condition, left, right):
        return np.where(condition, left, right)

    def maximum(self, left, right):
        return np.maximum(left, right)

    def minimum(self, left, right):
        return np.minimum(left, right)

    def abs(self, array):
        return np.abs(array)

    def sqrt(self, array):
        return np.sqrt(array)

    def exp(self, array):
        return np.exp(array)

    def cos(self, array):
        return np.cos(array)

    def sin(self, array):
        return np.sin(array)

    def deg2rad(self, array):
        return np.deg2rad(array)

    def arctan2(self, y, x):
        return np.arctan2(y, x)

    def isfinite(self, array):
        return np.isfinite(array)

    # ------------------------------------------------------------------ 归约与排序

    def sum(self, array, axis=None):
        return np.sum(array, axis=axis)

    def mean(self, array, axis=None):  # 布尔数组先转 float32，与 torch 门面对齐。
        data = np.asarray(array)
        if data.dtype == np.bool_:
            data = data.astype(np.float32)
        return np.mean(data, axis=axis)

    def max(self, array, axis=None):
        return np.max(array, axis=axis)

    def min(self, array, axis=None):
        return np.min(array, axis=axis)

    def argmax(self, array):
        return np.argmax(_download(array))

    def median(self, array, axis=None):
        return np.median(array, axis=axis)

    def sort(self, array, axis=-1):
        return np.sort(_download(array), axis=axis)

    def argsort(self, array, axis=-1, descending=False):
        # 稳定排序：并列值的相对次序固定为「下标升序」，与 torch 门面对齐（两边都不依赖排序实现细节）。
        order = np.argsort(_download(array), axis=axis, kind="stable")
        if not descending:
            return order
        return np.flip(order, axis=axis if axis is not None else 0)

    def searchsorted(self, sorted_sequence, values, side="left"):
        return np.searchsorted(np.asarray(_download(sorted_sequence)), np.asarray(_download(values)), side=side)

    def cumsum(self, array, axis=0):
        return np.cumsum(_download(array), axis=axis)

    def linalg_norm(self, array, axis=None):
        return np.linalg.norm(_download(array), axis=axis)

    # ------------------------------------------------------------------ 构造

    def linspace(self, start, stop, count, dtype=None):
        return np.linspace(start, stop, count, dtype=dtype or np.float32)

    def arange(self, count, dtype=None):
        return np.arange(count, dtype=dtype or np.float32)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)

    def zeros_like(self, array, dtype=None):
        data = _download(array)
        return np.zeros_like(data, dtype=dtype) if dtype is not None else np.zeros_like(data)

    def full(self, shape, value, dtype=None):
        return np.full(shape, value, dtype=dtype or np.float32)

    def full_like(self, array, value):
        return np.full_like(_download(array), value)

    def stack(self, arrays, axis=0):
        return np.stack([_download(item) for item in arrays], axis=axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate([_download(item) for item in arrays], axis=axis)


class TorchArrayApi:
    """torch CUDA 门面：把上面那套 numpy 风格 API 映射到 torch 算子。

    所有创建的张量都落在构造时选定的 device（默认 cuda），dtype 与 numpy 侧保持一致，
    这样同一份内核代码在两套门面下的数值行为只差浮点舍入。
    """

    name = "torch"  # 门面名。
    is_gpu = True  # 显卡门面。

    def __init__(self, torch_module, device=None):  # 构造门面：注入 torch 模块与目标设备。
        self.torch = torch_module  # 真正的 torch 模块，torch_flow 等需要裸算子的地方从这里取。
        self.device = torch_module.device(device or "cuda")  # 目标设备。
        self.float32 = torch_module.float32  # dtype 常量对齐 numpy 门面。
        self.float64 = torch_module.float64
        self.int32 = torch_module.int32
        self.int64 = torch_module.int64
        self.uint8 = torch_module.uint8  # 灰度帧的存储精度。
        self._grid_cache = {}  # (H, W) -> 常驻显存的采样网格，避免每帧重建。

    # ------------------------------------------------------------------ 数组搬运

    def asarray(self, array, dtype=None):  # 转成显存张量；已是本设备张量时直接复用。
        torch = self.torch
        if isinstance(array, torch.Tensor):
            if array.device != self.device or (dtype is not None and array.dtype != dtype):
                return array.to(device=self.device, dtype=dtype)
            return array
        if dtype is None and isinstance(array, (int, float)) and not isinstance(array, bool):
            # Python 标量按「弱类型」处理：默认 float32，避免与 float32 张量混算时被提升成 float64。
            return torch.as_tensor(array, dtype=torch.float32, device=self.device)
        data = np.asarray(array)
        if dtype is not None:
            data = data.astype(self._numpy_dtype(dtype), copy=False)
        return torch.as_tensor(np.ascontiguousarray(data), device=self.device)

    def _numpy_dtype(self, dtype):  # torch dtype -> numpy dtype，供上传前统一精度。
        mapping = {
            self.torch.float32: np.float32,
            self.torch.float64: np.float64,
            self.torch.int32: np.int32,
            self.torch.int64: np.int64,
            self.torch.uint8: np.uint8,
        }
        return mapping.get(dtype, np.float32)

    def astype(self, array, dtype):  # 显式类型转换。
        return self.asarray(array).to(dtype)

    def clone(self, array):  # 独立拷贝。
        return self.asarray(array).clone()

    def to_numpy(self, array):  # D2H 下载并转回 numpy。
        if isinstance(array, np.ndarray):
            return array
        return array.detach().to("cpu", copy=True).numpy()

    def fill(self, array, value):  # 就地填满同一个值。
        array.fill_(float(value))
        return array

    def copy_into(self, destination, source):  # 就地 D2D 拷贝，目标地址不变：CUDA Graph 的静态输入缓冲靠它刷新。
        destination.copy_(self.asarray(source))
        return destination

    def inference(self):  # 推理上下文：关掉自动求导，省掉建图开销。
        return self.torch.no_grad()

    # ------------------------------------------------------------------ 视图与索引

    def narrow(self, array, axis, start, length=None):  # 取某轴上 [start, start+length) 的视图。
        data = self.asarray(array)
        if length is None:
            length = data.shape[axis] - start
        return self.torch.narrow(data, axis, start, length)

    def take(self, array, indices, axis=0):  # 按索引数组取行/列。
        index_tensor = self._index_tensor(indices)  # 索引就是在显卡上算出来的（重采样）时只换 dtype，否则一次 H2D。
        return self.torch.index_select(self.asarray(array), axis, index_tensor)

    def select(self, array, axis, index):  # 取某轴上的单个下标并去掉该轴（等价 a[i]，返回的是**共享存储的视图**）。
        return self.torch.select(self.asarray(array), axis, int(index))

    def flip(self, array, axis=0):  # 沿某轴反转（torch 不支持 a[::-1]，必须走 flip）。
        return self.torch.flip(self.asarray(array), dims=[axis])

    def ravel(self, array):  # 展平成一维。
        return self.asarray(array).reshape(-1)

    def column(self, array, index):  # 取第 index 列并降成一维（替代 a[:, index]）。
        return self.ravel(self.narrow(array, 1, index, 1))

    def _index_tensor(self, indices):  # 索引统一成显卡上的 int64：设备侧索引不绕主存，NumPy 索引一次 H2D。
        torch = self.torch
        if isinstance(indices, torch.Tensor):
            return indices.to(device=self.device, dtype=torch.int64)
        return torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)

    def put_rows(self, array, indices, values):  # 按行索引就地覆写（index_copy_，不做主存同步）。
        data = self.asarray(array)
        data.index_copy_(0, self._index_tensor(indices), self.asarray(values).to(data.dtype))
        return data

    def warp(self, image, flow):  # 按光流重映射（显卡走 grid_sample，双线性 + 反射填充）。
        torch = self.torch
        data = self.asarray(image)  # (H,W,C) 或 (B,H,W,C)。
        source_dtype = data.dtype  # 记住输入 dtype，整型输入要按 cv2.remap 的语义还原回去。
        integer_output = not data.is_floating_point()  # grid_sample 只吃浮点，整型要先转。
        if integer_output:
            data = data.to(torch.float32)
        displacement = self.asarray(flow)
        height, width = data.shape[-3], data.shape[-2]
        base_x, base_y = self._meshgrid(height, width)
        # grid_sample 的 align_corners=False 约定：像素中心 p 对应归一化坐标 (2p+1)/size-1。
        grid_x = (2.0 * (base_x + displacement[..., 0]) + 1.0) / width - 1.0
        grid_y = (2.0 * (base_y + displacement[..., 1]) + 1.0) / height - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        batched = data.dim() == 4  # 支持 (B,H,W,C) 批量重映射。
        if not batched:
            data = data.unsqueeze(0)
            grid = grid.unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            data.permute(0, 3, 1, 2),  # grid_sample 要求 (B,C,H,W)。
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1)  # 换回 (B,H,W,C)。
        if integer_output:  # cv2.remap 对整型输入会四舍六入五成双后饱和截断回原 dtype。
            upper = float(torch.iinfo(source_dtype).max)
            sampled = torch.clamp(torch.round(sampled), 0.0, upper).to(source_dtype)
        return sampled if batched else sampled.squeeze(0)

    def to_gray(self, color):  # BGR → 灰度：显卡上用 OpenCV 的定点公式，全程不离开显存。
        torch = self.torch
        data = self.asarray(color).to(torch.float32)  # uint8 最大 255，乘上系数后仍远小于 2^24，float32 全程精确。
        value = (
            data[..., 0] * 1868.0  # B：0.114 * 2^14。
            + data[..., 1] * 9617.0  # G：0.587 * 2^14。
            + data[..., 2] * 4899.0  # R：0.299 * 2^14。三者之和恰为 2^14。
            + 8192.0  # 四舍五入偏置，等价于 CV_DESCALE 的 (1 << (shift-1))。
        ) / 16384.0  # 除以 2^14 即定点右移（2 的幂，float32 下精确）。
        return torch.clamp(value, 0.0, 255.0).to(torch.uint8)  # 截断取整等价于 >> 14（值非负）。

    def _meshgrid(self, height, width):  # 常驻显存的采样网格缓存：同一尺寸只建一次。
        key = (height, width)
        grids = self._grid_cache.get(key)
        if grids is None:
            torch = self.torch
            columns = torch.arange(width, device=self.device, dtype=torch.float32)
            rows = torch.arange(height, device=self.device, dtype=torch.float32)
            row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
            grids = (column_grid, row_grid)  # 与 numpy 门面的 np.meshgrid 一致，返回 (x, y)。
            self._grid_cache[key] = grids
        return grids

    # ------------------------------------------------------------------ 逐元素运算

    def rint(self, array):  # 四舍六入五成双：torch 没有 rint，torch.round 与 np.rint 同为 round-half-to-even。
        return self.torch.round(self.asarray(array))

    def clip(self, array, low, high):
        torch = self.torch
        data = self.asarray(array)
        if not data.is_floating_point():  # 整型数组：边界也要取整，否则 clamp 会把结果提升成 float32，后续当下标就报错。
            return torch.clamp(
                data,
                min=None if low is None else int(low),
                max=None if high is None else int(high),
            )
        low_value = self.asarray(low) if isinstance(low, (np.ndarray, torch.Tensor)) else None if low is None else float(low)
        high_value = self.asarray(high) if isinstance(high, (np.ndarray, torch.Tensor)) else None if high is None else float(high)
        return torch.clamp(data, min=low_value, max=high_value)

    def where(self, condition, left, right):
        torch = self.torch
        left_value = left if isinstance(left, (int, float)) or torch.is_tensor(left) else self.asarray(left)
        right_value = right if isinstance(right, (int, float)) or torch.is_tensor(right) else self.asarray(right)
        return torch.where(self.asarray(condition), left_value, right_value)

    def maximum(self, left, right):
        return self.torch.maximum(self.asarray(left), self.asarray(right))

    def minimum(self, left, right):
        return self.torch.minimum(self.asarray(left), self.asarray(right))

    def abs(self, array):
        return self.torch.abs(self.asarray(array))

    def sqrt(self, array):
        return self.torch.sqrt(self.asarray(array))

    def exp(self, array):
        return self.torch.exp(self.asarray(array))

    def cos(self, array):
        return self.torch.cos(self.asarray(array))

    def sin(self, array):
        return self.torch.sin(self.asarray(array))

    def deg2rad(self, array):
        return self.torch.deg2rad(self.asarray(array))

    def arctan2(self, y, x):
        return self.torch.arctan2(self.asarray(y), self.asarray(x))

    def isfinite(self, array):
        return self.torch.isfinite(self.asarray(array))

    # ------------------------------------------------------------------ 归约与排序

    def sum(self, array, axis=None):
        data = self._as_float(self.asarray(array))
        return data.sum() if axis is None else data.sum(dim=axis)

    def mean(self, array, axis=None):  # 布尔数组先转 float32，与 numpy 门面对齐。
        data = self._as_float(self.asarray(array))
        return data.mean() if axis is None else data.mean(dim=axis)

    def _as_float(self, data):  # bool -> float32，其余原样返回。
        if data.dtype == self.torch.bool:
            return data.to(self.torch.float32)
        return data

    def max(self, array, axis=None):
        data = self.asarray(array)
        return data.max() if axis is None else data.max(dim=axis).values

    def min(self, array, axis=None):
        data = self.asarray(array)
        return data.min() if axis is None else data.min(dim=axis).values

    def argmax(self, array):  # 展平后的最大元下标（与 np.argmax 一致，并列时取第一个）。
        return self.torch.argmax(self.ravel(array))

    def median(self, array, axis=None):  # 与 np.median 一致：偶数长度取中间两数均值。
        torch = self.torch
        data = self._as_float(self.asarray(array))
        if axis is None:
            data = data.reshape(-1)
            axis = 0
        data = data.movedim(axis, -1).contiguous()  # kthvalue 只支持最后一维。
        count = data.shape[-1]
        lower = data.kthvalue(count // 2, dim=-1).values  # 1-based 的第 count//2 个。
        if count % 2 == 1:  # 奇数长度：中位数就是正中间那个。
            return data.kthvalue(count // 2 + 1, dim=-1).values
        upper = data.kthvalue(count // 2 + 1, dim=-1).values
        return (lower + upper) * 0.5

    def sort(self, array, axis=-1):
        return self.torch.sort(self.asarray(array), dim=axis).values

    def argsort(self, array, axis=-1, descending=False):
        # 稳定排序 + 「升序后 flip」的降序：与 NumPy 门面的 kind="stable" + np.flip 同一套配方，
        # 也和改造前内核里的 np.argsort(x)[::-1] 结构一致。不用 torch.argsort(descending=True)：
        # 它在并列值上的取舍与 flip(升序) 不同，会平白多出一种后端间分歧。
        order = self.torch.argsort(self.asarray(array), dim=axis, stable=True)
        if not descending:
            return order
        return self.torch.flip(order, dims=[axis if axis is not None else 0])

    def searchsorted(self, sorted_sequence, values, side="left"):
        return self.torch.searchsorted(
            self.asarray(sorted_sequence).contiguous(),
            self.asarray(values).contiguous(),
            right=(side == "right"),
        )

    def cumsum(self, array, axis=0):
        return self.torch.cumsum(self.asarray(array), dim=axis)

    def linalg_norm(self, array, axis=None):
        data = self._as_float(self.asarray(array))
        if axis is None:
            return self.torch.linalg.vector_norm(data.reshape(-1))
        return self.torch.linalg.vector_norm(data, dim=axis)

    # ------------------------------------------------------------------ 构造

    def linspace(self, start, stop, count, dtype=None):
        return self.torch.linspace(start, stop, count, dtype=dtype or self.torch.float32, device=self.device)

    def arange(self, count, dtype=None):
        return self.torch.arange(count, dtype=dtype or self.torch.float32, device=self.device)

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(shape, dtype=dtype or self.torch.float32, device=self.device)

    def zeros_like(self, array, dtype=None):
        data = self.asarray(array)
        return self.torch.zeros_like(data, dtype=dtype or data.dtype)

    def full(self, shape, value, dtype=None):
        return self.torch.full(tuple(shape), float(value), dtype=dtype or self.torch.float32, device=self.device)

    def full_like(self, array, value):
        return self.torch.full_like(self.asarray(array), float(value))

    def stack(self, arrays, axis=0):
        return self.torch.stack([self.asarray(item) for item in arrays], dim=axis)

    def concatenate(self, arrays, axis=0):
        return self.torch.cat([self.asarray(item) for item in arrays], dim=axis)


def torch_gpu_available() -> bool:
    """探测 torch CUDA 是否可用：已装 torch、驱动可用、至少一块显卡。结果进程内缓存一次。

    首次探测会初始化 CUDA 上下文（约 1~3 秒），因此只在后端装配与 GUI 徽标刷新时调用。
    """

    global _TORCH_GPU_AVAILABLE  # 写探测缓存。
    if _TORCH_GPU_AVAILABLE is None:  # 尚未探测过。
        try:  # 未装 torch、驱动缺失、显存不足都会在导入或初始化时抛错。
            import torch  # 延迟导入：无 torch 环境不应影响服务启动。

            _TORCH_GPU_AVAILABLE = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
        except Exception:  # 任何异常都按不可用处理，调用方回落 CPU。
            _TORCH_GPU_AVAILABLE = False
    return _TORCH_GPU_AVAILABLE


_TORCH_GPU_AVAILABLE: bool | None = None  # 探测缓存，None 表示尚未探测。
_numpy_api: NumpyArrayApi | None = None  # CPU 门面单例。
_torch_api: TorchArrayApi | None = None  # 显卡门面单例。


def numpy_api() -> NumpyArrayApi:
    """取 CPU 门面单例。"""

    global _numpy_api
    if _numpy_api is None:
        _numpy_api = NumpyArrayApi()
    return _numpy_api


def torch_api() -> TorchArrayApi:
    """取显卡门面单例；torch 不可用时抛异常，由调用方决定降级。"""

    global _torch_api
    if _torch_api is None:
        import torch  # 延迟导入。

        _torch_api = TorchArrayApi(torch)
    return _torch_api


def torch_module() -> TorchArrayApi:
    """取显卡门面单例（gpu_shape_backend 装配显卡后端时调用）。

    刻意**不**碰任何 torch 的全局性能开关（``cudnn.benchmark`` / ``allow_tf32``），因为实测下来
    它们对本项目的算子零收益，却带着真实代价：

    - ``cudnn.benchmark=True`` 会为每个**新卷积形状**试算全部 cudnn 算法。测谎链路的形状随
      「可用 lag 数」变化（1/2/3 个 batch），实测每个新形状要卡 1.7~3.1 秒，而稳态耗时
      10.34ms vs 关闭后的 10.38ms——完全在噪声内。一局求解开头就会因此丢掉几十帧。
    - TF32 实测同样无差：稳态 6.95ms vs 6.43ms，与 cv2.calcOpticalFlowFarneback 对照的 EPE
      两者都是 0.0048px。而 ``matmul.allow_tf32`` 是进程全局的，改了会连带影响 YOLO 检测。

    不碰全局开关也意味着本模块对整个进程的其余部分零副作用。
    """

    import torch  # 延迟导入：无 torch 环境下这里抛异常，由调用方回落 CPU。

    torch.cuda.init()  # 显式创建 CUDA 上下文（约 1~3 秒），把这笔开销提到预热阶段而不是首帧。
    return torch_api()
