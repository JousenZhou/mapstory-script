"""稠密光流引擎：torch CUDA 上的 Farneback 等价实现 + 两个 CPU 回退引擎。

解测谎的时序证据链每帧要对多个 lag 各算一次稠密光流，改造前只有 ``cv2.DISOpticalFlow``
一条 CPU 路径。本模块提供三个接口一致（``calc_batch(current, previous) -> (B,H,W,2)``）的引擎：

- :class:`TorchFarnebackFlow`：显卡引擎，按 Farneback 的多项式展开 + 迭代位移求解，
  全部算子都是 torch 的卷积/采样/插值，多个 lag 合成一个 batch 一次算完；
- :class:`Cv2DisFlow`：CPU 引擎，包装 ``cv2.DISOpticalFlow``，打包版与无 torch 环境走它；
- :class:`Cv2FarnebackFlow`：CPU 引擎，包装 ``cv2.calcOpticalFlowFarneback``，
  只供单测做参照与 A/B 诊断，生产链路不用（3 个 lag 逐次调用撑不住 30fps 预算）。

**跨设备算法不一致是已知取舍**：显卡用 Farneback、CPU 用 DIS，两者的残差图在数值上不等价，
只在「运动边界处响应强」这个定性性质上一致。移植正确性由 tests/TestTorchFlow.py 里以
``cv2.calcOpticalFlowFarneback`` 为参照的 EPE 用例保证，对最终判定精度的影响由录像回归门控保证。

与 OpenCV ``FarnebackOpticalFlowImpl`` 的刻意偏离（都只影响极小的数值，不改变算法结构）：

1. **M 的更新方式**：OpenCV 在 ``FarnebackUpdateFlow_Blur`` 里边算 flow 边按条带重算 M
   （Gauss-Seidel，串行），显卡上改成整幅 M 用同一份 flow 算完再求解（Jacobi）。
   实测两者收敛到同一个不动点，差别在小数点后第三位像素量级。
2. **盒式滤波的边界初始化**：OpenCV 用滚动窗口，首行首列有一次 ``(m+2)`` 权重的近似初始化，
   这里直接用 ``F.pad(mode='replicate')`` + 可分离均匀卷积，是数学上干净的版本。
3. **多项式展开的边界外推**：OpenCV 在 warp 落到画面外时退化成「只取第 1 帧的 A、Δb 减半」，
   这里用 ``grid_sample(padding_mode='border')`` 钳制坐标。只影响最外圈 1~2 像素，
   而证据链随后会把 ``max(8, W*0.014)`` 宽的边缘带整体清零。
"""

from __future__ import annotations

import cv2  # CPU 回退引擎与单测参照。
import numpy as np  # 系数表在 CPU 上算好后一次性上传，避免每帧重建。

MIN_LEVEL_SIZE = 32  # OpenCV 的 min_size：金字塔某层任一边小于它就停止继续降采样。
TAPER_BORDER = 5  # FarnebackUpdateMatrices 的 BORDER：最外圈 5 像素对 M 做衰减。
TAPER_VALUES = (0.14, 0.14, 0.4472, 0.4472, 0.4472)  # 衰减系数表，与 OpenCV 逐位一致。
DETERMINANT_EPSILON = 1e-3  # 2x2 求解的正则项，OpenCV 直接加在缩放后的行列式上。

# OpenCV getGaussianKernel 在 sigma<=0 且 n 为奇数、n<=7 时查这张固定表（不是按 sigma 算），
# 金字塔最细层正好落在这个分支：smooth_sz=3、sigma=0 → [0.25, 0.5, 0.25]。
SMALL_GAUSSIAN_TABLE = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
}


def cv_round(value: float) -> int:
    """复刻 OpenCV 的 cvRound：四舍六入五成双（``np.rint`` 与 ``torch.round`` 同为该语义）。"""

    return int(np.rint(float(value)))


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """复刻 ``cv2.getGaussianKernel(size, sigma, CV_32F)``，含 sigma<=0 的两条分支。"""

    if sigma <= 0.0 and size % 2 == 1 and size <= 7:  # 小核固定表分支。
        return np.asarray(SMALL_GAUSSIAN_TABLE[size], dtype=np.float32)
    if sigma <= 0.0:  # 大核按核宽推一个默认 sigma。
        sigma = 0.3 * ((size - 1) * 0.5 - 1.0) + 0.8
    offsets = np.arange(size, dtype=np.float64) - (size - 1) * 0.5  # 以核中心为原点的偏移。
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def prepare_polynomial_kernels(poly_n: int, poly_sigma: float) -> tuple[np.ndarray, tuple[float, ...]]:
    """构造多项式展开所需的 6 张可分离卷积核与 Gram 逆的 4 个标量。

    复刻 ``FarnebackPrepareGaussian`` + ``FarnebackPolyExp``：
    基函数取 ``[1, x, y, x^2, y^2, xy]``，权重取各向同性高斯 ``g[y]*g[x]``。
    由于窗口与权重都与像素位置无关，Gram 矩阵是 6x6 常量，求逆后折进 4 个标量即可
    （OpenCV 利用了它的块结构，只有 ``invG(1,1)/(0,3)/(3,3)/(5,5)`` 四个独立元素）。

    返回 ``(kernels, (ig11, ig03, ig33, ig55))``，``kernels`` 形状 ``(6, 2n+1, 2n+1)``，
    顺序为 ``[b1, b2, b3, b4, b5, b6]``，第一维是 y、第二维是 x：

    - ``b1 = g_y * g_x * f``   （基 1）
    - ``b2 = g_y * xg_x * f``  （基 x）
    - ``b3 = xg_y * g_x * f``  （基 y）
    - ``b4 = g_y * xxg_x * f`` （基 x^2）
    - ``b5 = xxg_y * g_x * f`` （基 y^2）
    - ``b6 = xg_y * xg_x * f`` （基 xy）
    """

    radius = poly_n
    if poly_sigma < 1e-6:  # OpenCV：sigma 太小时按核宽推一个。
        poly_sigma = radius * 0.3
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(offsets * offsets) / (2.0 * poly_sigma * poly_sigma))
    g = g / g.sum()  # FarnebackPrepareGaussian 里显式归一化过。
    xg = offsets * g
    xxg = offsets * offsets * g

    gram = np.zeros((6, 6), dtype=np.float64)
    for y_index in range(2 * radius + 1):  # 双重循环只有 (2n+1)^2 次，n=5 时 121 次，构造期一次性开销。
        for x_index in range(2 * radius + 1):
            x = float(offsets[x_index])
            y = float(offsets[y_index])
            weight = g[y_index] * g[x_index]
            gram[0, 0] += weight
            gram[1, 1] += weight * x * x
            gram[3, 3] += weight * x ** 4
            gram[5, 5] += weight * x * x * y * y
    # 由对称性补齐其余元素（与 OpenCV 的赋值顺序一致）。
    gram[2, 2] = gram[0, 3] = gram[0, 4] = gram[3, 0] = gram[4, 0] = gram[1, 1]
    gram[4, 4] = gram[3, 3]
    gram[3, 4] = gram[4, 3] = gram[5, 5]
    inverse = np.linalg.inv(gram)

    kernels = np.stack(
        [
            np.outer(g, g),  # b1
            np.outer(g, xg),  # b2
            np.outer(xg, g),  # b3
            np.outer(g, xxg),  # b4
            np.outer(xxg, g),  # b5
            np.outer(xg, xg),  # b6
        ]
    ).astype(np.float32)
    return kernels, (
        float(inverse[1, 1]),
        float(inverse[0, 3]),
        float(inverse[3, 3]),
        float(inverse[5, 5]),
    )


def build_taper_1d(length: int) -> np.ndarray:
    """最外圈 5 像素的衰减系数：``FarnebackUpdateMatrices`` 用它压制不可靠的边界响应。"""

    scale = np.ones(length, dtype=np.float32)
    for index in range(length):
        if index < TAPER_BORDER:
            scale[index] *= TAPER_VALUES[index]
        if index >= length - TAPER_BORDER:
            scale[index] *= TAPER_VALUES[length - index - 1]
    return scale


class TorchFarnebackFlow:
    """显卡稠密光流引擎：Farneback 的 torch 张量等价实现。

    ``current`` 传 ``(H,W)``，``previous`` 传 ``(B,H,W)``（B 个历史帧的灰度），
    返回 ``(B,H,W,2)``，``[...,0]=dx``、``[...,1]=dy``，与 cv2 约定一致：
    ``previous[k](p + flow[k](p))`` 就是当前帧在 p 处的背景来源。

    方向约定与改造前一致：调用方传的是 ``calc(gray_current, gray_previous)``，
    等价于 ``cv2.calcOpticalFlowFarneback(prev=gray_current, next=gray_previous)``。
    """

    engine_name = "torch-farneback"

    def __init__(  # 构造引擎。
        self,
        xp,  # torch 数组门面（torch_array.TorchArrayApi），从这里取 torch 模块与设备。
        levels: int = 3,  # 金字塔层数（OpenCV 的 numLevels，实际会算 levels+1 层）。
        iterations: int = 2,  # 每层的迭代求解次数。
        winsize: int = 11,  # M 的盒式平滑窗口边长。
        poly_n: int = 5,  # 多项式展开的窗口半径。
        poly_sigma: float = 1.2,  # 多项式展开的高斯权重 sigma。
        pyr_scale: float = 0.5,  # 相邻金字塔层的尺度比。
    ):
        self.xp = xp
        self.torch = xp.torch  # 裸 torch 模块：卷积/采样/插值这些算子门面没有包装。
        self.levels = int(levels)
        self.iterations = int(iterations)
        self.winsize = int(winsize)
        self.poly_n = int(poly_n)
        self.poly_sigma = float(poly_sigma)
        self.pyr_scale = float(pyr_scale)

        kernels, inverse_gram = prepare_polynomial_kernels(self.poly_n, self.poly_sigma)
        self.ig11, self.ig03, self.ig33, self.ig55 = inverse_gram
        device = xp.device
        self._kernels = self.torch.as_tensor(kernels, device=device)  # (6, 2n+1, 2n+1)
        self._blur_cache: dict[tuple[int, int], object] = {}  # (C, winsize) -> 盒式卷积核
        self._taper_cache: dict[tuple[int, int], object] = {}  # (h, w) -> 衰减掩码
        self._grid_cache: dict[tuple[int, int], tuple] = {}  # (h, w) -> 采样基准网格

    # ------------------------------------------------------------------ 对外接口

    def calc_batch(self, current, previous):
        """一次算完全部 lag 的稠密光流，返回 ``(B,H,W,2)`` 显存张量。"""

        torch = self.torch
        current = self.xp.asarray(current)
        previous = self.xp.asarray(previous)
        if previous.dim() == 2:  # 单个 lag 也允许直接传 (H,W)。
            previous = previous.unsqueeze(0)
        count = previous.shape[0]

        with self.xp.inference():
            height0, width0 = current.shape[-2], current.shape[-1]
            levels = self._resolve_levels(height0, width0)
            # 第 0 张始终是当前帧（多项式展开的「第 1 帧」），后面 count 张是各 lag 的历史帧。
            images = torch.cat([current.unsqueeze(0), previous], dim=0).to(torch.float32)

            flow = None
            for k in range(levels, -1, -1):
                scale = self.pyr_scale ** k
                sigma = (1.0 / scale - 1.0) * 0.5
                smooth_size = max(cv_round(sigma * 5.0) | 1, 3)  # OpenCV：cvRound(sigma*5)|1，再兜底到至少 3。
                width = cv_round(width0 * scale)
                height = cv_round(height0 * scale)

                blurred = self._gaussian_blur(images, smooth_size, sigma)
                level_images = self._resize_images(blurred, height, width)
                expansion = self._polynomial_expansion(level_images)  # (1+B, 5, h, w)
                first = expansion[0:1]  # 当前帧的系数图，全部 lag 共用。
                second = expansion[1:]  # 各历史帧的系数图。

                if flow is None:  # 最粗层从零位移起步。
                    flow = torch.zeros((count, 2, height, width), device=images.device, dtype=torch.float32)
                else:  # 上一层的位移放大到本层尺度（1/pyr_scale）。
                    flow = self._resize_flow(flow, height, width) * (1.0 / self.pyr_scale)

                for _ in range(self.iterations):
                    flow = self._solve_once(first, second, flow, height, width)

            return flow.permute(0, 2, 3, 1).contiguous()  # (B,H,W,2)

    # ------------------------------------------------------------------ 金字塔

    def _resolve_levels(self, height: int, width: int) -> int:
        """复刻 OpenCV 的层数裁剪：某层任一边小于 32 就不再往下建。"""

        scale = 1.0
        levels = 0
        while levels < self.levels:
            scale *= self.pyr_scale
            if width * scale < MIN_LEVEL_SIZE or height * scale < MIN_LEVEL_SIZE:
                break
            levels += 1
        return levels

    def _gaussian_blur(self, images, size: int, sigma: float):
        """按层做全分辨率高斯预平滑（BORDER_REFLECT_101，与 cv2.GaussianBlur 默认一致）。

        ``images`` 是 ``(C,H,W)``：C 张单通道图当批量走，所以卷积只需一个核、groups=1。
        """

        torch = self.torch
        kernel = gaussian_kernel_1d(size, sigma)
        radius = size // 2
        data = images.unsqueeze(1)  # (C,1,H,W)
        padded = torch.nn.functional.pad(data, (0, 0, radius, radius), mode="reflect")  # 只填高度。
        weight_v = self._const_kernel((1, 1, size, 1), kernel.reshape(size, 1))
        blurred = torch.nn.functional.conv2d(padded, weight_v)
        padded = torch.nn.functional.pad(blurred, (radius, radius, 0, 0), mode="reflect")  # 再只填宽度。
        weight_h = self._const_kernel((1, 1, 1, size), kernel.reshape(1, size))
        return torch.nn.functional.conv2d(padded, weight_h).squeeze(1)

    def _const_kernel(self, shape, values: np.ndarray):
        """把 1D 核广播成卷积权重的常驻副本（带缓存，避免每帧重建）。"""

        key = ("kernel", tuple(shape), values.tobytes())  # 用核的原始字节做键，保证不同 sigma/winsize 不撞车。
        cached = self._blur_cache.get(key)
        if cached is None:
            base = self.torch.as_tensor(np.ascontiguousarray(values), device=self.xp.device).to(self.torch.float32)
            # shape 的前两维是「输出通道数 x 输入通道数」，depthwise 时把同一个 1D 核复制到每个通道。
            cached = base.reshape(1, 1, *shape[2:]).expand(shape).contiguous()
            self._blur_cache[key] = cached
        return cached

    def _resize_images(self, images, height: int, width: int):
        """把预平滑后的全分辨率图缩到本层尺寸（INTER_LINEAR == align_corners=False 双线性）。"""

        torch = self.torch
        if images.shape[-2] == height and images.shape[-1] == width:
            return images
        return torch.nn.functional.interpolate(
            images.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)

    def _resize_flow(self, flow, height: int, width: int):
        """把上一层位移场插值到本层尺寸。"""

        torch = self.torch
        if flow.shape[-2] == height and flow.shape[-1] == width:
            return flow.clone()
        return torch.nn.functional.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------ 多项式展开

    def _polynomial_expansion(self, images):
        """把 ``(C,h,w)`` 的灰度层展开成 ``(C,5,h,w)`` 系数图，通道序 ``[c_y, c_x, c_y2, c_x2, c_xy]``。

        6 个投影都是同一张图与不同可分离核的卷积，合成一次 6 输出通道的卷积跑完。
        """

        torch = self.torch
        radius = self.poly_n
        weight = self._kernels.unsqueeze(1)  # (6,1,2n+1,2n+1)：单输入通道 6 个输出通道。
        padded = torch.nn.functional.pad(
            images.unsqueeze(1), (radius, radius, radius, radius), mode="replicate"
        )
        projections = torch.nn.functional.conv2d(padded, weight)  # (C,6,h,w)
        b1, b2, b3, b4, b5, b6 = projections.unbind(1)
        return torch.stack(
            [
                b3 * self.ig11,  # c_y
                b2 * self.ig11,  # c_x
                b1 * self.ig03 + b5 * self.ig33,  # c_y2
                b1 * self.ig03 + b4 * self.ig33,  # c_x2
                b6 * self.ig55,  # c_xy
            ],
            dim=1,
        )

    # ------------------------------------------------------------------ 迭代求解

    def _solve_once(self, first, second, flow, height: int, width: int):
        """一次 Farneback 迭代：按当前位移 warp 第 2 帧系数 → 组 M → 盒式平滑 → 2x2 闭式求解。"""

        torch = self.torch
        warped = self._warp_channels(second, flow)  # (B,5,h,w)
        dx = flow[:, 0:1]
        dy = flow[:, 1:2]

        r2 = warped[:, 0:1]  # 第 2 帧平移后的 c_y。
        r3 = warped[:, 1:2]  # 第 2 帧平移后的 c_x。
        r4 = (first[:, 2:3] + warped[:, 2:3]) * 0.5  # c_y2 平均。
        r5 = (first[:, 3:4] + warped[:, 3:4]) * 0.5  # c_x2 平均。
        r6 = (first[:, 4:5] + warped[:, 4:5]) * 0.25  # c_xy 平均的一半（Hessian 的非对角项）。

        r2 = (first[:, 0:1] - r2) * 0.5  # Δb_y。
        r3 = (first[:, 1:2] - r3) * 0.5  # Δb_x。
        r2 = r2 + r4 * dy + r6 * dx  # 加上当前位移带来的 A·d 修正。
        r3 = r3 + r6 * dy + r5 * dx

        taper = self._taper(height, width)  # (1,1,h,w)
        r2 = r2 * taper
        r3 = r3 * taper
        r4 = r4 * taper
        r5 = r5 * taper
        r6 = r6 * taper

        moments = torch.cat(
            [
                r4 * r4 + r6 * r6,  # G_yy
                (r4 + r5) * r6,  # G_yx
                r5 * r5 + r6 * r6,  # G_xx
                r4 * r2 + r6 * r3,  # h_y
                r6 * r2 + r5 * r3,  # h_x
            ],
            dim=1,
        )
        blurred = self._box_blur(moments)
        g11 = blurred[:, 0:1]
        g12 = blurred[:, 1:2]
        g22 = blurred[:, 2:3]
        h1 = blurred[:, 3:4]
        h2 = blurred[:, 4:5]
        inverse_determinant = 1.0 / (g11 * g22 - g12 * g12 + DETERMINANT_EPSILON)
        return torch.cat(
            [(g11 * h2 - g12 * h1) * inverse_determinant, (g22 * h1 - g12 * h2) * inverse_determinant],
            dim=1,
        )

    def _box_blur(self, moments):
        """对 M 的 5 个通道做 winsize 见方的归一化盒式平滑（边界复制填充）。"""

        torch = self.torch
        radius = self.winsize // 2
        channels = moments.shape[1]
        padded = torch.nn.functional.pad(moments, (0, 0, radius, radius), mode="replicate")  # 只填高度。
        values = np.full(self.winsize, 1.0 / self.winsize, dtype=np.float32)
        weight_v = self._const_kernel((channels, 1, self.winsize, 1), values.reshape(self.winsize, 1))
        vertical = torch.nn.functional.conv2d(padded, weight_v, groups=channels)
        padded = torch.nn.functional.pad(vertical, (radius, radius, 0, 0), mode="replicate")  # 再只填宽度。
        weight_h = self._const_kernel((channels, 1, 1, self.winsize), values.reshape(1, self.winsize))
        return torch.nn.functional.conv2d(padded, weight_h, groups=channels)

    def _taper(self, height: int, width: int):
        """缓存的边界衰减掩码。"""

        key = (height, width)
        cached = self._taper_cache.get(key)
        if cached is None:
            rows = build_taper_1d(height)[:, None]
            columns = build_taper_1d(width)[None, :]
            mask = np.ascontiguousarray(rows * columns)
            cached = self.torch.as_tensor(mask, device=self.xp.device).reshape(1, 1, height, width)
            self._taper_cache[key] = cached
        return cached

    def _warp_channels(self, coefficients, flow):
        """按当前位移场对 ``(B,5,h,w)`` 的系数图做双线性采样。"""

        torch = self.torch
        height, width = coefficients.shape[-2], coefficients.shape[-1]
        base_y, base_x = self._meshgrid(height, width)
        # grid_sample 的 align_corners=False 约定：像素中心 p 对应归一化坐标 (2p+1)/size-1。
        grid_x = (2.0 * (base_x + flow[:, 0:1]) + 1.0) / width - 1.0
        grid_y = (2.0 * (base_y + flow[:, 1:2]) + 1.0) / height - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)  # (B,2,h,w) -> (B,h,w,2)
        return torch.nn.functional.grid_sample(
            coefficients, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    def _meshgrid(self, height: int, width: int):
        """常驻显存的像素坐标网格，同一尺寸只建一次。"""

        key = (height, width)
        grids = self._grid_cache.get(key)
        if grids is None:
            torch = self.torch
            columns = torch.arange(width, device=self.xp.device, dtype=torch.float32)
            rows = torch.arange(height, device=self.xp.device, dtype=torch.float32)
            grids = torch.meshgrid(rows, columns, indexing="ij")
            grids = (grids[0].reshape(1, 1, height, width), grids[1].reshape(1, 1, height, width))
            self._grid_cache[key] = grids
        return grids


class Cv2DisFlow:
    """CPU 稠密光流引擎：包装 ``cv2.DISOpticalFlow``，与改造前的行为逐位一致。

    打包版（``build_exe.spec`` 排除了 torch）与无显卡环境走这条路。
    """

    engine_name = "cv2-dis"

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):  # 构造引擎。
        self.preset = preset
        self.flow = cv2.DISOpticalFlow_create(preset)
        self.flow.setUseSpatialPropagation(True)  # 开启空间传播，提升弱纹理区域稳定性。

    def calc_batch(self, current, previous):
        """逐 lag 调 DIS，返回 ``(B,H,W,2)`` 的 numpy 数组。"""

        current = np.asarray(current)
        previous = np.asarray(previous)
        if previous.ndim == 2:  # 单 lag：(H,W) 补成 (1,H,W)。
            previous = previous[None]
        stacked = [self.flow.calc(current, previous[index], None) for index in range(previous.shape[0])]
        return np.stack(stacked, axis=0)


class Cv2FarnebackFlow:
    """CPU 版 Farneback：只供单测参照与 A/B 诊断，生产链路不用（逐 lag 调用太慢）。"""

    engine_name = "cv2-farneback"

    def __init__(  # 构造引擎。
        self,
        levels: int = 3,
        iterations: int = 2,
        winsize: int = 11,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        pyr_scale: float = 0.5,
    ):
        self.levels = int(levels)
        self.iterations = int(iterations)
        self.winsize = int(winsize)
        self.poly_n = int(poly_n)
        self.poly_sigma = float(poly_sigma)
        self.pyr_scale = float(pyr_scale)

    def calc_batch(self, current, previous):
        """逐 lag 调 cv2.calcOpticalFlowFarneback，返回 ``(B,H,W,2)`` 的 numpy 数组。"""

        current = np.asarray(current)
        previous = np.asarray(previous)
        if previous.ndim == 2:
            previous = previous[None]
        stacked = []
        for index in range(previous.shape[0]):
            flow = np.empty(current.shape[:2] + (2,), dtype=np.float32)
            cv2.calcOpticalFlowFarneback(
                current,
                previous[index],
                flow,
                self.pyr_scale,
                self.levels,
                self.winsize,
                self.iterations,
                self.poly_n,
                self.poly_sigma,
                0,
            )
            stacked.append(flow)
        return np.stack(stacked, axis=0)


def flow_engine_name(is_gpu: bool) -> str:
    """按后端返回光流引擎名，供日志与 UI 显示（与 DenseTemporalAligner 的装配判据同源）。"""

    return TorchFarnebackFlow.engine_name if is_gpu else Cv2DisFlow.engine_name
