# GPU 模板匹配器：用 CuPy FFT 在显卡上批量计算 TM_CCOEFF_NORMED，等价于 cv2.matchTemplate。
# 一帧画面只做一次上传与一次帧 FFT，即可对全部模板批量匹配，适合每帧多次匹配的场景。
# 数学上与 OpenCV 对齐：归一化分子用零均值模板的互相关直接求得，避免大数相减的精度损失。
import cv2  # 导入 OpenCV，用于灰度转换与模板预处理。
import numpy as np  # 导入 NumPy，用于 CPU 侧结果整理。

try:  # CuPy 为可选依赖，未安装或无 NVIDIA 显卡时整个模块自动降级不可用。
    import cupy as cp  # 导入 CuPy，CUDA 上的数组运算与 FFT。
except Exception:  # 导入失败（未安装）。
    cp = None  # 标记 CuPy 不可用。


def gpu_available():  # 探测 GPU 匹配是否可用：CuPy 已安装且至少存在一块可用显卡。
    if cp is None:  # CuPy 未安装。
        return False  # 不可用。
    try:  # 部分机器有 CuPy 但没有 NVIDIA 驱动，初始化会抛异常。
        return bool(cp.cuda.is_available()) and cp.cuda.runtime.getDeviceCount() > 0  # 有可用设备才算可用。
    except Exception:  # 初始化失败。
        return False  # 不可用。


class GpuTemplateMatcher:  # GPU 模板匹配器：预先注册全部模板，每帧批量匹配。

    def __init__(self, gray=True):  # 构造函数，gray=True 表示灰度匹配（更快且对颜色差异更稳定）。
        self.gray = gray  # 记录匹配色彩模式。
        self.templates = {}  # key -> 模板缓存字典，包含零均值翻转核与统计量。

    def add_template(self, key, template_bgr):  # 注册一个模板（BGR 原图），按 key 唯一标识，重复注册会覆盖。
        if self.gray:  # 灰度模式先转灰度。
            t = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)  # 用双精度求统计量避免误差。
        else:  # 彩色模式保留三通道。
            t = template_bgr.astype(np.float64)  # 同样用双精度求统计量。
        t_c = t - t.mean()  # 零均值化模板：corr(I, T-均值) 直接等于归一化分子，避免大数相减。
        k = t_c[::-1, ::-1]  # 空间翻转，卷积即互相关。
        n = float(t.size)  # 模板像素总数（彩色含通道）。
        t_sum = float(t.sum())  # 模板像素总和。
        t_sq = float((t * t).sum())  # 模板像素平方和。
        self.templates[key] = {  # 缓存模板的全部预计算结果。
            "k_g": cp.ascontiguousarray(cp.asarray(k.astype(np.float32))),  # 零均值翻转核上传到显存，只传一次。
            "h": t.shape[0],  # 模板高。
            "w": t.shape[1],  # 模板宽。
            "channels": t.shape[2] if t.ndim == 3 else 1,  # 通道数。
            "var": t_sq - t_sum * t_sum / n,  # 模板方差乘 n：Σ(T-均值)^2，归一化分母的一部分。
            "fk_cache": {},  # fft_shape -> 补零核的 FFT 缓存，画面尺寸不变时只算一次。
        }

    def match_frame(self, frame_bgr):  # 上传一帧画面并返回该帧的批量匹配句柄。
        return _GpuFrameMatch(self, frame_bgr)  # 帧句柄内完成上传与帧 FFT，按需对单个模板计算。


class _GpuFrameMatch:  # 单帧匹配句柄：持有本帧的显存数据，对各个模板按需计算得分图。

    def __init__(self, matcher, frame_bgr):  # 构造函数：灰度转换、上传显存、做帧 FFT。
        if matcher.gray:  # 灰度模式。
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # CPU 上转灰度，耗时可忽略。
        else:  # 彩色模式。
            img = frame_bgr  # 直接用原图。
        self.ig = cp.asarray(np.ascontiguousarray(img)).astype(cp.float32)  # 画面上传显存。
        self.H, self.W = self.ig.shape[:2]  # 画面高宽。
        self.matcher = matcher  # 记录所属匹配器，取模板缓存。
        self.scores = {}  # key -> 得分图缓存，同一模板同一帧只算一次。
        self.window_cache = {}  # (h, w) -> (s, sq) 滑动窗口积分统计缓存。
        max_h = max(td["h"] for td in matcher.templates.values())  # 最大模板高，统一补零尺寸。
        max_w = max(td["w"] for td in matcher.templates.values())  # 最大模板宽。
        self.fft_shape = (self.H + max_h - 1, self.W + max_w - 1)  # 全部模板共用的补零尺寸。
        self.ig_fft = cp.fft.rfft2(self.ig, s=self.fft_shape, axes=(0, 1))  # 帧 FFT 只做一次，全部模板复用。

    def _window_stats(self, h, w):  # 用积分图求 (h, w) 滑动窗口内的像素和与平方和，同尺寸只算一次。
        cached = self.window_cache.get((h, w))  # 先查缓存。
        if cached is not None:  # 已有该尺寸的统计。
            return cached  # 直接复用。
        ig64 = self.ig.astype(cp.float64)  # 双精度积分图避免累积误差。
        pad = cp.zeros((self.H + 1, self.W + 1), dtype=cp.float64)  # 积分图补零边。
        pad[1:, 1:] = cp.cumsum(cp.cumsum(ig64, axis=0), axis=1)  # 二维前缀和。
        s = pad[h:, w:] - pad[:-h, w:] - pad[h:, :-w] + pad[:-h, :-w]  # 窗口像素和。
        pad2 = cp.zeros((self.H + 1, self.W + 1), dtype=cp.float64)  # 平方积分图。
        pad2[1:, 1:] = cp.cumsum(cp.cumsum(ig64 * ig64, axis=0), axis=1)  # 平方二维前缀和。
        sq = pad2[h:, w:] - pad2[:-h, w:] - pad2[h:, :-w] + pad2[:-h, :-w]  # 窗口像素平方和。
        self.window_cache[(h, w)] = (s, sq)  # 写入缓存。
        return s, sq  # 返回统计量。

    def _score_map(self, key):  # 计算指定模板的完整得分图（与 cv2.matchTemplate 同形状）。
        score = self.scores.get(key)  # 先查缓存。
        if score is not None:  # 本帧已算过。
            return score  # 直接复用。
        td = self.matcher.templates[key]  # 取模板缓存。
        h, w = td["h"], td["w"]  # 模板尺寸。
        fk = td["fk_cache"].get(self.fft_shape)  # 查补零核 FFT 缓存。
        if fk is None:  # 画面尺寸变化后才第一次遇到该尺寸。
            k_pad = cp.zeros(self.fft_shape if td["channels"] == 1 else self.fft_shape + (td["channels"],),
                             dtype=cp.float32)  # 按补零尺寸建零矩阵。
            k_pad[:h, :w] = td["k_g"]  # 零均值翻转核放左上角。
            fk = cp.fft.rfft2(k_pad, axes=(0, 1))  # 核 FFT 只算一次，之后每帧复用。
            td["fk_cache"][self.fft_shape] = fk  # 写入缓存。
        cross = cp.fft.irfft2(self.ig_fft * fk, s=self.fft_shape, axes=(0, 1))  # 频域相乘再逆变换得互相关。
        if cross.ndim == 3:  # 彩色模板多通道结果求和。
            cross = cross.sum(axis=2)  # 通道维累加。
        cross = cross[h - 1:self.H, w - 1:self.W]  # 完整卷积中互相关的有效区从 (h-1, w-1) 开始。
        s, sq = self._window_stats(h, w)  # 取该模板尺寸的窗口统计。
        num = cross.astype(cp.float64)  # 零均值模板互相关已直接等于归一化分子。
        den_sq = (sq - s * s / (h * w * td["channels"])) * td["var"]  # 分母平方：窗口方差 × 模板方差。
        score = cp.where(den_sq > 1e-8, num / cp.sqrt(den_sq), cp.float64(0))  # 分母退化（纯色窗口）记 0 分。
        self.scores[key] = score  # 写入本帧缓存。
        return score  # 返回得分图。

    def best(self, key):  # 取指定模板的最高分与位置，返回 (x, y, score)。
        score = self._score_map(key)  # 计算得分图。
        pos = int(cp.argmax(score))  # 展平后的最大值下标。
        width = score.shape[1]  # 得分图宽。
        return pos % width, pos // width, float(score.ravel()[pos])  # 返回 (x, y, 得分)。

    def above(self, key, threshold):  # 取指定模板全部不低于阈值的位置，返回 (N, 3) 数组 [x, y, score]。
        score = self._score_map(key).astype(cp.float32)  # 计算得分图。
        mask = score >= threshold  # 阈值掩码。
        if not bool(mask.any()):  # 没有任何达标位置。
            return np.zeros((0, 3), dtype=np.float32)  # 返回空数组。
        ys, xs = cp.nonzero(mask)  # 达标位置的坐标。
        vals = score[mask]  # 达标位置的分数。
        return cp.stack([xs, ys, vals], axis=-1).get()  # 打包传回 CPU，一次传输。
