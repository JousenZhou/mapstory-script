"""时序残差证据链：唯一一份实现，全部走数组门面，CPU 与 CUDA 共用。

改造前这段逻辑（灰度 → 多 lag 光流对齐 → 绝对差 → median/MAD 鲁棒归一化 → 3x3 高斯 →
边缘带压制）写死在 ``shape_tracking.DenseTemporalAligner`` 里，只能用 cv2+numpy 跑 CPU。
本模块把它抽成 :class:`TensorTemporalAligner`，所有算子都通过注入的数组门面 ``xp`` 执行：

- 有 N 卡：``xp`` 是 torch 门面，帧缓冲常驻显存，每帧只上传一次当前 crop（320×214×3 ≈ 205KB），
  历史帧不再重复搬运，全帧只下载一次 ``raw_mean`` 标量；
- 无 N 卡：``xp`` 是 numpy 门面，光流引擎换成 ``cv2.DISOpticalFlow``，行为与改造前一致。

光流算法本身在 ``torch_flow.py``，本模块只负责调用 ``flow_engine.calc_batch`` 并把结果
加工成证据图，因此「显卡用 Farneback、CPU 用 DIS」这个跨设备差异被隔离在一个注入点里。

与改造前逐位对齐的三个细节：

1. **灰度**交给门面的 ``to_gray``：CPU 直接转 ``cv2.cvtColor(COLOR_BGR2GRAY)``（与改造前逐位一致，
   打包版行为零变化），显卡用 OpenCV 的定点公式 ``(1868B + 9617G + 4899R + 8192) >> 14``。
   两者只在 ``.5`` 灰阶边界上会差 1 个灰阶（实测随机图上约 0.15% 像素），传到证据图上
   最大差 0.25（一个 3x3 高斯量子），Pearson 相关仍为 1.000000。
2. **3x3 高斯**用移位加法实现：``cv2.GaussianBlur(x, (3,3), 0)`` 在 ``sigma<=0`` 且核宽 <=7 时
   查 OpenCV 的 ``small_gaussian_tab`` 固定表，3 宽正好是 ``[0.25, 0.5, 0.25]``
   （不是按 sigma=0.65 算出来的 ``[0.19, 0.62, 0.19]``），两个数都是 2 的幂，乘法精确。
   边界用 BORDER_REFLECT_101，靠 ``concatenate`` 头尾各补一行/列实现。
3. **重映射**交给门面的 ``warp``：CPU 直接转给 ``cv2.remap``，显卡用 ``grid_sample``
   并在整型输入时还原成整型，两条路径对 uint8 彩色帧的输出等价。

清零一律用「乘掩码」而不是切片赋值：显卡上就地写子区域会打断 CUDA Graph 捕获，
而掩码在会话内尺寸固定、可以常驻显存缓存。

**CUDA Graph（仅显卡路径）**：处理尺度只有 320×214，一帧要下发几百个小 kernel
（金字塔层数 × 迭代次数 × 每层 ~25 个算子），单个只有几微秒，launch 开销反而占了大头：
实测 high 档整链 eager 13.3ms → 整图 replay 4.7ms（2.85×），ultra 档 14.8ms → 5.2ms。
会话内区域尺寸固定 → 图内形状静态 → 天然可整图捕获。捕获只需 12ms、占 28MB 显存（且第二张
图只多 0.2MB，显存池复用），因此「每会话捕获」的代价可以忽略。具体约束见 :meth:`_try_capture`。
"""

from __future__ import annotations

from collections import deque  # 定长队列，保存光流对齐所需的历史帧。
from dataclasses import dataclass  # 数据类装饰器，承载算法中间结果。

import numpy as np  # 掩码在 CPU 上算好后一次性上传。


@dataclass
class TemporalEvidence:
    """多 lag 光流对齐后的时序残差证据。"""

    normalized: object  # 鲁棒归一化后的残差图（单通道 float32 后端数组，显卡时留在显存），边界处响应强。
    raw_mean: float  # 各 lag 归一化前残差均值的最小值，用于判断场景切换/静止。


def gaussian3x3(xp, data):
    """3x3 高斯平滑，核 ``[0.25, 0.5, 0.25]`` 可分离、边界 BORDER_REFLECT_101。

    与 ``cv2.GaussianBlur(data, (3, 3), 0)`` 等价，但只用数组门面的算子，因此两个设备都能跑
    且结果逐位一致（cv2 只能跑 CPU，会把显卡链路的中间结果拽回主存）。
    """

    for axis in (0, 1):  # 先列后行；可分离核两次一维卷积即可。
        size = int(data.shape[axis])
        padded = xp.concatenate(  # 头尾各补一行/列，用第 2 个和倒数第 2 个元素，即 REFLECT_101。
            [xp.narrow(data, axis, 1, 1), data, xp.narrow(data, axis, size - 2, 1)],
            axis=axis,
        )
        data = (
            xp.narrow(padded, axis, 0, size) * 0.25  # 上一行/左一列。
            + xp.narrow(padded, axis, 1, size) * 0.5  # 本行/本列。
            + xp.narrow(padded, axis, 2, size) * 0.25  # 下一行/右一列。
        )
    return data


class TensorTemporalAligner:
    """用稠密光流把历史帧对齐到当前帧，生成鲁棒的时序残差证据图（双后端）。

    契约与改造前的 ``DenseTemporalAligner`` 完全一致：``reset()`` 清空历史、
    ``update(frame)`` 喂一帧 BGR uint8 返回 :class:`TemporalEvidence` 或 None。
    """

    def __init__(  # 构造对齐器。
        self,
        lags,  # 使用的时间基线（帧间隔），例如 (1, 2, 4)。
        play_height_ratio: float,  # 有效高度比例，其下部分在证据图里清零。
        flow_engine,  # 稠密光流引擎，需提供 ``calc_batch(current, previous) -> (B,H,W,2)``。
        backend,  # 打分后端（gpu_shape_backend.ShapeScoreBackend），从它取数组门面。
        fallback=None,  # 零参回调：显卡运行期异常时用它建一个 CPU 引擎的对齐器，None 表示不降级直接抛。
    ):
        self.lags = tuple(lags)  # 保存时间基线。
        self.play_height_ratio = float(play_height_ratio)  # 保存有效高度比例。
        self.flow_engine = flow_engine  # 光流引擎：显卡 Farneback 或 CPU DIS。
        self.backend = backend  # 打分后端。
        self.xp = backend.xp  # 数组门面。
        self._fallback = fallback  # 降级构造器。
        self._degraded = None  # 降级后的 CPU 对齐器；一旦置位，后续帧全部转给它。
        self._history = deque(maxlen=max(self.lags) + 1)  # 定长历史队列，存 (彩色帧, 灰度帧)，都在后端设备上。
        self._mask_cache: dict[tuple[int, int, int], tuple] = {}  # (H, W, play_height) -> (有效带掩码, 边缘带掩码)
        # ---- CUDA Graph 状态（只在显卡门面下启用，见 _try_capture）----
        self._graph = None  # 捕获好的整图；None 表示当前走 eager。
        self._graph_evidence = None  # 图的静态输出：证据图（下一帧 replay 前有效）。
        self._graph_raw_mean = None  # 图的静态输出：残差均值（0 维）。
        self._graph_disabled = False  # 捕获失败一次就永久回退 eager，不再重试。
        self._static_current_color = None  # (H,W,3) 当前帧彩色的固定地址缓冲。
        self._static_current_gray = None  # (H,W) 当前帧灰度的固定地址缓冲。
        self._static_history_color = None  # (B,H,W,3) 各 lag 历史彩色的固定地址缓冲。
        self._static_history_gray = None  # (B,H,W) 各 lag 历史灰度的固定地址缓冲。

    @property
    def history(self):
        """历史帧队列（降级后指向 CPU 对齐器的那一份）。"""

        return self._degraded.history if self._degraded is not None else self._history

    @property
    def engine_name(self) -> str:
        """当前光流引擎名（``torch-farneback`` / ``cv2-dis`` / ``cv2-farneback``），供日志与测试断言。"""

        engine = self._degraded.flow_engine if self._degraded is not None else self.flow_engine
        return getattr(engine, "engine_name", "")

    def reset(self) -> None:
        """清空历史帧队列；跨局/区域位移后必须调用，否则残差会被上一段画面污染。"""

        self._history.clear()  # 丢弃全部历史帧（显卡上即释放对应的显存引用）。
        self._release_graph()  # 区域尺寸可能变了：静态缓冲与整图一并丢弃，下一段进稳态时重新捕获。
        if self._degraded is not None:  # 降级后的 CPU 对齐器也要同步清。
            self._degraded.reset()

    def update(self, frame: np.ndarray) -> TemporalEvidence | None:
        """喂入一帧，返回多 lag 合并后的时序残差证据；历史不足时返回 None。

        显卡运行期异常（显存不足/驱动重置）不能中断求解：有 ``fallback`` 时当场建成 CPU
        对齐器并把后续帧全部转给它（与打分内核的永久降级语义一致）。降级会丢掉显存里的
        历史帧，因此紧接的 ``max(lags)`` 帧返回 None，会话侧退化为 prediction 而非报错。
        """

        if self._degraded is not None:  # 已永久降级。
            return self._degraded.update(frame)
        try:
            return self._compute(frame)
        except Exception:  # 显卡链路异常。
            if self._fallback is None:  # 没配降级器（CPU 路径或单测），照旧报错。
                raise
            self._degraded = self._fallback()  # 建一个 CPU 引擎的对齐器，本帧起由它接管。
            return self._degraded.update(frame)

    def _compute(self, frame: np.ndarray) -> TemporalEvidence | None:
        """上传当前帧 → 走整图 replay（已捕获）或 eager 链 → 下载残差均值。
    
        刚启动的几帧 lag 还不齐、形状每帧在变，不能捕获；直到第一个「全部 lag 都可用」
        的稳态帧才尝试捕获一次，本帧仍用 eager 结果（避免白算一遂），下一帧起走 replay。
        """
    
        xp = self.xp
        with xp.inference():  # 显卡侧关掉自动求导，省掉建图开销。
            color = xp.clone(xp.asarray(frame))  # 上传当前 crop（显卡）或直接拷贝（CPU），本帧唯一一次 H2D。
            gray = xp.to_gray(color)  # 灰度：CPU 走 cv2（逐位一致），显卡走定点公式。
            self._history.append((color, gray))  # 入队。
            available = [lag for lag in self.lags if len(self._history) > lag]  # 筛出历史长度已经足够的 lag。
            if not available:  # 一个 lag 都算不了（刚启动）。
                return None  # 返回空证据。
    
            height = int(color.shape[0])  # 帧高。
            width = int(color.shape[1])  # 帧宽。
            play_height = int(height * self.play_height_ratio)  # 有效高度。
            play_mask, band_mask = self._masks(height, width, play_height)  # 取（或建）两个常驻掩码。
            steady = len(available) == len(self.lags)  # 稳态：全部 lag 都可用，从此图内形状不再变。
    
            if self._graph is not None and steady:  # 已捕获且处于稳态：走整图 replay。
                return self._replay_graph(color, gray)
    
            combined, raw_mean = self._chain(  # eager 链：与图内跑的是同一份代码。
                gray,
                [self._history[-1 - lag][1] for lag in available],
                color,
                [self._history[-1 - lag][0] for lag in available],
                play_mask,
                band_mask,
                play_height,
            )
            if steady:  # 刚进稳态：把这条链整图捕获，下一帧起生效。
                self._try_capture(height, width, play_height, play_mask, band_mask)
            return TemporalEvidence(combined, float(xp.to_numpy(raw_mean)))  # 本帧唯一一次 D2H。
    
    def _chain(self, current_gray, history_gray, current_color, history_color, play_mask, band_mask, play_height):
        """证据链本体：多 lag 合批光流 → 重建背景作差 → 鲁棒归一化 → 合并平滑。
    
        eager 与 CUDA Graph **共用这一份代码**，两条路径因此逐位一致。入参约定：
        ``history_gray`` / ``history_color`` 是「按 lag 顺序排列的单帧数组序列」（灰度 ``(H,W)``、
        彩色 ``(H,W,3)``）；eager 时它们来自历史队列，捕获后来自静态缓冲的固定视图，形状完全相同。
    
        返回 ``(证据图, 残差均值张量)``；均值故意留成 0 维张量而不提前下载，这样它也能被捕获进图。
        """
    
        xp = self.xp
        # 全部 lag 合成一个 batch，一次算完光流；历史帧的灰度已经在设备上，无需再搬运。
        flows = self.flow_engine.calc_batch(current_gray, xp.stack(history_gray, axis=0))  # (B,H,W,2)。
        current = xp.astype(current_color, xp.float32)  # 残差在 float32 下算，与 cv2.absdiff 对 uint8 的结果一致。
        normalized_maps = []  # 收集每个 lag 的归一化残差图。
        raw_means = []  # 收集每个 lag 归一化前的残差均值（留成 0 维张量，最后一次性归约）。
    
        for index in range(len(history_color)):
            # 按光流重建「当前帧的背景」，再与当前帧作差：
            # 静止背景被抵消，真实运动的边界留下残差。
            aligned = xp.astype(xp.warp(history_color[index], xp.select(flows, 0, index)), xp.float32)
            delta = xp.max(xp.abs(current - aligned), axis=2)  # 逐像素绝对差取三通道最大值。
            valid_delta = xp.narrow(delta, 0, 0, play_height)  # 只统计有效带内的残差。
            median = xp.astype(xp.median(valid_delta), xp.float32)  # 残差中位数，作为鲁棒基线。
            mad = xp.astype(xp.median(xp.abs(valid_delta - median)), xp.float32)  # 中位数绝对偏差。
            # 鲁棒标准差，下限 1.0 防止过度放大（用 clip 而不是 maximum，两个门面都保留 float32）。
            sigma = xp.clip(mad * 1.4826, 1.0, None)
            normalized = xp.clip((delta - median) / sigma, 0.0, 12.0) * play_mask  # 鲁棒归一化 + 截断 + 有效带以下清零。
            normalized_maps.append(normalized)
            raw_means.append(xp.mean(valid_delta))
    
        # 真实运动的边界只需要在某一个时间基线上清楚即可，因此多 lag 取最大。
        combined = xp.max(xp.stack(normalized_maps, axis=0), axis=0)  # 逐像素取各 lag 的最大响应。
        combined = gaussian3x3(xp, combined) * band_mask  # 轻度平滑抑制单像素噪声，再压制边缘带。
        # 残差均值取最小，避免单一 lag 的抖动误判场景切换。
        return combined, xp.min(xp.stack(raw_means, axis=0), axis=0)
    
    # ------------------------------------------------------------------ CUDA Graph
    
    def _try_capture(self, height: int, width: int, play_height: int, play_mask, band_mask) -> None:
        """把稳态证据链整图捕获成 CUDA Graph（只在显卡门面下生效）。
    
        捕获的硬性前提，任何一条不满足就置 ``_graph_disabled`` 并**永久回退 eager**：
    
        1. 门面必须是 torch CUDA（CPU 门面没有 launch 开销问题，也没有 ``cuda.CUDAGraph``）；
        2. 已进入稳态（全部 lag 可用、历史队列已满），图内形状从此不再变；调用方保证；
        3. 图内只能读**固定地址**：因此先建四个静态缓冲，每帧在图外把历史队列的内容
           ``copy_into`` 进去；``select`` 取出的单帧视图与缓冲共享存储，地址同样固定；
        4. 捕获期内不允许再出现 pageable H2D 或新缓存分配：先在**侧流**上跑 3 次把光流引擎的
           核缓存/衰减掩码/采样网格全填满，再 ``synchronize`` 后才开始捕获；
        5. 掩码已常驻设备（``_masks`` 在调用前已执行）。
    
        捕获失败**不**降级到 CPU：这只是省不掉 launch 开销，算子本身在显卡上仍是好的，
        因此与 ``update()`` 里「显卡运行期异常 → 转 CPU 对齐器」的语义分开处理。
        """
    
        if self._graph_disabled:  # 已经失败过，不再试。
            return
        torch = getattr(self.xp, "torch", None)  # CPU 门面没有这个属性。
        if torch is None or not self.xp.is_gpu:  # 不是显卡门面：永久走 eager。
            self._graph_disabled = True
            return
        xp = self.xp
        count = len(self.lags)  # 稳态下的 lag 个数，图内 batch 维固定为它。
        try:
            self._static_current_color = xp.zeros((height, width, 3), dtype=xp.uint8)
            self._static_current_gray = xp.zeros((height, width), dtype=xp.uint8)
            self._static_history_color = xp.zeros((count, height, width, 3), dtype=xp.uint8)
            self._static_history_gray = xp.zeros((count, height, width), dtype=xp.uint8)
            # 单帧视图只建一次：图内读的必须是这些固定地址，与缓冲共享存储。
            history_color = [xp.select(self._static_history_color, 0, index) for index in range(count)]
            history_gray = [xp.select(self._static_history_gray, 0, index) for index in range(count)]
    
            def body():  # 图体：与 eager 完全同一份 _chain，只是输入换成静态缓冲。
                return self._chain(
                    self._static_current_gray,
                    history_gray,
                    self._static_current_color,
                    history_color,
                    play_mask,
                    band_mask,
                    play_height,
                )
    
            self._fill_static(*self._history[-1])  # 先用当前真实帧填满，预热与捕获读到的都是有效数据。
            stream = torch.cuda.Stream()  # 侧流预热：torch 的捕获要求先在非默认流上跑几次。
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()  # 捕获前必须无待完成的显卡工作。
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):  # 捕获不执行，输出张量要到首次 replay 才有值。
                evidence, raw_mean = body()
            torch.cuda.synchronize()
            self._graph = graph
            self._graph_evidence = evidence
            self._graph_raw_mean = raw_mean
        except Exception:  # 显存不足/驱动不支持捕获/图内含同步点，一律永久回退 eager。
            self._release_graph()
            self._graph_disabled = True
    
    def _fill_static(self, color, gray) -> None:
        """把当前帧与各 lag 历史帧搬进固定地址的静态缓冲（图外执行，全是 D2D）。"""
    
        xp = self.xp
        xp.copy_into(self._static_current_color, color)
        xp.copy_into(self._static_current_gray, gray)
        for slot, lag in enumerate(self.lags):
            previous_color, previous_gray = self._history[-1 - lag]
            xp.copy_into(xp.narrow(self._static_history_color, 0, slot, 1), previous_color)
            xp.copy_into(xp.narrow(self._static_history_gray, 0, slot, 1), previous_gray)
    
    def _replay_graph(self, color, gray) -> TemporalEvidence:
        """刷新静态缓冲 → replay 整图 → 下载残差均值（本帧唯一一次 D2H）。
    
        返回的 ``normalized`` 是图的**静态输出缓冲**，只在下一帧 replay 之前有效；
        会话侧当帧就把它喂给 ``observe_border`` 消费掉了，不会跳帧持有。
        """
    
        self._fill_static(color, gray)
        self._graph.replay()
        return TemporalEvidence(self._graph_evidence, float(self.xp.to_numpy(self._graph_raw_mean)))
    
    def _release_graph(self) -> None:
        """丢弃已捕获的整图与静态缓冲（显存随即归还给 torch 的缓存分配器）。"""
    
        self._graph = None
        self._graph_evidence = None
        self._graph_raw_mean = None
        self._static_current_color = None
        self._static_current_gray = None
        self._static_history_color = None
        self._static_history_gray = None

    # ------------------------------------------------------------------ 内部工具

    def _masks(self, height: int, width: int, play_height: int) -> tuple:
        """构造并缓存两个清零掩码：有效带掩码（每 lag 用）与边缘带掩码（合并后用）。

        掩码只依赖 ``(H, W, play_height)``，会话内固定不变，因此建一次就常驻设备。
        """

        key = (height, width, play_height)
        cached = self._mask_cache.get(key)
        if cached is not None:  # 已建过。
            return cached
        rows = np.arange(height, dtype=np.int32)[:, None]  # 行坐标。
        columns = np.arange(width, dtype=np.int32)[None, :]  # 列坐标。
        play = np.zeros((height, width), dtype=np.float32)  # 有效带：play_height 以下为 0。
        if play_height > 0:
            play[:play_height] = 1.0
        # remap 的边界填充会产生又长又笔直的假边界。只压制这条窄的不可靠带：
        # 贴到画面边缘的目标仍能靠它剩下的两三条边被找回来。
        edge_band = max(8, int(round(width * 0.014)))  # 边缘带宽度，随画面宽度自适应。
        band = (
            (rows >= edge_band)  # 上边缘带以外。
            & (rows < max(0, play_height - edge_band))  # 有效带下边缘以内。
            & (columns >= edge_band)  # 左边缘带以外。
            & (columns < max(0, width - edge_band))  # 右边缘带以内。
        ).astype(np.float32)
        cached = (self.xp.asarray(play), self.xp.asarray(band))  # 上传并常驻。
        self._mask_cache[key] = cached
        return cached
