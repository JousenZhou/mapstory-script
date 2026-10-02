"""测谎检验「找目标」在线编排：把算法内核组织成一局录像的状态机，并附带离线回看修正。

本模块对应外部验证脚本 track_transparent_shape.py 的 run() 主循环，
剔除了视频读写、CSV 输出与命令行参数，改成由 GUI 页签逐帧喂入裁剪区域。

两遍流程：
- Pass 1（在线）：ShapeTrackSession.update 逐帧返回 ShapeFrameResult，实时预览。
- Pass 2（离线）：repair_shape_gaps 对短时低置信区间做双向回看插值，标记 interpolated。

外部 README 明确指出回看插值属于离线手段（会引入最多 max_gap_seconds 的回看延迟），
因此它只在 Pass 2 里使用，不参与 Pass 1 的实时判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field  # 数据类：参数集中与单帧输出。
from typing import Callable, Optional  # 类型标注：可选的日志回调。

import cv2  # OpenCV：DIS 光流档位常量（CPU 引擎用）。
import numpy as np  # 数值计算：随机数发生器与坐标数组。

from src.liedetector.gpu_shape_backend import numpy_backend, require_gpu_backend  # 打分后端探测：有 N 卡走 torch CUDA，没有/关闭则 NumPy。
from src.liedetector.shape_tracking import (  # 算法内核，纯计算无 I/O。
    DenseTemporalAligner,  # 多 lag 稠密光流时序残差对齐器（按后端装配的薄工厂）。
    ParticleShapeTracker,  # 6 维状态粒子跟踪器。
    ShapeTemplate,  # 学习到的形状模板。
    build_shape_template,  # 从白色轮廓构建模板。
    choose_shape_candidate,  # 从白色候选里挑出被跟踪目标。
    choose_template_detection,  # 选窗窗口内挑最稳定的一条检测学模板（最佳帧学习）。
    detect_white_shapes,  # 白色目标检测。
    resize_for_processing,  # 缩放到处理尺度。
    transformed_contour,  # 按当前姿态还原轮廓点集。
)
from src.liedetector.torch_flow import flow_engine_name  # 光流引擎名（日志用，与装配判据同源）。

# source 字段的全部合法取值，与外部脚本保持一致，便于回归对照。
SOURCE_WAITING = "waiting"  # 还没学到模板，本帧没有任何输出。
SOURCE_COLOR = "color"  # 白色轮廓可见，走强测量。
SOURCE_BORDER = "border"  # 目标已透明，靠时序残差证据可靠定位。
SOURCE_PREDICTION = "prediction"  # 证据不足，仅靠粒子运动预测外推。
SOURCE_SCENE_ENDED = "scene-ended"  # 场景结束（切场景或出现结算文字）。
SOURCE_INTERPOLATED = "interpolated"  # Pass 2 回看插值回填的帧。

# 精度档预设：把「低/中等/高/极高/最强」映射到一组跟踪参数。GPU 只决定打分走 torch CUDA 还是 numpy，
# 精度档由用户显式选择；无 N 卡（或运行期降级）时重载档会被 _resolve_scoring_backend 门控回落 medium。
# 光流参数分两套：dis_preset 给 CPU 引擎（cv2 DIS），flow_* 给显卡引擎（torch Farneback）。
# poly_n/poly_sigma/pyr_scale 全档统一（= dataclass 默认值），故不入档。
#
# 分档思路（纯 GPU CUDA 路径，CUDA Graph 已把证据链的 kernel launch 开销摊平）：
# - 图形重叠遮挡时，四个杠杆按收益排序是：处理分辨率 > 对照组 > 重定位漏斗宽度 > 粒子数。
#   分辨率把粘连成一个连通域的相邻图形真正分开；对照组决定背景得分分布会不会被邻近图形污染；
#   漏斗（coarse_top × refine_per_top）决定目标被完全遮住后能否在全局重定位里被重新捞回；
#   粒子数只是让状态空间覆盖更密。所以最高档同时抬分辨率与漏斗，而不是只堆粒子。
# - 实测打分 batch（= 粒子 + 3×对照）从 1200 涨到 1960 时 observe 耗时不升反降，说明打分在显卡上
#   由每帧固定开销主导，加大粒子/对照近乎免费；真正随粒子数线性增长的是 propagate/estimate 里的
#   NumPy 随机数与逐列运算（CPU 侧），故最高档把粒子压在 1400 以内以守住 30fps（33ms）预算。
PRECISION_TIER_KEYS = ("low", "medium", "high", "ultra", "extreme")  # 合法精度档 key，顺序即由低到高。
PRECISION_TIER_DEFAULT = "extreme"  # 默认精度档：GPU 机直接生效（最强），CPU 机运行时回落 medium。
PRECISION_TIER_GPU_ONLY = ("high", "ultra", "extreme")  # 仅显卡可选的重载档：算法层与 UI 下拉共用这一份名单，避免两侧漂移。
PRECISION_TIER_CPU_FALLBACK = "medium"  # CPU 机（或运行期永久降级）时重载档的回落目标。
PRECISION_TIERS = {  # 各档参数：dataclass 默认值即 medium 档，故 medium 应用时为空操作。
    "low": {  # 低：弱机保底，降分辨率到 256 并压缩粒子/对照/粗扫，最省算力。
        "dis_preset": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "temporal_lags": (1, 2),
        "particle_count": 220,
        "control_count": 110,
        "global_proposals": 400,
        "coarse_top": 30,
        "refine_per_top": 8,
        "max_process_side": 256.0,
        "flow_levels": 2,  # 显卡光流：金字塔只建 2 层，最省显存与 kernel 启动。
        "flow_iterations": 2,
        "flow_winsize": 9,
    },
    "medium": {  # 中等：CPU 实时档（30fps 逐帧处理基线），也是 dataclass 默认值；重载档在 CPU 机上回落到此。
        "dis_preset": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "temporal_lags": (1, 2),
        "particle_count": 320,
        "control_count": 140,  # 对照组要按 3 个尺度重复计入 batch，是 CPU 上最贵的一项，故本档只温和上调。
        "global_proposals": 600,
        "coarse_top": 30,
        "refine_per_top": 8,
        "max_process_side": 320.0,
        "flow_levels": 3,  # 显卡光流：与 dataclass 默认值一致，应用本档为空操作。
        "flow_iterations": 2,
        "flow_winsize": 11,
    },
    "high": {  # 高：显卡常规档，光流 MEDIUM 级参数 + 第三个 lag，并首次加宽重定位漏斗。
        "dis_preset": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "temporal_lags": (1, 2, 4),
        "particle_count": 600,
        "control_count": 300,
        "global_proposals": 1500,
        "coarse_top": 36,  # 粗扫进精扫的邻域数：遮挡场景下放宽，免得真目标在粗排就被相似邻居挤掉。
        "refine_per_top": 10,
        "max_process_side": 320.0,
        "flow_levels": 3,  # 显卡光流：= cv2 Farneback 默认档（iterations=3, winsize=15）。
        "flow_iterations": 3,
        "flow_winsize": 15,
    },
    "ultra": {  # 极高：显卡重载档，在高基础上大幅加粒子/对照/粗扫与光流金字塔层数，仍不升分辨率。
        "dis_preset": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "temporal_lags": (1, 2, 4),
        "particle_count": 1000,
        "control_count": 500,
        "global_proposals": 2400,
        "coarse_top": 44,
        "refine_per_top": 10,
        "max_process_side": 320.0,
        "flow_levels": 4,  # 显卡光流：多一层金字塔 + 四次迭代，大位移与弱纹理更稳。
        "flow_iterations": 4,
        "flow_winsize": 15,
    },
    "extreme": {  # 最强：唯一升分辨率的档，专治图形重叠遮挡；粒子/对照/漏斗全部拉满。
        # dis_preset 只服务 CPU 引擎，而本档在 CPU 机上会被门控回落 medium，因此它实际不生效；
        # 仍填 cv2 最高可用预设 MEDIUM（本 build 只有 ULTRAFAST/FAST/MEDIUM，没有 ULTRA），
        # 保证万一被直接拿去跑 CPU 引擎也是最高精度。
        "dis_preset": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "temporal_lags": (1, 2, 4),
        "particle_count": 1400,  # 再往上加就顶到 propagate/estimate 的 NumPy 侧线性开销，收益不抵 30fps 预算。
        "control_count": 720,
        "global_proposals": 3600,
        "coarse_top": 56,  # 最宽漏斗：目标被完全遮住后，重定位要能在多个相似邻居中间把它捞回来。
        "refine_per_top": 12,
        # 分辨率是分离粘连图形最根本的杠杆，但必须连 process_scale 一起抬：
        # effective_scale = min(process_scale, max_process_side / 区域长边)，
        # 实测测谎区域是 728x486，只把 max_process_side 提到 384 会被 process_scale=0.5 卡死在 364，
        # 等于白改一档；0.55 才能让 384 真正生效（处理尺度 384x256，面积是 320x214 的 1.44 倍）。
        "process_scale": 0.55,
        "max_process_side": 384.0,
        "flow_levels": 5,  # 显卡光流：384 尺度下多建一层金字塔（最粗 24px），大位移更稳。
        "flow_iterations": 4,
        "flow_winsize": 15,
    },
}


@dataclass
class ShapeTrackParams:
    """一局跟踪的全部参数。默认值 = 「中等」精度档（CPU 也能跑满 30fps 的保守基线）。

    本项目不新增任何任务配置项，参数一律以模块常量形式集中在这里。
    精度由 precision_tier 显式选择（低/中等/高/极高/最强，见 PRECISION_TIERS）：GPU 只决定打分与光流
    走 torch CUDA 还是 numpy/cv2，精度档由用户选定；无 N 卡时 high/ultra/extreme 会被门控回落到 medium。
    dataclass 默认值即 medium 档，_apply_precision_tier 只覆盖仍停在默认值的字段，尊重调用方显式自定义。
    """

    process_scale: float = 0.5  # 处理尺度上限：先把裁剪区域缩小再算，坐标输出时换算回全尺度（extreme 档升到 0.55 才能真正吃到 384 分辨率）。
    max_process_side: float = 320.0  # 处理分辨率上限（最长边像素）：大区域自动降低实际 scale；320 档实测三个参考视频跟踪结果与 400 档完全一致，但图形重叠粘连时更高分辨率才能把连通域分开。
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST  # CPU 光流引擎的 DIS 精度档位（medium 档用 FAST；high 及以上由档位升到 MEDIUM）。
    temporal_lags: tuple[int, ...] = (1, 2)  # 时序残差使用的时间基线（medium 档 2 个；high 及以上升到 (1,2,4)）。
    play_height_ratio: float = 1.0  # 有效高度比例。偏离脚本的 0.89：页签处理的是裁出的图形区域，内部没有 UI 需要排除。
    particle_count: int = 320  # 粒子数量（medium 档）。
    global_proposals: int = 600  # 重定位第一级粗扫候选数（两级粗到细，替代旧版 2400 一次性打分）。
    control_count: int = 140  # 对照组数量：估计背景得分分布的中位数/MAD；图形重叠时随机位姿常压在别的图形上，对照组越大背景基线越不被污染。
    coarse_top: int = 30  # 粗扫后进入精扫的 top-K 邻域数（medium 档；high 及以上加宽漏斗以扛住完全遮挡后的重定位）。
    refine_per_top: int = 8  # 每个粗扫邻域生成的精扫拖尾候选数（medium 档）。
    relocation_cooldown_frames: int = 15  # 两次全局重定位之间的最小间隔帧数（限频，约 0.5 秒@30fps）。
    confidence_threshold: float = 0.75  # Pass 2 回看的低置信阈值，低于它才考虑插值。
    max_gap_seconds: float = 1.25  # 允许回看修正的最长低置信区间（秒）。
    max_correction_ratio: float = 0.24  # 单帧证据修正上限（相对形状尺寸的比例），超限整体回滚。
    max_speed_ratio: float = 0.16  # 单帧中心移动速度上限（相对形状尺寸的比例）。
    template_learn_window: float = 0.4  # 开局最佳帧选窗时长（秒）：先攒一个窗口的检测再挑最干净的一条学模板，避免首帧被倒计时粘连/削顶污染。
    min_size_floor: float = 36.0  # 小目标阈值下限（处理尺度像素）：距离窗/限速/重定位半径/抖动/似然宽度按 max(标称尺寸, 此值) 缩放，
    # 否则小面积星星（nominal≈29）把所有窗口等比缩窄，目标稍一快跳就丢——大目标（≥此值）行为完全不变。
    random_seed: int = 20260902  # 随机种子，保证同一局录像结果可复现。
    gpu_scoring: bool = True  # 显卡开关：True 时探测 torch CUDA，可用则打分与光流都跑显卡，否则自动回落 NumPy + cv2 DIS（双编译结构）。仅决定后端，不改精度档。
    precision_tier: str = PRECISION_TIER_DEFAULT  # 精度档 key（low/medium/high/ultra/extreme）：由 GUI 精度下拉选定，无 N 卡时重载档运行时回落 medium。
    # 显卡光流（torch Farneback）参数；CPU 引擎用的是上面的 dis_preset，不看这几个字段。
    flow_levels: int = 3  # 金字塔层数（medium 档）。
    flow_iterations: int = 2  # 每层的迭代求解次数（medium 档）。
    flow_winsize: int = 11  # 位移场盒式平滑窗口边长（medium 档）。
    flow_poly_n: int = 5  # 多项式展开的窗口半径，全档统一。
    flow_poly_sigma: float = 1.2  # 多项式展开的高斯权重 sigma，全档统一。
    flow_pyr_scale: float = 0.5  # 相邻金字塔层的尺度比，全档统一。

    @property
    def max_gap_frames(self) -> int:
        """把回看窗口换算成帧数，至少 1 帧。"""

        return max(1, int(round(self.max_gap_seconds * 30)))  # 页签固定 30fps 节拍，直接按 30 换算。

    @property
    def gpu_flow_params(self) -> dict:
        """显卡 Farneback 引擎的参数字典，直接展开给 ``TorchFarnebackFlow``。"""

        return {  # 键名与 TorchFarnebackFlow 的构造参数一一对应。
            "levels": self.flow_levels,
            "iterations": self.flow_iterations,
            "winsize": self.flow_winsize,
            "poly_n": self.flow_poly_n,
            "poly_sigma": self.flow_poly_sigma,
            "pyr_scale": self.flow_pyr_scale,
        }


@dataclass
class ShapeFrameResult:
    """单帧跟踪输出。坐标一律已换算回裁剪区域的全尺度（除以 process_scale）。"""

    source: str = SOURCE_WAITING  # 本帧结论来源，取值见上面的 SOURCE_* 常量。
    center: Optional[np.ndarray] = None  # 目标中心 (2,)，区域全尺度坐标；未学到模板时为 None。
    velocity: Optional[np.ndarray] = None  # 目标速度 (2,)，区域全尺度像素/帧。
    scale: float = 1.0  # 目标相对模板的缩放估计。
    angle_deg: float = 0.0  # 目标角度（度），已折回对称周期域。
    confidence: float = 0.0  # 跟踪置信度，0~1。
    border_score: float = 0.0  # 上一次边界证据加权得分，诊断用。
    border_snr: float = 0.0  # 上一次边界证据信噪比，诊断用。
    search_radius: float = 0.0  # 当前搜索半径，区域全尺度，诊断用。
    flow_residual: Optional[float] = None  # 时序残差均值；历史不足（无光流）时为 None。
    contour: Optional[np.ndarray] = None  # 区域全尺度坐标下的轮廓点集 (N,1,2) int32；未学到模板时为 None。
    tracker_alive: bool = False  # 本帧是否有有效跟踪输出（已学到模板且场景未结束）。
    symmetry_period: float = 0.0  # 模板的旋转对称周期（度）。
    template_area: float = 0.0  # 模板面积，已换算回区域全尺度。
    use_rectangle_model: bool = False  # 是否启用了规则矩形边框评分模型。
    smoothed: bool = False  # Pass 2 回看插值后置 True。
    white_candidates: int = 0  # 本帧检测到的白色候选数量，诊断用。


@dataclass
class ShapeTrackSession:
    """一局录像的完整状态机：学习模板 → 光流证据 → 粒子跟踪 → 场景结束判定。"""

    params: ShapeTrackParams = field(default_factory=ShapeTrackParams)  # 本局使用的参数。
    logger: Optional[Callable[[str], None]] = None  # 可选日志回调，None 时静默。

    def __post_init__(self) -> None:
        """构造内部组件与状态；dataclass 的 __init__ 之后自动调用。"""

        self.backend = None  # 本局打分后端（torch CUDA 或 NumPy），创建跟踪器与对齐器时注入。
        self._resolve_scoring_backend()  # 先装配打分后端：显卡可用时同步恢复精度档参数（含光流档位/时间基线），对齐器据此创建。
        self.aligner = DenseTemporalAligner(  # 建时序残差对齐器（用已按后端恢复精度档的参数）。
            self.params.temporal_lags,  # 时间基线。
            self.params.play_height_ratio,  # 有效高度比例。
            self.params.dis_preset,  # CPU 引擎的 DIS 精度档位。
            backend=self.backend,  # 本局后端：显卡走 torch Farneback，CPU 走 cv2 DIS。
            flow_params=self.params.gpu_flow_params,  # 显卡 Farneback 的光流参数。
        )
        self.rng = np.random.default_rng(self.params.random_seed)  # 每局独立的随机数发生器，代替脚本里的全局 cv2.setRNGSeed。
        self.template: Optional[ShapeTemplate] = None  # 学习到的形状模板，未学习时为 None。
        self.tracker: Optional[ParticleShapeTracker] = None  # 粒子跟踪器，未学习时为 None。
        self.scene_change_run = 0  # 连续「残差过大」帧数，用于判定切场景。
        self.stagnant_scene_run = 0  # 连续「残差过小」帧数，用于判定画面静止。
        self.scene_ended = False  # 场景是否已结束，结束后不再输出跟踪。
        self.last_color_frame = -1  # 上一次测到白色轮廓的帧号，-1 表示从未测到。
        self._learn_buffer = []  # 最佳帧选窗的检测缓存（未学模板前逐帧攒）。
        self._prelearn_gone_run = 0  # 选窗期间检测缺失的连续帧数，白相过短时据此兜底学习。
        self.frame_index = 0  # 本局已处理帧数。
        self.source_counts: dict[str, int] = {}  # 各 source 的出现次数，收尾时输出分布。
        self.region_width = 0  # 最近一次 reset 传入的区域宽，诊断用。
        self.region_height = 0  # 最近一次 reset 传入的区域高，诊断用。
        self.effective_scale = self.params.process_scale  # 实际生效的处理尺度，大区域会被分辨率上限压低。

    def _log(self, message: str) -> None:
        """输出日志；未提供 logger 时静默丢弃。"""

        if self.logger is not None:  # 调用方注册了日志回调。
            self.logger(message)  # 转发消息。

    def _resolve_scoring_backend(self) -> None:
        """装配打分后端并按精度档设参：GPU 只决定走 torch CUDA/numpy，精度档由 precision_tier 显式决定。

        无 N 卡（或运行期永久降级）时重载档（PRECISION_TIER_GPU_ONLY）门控回落到 medium，
        避免 CPU 机负担不起高分辨率/大粒子档；这里用真实 backend 判定（含降级），是权威 clamp，
        GUI 侧的后端显示仅为最佳努力。
        """

        backend = require_gpu_backend() if self.params.gpu_scoring else numpy_backend()  # 开关关闭时显式走 NumPy。
        tier = self.params.precision_tier if self.params.precision_tier in PRECISION_TIERS else PRECISION_TIER_DEFAULT  # 非法档位回退默认。
        if not backend.is_gpu and tier in PRECISION_TIER_GPU_ONLY:  # CPU 机禁用高/极高/最强。
            tier = PRECISION_TIER_CPU_FALLBACK  # 回落中等（CPU 实时档）。
        self.params.precision_tier = tier  # 记录生效档位。
        self._apply_precision_tier(tier)  # 按档位设参（仅覆盖仍停在默认值的字段）。
        self.backend = backend  # 记录本局后端，创建跟踪器与对齐器时注入。
        flow = self.params.gpu_flow_params  # 显卡光流参数（CPU 引擎不用，但一并入日志便于排查）。
        self._log(
            f"SHAPE backend={backend.name} engine={flow_engine_name(backend.is_gpu)} tier={tier} "
            f"particles={self.params.particle_count} controls={self.params.control_count} "
            f"dis_preset={self.params.dis_preset} lags={self.params.temporal_lags} "
            f"flow(levels={flow['levels']},iterations={flow['iterations']},winsize={flow['winsize']}) "
            f"打分后端与精度档已装配"
        )  # 日志记录后端、光流引擎、生效档位与关键参数。

    def _apply_precision_tier(self, tier: str) -> None:
        """把 PRECISION_TIERS[tier] 应用到 params：仅覆盖仍等于 dataclass 默认值的字段，尊重调用方显式自定义。"""

        preset = PRECISION_TIERS.get(tier) or PRECISION_TIERS[PRECISION_TIER_DEFAULT]  # 取档位参数，非法档位回退默认。
        defaults = ShapeTrackParams.__dataclass_fields__  # 各字段的 dataclass 默认值。
        for name, value in preset.items():  # 逐参数应用。
            if getattr(self.params, name) == defaults[name].default:  # 该字段仍停在默认值（调用方未显式自定义）。
                setattr(self.params, name, value)  # 升到目标档位值。

    def reset(self, region_w: int, region_h: int) -> None:
        """区域位移或尺寸变化时重置整局状态。

        必须同时清空光流历史 deque、模板、跟踪器与场景计数，
        否则上一局的残差会污染新区域的证据图，导致跟踪器被拖到错误位置。
        """

        self.aligner.reset()  # 清空光流历史帧队列。
        self.template = None  # 丢弃已学模板，等待新区域重新学习。
        self.tracker = None  # 丢弃跟踪器。
        self.rng = np.random.default_rng(self.params.random_seed)  # 随机数发生器回到初始状态，保证可复现。
        self.scene_change_run = 0  # 清零切场景计数。
        self.stagnant_scene_run = 0  # 清零静止计数。
        self.scene_ended = False  # 场景结束标记复位。
        self.last_color_frame = -1  # 白色轮廓帧号复位。
        self._learn_buffer = []  # 清空最佳帧选窗缓存。
        self._prelearn_gone_run = 0  # 选窗缺失计数复位。
        self.frame_index = 0  # 帧计数复位。
        self.source_counts = {}  # 统计复位。
        self.region_width = int(region_w)  # 记录新区域宽。
        self.region_height = int(region_h)  # 记录新区域高。
        self.effective_scale = min(  # 实际处理尺度：不超过参数上限，且最长边不超分辨率上限。
            self.params.process_scale,
            self.params.max_process_side / max(region_w, region_h, 1),
        )
        self._log(f"SHAPE reset region={region_w}x{region_h} scale={self.effective_scale:.3f}")  # 日志记录重置。

    def update(self, crop_bgr: np.ndarray, fps: float) -> ShapeFrameResult:
        """喂入一帧裁剪区域，返回单帧结果。判定顺序严格对齐外部脚本 run() 主循环。"""

        process_frame = resize_for_processing(crop_bgr, self.effective_scale)  # 第 1 步：缩放到处理尺度。
        temporal = self.aligner.update(process_frame)  # 第 1 步：更新光流历史并算出时序残差证据。

        if temporal is not None and temporal.raw_mean > 24.0:  # 第 2 步：残差过大说明画面整体换了。
            self.scene_change_run += 1  # 累加切场景计数。
        else:
            self.scene_change_run = 0  # 中断计数。
        if temporal is not None and temporal.raw_mean < 0.22:  # 第 2 步：残差过小说明画面完全静止（多为结算画面）。
            self.stagnant_scene_run += 1  # 累加静止计数。
        else:
            self.stagnant_scene_run = 0  # 中断计数。
        run_limit = max(4, int(fps * 0.16))  # 触发场景结束所需的连续帧数，至少 4 帧。
        if self.frame_index > fps and (  # 第 2 步：跳过开局第一秒，避免启动抖动误判。
            self.scene_change_run >= run_limit  # 连续切场景。
            or self.stagnant_scene_run >= run_limit  # 或连续静止。
        ):
            self.scene_ended = True  # 判定场景结束。

        source = SOURCE_WAITING  # 第 3 步前先给默认来源。
        if self.tracker is not None and not self.scene_ended:  # 第 3 步：有跟踪器且场景未结束时推进粒子。
            self.tracker.propagate()  # 粒子按速度平移并扩散。
        center_hint = self.tracker.estimate()[0] if self.tracker is not None else None  # 位置先验取加权中心。
        candidates = detect_white_shapes(process_frame, self.params.play_height_ratio)  # 第 4 步：检测白色候选。
        detection = choose_shape_candidate(  # 第 4 步：从候选里挑出被跟踪目标。
            candidates,  # 白色候选列表。
            self.template,  # 已学模板（None 时直接取排序第一用于学习）。
            center_hint,  # 位置先验。
            self.tracker.confidence if self.tracker is not None else 0.0,  # 置信度决定距离窗口松紧。
            size_floor=self.params.min_size_floor,  # 小目标距离窗抬到绝对下限，强测量不被窄窗白白丢弃。
        )
        color_has_been_gone = (  # 第 5 步：目标是否已经消失超过一秒。
            self.tracker is not None  # 已经建过跟踪器。
            and self.last_color_frame >= 0  # 曾经测到过白色轮廓。
            and self.frame_index - self.last_color_frame > int(fps)  # 距上次测到颜色超过一秒。
        )
        if color_has_been_gone:  # 第 5 步：目标已淡出。
            # 目标淡出之后，后出现的白色覆盖物通常是 START/SUCCESS 文字，
            # 而不是同一个物体重新出现。多个同时出现的组件才作为保守的结束线索；
            # 孤立的背景高光只是被忽略掉。
            total_white_area = sum(item.area for item in candidates)  # 本帧全部白色候选的总面积。
            if (  # 三个条件同时满足才判定场景结束。
                self.template is not None  # 已学到模板。
                and len(candidates) >= 3  # 同时出现至少 3 个白色组件（结算文字的特征）。
                and total_white_area >= self.template.area * 1.35  # 白色总面积明显超过目标本身。
            ):
                self.scene_ended = True  # 判定场景结束。
            detection = None  # 强制丢弃检测，防止把 SUCCESS 文字当成目标重新初始化。
        learn_required = max(5, int(fps * self.params.template_learn_window))  # 最佳帧选窗需攒够的检测数（默认约 0.4 秒，至少 5 条才能算中位数）。
        if self.tracker is None and self._learn_buffer:  # 选窗尚未学模板：统计检测缺失的连续帧。
            self._prelearn_gone_run = self._prelearn_gone_run + 1 if detection is None else 0  # 缺检测累加，有检测归零。
        if not self.scene_ended and self.tracker is None and detection is not None:  # 第 6 步：首次学习改为最佳帧选窗，不再拿第一帧当场学模板。
            self._learn_buffer.append(detection)  # 缓存本帧检测，等窗口攒满再定夺。
            if len(self._learn_buffer) >= learn_required:  # 选窗攒够。
                detection = choose_template_detection(self._learn_buffer)  # 挑窗口内最干净的一条轮廓来学（首帧常被倒计时粘连拽大）。
                self._learn_buffer = []  # 清空缓存。
                self._prelearn_gone_run = 0  # 缺失计数复位。
            else:
                detection = None  # 窗口未满，本帧先不给出跟踪结论（source 保持 waiting）。
        elif (  # 兜底：白相过短，选窗未攒满目标就消失，用已缓存的检测学习，不丢模板学习机会。
            not self.scene_ended
            and self.tracker is None
            and self._learn_buffer
            and self._prelearn_gone_run >= max(3, int(fps * 0.3))
        ):
            detection = choose_template_detection(self._learn_buffer)  # 同样从缓存里挑最佳。
            self._learn_buffer = []  # 清空缓存。
        if not self.scene_ended and self.tracker is None and detection is not None:  # 第 6 步（续）：定下学习源，构建模板与跟踪器。
            self.template = build_shape_template(detection.contour)  # 从选中的白色轮廓构建模板。
            self.tracker = ParticleShapeTracker(  # 用首个可靠检测初始化粒子跟踪器。
                detection,  # 初始检测。
                self.template,  # 刚学到的模板。
                self.params.particle_count,  # 粒子数。
                process_frame.shape[1],  # 处理尺度下的帧宽。
                int(process_frame.shape[0] * self.params.play_height_ratio),  # 处理尺度下的有效高度。
                self.params.global_proposals,  # 重定位第一级粗扫候选数。
                self.rng,  # 本局随机数发生器。
                self.params.max_correction_ratio,  # 单帧修正上限比例。
                self.params.max_speed_ratio,  # 限速比例。
                control_count=self.params.control_count,  # 对照组数量。
                coarse_top=self.params.coarse_top,  # 精扫 top-K 邻域数。
                refine_per_top=self.params.refine_per_top,  # 每邻域精扫候选数。
                relocation_cooldown=self.params.relocation_cooldown_frames,  # 重定位限频间隔。
                min_size_floor=self.params.min_size_floor,  # 小目标阈值绝对下限：窗口/限速/半径/抖动不随小尺寸等比缩窄。
                backend=self.backend,  # 打分后端（torch CUDA 或 NumPy）。
            )
            source = SOURCE_COLOR  # 本帧来源为白色强测量。
            self.last_color_frame = self.frame_index  # 记录测到颜色的帧号。
            self._log(  # 输出学习结果，便于与外部脚本回归对照。
                f"SHAPE learned area={self.template.area:.1f} points={len(self.template.points)} "
                f"symmetry={self.template.symmetry_period:g}deg window={learn_required:02d}帧选优 "
                f"model={'rectangle' if self.template.use_rectangle_model else 'contour'}"
            )
        elif not self.scene_ended and self.tracker is not None:  # 第 7 步：已学习，逐帧观测。
            if detection is not None:  # 白色轮廓仍然可见。
                self.tracker.observe_color(detection)  # 强测量直接修正粒子群。
                source = SOURCE_COLOR  # 来源为颜色。
                self.last_color_frame = self.frame_index  # 更新颜色帧号。
            elif temporal is not None:  # 目标已透明但有时序残差证据。
                source = SOURCE_BORDER if self.tracker.observe_border(temporal.normalized) else SOURCE_PREDICTION  # 弱测量成功为 border，失败退化为 prediction。
            else:  # 历史不足，连证据图都算不出来。
                self.tracker.confidence *= 0.965  # 置信度缓慢衰减。
                self.tracker.frames_since_reliable += 1  # 不可靠帧数 +1。
                source = SOURCE_PREDICTION  # 来源为纯预测。
        elif self.scene_ended:  # 第 8 步：场景已结束。
            source = SOURCE_SCENE_ENDED  # 来源为场景结束。

        self.source_counts[source] = self.source_counts.get(source, 0) + 1  # 第 9 步：统计来源分布。
        draw_tracker = None if self.scene_ended else self.tracker  # 场景结束后不再输出跟踪（对齐脚本的 draw_tracker 语义）。
        result = ShapeFrameResult(  # 第 9 步：组装单帧结果。
            source=source,  # 来源。
            tracker_alive=draw_tracker is not None,  # 是否有有效输出。
            flow_residual=temporal.raw_mean if temporal is not None else None,  # 残差均值，无光流时为 None。
            white_candidates=len(candidates),  # 白色候选数，诊断用。
        )
        if draw_tracker is not None:  # 有有效输出时填充全部跟踪量。
            center, velocity, angle = draw_tracker.estimate()  # 取加权估计。
            scale = self.effective_scale  # 实际处理尺度，用于把坐标换算回区域全尺度。
            result.center = (center / scale).astype(np.float32)  # 中心换算回全尺度。
            result.velocity = (velocity / scale).astype(np.float32)  # 速度换算回全尺度。
            result.scale = float(draw_tracker.scale)  # 目标缩放估计（无量纲）。
            result.angle_deg = float(angle)  # 角度。
            result.confidence = float(draw_tracker.confidence)  # 置信度。
            result.border_score = float(draw_tracker.last_border_score)  # 边界得分。
            result.border_snr = float(draw_tracker.last_border_snr)  # 边界信噪比。
            result.search_radius = float(draw_tracker.search_radius / scale)  # 搜索半径换算回全尺度。
            result.symmetry_period = float(draw_tracker.template.symmetry_period)  # 对称周期。
            result.template_area = float(draw_tracker.template.area / (scale * scale))  # 面积按尺度平方换算回全尺度。
            result.use_rectangle_model = bool(draw_tracker.template.use_rectangle_model)  # 是否矩形模型。
            contour = transformed_contour(  # 还原当前姿态下的轮廓，供叠加绘制。
                draw_tracker.template, center, angle, draw_tracker.scale  # 模板、处理尺度中心、角度、缩放。
            ) / scale  # 换算回区域全尺度。
            result.contour = np.rint(contour).astype(np.int32).reshape(-1, 1, 2)  # 取整并整理成 polylines 需要的 (N,1,2)。
        self.frame_index += 1  # 帧计数推进（必须在判定之后，保证与脚本的 frame_index 语义一致）。
        return result  # 返回单帧结果。


def repair_shape_gaps(
    results: list[ShapeFrameResult],
    threshold: float,
    max_gap_frames: int,
    angle_period: float,
) -> int:
    """离线回看：把两端都可靠、长度不超过 max_gap_frames 的低置信区间线性插值回填。

    移植自外部脚本 repair_shape_gaps，操作对象从 CSV 字典行改成 ShapeFrameResult 列表。
    返回被回填的帧数。
    """

    def has_track(result: ShapeFrameResult) -> bool:
        """判断一帧是否有可插值的跟踪输出。"""

        return bool(result.tracker_alive) and result.source != SOURCE_SCENE_ENDED  # 必须有输出且不是场景结束帧。

    def confidence_of(result: ShapeFrameResult) -> float:
        """取一帧的置信度。"""

        return float(result.confidence)  # 直接读字段。

    repaired = 0  # 已回填帧数。
    index = 0  # 扫描游标。
    period = max(1e-6, float(angle_period))  # 角度周期，下限防除零。
    while index < len(results):  # 扫描全部帧。
        if not has_track(results[index]) or confidence_of(results[index]) >= threshold:  # 无输出或已经足够可靠。
            index += 1  # 跳过。
            continue
        start = index  # 低置信区间起点。
        while (  # 向后延伸，直到离开低置信区间。
            index < len(results)  # 未越界。
            and has_track(results[index])  # 仍有输出。
            and confidence_of(results[index]) < threshold  # 仍是低置信。
        ):
            index += 1  # 继续延伸。
        end = index - 1  # 低置信区间终点。
        left = start - 1  # 区间左侧的可靠帧。
        right = end + 1  # 区间右侧的可靠帧。
        if (  # 任一边界不满足就放弃这段区间。
            left < 0  # 左边越界（区间从第 0 帧开始）。
            or right >= len(results)  # 右边越界（区间延伸到最后一帧）。
            or not has_track(results[left])  # 左帧没有输出。
            or not has_track(results[right])  # 右帧没有输出。
            or confidence_of(results[left]) < threshold  # 左帧本身也不可靠。
            or confidence_of(results[right]) < threshold  # 右帧本身也不可靠。
            or right - left - 1 > max_gap_frames  # 区间过长，插值不可信。
        ):
            continue  # 放弃这段区间，index 已停在区间之后。
        for current in range(start, right):  # 回填区间内每一帧。
            fraction = (current - left) / (right - left)  # 归一化插值位置，0~1。
            first_center = results[left].center  # 左端中心。
            last_center = results[right].center  # 右端中心。
            results[current].center = (first_center + fraction * (last_center - first_center)).astype(np.float32)  # 中心线性插值。
            first_velocity = results[left].velocity  # 左端速度。
            last_velocity = results[right].velocity  # 右端速度。
            results[current].velocity = (first_velocity + fraction * (last_velocity - first_velocity)).astype(np.float32)  # 速度线性插值。
            first_scale = results[left].scale  # 左端缩放。
            last_scale = results[right].scale  # 右端缩放。
            results[current].scale = first_scale + fraction * (last_scale - first_scale)  # 缩放线性插值。
            first_angle = float(results[left].angle_deg)  # 左端角度。
            last_angle = float(results[right].angle_deg)  # 右端角度。
            angle_delta = (  # 角度差必须走周期域最短弧，否则 89°→1° 会被插成绕远路。
                last_angle - first_angle + period * 0.5  # 平移半周期。
            ) % period - period * 0.5  # 取模后折回，得到最短角差。
            results[current].angle_deg = float((first_angle + fraction * angle_delta) % period)  # 沿最短弧插值并折回周期域。
            results[current].source = SOURCE_INTERPOLATED  # 标记为回看插值帧（叠加绘制时用蓝色轮廓）。
            results[current].smoothed = True  # 标记已被平滑处理。
            repaired += 1  # 计数。
    return repaired  # 返回回填总帧数。
