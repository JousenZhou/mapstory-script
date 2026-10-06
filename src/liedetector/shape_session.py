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
    RotationWitness,  # 旋转见证器：从证据图+光流独立识别唯一在旋转的团块。
    ShapeTemplate,  # 学习到的形状模板。
    _to_numpy_array,  # 后端数组统一转回 NumPy（见证器消费证据/光流需下载）。
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
SOURCE_ROTATION = "rotation"  # 透明阶段证据不可信，由旋转见证器重锚到唯一在旋转的团块。
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
        "witness_check_interval": 8,  # 弱机把见证器检查间隔拉开，进一步控预算。
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
        "witness_check_interval": 5,
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
        "witness_check_interval": 5,
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
        "witness_check_interval": 5,
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
        "witness_check_interval": 5,
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
    # 旋转见证器（RotationWitness）：测谎场景只有目标旋转、干扰物仅平移，透明阶段靠旋转量区分目标。
    witness_enabled: bool = True  # 见证器总开关。
    witness_evidence_threshold: float = 3.0  # 证据分割的绝对阈值下限（与 0.995 分位取大者），归一化残差 0~12。
    witness_check_interval: int = 5  # 每几帧喂一次见证器（控预算）；重锚/防假锁与滞回、冷却计数都在检查点按帧长推进。
    witness_recover_conf: float = 0.80  # 低置信重锚的置信度阈：低于它才算进入低置信窗口。
    witness_recover_frames: int = 20  # 低置信连续多少帧才触发重锚（滞回，避免单帧抖动误抢）。
    witness_min_cum_deg: float = 60.0  # 候选成为真目标所需的最小累计转角（度）。
    witness_min_samples: int = 6  # 候选所需最少 ω 采样数（每检查点一次，即至少关联满 30 帧）。
    # 曾试过直接降到 3：回放实测让真目标（旋转末期 (594,413)，折算转角 96~160°）在 t756 就进 top-1，但同时把
    # 慢漂干扰物的闸门一起放开（extreme 的 prediction 帧数 47→70、错误重锚照旧），净效果变差，故回到 6。
    # 真正的问题不是采样下限本身，而是同一物理目标被碎成多条轨迹后各自攒不满门槛（取证实测：到帧 700 已建档
    # 209 条、同期存活 44~50 条，真目标那条只关联到 3~4 次）——由下面的证据簇聚合解决，而不是降采样下限。
    witness_cluster_enabled: bool = True  # 是否按空间簇聚合旋转证据后再卡门槛（关掉即退回逐轨迹口径，供回滚对比）。
    witness_cluster_radius: float = 24.0  # 入簇半径（处理尺度像素，约 0.8 倍目标尺寸）：同一目标碎裂轨迹的分布尺度。
    # 簇口径的采样下限实测：3 比 6 更差（回放判据 median 25.4→31.7、applied 检查点从 [612,642,672] 退成 [607,657]），
    # 因为放到 3 之后邻近慢漂碎片也能成簇、把簇质心拽偏，故回到 6（簇聚合本身仍保留：27.5→25.4 是有效的）。
    witness_cluster_min_samples: int = 6  # 簇候选的采样下限（簇内合计 ω 采样数）。
    witness_margin_deg: float = 40.0  # 最优候选与当前跟踪位置转角的差门槛（度）。
    witness_dir_consistency: float = 0.8  # 候选旋转方向一致性下限（单一定向旋转接近 1）。
    witness_min_age_frames: float = 12.0  # 候选最短年龄（帧）：与采样下限配套，压制刚建档就把小角高倍外推的单点碎片。
    # 旋转锚点缰绳（方案 F）：旋转证据是「哪个团块是答案」的唯一身份判据，一旦确认过，就不允许
    # 后续的时序残差弱测量把中心拖走。实测失败素材在旋转窗（t702–776）结束后，弱测量把中心从
    # (616,388) 一路拖到 (689,451)（真目标始终在 (606,401) 附近），偏差 90px 且置信度还高达 0.95，
    # 而规则①②因为要求「别处转角比当前多 40°」在跟踪点已经压在答案上时永不成立（自我锁死）。
    witness_anchor_enabled: bool = True  # 缰绳总开关（关掉即退回只有规则①②的旧行为，供回滚对比）。
    witness_anchor_tol_ratio: float = 0.25  # 允许离开锚点的最大距离（相对 size_ref）：超过就拉回锚点。取 0.25 而非 0.6：实测弱测量能以 ≈3.7 处理像素/帧回拖，0.6（≙43 全像素）的牙齿比一个检查区间（5 帧）的摆幅还大，17s 很容易落在波峰上。
    witness_anchor_hold_frames: int = 150  # 锚点保鲜期（帧）：旋转证据停更后仍继续缰绳的时长，超时说明目标身份已不可信。
    witness_anchor_min_checks: int = 2  # 锚点生效前需要连续确认的检查点数（滞回，压制单帧误分割抢锚）。
    # 锁定只在「旋转相位已结束」时收紧：恒定转速下簇的折算窗内转角是常数，只有目标停止旋转才会让读数
    # 衰减跌落。取证实测真答案簇 C2 从 78.5 跌到 24.0（t782→t792）正是相位结束时刻；而相位还在跑时
    # （t777–t787 读数 75.8/78.5/75.7）钉死会拦住本该跟随目标的正确漂移，故必须等衰减到位。
    witness_anchor_decay: float = 0.5  # 锚点簇读数跌到峰值的这一比例以下才判定「旋转相位结束」、上锁定。
    witness_anchor_dead_checks: int = 3  # 锚点处连续多少个检查点读不到可信簇就丢弃锚点（宁可不钉也不钉残影）。
    witness_anchor_conf: float = 0.85  # 拉回锚点时赋予的置信度：高于 witness_recover_conf，不让锁定误触发规则①。
    witness_anchor_lock_enabled: bool = True  # 锁定模式：生效后暂停 border 弱测量（保留逐帧外推），而不只靠周期拉回。
    # 选锁而不选持续拉回的理由：拉回是一次全量重摊粒子（角度/权重/速度清零），每检查点执行一遍会把
    # 已学到的模板姿态反复抹掉；暂停弱测量后粒子仅按 ≈零均值速度外推，60 帧内的随机漂移远小于测量拖拽。
    # 准星足迹掩膜（9s 缺陷）：测谎回放画面里烧进了上一轮操作留下的绿色准星轨迹，它自己会动，
    # 是透明相位里最强的时序残差源（取证：长基线相干积分里全场最强轨迹峰全落在准星位置带，比真值处高
    # 3 个数量级）。实测第 9s（t452）起弱测量把中心钉在准星上 160 帧不分离（距准星中位 16.7px、重合 100%）。
    # 只遮弱测量、不遮见证器：见证器按「旋转」认答案，平移的准星本就不是它的候选；而一旦把准星证据
    # 从见证器也剥掉，答案残差簇会提前 ≈25 帧出生，`windowed_deg` 是「累计转角 × 参考窗 / 年龄」，年龄翻倍
    # 就把读数从 78° 稀释到 44°，跌过 witness_min_cum_deg 门槛 → 锚点永不升格 → 缰绳失效、尾部回退（实测 t853
    # 7.4px → 100.5px）。A/B 三档（all/border/witness）定案：border 档同时拿到 9s 修复与 17s 收官。
    cursor_mask_enabled: bool = True  # 准星足迹掩膜总开关（只影响透明帧的 border 弱测量入参，见证器与强测量不变）。
    cursor_green_min: int = 90  # 绿色判据：G 通道下限（准星是高对比绿十字，缩放会混色故门槛取得比肉眼松）。
    cursor_green_dominance: int = 12  # 绿色判据：G - max(R,B) 的下限，排除发白的黄色与灰绿背景。
    cursor_mask_min_pixels: int = 15  # 绿色像素少于此数视为本帧无准星（不遮，退回原行为）。
    cursor_mask_dilate: int = 7  # 足迹膨胀核边长（奇数，处理尺度）：盖住抗锯齿边与拖影；实测 9 以上开始挖掉答案星自己的残差环。
    cursor_mask_max_frac: float = 0.05  # 足迹（膨胀后）占处理帧面积的上限，超出视为「这不是准星而是场景绿块」，本帧不遮。
    #   实测依据：11 个素材里准星足迹稳定在 0.04%~2.6%（十字约 18×18 处理像素），但有两个素材的个别帧
    #   最大绿块飙到 9.3% / 11.1%（`20261001_133728`、`20261001_165429`）——那是被绿判据误收的地形/技能光效，
    #   照遮就等于把一成证据抹掉，属于自残。上限取 5%：既放行全部真准星帧，又挡住大面绿块。

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
    witness_center: Optional[np.ndarray] = None  # 旋转见证器最优候选中心 (2,)，区域全尺度；无候选时为 None。
    witness_cum_deg: float = 0.0  # 旋转见证器最优候选的累计转角绝对值（度），诊断用。
    witness_applied: bool = False  # 本帧是否由见证器重锚触发（source == rotation）。


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
        self.witness: Optional[RotationWitness] = (  # 旋转见证器：不依赖模板/跟踪器，开局即创建；关闭时为 None。
            RotationWitness(
                threshold=self.params.witness_evidence_threshold,  # 证据分割绝对阈值下限。
                check_interval=self.params.witness_check_interval,  # 检查节奏（缺省 dt）。
                min_track_samples=self.params.witness_min_samples,  # 采样下限（按证据簇合计采样计，碎裂轨迹不再单独卡门）。
                min_track_age=self.params.witness_min_age_frames,  # 年龄下限（与采样下限配套防单点碎片）。
                cluster_radius=self.params.witness_cluster_radius,  # 同一物理目标的轨迹聚合半径（处理尺度）。
                cluster_min_samples=self.params.witness_cluster_min_samples,  # 簇口径的采样下限（合计采样）。
                cluster_enabled=self.params.witness_cluster_enabled,  # 关掉即退回逐轨迹口径（回滚开关）。
            )
            if self.params.witness_enabled
            else None
        )
        self._witness_since_check = 0  # 距上次喂见证器经过的帧数。
        self._witness_low_run = 0  # 连续低置信帧数（规则①滞回）。
        self._witness_static_run = 0  # 连续「当前位置静止而别处在转」检查次数（规则②）。
        self._witness_cooldown = 0  # 重锚后的冷却帧数，限频避免反复抢锚。
        self._witness_center: Optional[np.ndarray] = None  # 最近一次查询的最优候选中心（处理尺度），供结果回显。
        self._witness_cum_deg = 0.0  # 最近一次查询的最优候选累计转角（度）。
        self._witness_anchor: Optional[np.ndarray] = None  # 旋转锚点：最后一次被旋转证据确认的目标中心（处理尺度）。
        self._witness_anchor_age = 10 ** 9  # 锚点距最后一次确认经过的帧数（初值极大 = 尚无锚点）。
        self._witness_anchor_run = 0  # 锚点候选连续确认的检查点数（滞回）。
        self._witness_anchor_pending: Optional[np.ndarray] = None  # 滞回期间暂存的候选锚点位置。
        self._witness_anchor_pending_samples = 0  # 上次确认时该簇的合计采样数：采样不涨＝死簇外推，不能计入滞回（区分 C1/C2 的活度判据）。
        self._witness_anchor_dead_run = 0  # 锚点处连续读不到可信簇的检查点数（达阈丢弃锚点）。
        self._witness_anchor_lock = False  # 锁定中：True 时暂停 border 弱测量，中心由锚点接管理。
        self._witness_anchor_peak = 0.0  # 锚点簇的历史峰值转角，作为缰绳 engage 的衰减基准：恒定转速下 windowed_deg 为常数，只有旋转相位结束才会跌到峰值一半以下。
        self._witness_anchor_cooldown = 0  # 缰绳拉回后的冷却帧数，避免同一检查点反复拉扯。
        self._witness_trace: dict = {}  # 最近一个检查点的见证器内部量快照（诊断用，见 _apply_rotation_witness）。
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
        if self.witness is not None:  # 见证器存在。
            self.witness.reset()  # 丢弃全部旋转轨迹，避免上一局的旋转残留污染新区域。
        self._witness_since_check = 0  # 检查倒计时复位。
        self._witness_low_run = 0  # 低置信计数复位。
        self._witness_static_run = 0  # 防假锁计数复位。
        self._witness_cooldown = 0  # 冷却计数复位。
        self._witness_center = None  # 候选回显复位。
        self._witness_cum_deg = 0.0  # 候选转角回显复位。
        self._witness_anchor = None  # 旋转锚点复位。
        self._witness_anchor_age = 10 ** 9  # 锚点保鲜计数复位（无锚点）。
        self._witness_anchor_run = 0  # 锚点滞回计数复位。
        self._witness_anchor_pending = None  # 暂存候选复位。
        self._witness_anchor_pending_samples = 0  # 采样数基准复位。
        self._witness_anchor_dead_run = 0  # 死簇计数复位。
        self._witness_anchor_lock = False  # 锁定标志复位。
        self._witness_anchor_peak = 0.0  # 锚点峰值读数复位。
        self._witness_anchor_cooldown = 0  # 缰绳冷却复位。
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
                self._witness_anchor_lock = False  # 白色强测量重新接管：锁定让位（否则目标重新可见后会被钉在旧锚点）。
            elif temporal is not None:  # 目标已透明但有时序残差证据。
                if self._witness_anchor_lock:  # 缰绳锁定中：弱测量已被证明会把中心拖离旋转确认位置，本帧不改粒子状态，仅由逐帧 propagate 接管（速度接近零均值，不会系统性跑偏）。
                    source = SOURCE_ROTATION  # 来源仍计作旋转证据（中心理想位置就是锚点）。
                else:
                    # 弱测量入口遮准星足迹：透明相位只有残差可看，烧录的绿色准星是最强残差源，不遮就会被它抢走。
                    # 见证器仍读 temporal.normalized 原图（旋转判据不受准星影响，且遮了会拖花锚点年龄口径）。
                    border_map = self._mask_cursor_footprint(temporal.normalized, process_frame)
                    source = SOURCE_BORDER if self.tracker.observe_border(border_map) else SOURCE_PREDICTION  # 弱测量成功为 border，失败退化为 prediction。
            else:  # 历史不足，连证据图都算不出来。
                self.tracker.confidence *= 0.965  # 置信度缓慢衰减。
                self.tracker.frames_since_reliable += 1  # 不可靠帧数 +1。
                source = SOURCE_PREDICTION  # 来源为纯预测。
        elif self.scene_ended:  # 第 8 步：场景已结束。
            source = SOURCE_SCENE_ENDED  # 来源为场景结束。

        # 第 7.5 步：旋转见证器——仅透明帧（非 color）、非场景结束、有跟踪器与证据时生效。
        witness_applied = False
        if (
            self.witness is not None
            and self.tracker is not None
            and not self.scene_ended
            and temporal is not None
            and source != SOURCE_COLOR
        ):
            witness_applied = self._apply_rotation_witness(temporal, process_frame)  # 跑低置信重锚与防假锁两条规则。
            if witness_applied:  # 见证器把跟踪器重锚到了旋转团块。
                source = SOURCE_ROTATION

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
        if self.witness is not None:  # 回显旋转见证器诊断字段（候选中心换算回区域全尺度）。
            if self._witness_center is not None:  # 有旋转候选。
                result.witness_center = (self._witness_center / self.effective_scale).astype(np.float32)
            result.witness_cum_deg = float(self._witness_cum_deg)  # 最优候选累计转角。
            result.witness_applied = bool(witness_applied)  # 本帧是否重锚。
        self.frame_index += 1  # 帧计数推进（必须在判定之后，保证与脚本的 frame_index 语义一致）。
        return result  # 返回单帧结果。

    def _mask_cursor_footprint(self, evidence_map, process_frame):
        """把烧进画面的绿色准星足迹内的时序残差置零，返回**新**证据图（绝不就地写原图）。

        只在透明帧的弱测量入口调用：粒子靠残差加权找目标，而准星自己会动、对比度极高，
        是透明相位里最强的残差源，不遮就会把中心钉在准星上（9s 缺陷）。见证器不吃这张遮过的图：
        它按旋转量认答案、平移的准星本就不是候选，且遮了会提前答案残差簇的出生时刻、把
        `windowed_deg`（累计转角×参考窗/年龄）稀释到门槛以下，锚点升格链直接断裂（实测尾部 7.4→100.5px）。

        宁可不遮的退路有五条：开关关闭、本帧绿色像素不足、检不出连通块、足迹面积超上限（大面绿块不是
        准星）、足迹与证据尺寸不同步。检不出准星时原样返回，行为完全退回无掩膜口径（白相强测量帧不走这里，不受影响）。
        """

        params = self.params  # 参数引用。
        if not params.cursor_mask_enabled or process_frame is None or process_frame.ndim != 3:  # 关闭或非彩色帧。
            return evidence_map
        b = process_frame[:, :, 0].astype(np.int16)  # B 通道（转带符号：uint8 相减会回绕）。
        g = process_frame[:, :, 1].astype(np.int16)  # G 通道。
        r = process_frame[:, :, 2].astype(np.int16)  # R 通道。
        green = (((g > params.cursor_green_min) & (g - np.maximum(r, b) > params.cursor_green_dominance))
                 .astype(np.uint8)) * 255  # 绿主导像素掩膜。
        if int(cv2.countNonZero(green)) < params.cursor_mask_min_pixels:  # 本帧无准星。
            return evidence_map
        count, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)  # 连通块：最大块即准星本体。
        if count <= 1:  # 只有背景。
            return evidence_map
        blob = (labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)  # 准星足迹。
        kernel = max(1, int(params.cursor_mask_dilate))  # 膨胀核边长。
        if kernel > 1:  # 盖住抗锯齿边与拖影（足迹是细十字，不挖整盘）。
            blob = cv2.dilate(blob * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))) > 0
        # 面积上限护栏：真准星是十来个像素的小十字，而地形/光效被绿判据误收时动辄占一成画面，
        # 遮它等于自断证据；这种帧宁可不遮（维持原行为），也不冒挖掉大片残差的风险。
        if float(np.count_nonzero(blob)) > params.cursor_mask_max_frac * float(blob.size):
            return evidence_map
        keep = (1.0 - blob.astype(np.float32))  # 足迹内 0（此处无信息）、足迹外 1（原样保留）。
        if evidence_map.shape[0] != keep.shape[0] or evidence_map.shape[1] != keep.shape[1]:
            return evidence_map  # 证据图与处理帧尺寸不同步：宁可误锁准星也不遮错地方。
        if isinstance(evidence_map, np.ndarray):  # NumPy 后端：乘出新数组。
            return evidence_map * keep.astype(evidence_map.dtype, copy=False)
        try:  # torch 后端：掩膜搬到同设备同类型再相乘，产出新张量（normalized 是 CUDA Graph 的静态输出缓冲，不可就地写）。
            return evidence_map * evidence_map.new_tensor(keep)
        except Exception:  # 未知后端：不遮，原样交给弱测量。
            return evidence_map

    def _apply_rotation_witness(self, temporal, process_frame) -> bool:
        """按节奏喂证据+当前帧灰度给旋转见证器，跑低置信重锚与防假锁两条规则；返回本帧是否触发重锚。

        仅在透明帧调用。两条规则都以「存在一个强旋转、不在当前跟踪位置、且转角明显超过当前位置」
        的候选为前提：
        ① 低置信重锚：置信度低于阈值连续足够帧 → observe_rotation 重锚到新位置（滞回）。
        ② 防假锁复查：即使置信度高，若当前位置持续 ≈0 旋转而另一团块持续旋转（连续 ≥3 次检查）
          → 降信并重锚；两条规则共用一个重锚后冷却（interval×6 帧）。
        所有坐标均为处理尺度（见证器与粒子跟踪器同一尺度），结果字段换算回全尺度。
        """

        params = self.params  # 参数引用。
        tracker = self.tracker  # 当前跟踪器（调用方保证非 None）。
        interval = max(1, int(params.witness_check_interval))  # 检查间隔（帧）。
        # 检查节奏：重分割累计与两条重锚规则都只在检查点评估（控预算，与 witness_check_interval 注释同口径）。
        # 早先规则评估是逐帧做的，而滞回/冷却计数按「每检查点 += interval 帧」记账，于是
        # witness_recover_frames=20 实际 4 帧就满足、interval×6=30 帧的冷却实际 6 帧就到期，
        # 表现为重锚每几帧抢一次并把置信度反复压回 0.70，等于没有滞回。故整体收拢到检查点。
        self._witness_since_check += 1  # 逐帧累加。
        if self._witness_since_check < interval:  # 未到检查点：本帧不重分割、不评估重锚/防假锁。
            return False
        self._witness_since_check = 0  # 倒计时归零。
        evidence_np = np.asarray(_to_numpy_array(temporal.normalized), dtype=np.float32)  # 证据图下载主存。
        # 见证器自维灰度环形缓冲、每检查点算一次 ~12 帧长基线 Farneback（lag=1 测不到透明旋转）。
        gray_np = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)  # 处理尺度灰度（纯 CPU/cv2 侧）。
        self.witness.update(evidence_np, gray_np, dt=float(interval))  # 累计一个检查区间的旋转。
        if self._witness_cooldown > 0:  # 重锚冷却递减（每检查点减一个检查区间，冷却起点 interval×6 帧）。
            self._witness_cooldown -= interval
        current_center = tracker.estimate()[0]  # 当前跟踪中心（处理尺度）。
        radius = max(float(tracker.size_ref) * 0.6, 12.0)  # 查询半径：至少半个目标尺寸。
        current_cluster = self.witness.cluster_at(current_center, radius)  # 脚下的证据簇对象（供缰绳区分衰减/消失）。
        current_cum = 0.0 if current_cluster is None else float(current_cluster.windowed_deg)  # 当前位置团块的折算窗内转角。
        candidates = self.witness.query()  # 按质量分降序的旋转候选。
        best = candidates[0] if candidates else None  # 最优候选。
        self._witness_center = None if best is None else np.asarray(best.center, dtype=np.float32)  # 回显候选中心。
        best_windowed = 0.0 if best is None else float(best.windowed_deg)  # 最优候选折算窗内转角（旋转速率判据）。
        self._witness_cum_deg = best_windowed  # 回显候选转角（折算窗内）。
        anchor_radius = max(float(tracker.size_ref) * params.witness_anchor_tol_ratio, 12.0)  # 锚点容差（处理尺度）。
        # 诊断快照（处理尺度）：仅供回放脚本逐检查点取证，生产链路不读。放在这里是因为它是
        # 唯一能同时看到「当前位置转角 / 最优候选 / 锚点 / 中心」四个量的位置，缺一个就没法定位误抢锚。
        # 方向一致性/采样数/年龄是区分「稳定旋转干扰簇 C1」与「真答案簇 C2」的唯一依据（取证定案）。
        self._witness_trace = {  # 每检查点覆写一次。
            "frame": int(self.frame_index),  # 帧号。
            "current_cum": float(current_cum),  # 当前跟踪位置的折算窗内转角。
            "current_dir": 0.0 if current_cluster is None else float(current_cluster.direction_consistency),  # 脚下簇方向一致性。
            "current_samples": 0 if current_cluster is None else int(current_cluster.samples),  # 脚下簇合计采样数。
            "best": None if best is None else [round(float(best.center[0]), 1), round(float(best.center[1]), 1), round(best_windowed, 1)],
            "best_dir": 0.0 if best is None else float(best.direction_consistency),  # 最优候选方向一致性。
            "best_samples": 0 if best is None else int(best.samples),  # 最优候选采样数。
            "best_age": 0.0 if best is None else float(best.age),  # 最优候选年龄（帧）。
            "n_cand": int(len(candidates)),  # 过门槛的候选簇数。
            "center": [round(float(current_center[0]), 1), round(float(current_center[1]), 1)],  # 当前中心。
            "anchor": None if self._witness_anchor is None else [round(float(self._witness_anchor[0]), 1), round(float(self._witness_anchor[1]), 1)],
            "anchor_age": int(self._witness_anchor_age),  # 锚点保鲜计数。
            "tol": round(float(anchor_radius), 2),  # 锚点容差。
        }
        # 锚点维护（自确认口径）：锚点只能来自「跟踪器自己脚下读到强旋转，且脚下簇就是全场最优」，
        # 远处冒出来的候选一律不算。第一版正是拿 query() 最优候选当确认，在 t612 被画面顶部的干扰簇
        # (585,162) 抢走锚点，把跟踪器钉在顶部一路毁到结尾（17s 偏差 96→233px）。
        # 第二道闸门是「活度」：采样数必须比上次确认时增长。取证实测干扰簇 C1 在 t647/t652 两个自确认
        # 检查点上 samples 恒为 6、age 恒为 35.0（靠 lost_frames 苟活的死簇外推），而真答案簇 C2 在
        # t777→t797 的 samples 是 6→7→8→10→11。另外 C2 的 direction_consistency 只有 0.67~0.75，低于
        # witness_dir_consistency=0.8——这解释了规则①②为何在整个旋转窗里一次都没救过它，
        # 所以锚点判据不能沿用 0.8 的一致性门槛（否则永远只能看到 C1）。
        confirmed_center = None
        confirmed_samples = 0
        if (
            current_cluster is not None
            and current_cum >= params.witness_min_cum_deg  # 脚下读数达标。
            and best is not None
            and float(np.linalg.norm(best.center - current_center)) <= radius  # 脚下簇即全场最优：自己正踩在答案上。
        ):
            confirmed_center = np.asarray(current_cluster.center, dtype=np.float32)  # 确认位置取簇质心（t787 距 17s 真值仅 4px，比跟踪中心准）。
            confirmed_samples = int(current_cluster.samples)
        if params.witness_anchor_enabled:  # 锚点总开关关闭时完全不维护锚点，行为退回只有规则①②。
            self._witness_anchor_age += interval  # 先按一个检查区间老化，命中确认再清零。
            if self._witness_anchor_cooldown > 0:
                self._witness_anchor_cooldown -= interval  # 拉回冷却递减。
            if confirmed_center is None:
                self._witness_anchor_run = 0  # 无自确认：滞回计数归零，已有锚点靠保鲜期继续撑着。
                self._witness_anchor_pending = None
                self._witness_anchor_pending_samples = 0
            else:
                pending = self._witness_anchor_pending
                if pending is None or float(np.linalg.norm(confirmed_center - pending)) > anchor_radius:
                    self._witness_anchor_run = 1  # 确认位置换了地方：从本次重新起算滞回。
                    self._witness_anchor_pending_samples = confirmed_samples
                elif confirmed_samples > self._witness_anchor_pending_samples:
                    self._witness_anchor_run += 1  # 同一处且采样在涨：活簇确认。
                    self._witness_anchor_pending_samples = confirmed_samples
                # 采样不涨：既不 +1 也不清零——死簇外推不该计入滞回，但单检查点的 ω 估计抖动也不该抹掉已有进度。
                self._witness_anchor_pending = confirmed_center
                if self._witness_anchor_run >= params.witness_anchor_min_checks:
                    if self._witness_anchor is not None and float(np.linalg.norm(confirmed_center - self._witness_anchor)) > anchor_radius:
                        self._witness_anchor_peak = current_cum  # 换锚：衰减基准重新起算。
                    else:
                        self._witness_anchor_peak = max(self._witness_anchor_peak, current_cum)  # 同锚：峰值只抬不降。
                    self._witness_anchor = confirmed_center  # 升格/刷新锚点。
                    self._witness_anchor_age = 0  # 保鲜计数归零。
                    self._witness_anchor_dead_run = 0  # 刚被活簇确认：丢锚计数复位。
        # 锚点处的读数必须在维护之后取，才能反映「本检查点刚确认到的锚点」；用 radius（而非更紧的
        # anchor_radius）检索，是因为簇质心会被邻近碎片拽偏几像素，读数容差应与查询口径一致。
        anchor_cluster = None if self._witness_anchor is None else self.witness.cluster_at(self._witness_anchor, radius)  # 锚点处的簇仍在否。
        anchor_cum = 0.0 if anchor_cluster is None else float(anchor_cluster.windowed_deg)  # 锚点簇当前读数。
        self._witness_trace["anchor_cum"] = float(anchor_cum)  # 补写诊断：衰减=该收紧，归零=该丢锚。
        self._witness_trace["anchor_peak"] = float(self._witness_anchor_peak)  # 补写诊断：衰减基准。
        self._witness_trace["anchor_confirmed"] = confirmed_center is not None  # 补写诊断：本检查点是否自确认成功。
        # 强旋转候选条件：折算窗内转角、方向一致性、非当前跟踪位置、与当前位置转角差达门槛。
        strong = (
            best is not None
            and best_windowed >= params.witness_min_cum_deg
            and best.direction_consistency >= params.witness_dir_consistency
            and float(np.linalg.norm(best.center - current_center)) > radius  # 与当前位置不同团块。
            and (best_windowed - current_cum) >= params.witness_margin_deg  # 差门槛。
        )
        # 规则①：低置信重锚（计数按帧记，只在检查点累加一个检查区间的长度）。锁定中不跑①②：
        # 锁定的依据是「自己脚下强旋转过的簇仍在原地」，比远处一个强候选更可信，否则两条规则会把
        # 刚钉好的中心抢走（实测失败素材里 rot 窗后的全场最优候选是一个读数冻结在 30.6 的死簇）。
        if tracker.confidence < params.witness_recover_conf and not self._witness_anchor_lock:  # 本帧低置信。
            self._witness_low_run += interval  # 累加低置信帧数。
        else:
            self._witness_low_run = 0  # 一旦回到高置信就清零（滞回）。
        if strong and not self._witness_anchor_lock and self._witness_low_run >= params.witness_recover_frames and self._witness_cooldown <= 0:
            tracker.observe_rotation(best.center)  # 以候选中心为位置先验重撒粒子。
            self._witness_low_run = 0  # 重锚后清低置信计数。
            self._witness_static_run = 0  # 清防假锁计数。
            self._witness_cooldown = interval * 6  # 进入冷却。
            return True
        # 规则②：防假锁复查——当前位置持续 ≈0 旋转（即使置信度高）而另一团块持续旋转。
        static_now = current_cum < max(12.0, params.witness_min_cum_deg * 0.25)  # 当前位置几乎不转。
        if strong and static_now and not self._witness_anchor_lock:  # 满足“别处在转、自己不转”。
            self._witness_static_run += 1  # 连续检查次数 +1。
        else:
            self._witness_static_run = 0  # 否则重置。
        if strong and static_now and not self._witness_anchor_lock and self._witness_static_run >= 3 and self._witness_cooldown <= 0:
            tracker.confidence = min(tracker.confidence, 0.6)  # 先降信，重锚只抬到温和值。
            tracker.observe_rotation(best.center)  # 重锚到持续旋转的团块。
            self._witness_static_run = 0  # 清计数。
            self._witness_low_run = 0  # 清低置信计数。
            self._witness_cooldown = interval * 6  # 进入冷却。
            return True
        # 规则③（旋转锚点约束）：见 ShapeTrackParams.witness_anchor_enabled 的注释。
        # 和规则①②的本质区别：不再要求「别处转得比当前多」，而是要求「当前中心不得离开自确认过的
        # 锚点」；并且必须等到锚点簇读数衰减（旋转相位结束）才收紧：相位还在跑时钉死会拦住本该
        # 跟随目标的正确漂移（实测 t692→t772 正是靠弱测量把中心从 (585,255) 拉到真目标跟上的）。
        if params.witness_anchor_enabled and self._witness_anchor is not None:
            if anchor_cum <= 0.0:  # 锚点处已读不到可信簇（簇碎裂/消失）。
                self._witness_anchor_dead_run += 1  # 连续计数，容忍单检查点的采样抖动。
                if self._witness_anchor_dead_run >= params.witness_anchor_dead_checks:
                    self._witness_anchor = None  # 彻底丢弃：宁可不钉，也不钉住一具残影。
                    self._witness_anchor_peak = 0.0
                    self._witness_anchor_lock = False
            else:
                self._witness_anchor_dead_run = 0  # 锚点簇仍在。
                expired = self._witness_anchor_age > params.witness_anchor_hold_frames  # 超出保鲜期：目标身份不可信。
                decayed = anchor_cum <= params.witness_anchor_decay * self._witness_anchor_peak  # 读数跌到峰值一半以下：相位结束。
                if self._witness_anchor_lock and (expired or not decayed):
                    self._witness_anchor_lock = False  # 相位重新开转/锚点过期：交还给正常观测。
                if not expired and decayed and self._witness_anchor_cooldown <= 0:
                    drift = float(np.linalg.norm(tracker.estimate()[0] - self._witness_anchor))  # 离开锚点的距离。
                    if drift > anchor_radius:  # 被弱测量拖跑：以锚点为位置先验重摊粒子拉回，并转入锁定。
                        tracker.observe_rotation(self._witness_anchor, confidence=params.witness_anchor_conf)  # 拉回锚点。
                        self._witness_anchor_cooldown = interval  # 下个检查点即可再次收紧（锁定后基本不再漂）。
                        self._witness_anchor_lock = bool(params.witness_anchor_lock_enabled)  # 锁定：暂停 border 弱测量。
                        return True
        return False  # 本帧未触发重锚。


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
