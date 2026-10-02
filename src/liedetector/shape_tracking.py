"""测谎检验「找目标」算法内核：无神经网络的轮廓学习 + 稠密光流 + 粒子滤波。

本模块从外部验证脚本 track_transparent_shape.py / track_transparent_square_v2.py
移植而来，只做纯算法计算，不含任何视频读写、CSV 输出、GUI 绘制与命令行参数。

整体分三段：
1. 学习段：首帧从白色区域提取轮廓，重采样成 120 点模板（含法向、内部采样点、
   旋转对称周期、矩形度），满足四重对称 + 高矩形度时自动切换规则矩形边框评分模型。
2. 证据段：用稠密光流（显卡走 torch Farneback，CPU 走 cv2 DIS）把多个 lag 的历史帧
   对齐到当前帧，取绝对差得到「时序残差证据图」，目标透明化后仍会在残差图边界留下痕迹。
3. 跟踪段：粒子群在 6 维状态空间 (x, y, vx, vy, angle, 角速度) 上联合轮廓边界与
   内部残差打分，输出位置/角度/小范围缩放。

刻意与外部脚本的偏离：不调用全局 cv2.setNumThreads(1) / cv2.ocl.setUseOpenCL(False)
/ cv2.setRNGSeed()，因为这些全局设置会拖慢整个应用；可复现性改由每局独立的
np.random.default_rng(seed) 保证。
"""

from __future__ import annotations

import math  # 数学函数：三角、开方、弧度角度互转。
from dataclasses import dataclass  # 数据类装饰器，承载算法中间结果。

import cv2  # OpenCV：HSV 转换、形态学、轮廓、光流、重映射。
import numpy as np  # 数值计算：向量化旋转、打分、粒子群运算。

from src.liedetector.gpu_shape_backend import (  # 打分内核双后端：同一份算法代码，有 N 卡走 torch CUDA，没有则 NumPy。
    BorderEvidence,  # 矩形四边打分结果（定义迁至后端模块，此处 re-export 保持旧导入路径可用）。
    ShapeEvidence,  # 轮廓打分结果（同上）。
    ShapeScoreBackend,  # 打分后端类型。
    clear_gpu_failure,  # 打分成功后清零显卡连续失败计数。
    mark_gpu_failure,  # 记录显卡运行期异常，连续失败达上限后永久回退 NumPy。
    numpy_backend,  # NumPy 后端单例。
    require_gpu_backend,  # 请求显卡后端（不可用/已降级时返回 NumPy 后端）。
    sample_map_core,  # 采样内核（后端版）。
    score_rotated_borders_core,  # 矩形四边打分内核（后端版）。
    score_shape_contours_core,  # 通用轮廓打分内核（后端版）。
    transform_template_points_core,  # 模板点变换内核（后端版）。
)
from src.liedetector.tensor_evidence import (  # 时序证据链：唯一一份实现，CPU 与 CUDA 共用。
    TemporalEvidence,  # 多 lag 光流对齐后的时序残差证据（定义迁至证据链模块，此处 re-export）。
    TensorTemporalAligner,  # 双后端对齐器本体，DenseTemporalAligner 只做装配。
)
from src.liedetector.torch_flow import Cv2DisFlow, TorchFarnebackFlow  # 光流引擎：显卡 Farneback / CPU DIS。


def _resolve_backend(backend):  # 解析本次打分使用的后端：显式传入优先，否则取全局生效后端（torch CUDA 或 NumPy）。
    if backend is not None:  # 调用方显式指定了后端（会话装配时注入）。
        return backend  # 直接使用。
    return require_gpu_backend()  # 未指定：请求显卡后端，不可用/已降级时返回 NumPy 后端。


def _to_numpy_array(item):  # 把后端数组统一转回 NumPy（torch 张量 detach 后 D2H，NumPy 数组直接复用）。
    if isinstance(item, np.ndarray):  # 已是 NumPy 数组。
        return item  # 直接返回。
    detach = getattr(item, "detach", None)  # torch 张量的去求导方法。
    if callable(detach):  # 显卡张量。
        return detach().cpu().numpy()  # 下载到主存并转 numpy。
    return np.asarray(item)  # 其余情况兼容处理。


def _pack_scoring(result, to_numpy):  # 把内核返回的打分结果重新打包；to_numpy 为真时下载成 NumPy 版数据类。
    if not to_numpy:  # 调用方自己持有后端（粒子滤波），结果原样留在设备上，省掉一次 D2H。
        return result  # 直接返回。
    return type(result)(_to_numpy_array(result.scores), _to_numpy_array(result.coverage))  # 下载并按原数据类重新打包。


def _run_scoring(core, backend, *args, to_numpy=True):  # 统一的打分调度：显卡后端异常时自动改走 NumPy 重算本帧。
    chosen = _resolve_backend(backend)  # 本次生效的后端。
    if chosen.is_gpu and chosen is require_gpu_backend():  # 全局显卡后端：运行期异常（显存不足/驱动重置等）不能中断求解。
        try:  # 显卡执行段。
            result = core(chosen.xp, *args)  # 在显卡上执行打分内核。
            clear_gpu_failure()  # 打分成功，清零连续失败计数。
            return _pack_scoring(result, to_numpy)  # 按需下载后返回。
        except Exception:  # 显卡运行期异常。
            mark_gpu_failure()  # 计数，连续 3 次后本进程永久回退 NumPy。
            chosen = numpy_backend()  # 本帧改走 CPU 重算，调用方拿到的结果与显卡正常时等价（门面的 asarray 会把显存数组降级下载）。
    return _pack_scoring(core(chosen.xp, *args), to_numpy)  # NumPy 后端（或显卡降级后）直接计算。


@dataclass
class ShapeDetection:
    """一帧里检测到的单个白色候选目标。"""

    center: np.ndarray  # 候选目标的质心坐标 (2,)，处理尺度坐标系。
    contour: np.ndarray  # 候选目标的原始轮廓点集 (N,1,2)。
    area: float  # 轮廓面积（像素^2）。
    nominal_size: float  # 标称尺寸 = sqrt(area)，后续所有距离阈值都以它为基准。
    confidence: float  # 白色检测置信度，由白度/亮度/实心度/延展度加权得到。
    angle: float = 0.0  # 与模板做循环 Procrustes 对齐后得到的旋转角（度）。
    shape_distance: float = 0.0  # 对齐残差距离，越小说明形状越像模板。


@dataclass
class ShapeTemplate:
    """从首个可靠白色轮廓学习到的形状模板。"""

    points: np.ndarray  # 等周长重采样的 120 个轮廓点，已减去质心（局部坐标）。
    normals: np.ndarray  # 每个轮廓点对应的单位法向量，用于沿法向采样证据。
    interior_points: np.ndarray  # 17x17 网格中落在轮廓内部的采样点（局部坐标）。
    area: float  # 模板面积（像素^2）。
    nominal_size: float  # 模板标称尺寸 = max(4.0, sqrt(area))。
    radius: float  # 模板外接半径 = 局部点到质心的最大距离。
    symmetry_period: float  # 推断出的最小旋转对称周期（度），圆为 30、正方形为 90。
    rectangularity: float  # 矩形度 = 轮廓面积 / 最小外接矩形面积，范围 0~1。
    rectangle_side: float  # 最小外接矩形的平均边长，矩形模型专用。
    rectangle_angle: float  # 最小外接矩形主边相对水平轴的偏置角（度，模 90）。
    use_rectangle_model: bool  # 是否启用规则矩形边框评分模型。


# 倒计时数字的冷色光环（蓝灰边框阴影）判别阈值与排除区参数，取自生产录像实测（lie_records/20260930_*）：
# 倒计时数字被一圈 H∈[40,130]、S∈[15,140]、V≥90 的冷色光环包围（处理尺度下连通域 700~1600px），
# 而目标图形周围环带该特征占比≈0；无倒计时帧的全图冷色连通域均 <60px，故 HALO_MIN_AREA=200 留 3 倍余量。
HALO_HUE_MIN = 40  # 冷色光环色相下限（OpenCV 色相 0~179，40~130 覆盖绿-青-蓝）。
HALO_HUE_MAX = 130  # 冷色光环色相上限。
HALO_SAT_MIN = 15  # 冷色光环饱和度下限：排除低饱和的白色数字本体与纯灰高光。
HALO_SAT_MAX = 140  # 冷色光环饱和度上限：阴影是淡蓝灰，不是高饱和纯蓝。
HALO_VALUE_MIN = 90  # 冷色光环明度下限：排除暗色描边。
HALO_DILATE_SIZE = 5  # 防线一膨胀核边长：从光环向白像素吃入 2px，吃掉数字本体外缘并切断数字↔图形粘连桥。
HALO_MIN_AREA = 200.0  # 光环连通域成排除区的最小面积（处理尺度像素）：背景冷色噪声连通域实测 <60px。
HALO_MIN_FILL = 0.2  # 光环连通域入选防线的最小填充率（面积/bbox 面积）：倒计时环实测 ≈0.52、数字过渡碎片 0.32~0.36、合成帧环 ≈0.27，而旧录像背景散点噪声会连成整帧级连通域（bbox≈384x254、面积 ≈950、填充率 ≈0.01）；入选门槛同时约束防线一与防线二——散点噪声若参与防线一膨胀，会随机蚀穿全图白色掩码（实测星形面积 813→354、质心偏 9px），参与防线二则排除区覆盖全帧误杀所有候选。
HALO_ZONE_PAD = 4  # 排除区 bbox 外扩边距（像素）：罩住防线一吃剩的数字核心残片。


def resize_for_processing(frame: np.ndarray, scale: float) -> np.ndarray:
    """把帧缩放到算法处理尺度；尺度为 1 时原样返回以避免无谓拷贝。"""

    if math.isclose(scale, 1.0):  # 处理尺度为 1 时无需缩放。
        return frame  # 直接返回原帧。
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # 缩小用 INTER_AREA 抗锯齿。


def contour_center(contour: np.ndarray) -> np.ndarray:
    """求轮廓质心：优先用图像矩，退化时用点集算术均值。"""

    moments = cv2.moments(contour)  # 计算轮廓的图像矩。
    if abs(moments["m00"]) > 1e-6:  # 零阶矩非零，质心可用。
        return np.array(  # 用一阶矩除以零阶矩得到质心。
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],  # x、y 质心分量。
            dtype=np.float32,  # 统一成 float32 便于后续向量运算。
        )
    return np.mean(contour.reshape(-1, 2), axis=0).astype(np.float32)  # 退化轮廓用点集均值兜底。


def resample_closed_contour(contour: np.ndarray, count: int = 120) -> np.ndarray:
    """沿轮廓周长等距重采样出 count 个点，让不同分辨率的形状可以直接比较。"""

    points = contour.reshape(-1, 2).astype(np.float32)  # 展平成 (N,2) 点集。
    if len(points) < 3:  # 少于三点无法构成闭合轮廓。
        raise ValueError("A shape contour needs at least three points")  # 抛出异常由调用方处理。
    closed = np.vstack([points, points[0]])  # 首尾相接，保证周长闭合。
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)  # 每段折线的长度。
    perimeter = float(np.sum(segment_lengths))  # 总周长。
    if perimeter < 4.0:  # 周长过小说明是噪点轮廓。
        raise ValueError("Shape contour is too small")  # 拒绝过小的形状。
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # 累积弧长，用于二分定位。
    distances = np.linspace(0.0, perimeter, count, endpoint=False)  # 等弧长目标位置（不含终点避免重复）。
    segment_indexes = np.searchsorted(cumulative, distances, side="right") - 1  # 每个目标位置落在哪一段。
    segment_indexes = np.clip(segment_indexes, 0, len(points) - 1)  # 防止浮点误差越界。
    within = distances - cumulative[segment_indexes]  # 在本段内的已走弧长。
    fractions = within / np.maximum(segment_lengths[segment_indexes], 1e-6)  # 换算成段内插值比例。
    return (  # 段起点 + 比例 * (段终点 - 段起点)，得到重采样点。
        closed[segment_indexes]
        + fractions[:, None]
        * (closed[segment_indexes + 1] - closed[segment_indexes])
    ).astype(np.float32)  # 统一 float32 输出。


def infer_symmetry_period(points: np.ndarray, radius: float) -> float:
    """推断轮廓的最小旋转对称周期：从小到大试候选角，看旋转后是否与原轮廓重合。"""

    candidates = (30.0, 36.0, 45.0, 60.0, 72.0, 90.0, 120.0, 180.0)  # 候选对称周期（度），按从小到大排列。
    threshold = max(1.0, radius * 0.040)  # 容差随尺寸放大，至少 1 像素。
    for period in candidates:  # 依次尝试每个候选周期。
        radians = math.radians(period)  # 转成弧度。
        rotation = np.array(  # 构造二维旋转矩阵。
            [[math.cos(radians), -math.sin(radians)],  # 第一行。
             [math.sin(radians), math.cos(radians)]],  # 第二行。
            dtype=np.float32,  # 与点集同精度。
        )
        rotated = points @ rotation.T  # 把全部采样点旋转该角度。
        distances = np.linalg.norm(  # 旋转后每个点到原始点集的最短距离矩阵。
            rotated[:, None, :] - points[None, :, :], axis=2
        )
        if float(np.mean(np.min(distances, axis=1))) <= threshold:  # 平均最近距离足够小说明对称成立。
            return period  # 返回该周期。
    return 360.0  # 没有任何对称性，退化为无周期（360 度）。


def build_shape_template(contour: np.ndarray, sample_count: int = 120) -> ShapeTemplate:
    """从一条白色轮廓构建完整模板：采样点、法向、内部点、对称周期与矩形度。"""

    center = contour_center(contour)  # 轮廓质心，作为局部坐标原点。
    points = resample_closed_contour(contour, sample_count) - center  # 等周重采样并平移到局部坐标。
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)  # 中心差分求切向。
    tangent_length = np.maximum(np.linalg.norm(tangent, axis=1), 1e-6)  # 切向长度，下限防除零。
    tangent /= tangent_length[:, None]  # 归一化成单位切向。
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)  # 切向逆时针旋转 90 度得到法向。
    area = float(abs(cv2.contourArea(contour)))  # 轮廓面积。
    nominal_size = max(4.0, math.sqrt(area))  # 标称尺寸，下限 4 像素避免过小目标除零。
    radius = float(np.max(np.linalg.norm(points, axis=1)))  # 外接半径。
    rectangle = cv2.minAreaRect(contour.astype(np.float32))  # 最小外接旋转矩形。
    rectangle_width, rectangle_height = rectangle[1]  # 取出矩形的宽和高。
    rectangle_area = max(1.0, rectangle_width * rectangle_height)  # 矩形面积，下限防除零。
    rectangularity = float(np.clip(area / rectangle_area, 0.0, 1.0))  # 矩形度：越接近 1 越像实心矩形。
    rectangle_side = float((rectangle_width + rectangle_height) * 0.5)  # 平均边长，矩形模型按正方形近似。
    rectangle_aspect = max(rectangle_width, rectangle_height) / max(  # 长宽比。
        1.0, min(rectangle_width, rectangle_height)  # 分母下限 1.0 防除零。
    )
    rectangle_points = cv2.boxPoints(rectangle)  # 最小外接矩形的四个角点。
    rectangle_edges = np.roll(rectangle_points, -1, axis=0) - rectangle_points  # 四条边向量。
    reference_edge = rectangle_edges[int(np.argmax(np.linalg.norm(rectangle_edges, axis=1)))]  # 取最长边作为角度基准。
    rectangle_angle = float(  # 基准边相对水平轴的偏置角，模 90 归一到正方形对称域。
        math.degrees(math.atan2(reference_edge[1], reference_edge[0])) % 90.0
    )
    symmetry_period = infer_symmetry_period(points, radius)  # 推断旋转对称周期。
    use_rectangle_model = bool(  # 只有「四重对称 + 高矩形度 + 接近正方形」才切换到矩形模型。
        symmetry_period == 90.0  # 条件一：90 度对称。
        and rectangularity >= 0.78  # 条件二：矩形度足够高。
        and rectangle_aspect <= 1.30  # 条件三：长宽比接近 1。
    )
    # 稀疏内部采样点用于捕捉半透明形状内部的运动残差；
    # 边界采样点仍然是形状特异性的主要证据来源。
    grid_x = np.linspace(float(np.min(points[:, 0])), float(np.max(points[:, 0])), 17)  # 局部 x 方向 17 档网格。
    grid_y = np.linspace(float(np.min(points[:, 1])), float(np.max(points[:, 1])), 17)  # 局部 y 方向 17 档网格。
    local_contour = points.reshape(-1, 1, 2)  # pointPolygonTest 需要 (N,1,2) 形状。
    interior_points = np.asarray(  # 只保留落在轮廓内部（含边界）的网格点。
        [
            (x, y)  # 一个内部采样点。
            for y in grid_y  # 遍历 y 网格。
            for x in grid_x  # 遍历 x 网格。
            if cv2.pointPolygonTest(local_contour, (float(x), float(y)), False) >= 0  # 在轮廓内或边界上。
        ],
        dtype=np.float32,  # 统一精度。
    )
    if len(interior_points) == 0:  # 极细长形状可能一个内部点都取不到。
        interior_points = np.zeros((1, 2), dtype=np.float32)  # 放一个原点兜底，避免后续空数组报错。
    return ShapeTemplate(  # 组装模板。
        points=points,  # 120 个局部轮廓采样点。
        normals=normals,  # 对应法向。
        interior_points=interior_points,  # 内部残差采样点。
        area=area,  # 面积。
        nominal_size=nominal_size,  # 标称尺寸。
        radius=radius,  # 外接半径。
        symmetry_period=symmetry_period,  # 对称周期。
        rectangularity=rectangularity,  # 矩形度。
        rectangle_side=rectangle_side,  # 矩形平均边长。
        rectangle_angle=rectangle_angle,  # 矩形角度偏置。
        use_rectangle_model=use_rectangle_model,  # 是否走矩形模型。
    )


def transformed_contour(
    template: ShapeTemplate, center: np.ndarray, angle: float, scale: float
) -> np.ndarray:
    """按中心/角度/缩放把模板还原成当前姿态的轮廓点集，供叠加绘制使用。"""

    local_points = template.points  # 默认使用学习到的 120 点轮廓。
    display_angle = angle  # 显示用角度初值。
    if template.use_rectangle_model:  # 矩形模型改用规则四角点，避免白色蒙版毛边误导。
        half_side = template.rectangle_side * 0.5  # 半边长。
        local_points = np.array(  # 规则正方形的四个角点（局部坐标）。
            [
                [-half_side, -half_side],  # 左上。
                [half_side, -half_side],  # 右上。
                [half_side, half_side],  # 右下。
                [-half_side, half_side],  # 左下。
            ],
            dtype=np.float32,  # 统一精度。
        )
        display_angle += template.rectangle_angle  # 叠加矩形的角度偏置。
    radians = math.radians(display_angle)  # 转弧度。
    rotation = np.array(  # 二维旋转矩阵。
        [[math.cos(radians), -math.sin(radians)],  # 第一行。
         [math.sin(radians), math.cos(radians)]],  # 第二行。
        dtype=np.float32,  # 统一精度。
    )
    return (local_points @ rotation.T * scale + center).astype(np.float32)  # 旋转 -> 缩放 -> 平移到目标中心。


def detect_white_shapes(
    frame: np.ndarray, play_height_ratio: float
) -> list[ShapeDetection]:
    """在一帧里找出所有「低饱和高亮」的白色候选目标，按面积*置信度降序返回。

    测谎弹窗里的倒计时数字同样是白色（带蓝灰冷色边框阴影），会被白色掩码检成候选，
    甚至与目标图形粘连成单一轮廓拉偏质心。这里用冷色光环特征加三道防线剔除它：
    防线零把真光环连通域填成「数字足迹」（凸包），形态学前整块删掉足迹内白像素——
    粗笔画数字本体被连核心一并移除、数字↔图形粘连桥被切断（旧版仅向白像素吃入 2px，
    够不到粗笔画核心，数字与图形粘连轮廓的质心被图形拽离排除区、被误学成畸形模板）；
    防线一再从光环膨胀蚀除残留薄桥，防线二把大光环连通域的 bbox 外扩成排除区，
    中心落在区内的候选（数字核心残片）直接丢弃。
    """

    height, width = frame.shape[:2]  # 帧的高和宽。
    play_height = int(height * play_height_ratio)  # 有效检测高度（比例以下视为无效带）。
    hsv = cv2.cvtColor(frame[:play_height], cv2.COLOR_BGR2HSV)  # 只在有效带内转 HSV。
    saturation = hsv[:, :, 1]  # 饱和度通道。
    value = hsv[:, :, 2]  # 明度通道。
    mask = ((saturation <= 92) & (value >= 188)).astype(np.uint8) * 255  # 低饱和 + 高亮 = 白色掩码。
    hue = hsv[:, :, 0]  # 色相通道：倒计时冷色光环判别用。
    halo = (  # 冷色光环掩码：倒计时数字独有的蓝灰边框阴影（实测 H∈[40,130]、S∈[15,140]、V≥90）。
        (hue >= HALO_HUE_MIN) & (hue <= HALO_HUE_MAX)
        & (saturation >= HALO_SAT_MIN) & (saturation <= HALO_SAT_MAX)
        & (value >= HALO_VALUE_MIN)
    ).astype(np.uint8)
    _, halo_labels, halo_stats, _ = cv2.connectedComponentsWithStats(halo, 8)  # 冷色光环连通域（8 邻域）。
    halo_clean = np.zeros_like(halo)  # 通过面积+填充率门槛的真光环：防线一只从它膨胀。
    digit_block = np.zeros_like(halo)  # 防线零数字足迹：真光环闭合圈的凸包实心域，罩住整条粗笔画数字本体。
    zones = []  # 防线二排除区：真光环 bbox 外扩一圈，防线一吃剩的数字核心残片会落在里面。
    for label, stat in enumerate(halo_stats[1:], start=1):  # 跳过背景连通域。
        halo_area = float(stat[cv2.CC_STAT_AREA])  # 连通域面积。
        halo_fill = halo_area / max(1.0, stat[cv2.CC_STAT_WIDTH] * stat[cv2.CC_STAT_HEIGHT])  # 填充率。
        if halo_area < HALO_MIN_AREA or halo_fill < HALO_MIN_FILL:  # 散点噪声/整帧级噪声连通域不入防线。
            continue  # 跳过。
        halo_clean[halo_labels == label] = 1  # 保留真光环供防线一膨胀。
        halo_points = cv2.findNonZero((halo_labels == label).astype(np.uint8))  # 本光环全部像素坐标。
        if halo_points is not None:  # 防线零：把闭合光环的凸包填成数字足迹（环+被圈住的数字本体）。
            cv2.fillConvexPoly(digit_block, cv2.convexHull(halo_points), 1)  # 凸包实心域≈罩住数字本体的圆盘。
        zones.append(  # 记录一个排除区（bbox 四界外扩）。
            (
                int(stat[cv2.CC_STAT_LEFT]) - HALO_ZONE_PAD,  # 左界外扩。
                int(stat[cv2.CC_STAT_TOP]) - HALO_ZONE_PAD,  # 上界外扩。
                int(stat[cv2.CC_STAT_LEFT]) + int(stat[cv2.CC_STAT_WIDTH]) + HALO_ZONE_PAD,  # 右界外扩。
                int(stat[cv2.CC_STAT_TOP]) + int(stat[cv2.CC_STAT_HEIGHT]) + HALO_ZONE_PAD,  # 下界外扩。
            )
        )
    mask[digit_block > 0] = 0  # 防线零：整块删掉数字足迹内的白像素，粗笔画数字本体与粘连桥一并移除，星形（在光环外）保持干净。
    mask[  # 防线一：再删掉贴着蓝灰阴影的白像素，吃掉足迹外溢出的数字边缘与残余薄桥。
        cv2.dilate(halo_clean, np.ones((HALO_DILATE_SIZE, HALO_DILATE_SIZE), np.uint8)) > 0
    ] = 0
    margin_x = max(2, int(width * 0.012))  # 左右边距，避开画面边框高光。
    margin_y = max(2, int(height * 0.012))  # 上边距。
    mask[:margin_y] = 0  # 清掉顶部边距。
    mask[:, :margin_x] = 0  # 清掉左边距。
    mask[:, width - margin_x :] = 0  # 清掉右边距。
    mask = cv2.morphologyEx(  # 开运算去孤立噪点。
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3x3 椭圆核。
    )
    mask = cv2.morphologyEx(  # 闭运算填补目标内部小孔洞。
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 7x7 椭圆核。
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 只取外轮廓，保留全部边界点。
    play_area = width * play_height  # 有效检测面积。
    minimum_area = max(160.0, play_area * 0.0014)  # 面积下限：太小的是噪点。
    maximum_area = play_area * 0.20  # 面积上限：太大的多半是背景/UI 白块。
    detections: list[ShapeDetection] = []  # 收集通过筛选的候选目标。
    for contour in contours:  # 逐个轮廓判定。
        area = float(abs(cv2.contourArea(contour)))  # 轮廓面积。
        if not minimum_area <= area <= maximum_area or len(contour) < 8:  # 面积不在区间内，或点数太少形状不可靠。
            continue  # 丢弃。
        x, y, box_width, box_height = cv2.boundingRect(contour)  # 轴对齐外接矩形。
        if min(box_width, box_height) < 14 or max(box_width, box_height) > min(width, play_height) * 0.78:  # 最短边过小或最长边过大。
            continue  # 丢弃。
        aspect = max(box_width, box_height) / max(1.0, min(box_width, box_height))  # 长宽比。
        if aspect > 4.2:  # 过于细长的条状物不是目标。
            continue  # 丢弃。
        center = contour_center(contour)  # 质心：先算出来供排除区判定与候选输出共用。
        if any(  # 防线二：质心落在倒计时排除区内的是数字核心残片，不是目标图形。
            left <= center[0] <= right and top <= center[1] <= bottom  # 点在扩边后的 bbox 内。
            for left, top, right, bottom in zones  # 逐个排除区判定。
        ):
            continue  # 丢弃。
        hull_area = float(abs(cv2.contourArea(cv2.convexHull(contour))))  # 凸包面积。
        solidity = area / max(hull_area, 1.0)  # 实心度：凸缺陷越少越接近 1。
        extent = area / max(1.0, box_width * box_height)  # 延展度：占外接矩形的比例。
        region_mask = np.zeros((play_height, width), dtype=np.uint8)  # 用于统计内部像素的单通道掩码。
        cv2.drawContours(region_mask, [contour], -1, 255, cv2.FILLED)  # 填充轮廓内部。
        inside = region_mask > 0  # 布尔索引：轮廓内部像素。
        whiteness = float(np.mean((255 - saturation[inside]) / 255.0))  # 白度：内部平均饱和度越低越白。
        brightness = float(np.mean(value[inside]) / 255.0)  # 亮度：内部平均明度。
        confidence = float(  # 四项指标加权得到置信度。
            np.clip(
                0.30 * whiteness  # 白度权重最高。
                + 0.28 * brightness  # 亮度次之。
                + 0.22 * np.clip(solidity, 0.0, 1.0)  # 实心度。
                + 0.20 * np.clip(extent / 0.70, 0.0, 1.0)  # 延展度按 0.70 归一（实心形状延展度约 0.7~1）。
                ,
                0.0,  # 下限。
                1.0,  # 上限。
            )
        )
        detections.append(  # 记录一个通过筛选的候选目标。
            ShapeDetection(
                center=center,  # 质心（排除区判定已算好）。
                contour=contour,  # 原始轮廓。
                area=area,  # 面积。
                nominal_size=math.sqrt(area),  # 标称尺寸。
                confidence=confidence,  # 置信度。
            )
        )
    return sorted(detections, key=lambda item: item.area * item.confidence, reverse=True)  # 面积*置信度降序，最像目标的排前面。


def periodic_angle_difference(angles, reference: float, period: float):
    """求一组角度与参考角在周期域内的最小角距离（后端无关：只用四则、取模与 abs）。"""

    return abs((angles - reference + period * 0.5) % period - period * 0.5)  # 平移半周期取模再折回，得到 [-period/2, period/2] 的绕对值。


def periodic_angle_from_components(cosine: float, sine: float, plain_mean: float, period: float) -> float:
    """由周期域的加权 cos/sin 合向量还原加权平均角。

    改造前这里写的是 ``np.sum(weights * np.exp(1j * radians))`` 再取 ``np.angle``；复数 dtype 是
    NumPy 专有的，数组门面上没有对应算子。拆成实部/虚部分别累加后数学上完全等价：
    ``abs(vector)`` 就是 ``hypot(cosine, sine)``，``np.angle(vector)`` 就是 ``atan2(sine, cosine)``。
    分量在后端上算（``ParticleShapeTracker._estimate_backend``），本函数只处理下载回来的标量。
    """

    if math.hypot(cosine, sine) < 1e-8:  # 合向量相互抵消，方向无意义。
        return float(plain_mean % period)  # 退化为算术平均。
    return float((math.atan2(sine, cosine) * period / (2.0 * math.pi)) % period)  # 取合向量方向并映射回周期域。


def estimate_detection_angle(
    template: ShapeTemplate, contour: np.ndarray
) -> tuple[float, float]:
    """循环 Procrustes 对齐：求新轮廓相对模板的旋转角与形状差异距离。"""

    candidate = resample_closed_contour(contour, len(template.points))  # 按模板点数等周重采样，保证点数一致。
    candidate -= np.mean(candidate, axis=0)  # 去质心，消除平移影响。
    candidate_scale = max(1e-6, float(np.sqrt(np.mean(candidate**2))))  # 候选轮廓的均方根半径，用于消除缩放影响。
    template_scale = max(1e-6, float(np.sqrt(np.mean(template.points**2))))  # 模板的均方根半径。
    candidate_complex = (candidate[:, 0] + 1j * candidate[:, 1]) / candidate_scale  # 候选点转复数并归一化尺度。
    template_complex = (  # 模板点转复数并归一化尺度。
        template.points[:, 0] + 1j * template.points[:, 1]
    ) / template_scale
    best_correlation = 0j  # 记录最优复相关值，其幅角即为旋转角。
    best_strength = -1.0  # 记录最优相关强度。
    for direction in (candidate_complex, candidate_complex[::-1]):  # 正反两个环绕方向都要试（轮廓起点/方向未知）。
        for shift in range(len(direction)):  # 遍历全部循环移位，对齐起点。
            correlation = np.vdot(template_complex, np.roll(direction, shift))  # 复数内积即相关值。
            strength = abs(correlation)  # 相关强度。
            if strength > best_strength:  # 找到更强的对齐。
                best_strength = strength  # 更新强度。
                best_correlation = correlation  # 更新相关值。
    angle = math.degrees(math.atan2(best_correlation.imag, best_correlation.real))  # 相关值幅角就是旋转角（度）。
    normalized_strength = best_strength / max(1.0, len(template.points) * 2.0)  # 按点数归一化相关强度。
    distance = float(np.clip(1.0 - normalized_strength, 0.0, 1.0))  # 强度越高距离越小。
    return angle % template.symmetry_period, distance  # 角度折回对称周期域，同时返回形状距离。


def choose_template_detection(detections: list[ShapeDetection]) -> ShapeDetection:
    """从选窗窗口内的检测序列里挑一条最稳定的轮廓学模板：先按面积中位数挡住双向离群，再取组内置信度最高。

    小目标开局的首帧常被倒计时数字粘连桥拽大（面积虚高），个别帧又被数字足迹削矮（面积偏小），
    都是窗口里的少数帧；面积中位数天然剔除双向畸形，置信度（白度/亮度/实心度/延展度加权）
    在正常形态组内越高说明轮廓越干净。
    """

    areas = sorted(item.area for item in detections)  # 面积排序，准备取中位数。
    mid_index = len(areas) // 2  # 中位数下标。
    median_area = 0.5 * (areas[mid_index] + areas[mid_index - 1]) if len(areas) % 2 == 0 else areas[mid_index]  # 中位面积。
    near = [item for item in detections if 0.80 * median_area <= item.area <= 1.25 * median_area]  # 贴近中位的正常形态组（粘连/被削的离群帧落在组外）。
    if not near:  # 容差带内一个都没有（面积全在跳变）。
        near = list(detections)  # 退回全集，由置信度兜底选优。
    return max(near, key=lambda item: item.confidence)  # 组内置信度最高的一条。


def choose_shape_candidate(
    candidates: list[ShapeDetection],
    template: ShapeTemplate | None,
    center: np.ndarray | None,
    confidence: float,
    size_floor: float = 0.0,
) -> ShapeDetection | None:
    """从白色候选里挑出最可能是被跟踪目标的那一个。

    距离窗口按 ``max(模板标称尺寸, size_floor)`` 缩放：小目标（如小面积星星）若继续
    按自身尺寸等比缩窗，目标稍微一跳就落在窗外被丢，颜色强测量白白浪费。
    """

    if not candidates:  # 本帧没有任何白色候选。
        return None  # 返回空。
    if template is None or center is None:  # 还没学到模板，或没有位置先验。
        return candidates[0]  # 直接取排序第一（面积*置信度最大）的候选，用于初始化学习。
    size_ref = max(float(template.nominal_size), float(size_floor or 0.0))  # 阈值基准：小目标抬到绝对下限。
    plausible: list[tuple[float, ShapeDetection]] = []  # 收集通过尺度/距离窗口筛选的候选及其得分。
    allowed_distance = size_ref * (1.5 if confidence >= 0.25 else 3.5)  # 置信度高时窗口收紧，丢失后放宽。
    for candidate in candidates:  # 逐个候选判定。
        scale = candidate.nominal_size / max(template.nominal_size, 1.0)  # 相对模板的尺度比。
        if not 0.68 <= scale <= 1.38:  # 尺度差太多，不可能是同一个目标。
            continue  # 丢弃。
        distance = float(np.linalg.norm(candidate.center - center))  # 与位置先验的距离。
        if distance > allowed_distance:  # 超出允许窗口。
            continue  # 丢弃。
        angle, shape_distance = estimate_detection_angle(template, candidate.contour)  # 对齐求角度与形状距离。
        candidate.angle = angle  # 写回候选，供 observe_color 使用。
        candidate.shape_distance = shape_distance  # 写回形状距离。
        match = float(cv2.matchShapes(candidate.contour, template.points.reshape(-1, 1, 2), cv2.CONTOURS_MATCH_I1, 0.0))  # Hu 矩形状相似度，越小越像。
        score = (  # 综合打分：置信度为主，距离/形状差为惩罚。
            candidate.confidence  # 白色检测置信度。
            - 0.18 * distance / max(size_ref, 1.0)  # 归一化距离惩罚（与窗口同基准，小目标不被惩罚主导）。
            - 0.16 * min(match, 2.0)  # Hu 矩形状惩罚，截断避免异常值主导。
            - 0.10 * shape_distance  # 循环对齐残差惩罚。
        )
        plausible.append((score, candidate))  # 记录候选。
    return max(plausible, key=lambda item: item[0])[1] if plausible else None  # 取得分最高者；全被筛掉则返回 None。


class DenseTemporalAligner(TensorTemporalAligner):
    """按后端装配的时序证据对齐器：只做装配，算法本体在 ``tensor_evidence``。

    - 显卡后端：``TorchFarnebackFlow``，帧缓冲常驻显存，全部 lag 合成一个 batch 一次算完；
    - CPU 后端：``Cv2DisFlow``，与改造前的 ``cv2.DISOpticalFlow`` 逐位一致（打包版行为零变化）。

    跨设备算法不同（Farneback vs DIS）是刻意取舍：DIS 在 CPU 上 3.06ms/次、Farneback 要 7.7ms/次，
    三个 lag 会吃掉 30fps 的全部预算；显卡侧反过来，Farneback 能用张量算子合批，DIS 没有可用的 GPU 实现。

    显卡运行期异常（显存不足/驱动重置）按打分内核同样的语义降级：记一次显卡失败并当场换成
    CPU DIS 对齐器接管后续帧，连续 3 次后本进程永久回退 NumPy。降级会丢掉显存里的历史帧，
    因此紧接的 ``max(lags)`` 帧返回 None，会话侧退化为 prediction 而不是报错。
    """

    def __init__(  # 构造对齐器。
        self,
        lags: tuple[int, ...],  # 使用的时间基线（帧间隔），例如 (1, 2, 4)。
        play_height_ratio: float,  # 有效高度比例，其下部分在证据图里清零。
        preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,  # CPU 引擎的 DIS 精度档位。
        backend=None,  # 打分后端；None 表示取全局生效后端。
        flow_params: dict | None = None,  # 显卡 Farneback 的光流参数，缺省项由引擎默认值（= medium 档）补齐。
    ):
        chosen = _resolve_backend(backend)  # 本次生效的后端。
        params = dict(flow_params or {})  # 参数副本：不污染调用方的 dict。
        if chosen.is_gpu:  # 显卡：torch Farneback，并挂上 CPU 降级器。
            super().__init__(
                lags,
                play_height_ratio,
                TorchFarnebackFlow(chosen.xp, **params),
                chosen,
                fallback=lambda: self._degrade_to_cpu(lags, play_height_ratio, preset),
            )
        else:  # CPU：DIS，行为与改造前一致，不需要降级器。
            super().__init__(lags, play_height_ratio, Cv2DisFlow(preset), numpy_backend())

    @staticmethod
    def _degrade_to_cpu(lags, play_height_ratio, preset):  # 显卡链路异常时建一个 CPU DIS 对齐器接管。
        mark_gpu_failure()  # 计数，连续 3 次后本进程永久回退 NumPy（与打分内核同一套语义）。
        return TensorTemporalAligner(lags, play_height_ratio, Cv2DisFlow(preset), numpy_backend())


def sample_map(image: np.ndarray, x: np.ndarray, y: np.ndarray, backend=None) -> np.ndarray:
    """在单通道证据图上做最近邻批量采样，越界位置返回 0（双后端薄壳，实现在 gpu_shape_backend）。"""

    chosen = _resolve_backend(backend)  # 解析本次生效的后端。
    return _to_numpy_array(sample_map_core(chosen.xp, image, x, y))  # 后端采样后统一转回 NumPy。


def score_rotated_borders(
    evidence: np.ndarray, states: np.ndarray, side: float, play_height: int, backend=None
) -> BorderEvidence:
    """矩形模型专用：给一批位姿假设的四条薄边打分（双后端薄壳，实现在 gpu_shape_backend）。"""

    return _run_scoring(score_rotated_borders_core, backend, evidence, states, side, play_height)  # 统一调度与降级。


def transform_template_points(
    template: ShapeTemplate, states: np.ndarray, scales: np.ndarray | float, backend=None
) -> tuple[np.ndarray, np.ndarray]:
    """把模板轮廓点与法向批量旋转缩放到每个粒子假设的当前姿态（双后端薄壳，实现在 gpu_shape_backend）。"""

    chosen = _resolve_backend(backend)  # 解析本次生效的后端。
    points, normals = transform_template_points_core(chosen.xp, template, states, scales)  # 后端上变换。
    return _to_numpy_array(points), _to_numpy_array(normals)  # 统一转回 NumPy。


def score_shape_contours(
    evidence: np.ndarray,
    states: np.ndarray,
    template: ShapeTemplate,
    scales: np.ndarray | float,
    play_height: int,
    backend=None,
) -> ShapeEvidence:
    """给一批粒子假设打分：联合轮廓边界对比度与半透明区域内部残差（双后端薄壳，实现在 gpu_shape_backend）。"""

    return _run_scoring(score_shape_contours_core, backend, evidence, states, template, scales, play_height)  # 统一调度与降级。


def score_shape_contours_on_backend(
    evidence,
    states,
    template,
    scales,
    play_height,
    backend=None,
) -> ShapeEvidence:
    """与 :func:`score_shape_contours` 同一套调度与降级，但结果**留在后端设备上**。

    粒子滤波每帧要拿得分继续做加权/重采样，全部在设备上算完后再一次性下载决策标量，
    因此不能像薄壳那样每次打分都把 (count,) 数组拽回主存。打分降级到 NumPy 时返回的就是
    NumPy 数组，调用方用 ``backend.to_backend`` 归位即可，两个方向都由门面的 ``asarray`` 兜住。
    """

    return _run_scoring(  # 复用同一套调度，只把下载关掉。
        score_shape_contours_core, backend, evidence, states, template, scales, play_height, to_numpy=False
    )


class ParticleShapeTracker:
    """多假设粒子跟踪器：6 维状态 (x, y, vx, vy, angle, 角速度) + 权重。

    强测量（白色轮廓可见）走 observe_color 直接修正；
    弱测量（目标已透明）走 observe_border 靠时序残差证据加权，
    并对单帧过大的修正整体回滚，避免被背景强轮廓拖跑。
    """

    def __init__(  # 构造跟踪器。
        self,
        detection: ShapeDetection,  # 初始化用的首个可靠白色检测。
        template: ShapeTemplate,  # 已学习的形状模板。
        particle_count: int,  # 粒子数量。
        frame_width: int,  # 处理尺度下的帧宽。
        play_height: int,  # 处理尺度下的有效高度。
        global_proposals: int,  # 重定位第一级粗扫候选数（实时档 400，远低于旧版 2400）。
        rng: np.random.Generator,  # 随机数发生器，每局独立以保证可复现。
        max_correction_ratio: float = 0.24,  # 单帧证据修正上限（相对形状尺寸的比例）。
        max_speed_ratio: float = 0.16,  # 单帧中心移动速度上限（相对形状尺寸的比例）。
        control_count: int = 260,  # 对照组数量：随机位姿，用于估计背景得分分布。
        coarse_top: int = 30,  # 粗扫后进入精扫的 top-K 邻域数。
        refine_per_top: int = 8,  # 每个粗扫邻域生成的精扫拖尾候选数。
        relocation_cooldown: int = 15,  # 两次全局重定位之间的最小间隔帧数（限频）。
        min_size_floor: float = 0.0,  # 阈值类尺寸的绝对下限（处理尺度像素）：小目标不再把窗口/限速/抖动等比缩窄。
        backend=None,  # 打分后端（gpu_shape_backend.ShapeScoreBackend），None 时用 NumPy 后端。
    ):
        self.template = template  # 保存模板。
        self.backend = numpy_backend() if backend is None else backend  # 打分后端：会话装配时注入，缺省 NumPy（有 N 卡时为 torch CUDA）。
        self.xp = self.backend.xp  # 数组门面：粒子群的全部向量运算都走它，CPU 与显卡共用同一份代码。
        self.count = particle_count  # 保存粒子数。
        self.frame_width = frame_width  # 保存帧宽。
        self.play_height = play_height  # 保存有效高度。
        self.global_proposals = global_proposals  # 保存粗扫候选数。
        self.rng = rng  # 保存随机数发生器。
        self.max_correction_ratio = max_correction_ratio  # 保存单帧修正上限比例。
        self.max_speed_ratio = max_speed_ratio  # 保存限速比例。
        self.control_count = control_count  # 保存对照组数量。
        self.coarse_top = coarse_top  # 保存粗扫 top-K。
        self.refine_per_top = refine_per_top  # 保存每邻域精扫候选数。
        self.relocation_cooldown = relocation_cooldown  # 保存重定位冷却帧数。
        # 阈值基准尺寸：几何类（粒子扩散/边界余量/尺度测量）仍用模板真实尺寸，
        # 窗口/限速/半径/抖动/似然宽度这类阈值用 size_ref，小目标抬到绝对下限，
        # 否则 nominal_size 越小重定位越容易失败，恰好是小星星难解的主因之一。
        self.size_ref = max(float(template.nominal_size), float(min_size_floor or 0.0), 4.0)  # 下限 4 像素防除零。
        self.scale = 1.0  # 当前估计的目标缩放，初值 1。
        # 随机数一律在 NumPy 侧生成（保住 random_seed 的可复现契约），整块搬上设备后不再回主存。
        states = np.zeros((particle_count, 6), dtype=np.float32)  # 粒子状态矩阵。
        states[:, :2] = detection.center + rng.normal(  # 位置围绕检测中心小幅扩散。
            0.0, template.nominal_size * 0.035, size=(particle_count, 2)  # 扩散标准差为标称尺寸的 3.5%。
        )
        states[:, 2:4] = rng.normal(0.0, 1.2, size=(particle_count, 2))  # 初速度零均值随机，标准差 1.2 像素/帧。
        states[:, 4] = rng.normal(0.0, 2.5, particle_count) % template.symmetry_period  # 初始角度围绕 0 小范围扰动后折回对称周期域。
        states[:, 5] = rng.normal(0.0, 0.6, particle_count)  # 初始角速度零均值随机。
        self.states = self.backend.to_backend(states)  # 状态常驻后端设备（float32）。
        # 权重用 float64：归一化后要反复乘除，float32 在 700 个粒子上会累积出可见偏差；
        # 显卡双精度在这个规模上不是瓶颈（每帧只有几个千元级的归约）。
        self.weights = self.backend.to_backend(np.full(particle_count, 1.0 / particle_count, dtype=np.float64))  # 权重均匀初始化。
        self.confidence = detection.confidence  # 初始置信度取白色检测置信度。
        self.last_reliable_center = detection.center.astype(np.float32).copy()  # 最后一个已确认可靠的中心。
        self.pending_reliable_center = self.last_reliable_center.copy()  # 正在累积命中次数的待定中心。
        self.pending_reliable_hits = 3  # 初始已有 3 次命中（白色检测本身很可靠）。
        self.frames_since_reliable = 0  # 距上次可靠估计的帧数。
        self.frames_since_relocation = relocation_cooldown  # 距上次重定位的帧数，初始即满允许首次立即重定位。
        self.last_border_score = 0.0  # 上一次边界加权得分，供诊断输出。
        self.last_border_snr = 0.0  # 上一次边界信噪比，供诊断输出。
        self.search_radius = self.size_ref * 0.5  # 当前搜索半径，供诊断输出。

    def _estimate_device(self):  # 在后端设备上算出全部加权估计量，不做任何主存同步。
        """返回 ``(center, velocity, cosine, sine, plain_mean)``，全部是设备上的张量。

        ``cosine``/``sine`` 是周期域加权合向量的两个分量：改造前这里写的是
        ``np.sum(weights * np.exp(1j * radians))``，复数 dtype 是 NumPy 专有的，拆成实部/虚部
        分别累加后与 ``abs(vector)`` / ``np.angle(vector)`` 完全等价。
        """

        xp = self.xp
        column_weights = xp.stack([self.weights], axis=1)  # (count,1)，供位置/速度加权。
        center = xp.sum(xp.narrow(self.states, 1, 0, 2) * column_weights, axis=0)  # 位置加权均值 (2,)。
        velocity = xp.sum(xp.narrow(self.states, 1, 2, 2) * column_weights, axis=0)  # 速度加权均值 (2,)。
        period = self.template.symmetry_period  # 对称周期。
        radians = xp.column(self.states, 4) * (2.0 * math.pi / period)  # 把周期域角度映射到 0~2pi。
        cosine = xp.sum(self.weights * xp.cos(radians))  # 合向量实部。
        sine = xp.sum(self.weights * xp.sin(radians))  # 合向量虚部。
        plain_mean = xp.astype(xp.mean(xp.column(self.states, 4)), xp.float64)  # 合向量抵消时的算术均值兜底。
        return center, velocity, cosine, sine, plain_mean

    def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """输出加权估计：中心、速度与周期域加权平均角（一次 D2H 取回全部标量）。"""

        xp = self.xp
        center, velocity, cosine, sine, plain_mean = self._estimate_device()  # 全部在设备上算完。
        period = self.template.symmetry_period  # 对称周期。
        packed = xp.concatenate([center, velocity, xp.stack([cosine, sine, plain_mean])])  # 攒成一个 (7,) 张量。
        values = xp.to_numpy(packed)  # 本函数唯一一次 D2H，避免逐标量同步。
        angle = periodic_angle_from_components(float(values[4]), float(values[5]), float(values[6]), period)  # 标量还原加权平均角。
        return values[:2].astype(np.float32), values[2:4].astype(np.float32), angle  # 返回三个估计量。

    def _apply_bounds(self, states):  # 施加边界、限速、角度取模与角速度截断约束，返回新状态。
        """与改造前的布尔掩码就地赋值等价，但只用门面的 ``where``，因此显卡上也能跑。

        改造前写的是 ``states[left, 0] = margin`` 这类花式索引赋值；``where`` 版本逐元素选出
        同一批值，数值结果一致，且不要求数组可写、不打断 CUDA Graph 捕获。
        """

        xp = self.xp
        margin = self.template.nominal_size * 0.50 * self.scale  # 边距为半个形状，避免中心贴到画面边缘。
        max_x = self.frame_width - margin  # x 上限。
        max_y = self.play_height - margin  # y 上限（以有效高度为准）。
        pos_x, pos_y, vel_x, vel_y, angle, spin = [xp.narrow(states, 1, index, 1) for index in range(6)]  # 六列各 (count,1)。
        left = pos_x < margin  # 越左界的粒子。
        right = pos_x > max_x  # 越右界的粒子。
        top = pos_y < margin  # 越上界的粒子。
        bottom = pos_y > max_y  # 越下界的粒子。
        pos_x = xp.where(left, xp.full_like(pos_x, margin), pos_x)  # 钳回左边界。
        vel_x = xp.where(left, xp.maximum(vel_x, 0.0), vel_x)  # 左界处不允许继续向左的速度。
        pos_x = xp.where(right, xp.full_like(pos_x, max_x), pos_x)  # 钳回右边界。
        vel_x = xp.where(right, xp.minimum(vel_x, 0.0), vel_x)  # 右界处不允许继续向右的速度。
        pos_y = xp.where(top, xp.full_like(pos_y, margin), pos_y)  # 钳回上边界。
        vel_y = xp.where(top, xp.maximum(vel_y, 0.0), vel_y)  # 上界处不允许继续向上的速度。
        pos_y = xp.where(bottom, xp.full_like(pos_y, max_y), pos_y)  # 钳回下边界。
        vel_y = xp.where(bottom, xp.minimum(vel_y, 0.0), vel_y)  # 下界处不允许继续向下的速度。
        max_speed = self.size_ref * self.max_speed_ratio  # 单帧最大位移（小目标抬到下限，不被限速锁死）。
        speed = xp.linalg_norm(xp.concatenate([vel_x, vel_y], axis=1), axis=1)  # 当前速度大小 (count,)。
        too_fast = speed > max_speed  # 超速的粒子。
        # 超速的按 max_speed/speed 等比缩放（保留方向），其余乘 1 保持原样；
        # 分母取 max(speed, 1e-9) 只为避开 where 两个分支都要算带来的除零，超速分支里 speed 远大于它。
        ratio = xp.where(too_fast, max_speed / xp.maximum(speed, 1e-9), 1.0)
        factor = xp.astype(xp.stack([ratio], axis=1), xp.float32)  # (count,1)，强制 float32 以与状态列同精度。
        vel_x = vel_x * factor  # 限速后的 x 速度。
        vel_y = vel_y * factor  # 限速后的 y 速度。
        angle = angle % self.template.symmetry_period  # 角度折回对称周期域（两套门面的 % 同为 floor-mod）。
        spin = xp.clip(spin, -6.0, 6.0)  # 角速度截断，避免无限累积。
        return xp.astype(xp.concatenate([pos_x, pos_y, vel_x, vel_y, angle, spin], axis=1), xp.float32)  # 状态恒为 float32。

    def propagate(self) -> None:
        """粒子推进一步：按当前速度平移，并按「1 - 置信度」放大扩散噪声。"""

        xp = self.xp
        uncertainty = 1.0 - float(np.clip(self.confidence, 0.0, 1.0))  # 不确定度：置信度越低扩散越大。
        # 噪声整块在 NumPy 侧生成后一次上传：随机数契约（random_seed=20260902 可复现）只在 NumPy 侧成立，
        # 且顺序与改造前逐列 rng.normal 完全一致，因此同种子下噪声逐位相同。
        noise = np.zeros((self.count, 6), dtype=np.float32)  # 六列噪声容器。
        noise[:, :2] = self.rng.normal(0.0, 0.45 + 2.8 * uncertainty, size=(self.count, 2))  # 位置扩散：基准 0.45，不确定度满时额外 2.8。
        noise[:, 2:4] = self.rng.normal(0.0, 0.18 + 0.85 * uncertainty, size=(self.count, 2))  # 速度随机游走：基准 0.18。
        noise[:, 4] = self.rng.normal(0.0, 0.35 + 2.0 * uncertainty, self.count)  # 角度扩散：基准 0.35 度。
        noise[:, 5] = self.rng.normal(0.0, 0.20 + 0.30 * uncertainty, self.count)  # 角速度随机游走：基准 0.20。
        columns = [xp.narrow(self.states, 1, index, 1) for index in range(6)]  # 当前六列。
        # 确定性推进用「加噪声之前」的速度/角速度，与改造前「先 += 速度列、再 += 噪声」的先后顺序一致。
        advanced = [
            columns[0] + columns[2],  # x += vx。
            columns[1] + columns[3],  # y += vy。
            columns[2],  # vx 本身不做确定性推进。
            columns[3],  # vy 同上。
            columns[4] + columns[5],  # angle += 角速度。
            columns[5],  # 角速度本身不做确定性推进。
        ]
        self.states = xp.astype(xp.concatenate(advanced, axis=1) + xp.asarray(noise), xp.float32)  # 推进 + 噪声，一次上传。
        self.states = self._apply_bounds(self.states)  # 推进后统一施加约束。
        self.frames_since_relocation += 1  # 重定位冷却计数 +1。

    def observe_color(self, detection: ShapeDetection) -> None:
        """强测量：白色轮廓仍然可见，直接用检测结果修正粒子群与尺度。"""

        xp = self.xp
        with xp.inference():  # 显卡侧关掉自动求导。
            measured_center = xp.asarray(detection.center)  # 检测中心 (2,)，一次 8 字节 H2D。
            positions = xp.narrow(self.states, 1, 0, 2)  # 当前粒子位置 (count,2)。
            distance = xp.linalg_norm(positions - measured_center, axis=1)  # 每个粒子到检测中心的距离。
            angles = xp.column(self.states, 4)  # 当前粒子角度 (count,)。
            angle_distance = periodic_angle_difference(  # 周期域角度偏差。
                angles, detection.angle, self.template.symmetry_period  # 粒子角度 vs 检测角度。
            )
            position_sigma = max(5.0, self.size_ref * 0.22)  # 位置似然标准差，随阈值基准尺寸自适应（小目标有下限）。
            angle_sigma = max(8.0, self.template.symmetry_period * 0.15)  # 角度似然标准差，随对称周期自适应。
            likelihood = xp.exp(-0.5 * (distance / position_sigma) ** 2)  # 位置高斯似然。
            # 对称/接近圆形的轮廓携带的角度信息很弱，必须降低角度项权重。
            angle_weight = 0.35 if self.template.symmetry_period <= 60.0 else 0.75  # 周期≤ 60 度时降权到 0.35。
            likelihood = likelihood * ((1.0 - angle_weight) + angle_weight * xp.exp(  # 角度高斯似然按权重混合进去。
                -0.5 * (angle_distance / angle_sigma) ** 2  # 角度高斯项。
            ))
            self.weights = self.weights * xp.astype(likelihood + 1e-9, xp.float64)  # 权重乘似然，加极小量避免全零（权重恒为 float64）。
            self._normalize_weights()  # 归一化。
            innovation = measured_center - positions  # 位置新息（测量 - 预测）。
            shift_x = xp.narrow(innovation, 1, 0, 1)  # 新息 x 分量 (count,1)。
            shift_y = xp.narrow(innovation, 1, 1, 1)  # 新息 y 分量 (count,1)。
            columns = [xp.narrow(self.states, 1, index, 1) for index in range(6)]  # 修正前的六列快照。
            period = self.template.symmetry_period  # 对称周期。
            angular_innovation = (  # 角度新息，必须走周期域最短弧。
                detection.angle  # 测量角度。
                - angles  # 减去粒子角度。
                + period * 0.5  # 平移半周期。
            ) % period - period * 0.5  # 取模后折回，得到最短角差。
            corrected_angle = (  # 角度按权重修正，并折回对称周期域。
                angles + 0.28 * angle_weight * angular_innovation  # 修正量同时受对称性降权影响。
            ) % period  # 取模保证角度始终在周期域内。
            # 位置强修正（快速贴向测量）与速度弱修正（保留惯性）共用同一份修正前的新息，
            # 与改造前两句就地 += 的数值语义一致。
            self.states = xp.astype(xp.concatenate([
                columns[0] + 0.52 * shift_x,  # x 位置强修正。
                columns[1] + 0.52 * shift_y,  # y 位置强修正。
                columns[2] + 0.14 * shift_x,  # vx 弱修正。
                columns[3] + 0.14 * shift_y,  # vy 弱修正。
                xp.stack([corrected_angle], axis=1),  # 修正后的角度。
                columns[5],  # 角速度不变。
            ], axis=1), xp.float32)
            measured_scale = detection.nominal_size / max(self.template.nominal_size, 1.0)  # 测量到的相对尺度。
            if 0.86 <= measured_scale <= 1.16:  # 只接受合理范围内的尺度测量。
                self.scale = float(np.clip(0.94 * self.scale + 0.06 * measured_scale, 0.86, 1.16))  # 低通滤波平滑尺度，并限制上下限。
            self.states = self._apply_bounds(self.states)  # 修正后重新施加约束。
        self.confidence = max(self.confidence * 0.55, detection.confidence)  # 置信度：旧值衰减后与测量值取大。
        self.last_reliable_center = detection.center.astype(np.float32).copy()  # 白色检测就是可靠中心。
        self.pending_reliable_center = self.last_reliable_center.copy()  # 待定中心同步。
        self.pending_reliable_hits = 3  # 重置为 3，再命中一次即达 4 可确认。
        self.frames_since_reliable = 0  # 可靠计数归零。
        self.last_border_score = 0.0  # 清零边界诊断值。
        self.last_border_snr = 0.0  # 清零信噪比诊断值。
        self.search_radius = self.size_ref * 0.5  # 搜索半径回到默认值。
        self._resample_if_needed(force=False)  # 按需重采样，不强制。

    def _proposal_states(self, reference_center: np.ndarray, velocity: np.ndarray, current_angle: float) -> np.ndarray:
        """生成全局重定位候选位姿：30% 局部圆域 + 70% 全画面网格×角度离散。

        整段留在 NumPy：候选生成要消耗随机数，而随机数的可复现契约只在 NumPy 侧成立。
        返回的 ``(proposal_count, 6)`` 数组由调用方一次上传到后端并施加边界约束；
        ``velocity`` / ``current_angle`` 也从调用方传入（取自 :meth:`estimate`），
        这样整次重定位只需一次 D2H，不必在候选生成里再同步一遍。
        """

        proposal_count = self.global_proposals  # 候选总数。
        lost = max(1, self.frames_since_reliable)  # 已丢失帧数，下限 1 避免除零。
        max_speed = self.size_ref * self.max_speed_ratio  # 单帧最大位移（与限速约束同基准）。
        radius = min(  # 局部搜索半径：丢失越久半径越大，但不超过画面对角线。
            math.hypot(self.frame_width, self.play_height),  # 上限：画面对角线。
            self.size_ref * 0.55 + lost * max_speed,  # 基础半径 + 丢失期间可走的最大距离。
        )
        self.search_radius = radius  # 记录搜索半径供诊断输出。
        local_count = max(120, int(proposal_count * 0.30))  # 局部候选数，至少 120 个。
        if local_count > proposal_count:  # 总候选数比局部下限还小（测试/低配置场景）。
            local_count = proposal_count  # 全部给局部。
        global_count = proposal_count - local_count  # 剩下的给全画面网格。
        directions = self.rng.uniform(0.0, 2.0 * math.pi, local_count)  # 局部候选的方向角均匀采样。
        distances = radius * np.sqrt(self.rng.random(local_count))  # 开方使圆域内位置分布均匀（而非向圆心聚集）。
        local_positions = reference_center + np.column_stack(  # 极坐标转直角坐标得到局部候选位置。
            [np.cos(directions) * distances, np.sin(directions) * distances]  # x/y 偏移。
        )

        period = self.template.symmetry_period  # 对称周期。
        angle_count = max(3, int(math.ceil(period / 30.0)))  # 周期域内离散的角度个数，至少 3 个。
        angle_variants = np.linspace(0.0, period, angle_count, endpoint=False).astype(  # 均匀分布的角度候选。
            np.float32  # 统一精度。
        )
        lattice_points = max(1, math.ceil(global_count / angle_count))  # 每个角度需要的网格点数。
        margin = self.template.nominal_size * 0.50  # 网格边距，与 _apply_bounds 保持一致。
        aspect = max(  # 可用区域的长宽比，用于算出接近正方形的网格划分。
            0.5,  # 下限避免过于狭长。
            (self.frame_width - 2 * margin)  # 可用宽度。
            / max(1.0, self.play_height - 2 * margin),  # 可用高度，下限防除零。
        )
        columns = max(1, math.ceil(math.sqrt(lattice_points * aspect)))  # 网格列数。
        rows = max(1, math.ceil(lattice_points / columns))  # 网格行数。
        cell_width = (self.frame_width - 2 * margin) / columns  # 单元格宽。
        cell_height = (self.play_height - 2 * margin) / rows  # 单元格高。
        grid_x = margin + (np.arange(columns, dtype=np.float32) + 0.5) * cell_width  # 各列中心 x。
        grid_y = margin + (np.arange(rows, dtype=np.float32) + 0.5) * cell_height  # 各行中心 y。
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)  # 展开成网格。
        lattice = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])[:lattice_points]  # 取前 lattice_points 个网格点。
        global_positions = np.repeat(lattice, angle_count, axis=0)[:global_count] if global_count > 0 else np.zeros((0, 2), dtype=np.float32)  # 每个网格点配上全部角度候选，再截到 global_count；为 0 时给空数组。
        positions = np.vstack([local_positions, global_positions]) if global_count > 0 else local_positions  # 合并局部与全局位置；全局为空时只用局部。

        states = np.zeros((proposal_count, 6), dtype=np.float32)  # 候选状态矩阵。
        states[:, :2] = positions  # 写入位置。
        states[:, 2:4] = 0.55 * (  # 速度主项：由「候选位置相对参考中心的位移 / 丢失帧数」推出的隐含速度。
            (positions - reference_center) / max(lost, 1)  # 平均到每帧。
        ) + 0.45 * velocity  # 副项：保留当前速度估计。
        states[:, 2:4] += self.rng.normal(0.0, 1.5, size=(proposal_count, 2))  # 速度加噪声，避免候选过于集中。
        states[:local_count, 4] = (  # 局部候选：围绕当前角度扰动。
            current_angle + self.rng.normal(0.0, max(8.0, period * 0.12), local_count)  # 扰动标准差随周期自适应。
        ) % period  # 折回周期域。
        global_angles = np.tile(angle_variants, lattice_points)[:global_count]  # 全局候选：角度均匀铺满周期域。
        states[local_count:, 4] = (  # 写入全局候选角度。
            global_angles + self.rng.normal(0.0, min(3.0, period * 0.04), global_count)  # 加微小抖动，但保留网格覆盖性。
        ) % period  # 折回周期域。
        states[:, 5] = self.rng.normal(0.0, 1.5, proposal_count)  # 角速度随机初始化。
        return states  # 返回全部候选位姿（边界约束改由调用方在后端上施加，省一次主存往返）。

    def _relocate(  # 两级粗到细全局重定位。
        self,
        temporal_map,  # 时序残差证据图（CPU 时是 NumPy 数组，显卡时是显存张量）。
        particle_scores,  # 当前粒子得分（后端数组），拉回的候选要同步写回。
        particle_coverage,  # 当前粒子覆盖率（后端数组），同上。
        control_median,  # 对照组得分中位数（后端 float64 标量）。
        control_sigma,  # 对照组鲁棒标准差（后端 float64 标量）。
    ):
        """第一级用有限候选粗扫全图（单尺度），第二级对 top-K 邻域拖尾细化（3 尺度）。
        替代旧版 2400×3 尺度一次性打分，把单次重定位从 ~240ms 降到 ~30ms。

        打分与排序全部留在后端设备上，只有两处必须回主存：
        一是粗扫 top-K 的基准位姿（拖尾抖动要消耗随机数，而随机数契约只在 NumPy 侧成立），
        二是精扫的 ``order + 候选位置 + best_snr + best_scale``——后者攒成一个张量一次下载，
        最小间距贪婪去重仍在 CPU 上按原算法跑（算法一字不改），选中的索引再上传回写。
        返回可能被覆写过的 ``(particle_scores, particle_coverage)``。
        """

        xp = self.xp  # 数组门面。
        scale_factors = (0.90, 1.0, 1.10)  # 精扫跑三个尺度。
        top_k = min(self.coarse_top, self.global_proposals)  # 进入精扫的邻域数，不超过粗扫候选总数。
        if top_k == 0:  # 没有任何候选（理论上不会出现），直接放弃重定位。
            return particle_scores, particle_coverage  # 保持冷却计数不变，下一帧再试。
        predicted_center, velocity, current_angle = self.estimate()  # 一次 D2H：候选生成需要主存侧的运动先验。
        predicted_device = xp.asarray(predicted_center)  # 预测中心的设备副本，供两次距离惩罚复用。
        coarse_states = self._apply_bounds(self.backend.to_backend(  # 候选在主存生成后一次上传，再施加边界/限速约束。
            self._proposal_states(predicted_center, velocity, current_angle)  # 第一级粗扫候选，数量 = global_proposals（实时档 400）。
        ))
        coarse_result = score_shape_contours_on_backend(  # 粗扫只在 1.0 尺度下打分，结果留在设备上。
            temporal_map, coarse_states, self.template, 1.0, self.play_height, self.backend  # 证据图、候选、模板、尺度、有效高度与打分后端。
        )
        prior_radius = self.size_ref * (  # 距离先验半径：丢失越久容忍越宽（小目标抬到下限，重定位不被先验锁死）。
            0.55 + 0.060 * min(self.frames_since_reliable, 10)  # 基础 0.55，每丢失一帧 +0.06，上限 10 帧。
        )
        coarse_distance = xp.linalg_norm(  # 候选到预测中心的距离。
            xp.narrow(coarse_states, 1, 0, 2) - predicted_device, axis=1  # 二维距离。
        )
        coarse_ranking = coarse_result.scores - 2.20 * (  # 排序得分 = 证据得分 - 平方距离惩罚。
            coarse_distance / max(prior_radius, 1.0)  # 归一化距离，下限防除零。
        ) ** 2
        top_indices = xp.narrow(xp.argsort(coarse_ranking, descending=True), 0, 0, top_k)  # 粗扫排序得分最高的 top-K 索引。
        bases = xp.to_numpy(xp.take(coarse_states, top_indices)).astype(np.float32)  # 一次 D2H：只把 top-K 行拉回主存做拖尾抖动。

        period = self.template.symmetry_period  # 对称周期。
        position_jitter = max(4.0, self.size_ref * 0.14)  # 位置抖动标准差：阈值基准的 14%（小目标有下限）。
        angle_jitter = max(6.0, period * 0.10)  # 角度抖动标准差：周期的 10%。
        refined_list: list[np.ndarray] = []  # 收集每个邻域的抖动候选组。
        for base in bases:  # 遍历粗扫 top-K 邻域（顺序即排序得分降序，随机数消耗次序与改造前一致）。
            variants = np.tile(base, (self.refine_per_top, 1))  # 复制基准位姿成一组。
            variants[:, :2] += self.rng.normal(  # 位置抖动。
                0.0, position_jitter, size=(self.refine_per_top, 2)  # x/y 独立抖动。
            )
            variants[:, 4] = (  # 角度抖动后折回周期域。
                base[4] + self.rng.normal(0.0, angle_jitter, self.refine_per_top)  # 基准角度 + 抖动。
            ) % period  # 取模保证在周期域内。
            refined_list.append(variants)  # 收集本邻域候选。
        proposals = np.vstack(refined_list).astype(np.float32)  # 合并成精扫候选集。
        proposal_rows = len(proposals)  # 精扫候选行数（= top_k * refine_per_top）。
        proposals_device = self._apply_bounds(self.backend.to_backend(proposals))  # 一次上传并统一施加边界约束。
        # 三个尺度的精扫合并成一次打分：候选按「尺度段 × 候选」排列，与旧版逐尺度拼接顺序一致，
        # 后续 ranking/selected 索引到 all_states 的映射语义不变。
        all_states = xp.concatenate([proposals_device] * len(scale_factors), axis=0)  # 同一批候选在三个尺度下各记一次。
        all_scales = xp.asarray(np.concatenate(  # 逐候选尺度：每段 proposals 对应一个尺度因子。
            [np.full(proposal_rows, float(np.clip(factor, 0.86, 1.16)), dtype=np.float32) for factor in scale_factors]  # 尺度限在合法区间内。
        ))
        refine_evidence = score_shape_contours_on_backend(  # 一次合批对全部尺度候选打分，结果留在设备上。
            temporal_map,  # 证据图。
            all_states,  # 三尺度合并候选。
            self.template,  # 模板。
            all_scales,  # 逐假设尺度。
            self.play_height,  # 有效高度。
            self.backend,  # 打分后端。
        )
        all_scores = refine_evidence.scores  # 全部候选得分。
        all_coverages = refine_evidence.coverage  # 全部候选覆盖率。
        top_count = min(self.count, max(42, self.count // 5))  # 要拉回的候选数，至少 42 个，但不超过粒子总数。
        # 重定位候选的排序要围绕当前运动预测，而不是围绕一个陈旧的已确认点。
        # 这样既能保持运动平滑，又能让远处的背景轮廓付出很高代价。
        distance_from_prediction = xp.linalg_norm(  # 候选到预测中心的距离。
            xp.narrow(all_states, 1, 0, 2) - predicted_device, axis=1  # 二维距离。
        )
        ranking_scores = all_scores - 2.20 * (  # 排序得分 = 证据得分 - 平方距离惩罚。
            distance_from_prediction / max(prior_radius, 1.0)  # 归一化距离，下限防除零。
        ) ** 2
        order = xp.argsort(ranking_scores, descending=True)  # 排序得分降序索引。
        best_slot = xp.stack([xp.argmax(ranking_scores)])  # 精扫集内最优候选索引，包成 (1,) 便于 take 出标量段。
        best_snr = (  # 最优候选的信噪比（float64，与改造前 float(...) 的精度一致）。
            xp.take(xp.astype(all_scores, xp.float64), best_slot) - control_median  # 最优得分减背景基线。
        ) / control_sigma  # 除以鲁棒标准差。
        # 本帧重定位的唯一一次批量 D2H：排序索引 + 精扫候选位置 + 最优信噪比 + 最优尺度。
        # 候选位置只取 proposals 的一段（三个尺度段共用同一批位置），下载量从 720x6 降到 240x2 加几个标量。
        payload = xp.concatenate([
            xp.astype(order, xp.float64),  # 排序索引段。
            xp.astype(xp.ravel(xp.narrow(proposals_device, 1, 0, 2)), xp.float64),  # 候选位置段（展平）。
            best_snr,  # 信噪比标量。
            xp.astype(xp.take(all_scales, best_slot), xp.float64),  # 最优尺度标量。
        ])
        host = xp.to_numpy(payload)  # 一次下载全部决策数据。
        total_rows = proposal_rows * len(scale_factors)  # 精扫候选总数。
        order_host = host[:total_rows].astype(np.int64)  # 排序索引（主存）。
        positions_host = host[total_rows : total_rows + proposal_rows * 2].reshape(-1, 2).astype(np.float32)  # 候选位置（主存）。
        best_snr_host = float(host[-2])  # 最优信噪比（主存）。
        best_scale_host = float(host[-1])  # 最优尺度（主存）。
        all_positions = np.tile(positions_host, (len(scale_factors), 1))  # 还原 all_states 的位置列，与设备侧逐行一致。
        selected: list[int] = []  # 已选中的候选索引。
        minimum_separation = self.template.nominal_size * 0.24  # 候选之间的最小间距，避免拉回一堆重复位姿。
        for candidate in order_host:  # 按排序依次尝试选取。
            if all(  # 与所有已选候选的距离都达标准。
                np.linalg.norm(  # 两点距离。
                    all_positions[candidate] - all_positions[chosen]  # 当前候选 vs 已选候选。
                )
                >= minimum_separation  # 不小于最小间距。
                for chosen in selected  # 遍历已选集合。
            ):
                selected.append(int(candidate))  # 选入。
                if len(selected) >= top_count:  # 已经选够。
                    break  # 提前结束循环。
        if len(selected) < top_count:  # 去重后不够 top_count 个。
            used = set(selected)  # 已用索引集合。
            selected.extend(int(item) for item in order_host if int(item) not in used)  # 按排序补齐剩下的名额。
        top = np.asarray(selected[:top_count], dtype=np.int64)  # 最终拉回的候选索引。
        replace = xp.narrow(xp.argsort(self.weights), 0, 0, top_count)  # 权重最低的 top_count 个粒子被替换。
        weight_median = xp.median(self.weights)  # 新粒子给中位数权重，必须在覆写之前算出来。
        replaced_weights = xp.take(self.weights, replace)  # 被替换粒子的旧权重，只当 zeros_like 的形状/dtype 模板用。
        self.states = xp.put_rows(self.states, replace, xp.take(all_states, top))  # 用高分候选替换低权重粒子的状态。
        particle_scores = xp.put_rows(particle_scores, replace, xp.take(all_scores, top))  # 同步替换得分。
        particle_coverage = xp.put_rows(particle_coverage, replace, xp.take(all_coverages, top))  # 同步替换覆盖率。
        self.weights = xp.put_rows(self.weights, replace, xp.zeros_like(replaced_weights) + weight_median)  # 新粒子给中位数权重，避免它们直接主导。
        if best_snr_host >= 4.0:  # 信噪比足够高才相信它的尺度。
            self.scale = float(  # 尺度低通滤波。
                np.clip(0.90 * self.scale + 0.10 * best_scale_host, 0.86, 1.16)  # 90% 保留旧值 + 10% 吸收测量，并限制上下限。
            )
            self.states = self._apply_bounds(self.states)  # 尺度变化后边界也变，重新施加约束。
        self.frames_since_relocation = 0  # 重定位完成，冷却计数归零。
        return particle_scores, particle_coverage  # 返回可能被覆写过的得分与覆盖率。

    def observe_border(self, temporal_map) -> bool:
        """弱测量：目标已透明，只能靠时序残差证据加权。返回本帧估计是否可靠。

        全部向量运算都在后端设备上完成，整帧只有一次批量 D2H：把加权中心、加权得分/覆盖率、
        信噪比、证据置信度、修正幅度与有效样本数攒成一个 float64 张量一次取回，之后的可靠性判定、
        回滚与簿记全在 CPU 上跑（都是标量分支，放到设备上没有意义）。触发全局重定位的帧会额外
        多两次 D2H（运动先验 + 精扫决策），但重定位本身有冷却限频。
        """

        xp = self.xp  # 数组门面。
        with xp.inference():  # 显卡侧关掉自动求导。
            # 把推进后的状态先存一份作为回滚点。纹理丰富的场景里经常存在
            # 另一条更强的轮廓；不能让单独一帧把跟踪器「传送」过去。
            predicted_states = xp.clone(self.states)  # 回滚用状态快照。
            predicted_weights = xp.clone(self.weights)  # 回滚用权重快照。
            predicted_scale = self.scale  # 回滚用尺度快照。
            # 推进后的预测中心，作为运动先验的参考点。全程留在设备上（float32，与改造前
            # estimate() 的返回精度一致），避开一次只为拿两个浮点数的主存同步。
            predicted_device = xp.astype(self._estimate_device()[0], xp.float32)
            control_count = self.control_count  # 对照组数量：随机位姿，用于估计背景得分分布。
            margin = self.template.nominal_size * 0.50  # 与 _apply_bounds 一致的边距。
            controls = np.zeros((control_count, 6), dtype=np.float32)  # 对照组状态矩阵（随机数只在 NumPy 侧生成）。
            controls[:, 0] = self.rng.uniform(margin, self.frame_width - margin, control_count)  # 随机 x。
            controls[:, 1] = self.rng.uniform(margin, self.play_height - margin, control_count)  # 随机 y。
            controls[:, 4] = self.rng.uniform(  # 随机角度，铺满对称周期域。
                0.0, self.template.symmetry_period, control_count  # 角度上下限。
            )
            scale_factors = (0.90, 1.0, 1.10)  # 对照组与重定位都跑三个尺度。
            control_states = xp.asarray(controls)  # 对照组一次上传（3 个尺度段共用同一份）。
            # 粒子 + 对照组×3 尺度合并成一次打分：显卡上加大 batch 几乎免费，CPU 上也省掉多次函数调用与证据图重复处理。
            batch_states = xp.concatenate(  # 合并状态：粒子在前、对照组按尺度重复三次在后。
                [self.states] + [control_states] * len(scale_factors), axis=0  # 拼接顺序与改造前一致。
            )
            batch_scales = xp.asarray(np.concatenate(  # 合并尺度：粒子段用当前估计尺度，对照组三段逐尺度铺开。
                [np.full(self.count, self.scale, dtype=np.float32)]  # 粒子段。
                + [np.full(control_count, factor, dtype=np.float32) for factor in scale_factors]  # 对照组三段。
            ))
            batch_evidence = score_shape_contours_on_backend(  # 一次合批打分，结果留在设备上。
                temporal_map, batch_states, self.template, batch_scales, self.play_height, self.backend  # 合并后的状态与逐假设尺度。
            )
            particle_scores = xp.narrow(batch_evidence.scores, 0, 0, self.count)  # 前 count 个是粒子（视图，重定位会就地覆写）。
            particle_coverage = xp.narrow(batch_evidence.coverage, 0, 0, self.count)  # 覆盖率同段。
            control_scores = xp.narrow(batch_evidence.scores, 0, self.count, None)  # 其余是三个尺度段的对照组得分（拼接顺序与旧版一致）。
            control_median = xp.astype(xp.median(control_scores), xp.float64)  # 对照组得分中位数，作为背景基线。
            control_mad = xp.astype(xp.median(xp.abs(control_scores - control_median)), xp.float64)  # 中位数绝对偏差。
            control_sigma = xp.clip(control_mad * 1.4826, 0.12, None)  # 鲁棒标准差，下限 0.12 避免除零放大。

            if (  # 置信度偏低或已不是刚刚可靠，且冷却期已满：启动两级粗到细全局重定位。
                self.confidence < 0.56 or self.frames_since_reliable > 0
            ) and self.frames_since_relocation >= self.relocation_cooldown:  # 限频：避免透明阶段每帧都花几十毫秒重定位。
                particle_scores, particle_coverage = self._relocate(  # 两级粗到细重定位，拿回可能被覆写的粒子得分与覆盖率。
                    temporal_map, particle_scores, particle_coverage, control_median, control_sigma  # 证据、粒子得分/覆盖率与对照统计。
                )

            robust_z = (xp.astype(particle_scores, xp.float64) - control_median) / control_sigma  # 每个粒子的鲁棒 z 分数。
            likelihood = xp.exp(xp.clip(0.48 * robust_z, -3.5, 3.8))  # z 分数转似然，上下限截断避免指数爆炸/消失。
            # 覆盖率调制似然：边界命中越多越可信。刻意先在 float32 里算完再放宽到 float64，与改造前的提升次序一致。
            likelihood = likelihood * xp.astype(0.35 + 0.65 * xp.clip(particle_coverage, 0.0, 1.0), xp.float64)
            continuity_sigma = max(  # 运动连续性先验的标准差：丢失越久容忍越宽。
                10.0,  # 下限 10 像素。
                self.template.nominal_size  # 随形状尺度自适应。
                * (0.16 + 0.035 * min(self.frames_since_reliable, 12)),  # 基础 0.16，每丢失一帧 +0.035，上限 12 帧。
            )
            prior_distance = xp.linalg_norm(  # 粒子到预测中心的距离。
                xp.narrow(self.states, 1, 0, 2) - predicted_device, axis=1  # 二维距离（重定位可能已改写 states，因此放在它后面）。
            )
            motion_prior = xp.exp(  # 运动先验：离预测越远可能性越低。
                -0.5 * (prior_distance / max(continuity_sigma, 1.0)) ** 2  # 高斯衰减。
            )
            prior_floor = min(  # 先验下限：丢失很久后必须允许「完全不看运动先验」的重定位。
                0.18, 0.015 + 0.012 * min(self.frames_since_reliable, 14)  # 从 0.015 逐步括到上限 0.18。
            )
            likelihood = likelihood * (prior_floor + (1.0 - prior_floor) * motion_prior)  # 先验按下限混合，保留一定的全局探索能力。
            self.weights = self.weights * (likelihood + 1e-10)  # 权重乘似然，加极小量避免全零。
            self._normalize_weights()  # 归一化。
            weighted_score = xp.sum(self.weights * xp.astype(particle_scores, xp.float64))  # 加权证据得分。
            weighted_coverage = xp.sum(self.weights * xp.astype(particle_coverage, xp.float64))  # 加权覆盖率。
            border_snr = (weighted_score - control_median) / control_sigma  # 相对对照组的信噪比。
            evidence_confidence = (  # 证据置信度：信噪比与覆盖率两项相乘，缺一不可。
                xp.clip((border_snr - 1.15) / 4.5, 0.0, 1.0)  # 信噪比项：1.15 起步，4.5 跨度。
                * xp.clip((weighted_coverage - 0.16) / 0.50, 0.0, 1.0)  # 覆盖率项：0.16 起步，0.50 跨度。
            )
            center = xp.astype(self._estimate_device()[0], xp.float32)  # 加权后的新中心。
            correction = xp.astype(xp.linalg_norm(center - predicted_device), xp.float64)  # 本帧修正幅度。
            maximum_correction = max(  # 允许的最大单帧修正。
                20.0, self.size_ref * self.max_correction_ratio  # 下限 20 像素，否则按阈值基准尺寸比例。
            )
            evidence_confidence = evidence_confidence * xp.exp(  # 修正幅度越大，置信度扣得越狠（软惩罚）。
                -0.5 * (correction / max(maximum_correction * 0.75, 1.0)) ** 2  # 高斯惩罚，标准差为上限的 75%。
            )
            effective = 1.0 / xp.sum(self.weights * self.weights)  # 有效样本数（Kish），一并下载给重采样判定复用，省一次同步。
            summary = xp.to_numpy(xp.concatenate([  # 本帧唯一一次批量 D2H：8 个决策量一次取回。
                xp.astype(center, xp.float64),  # 加权中心 (2,)。
                xp.stack([weighted_score, weighted_coverage, border_snr, evidence_confidence, correction, effective]),  # 六个标量。
            ]))
            center_host = summary[:2].astype(np.float32)  # 加权中心（主存，float32 与改造前一致）。
            weighted_score_host = float(summary[2])  # 加权证据得分（主存）。
            weighted_coverage_host = float(summary[3])  # 加权覆盖率（主存）。
            border_snr_host = float(summary[4])  # 信噪比（主存）。
            evidence_confidence_host = float(summary[5])  # 证据置信度（主存）。
            correction_host = float(summary[6])  # 修正幅度（主存）。
            effective_host = float(summary[7])  # 有效样本数（主存）。
        self.last_border_score = weighted_score_host  # 记录得分供诊断。
        self.last_border_snr = border_snr_host  # 记录信噪比供诊断。
        if correction_host > maximum_correction:  # 单帧修正超限：认定这一帧证据不可信，整体回滚。
            self.states = predicted_states  # 恢复状态。
            self.weights = predicted_weights  # 恢复权重。
            self.scale = predicted_scale  # 恢复尺度。
            self.pending_reliable_hits = 0  # 清零待定命中。
            self.confidence *= 0.94  # 置信度衰减。
            self.frames_since_reliable += 1  # 不可靠帧数 +1。
            return False  # 返回不可靠。
        reliable = evidence_confidence_host >= 0.18  # 置信度过阈即认为本帧可靠。
        if reliable:  # 可靠分支。
            self.confidence = 0.40 * self.confidence + 0.60 * evidence_confidence_host  # 置信度低通吸收证据置信度。
            if evidence_confidence_host >= 0.72:  # 强可靠：才有资格推进「已确认可靠中心」。
                if (  # 与待定中心的距离足够近，说明连续多帧指向同一位置。
                    np.linalg.norm(center_host - self.pending_reliable_center)  # 两者距离。
                    <= self.template.nominal_size * 0.70  # 阈值为标称尺寸的 70%。
                ):
                    self.pending_reliable_hits += 1  # 命中次数 +1。
                else:
                    self.pending_reliable_hits = 1  # 位置跳了，重新开始计数。
                self.pending_reliable_center = center_host.copy()  # 更新待定中心。
                if self.pending_reliable_hits >= 4:  # 连续 4 帧强可靠且位置一致才正式确认。
                    self.last_reliable_center = center_host.copy()  # 更新已确认可靠中心。
                    self.frames_since_reliable = 0  # 不可靠帧数归零。
                else:
                    self.frames_since_reliable += 1  # 尚未达 4 次，仍算不可靠。
            else:
                self.pending_reliable_hits = 0  # 不够强，清零待定命中。
                self.frames_since_reliable += 1  # 不可靠帧数 +1。
        else:  # 不可靠分支。
            self.pending_reliable_hits = 0  # 清零待定命中。
            self.confidence *= 0.965  # 置信度缓慢衰减。
            self.frames_since_reliable += 1  # 不可靠帧数 +1。
        self._resample_if_needed(force=reliable, effective=effective_host)  # 可靠时强制重采样以集中粒子，否则按需。
        return reliable  # 返回本帧是否可靠。

    def _normalize_weights(self) -> None:
        """归一化权重；全部退化时重置为均匀分布，避免数值崩溃。

        写成无分支的 ``where`` 形式：退化判定需要读回权重和，那就等于每帧多一次主存同步。
        两个分支都算一遍再由 ``where`` 选，代价是几次千元级的逐元素运算，换来判定完全留在设备上。
        """

        xp = self.xp  # 数组门面。
        total = xp.sum(self.weights)  # 权重总和（0 维）。
        # NaN/Inf 过不了 <= 比较，先把它换成 0 让它落进退化分支，语义与改造前的 not np.isfinite(total) 一致。
        finite_total = xp.where(xp.isfinite(total), total, xp.zeros_like(total))  # 非有限值归零。
        degenerate = finite_total <= 1e-18  # 出现 NaN/Inf 或权重全部衰减到 0。
        safe_total = xp.where(degenerate, xp.full_like(total, 1.0), total)  # 退化时用一个安全分母占位，它的结果会被下面的 where 丢弃。
        uniform = xp.full_like(self.weights, 1.0 / self.count)  # 均匀分布，等价于放弃当前假设重新开始。
        self.weights = xp.where(degenerate, uniform, self.weights / safe_total)  # 退化走均匀重置，否则正常归一化。

    def _resample_if_needed(self, force: bool, effective: float | None = None) -> None:
        """系统重采样：有效样本数过低或强制时重建粒子群，并对一部分粒子加抖动。

        ``effective`` 由调用方传入时可以省掉一次主存同步（observe_border 已经把它并进当帧的批量下载），
        传 None 则就地算并下载。重采样索引在设备上用 cumsum + searchsorted 求出，
        随机数（分层位置与抖动噪声）仍全部在 NumPy 侧生成，以保住可复现契约。
        """

        xp = self.xp  # 数组门面。
        if effective is None:  # 调用方没有预算好有效样本数。
            effective = float(xp.to_numpy(1.0 / xp.sum(self.weights * self.weights)))  # 有效样本数（Kish），一次 D2H 取回标量。
        if not force and effective >= self.count * 0.56:  # 未强制且粒子多样性还够。
            return  # 不重采样，保留现有假设分布。
        positions = (self.rng.random() + np.arange(self.count)) / self.count  # 系统重采样的均匀分层位置。
        cumulative = xp.cumsum(self.weights)  # 权重累积分布。
        indexes = xp.searchsorted(cumulative, xp.asarray(positions), side="right")  # 查找每个分层位置对应的粒子索引。
        indexes = xp.clip(indexes, 0, self.count - 1)  # 防止浮点误差越界。
        self.states = xp.take(self.states, indexes, axis=0)  # 按权重重建粒子群（take 返回新数组，不与原状态共享存储）。
        self.weights = xp.full_like(self.weights, 1.0 / self.count)  # 重采样后权重重新均匀。
        jitter_count = max(10, self.count // 16)  # 需要抖动的粒子数，保留探索能力。
        chosen = self.rng.choice(self.count, jitter_count, replace=False)  # 不重复地选出抖动粒子。
        uncertainty = 1.0 - float(np.clip(self.confidence, 0.0, 1.0))  # 不确定度：置信度低时抖动更大。
        # 抖动先在主存侧散播成 (count,6) 的整块增量再上传：显卡上不做花式索引赋值，
        # 且 float64 增量与 float32 状态相加后只舍入一次，与改造前逐列就地 += 的精度语义一致。
        delta = np.zeros((self.count, 6), dtype=np.float64)  # 抖动增量容器。
        delta[chosen, :2] = self.rng.normal(  # 位置抖动。
            0.0,  # 零均值。
            1.0 + self.template.nominal_size * 0.10 * uncertainty,  # 基准 1 像素 + 随不确定度放大的形状尺度项。
            size=(jitter_count, 2),  # 二维噪声。
        )
        delta[chosen, 2:4] = self.rng.normal(  # 速度抖动。
            0.0, 0.8 + uncertainty, size=(jitter_count, 2)  # 基准 0.8。
        )
        delta[chosen, 4] = self.rng.normal(  # 角度抖动。
            0.0, 2.0 + min(10.0, self.template.symmetry_period * 0.10) * uncertainty,  # 基准 2 度，周期项上限 10。
            jitter_count,  # 一维噪声。
        )
        self.states = xp.astype(self.states + xp.asarray(delta), xp.float32)  # 一次上传并叠加抖动，状态回到 float32。
        self.states = self._apply_bounds(self.states)  # 抖动后重新施加约束。
