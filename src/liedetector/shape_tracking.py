"""测谎检验「找目标」算法内核：无神经网络的轮廓学习 + DIS 光流 + 粒子滤波。

本模块从外部验证脚本 track_transparent_shape.py / track_transparent_square_v2.py
移植而来，只做纯算法计算，不含任何视频读写、CSV 输出、GUI 绘制与命令行参数。

整体分三段：
1. 学习段：首帧从白色区域提取轮廓，重采样成 120 点模板（含法向、内部采样点、
   旋转对称周期、矩形度），满足四重对称 + 高矩形度时自动切换规则矩形边框评分模型。
2. 证据段：用 DIS 稠密光流把多个 lag 的历史帧对齐到当前帧，取绝对差得到「时序残差
   证据图」，目标透明化后仍会在残差图边界留下痕迹。
3. 跟踪段：粒子群在 6 维状态空间 (x, y, vx, vy, angle, 角速度) 上联合轮廓边界与
   内部残差打分，输出位置/角度/小范围缩放。

刻意与外部脚本的偏离：不调用全局 cv2.setNumThreads(1) / cv2.ocl.setUseOpenCL(False)
/ cv2.setRNGSeed()，因为这些全局设置会拖慢整个应用；可复现性改由每局独立的
np.random.default_rng(seed) 保证。
"""

from __future__ import annotations

import math  # 数学函数：三角、开方、弧度角度互转。
from collections import deque  # 定长队列，保存光流对齐所需的历史帧。
from dataclasses import dataclass  # 数据类装饰器，承载算法中间结果。

import cv2  # OpenCV：HSV 转换、形态学、轮廓、光流、重映射。
import numpy as np  # 数值计算：向量化旋转、打分、粒子群运算。

from src.liedetector.gpu_shape_backend import (  # 打分内核双后端：同一份算法代码，有 N 卡走 CuPy，没有则 NumPy。
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


def _resolve_backend(backend):  # 解析本次打分使用的后端：显式传入优先，否则取全局生效后端（CuPy 或 NumPy）。
    if backend is not None:  # 调用方显式指定了后端（会话装配时注入）。
        return backend  # 直接使用。
    return require_gpu_backend()  # 未指定：请求显卡后端，不可用/已降级时返回 NumPy 后端。


def _to_numpy_array(item):  # 把后端数组统一转回 NumPy（CuPy 数组用 .get() 下载，NumPy 数组直接复用）。
    if isinstance(item, np.ndarray):  # 已是 NumPy 数组。
        return item  # 直接返回。
    getter = getattr(item, "get", None)  # CuPy 数组的 D2H 下载方法。
    if callable(getter):  # 显卡数组。
        return getter()  # 下载到主存。
    return np.asarray(item)  # 其余情况兼容处理。


def _run_scoring(core, kind, backend, *args):  # 统一的打分调度：显卡后端异常时自动改走 NumPy 重算本帧，返回 NumPy 版结果数据类。
    chosen = _resolve_backend(backend)  # 本次生效的后端。
    if chosen.is_gpu and chosen is require_gpu_backend():  # 全局显卡后端：运行期异常（显存不足/驱动重置等）不能中断求解。
        try:  # 显卡执行段。
            result = core(chosen.xp, *args)  # 在显卡上执行打分内核。
            packed = kind(_to_numpy_array(result.scores), _to_numpy_array(result.coverage))  # 下载并重新打包成 NumPy 版结果。
            clear_gpu_failure()  # 打分成功，清零连续失败计数。
            return packed  # 返回结果。
        except Exception:  # 显卡运行期异常。
            mark_gpu_failure()  # 计数，连续 3 次后本进程永久回退 NumPy。
            chosen = numpy_backend()  # 本帧改走 CPU 重算，调用方拿到的结果与显卡正常时等价。
    result = core(chosen.xp, *args)  # NumPy 后端（或显卡降级后）直接计算。
    return kind(_to_numpy_array(result.scores), _to_numpy_array(result.coverage))  # 打包成 NumPy 版结果返回。


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


@dataclass
class TemporalEvidence:
    """多 lag 光流对齐后的时序残差证据。"""

    normalized: np.ndarray  # 鲁棒归一化后的残差图（单通道 float32），边界处响应强。
    raw_mean: float  # 各 lag 归一化前残差均值的最小值，用于判断场景切换/静止。


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
    """在一帧里找出所有「低饱和高亮」的白色候选目标，按面积*置信度降序返回。"""

    height, width = frame.shape[:2]  # 帧的高和宽。
    play_height = int(height * play_height_ratio)  # 有效检测高度（比例以下视为无效带）。
    hsv = cv2.cvtColor(frame[:play_height], cv2.COLOR_BGR2HSV)  # 只在有效带内转 HSV。
    saturation = hsv[:, :, 1]  # 饱和度通道。
    value = hsv[:, :, 2]  # 明度通道。
    mask = ((saturation <= 92) & (value >= 188)).astype(np.uint8) * 255  # 低饱和 + 高亮 = 白色掩码。
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
                center=contour_center(contour),  # 质心。
                contour=contour,  # 原始轮廓。
                area=area,  # 面积。
                nominal_size=math.sqrt(area),  # 标称尺寸。
                confidence=confidence,  # 置信度。
            )
        )
    return sorted(detections, key=lambda item: item.area * item.confidence, reverse=True)  # 面积*置信度降序，最像目标的排前面。


def periodic_angle_difference(
    angles: np.ndarray, reference: float, period: float
) -> np.ndarray:
    """求一组角度与参考角在周期域内的最小角距离。"""

    return np.abs((angles - reference + period * 0.5) % period - period * 0.5)  # 平移半周期取模再折回，得到 [-period/2, period/2] 的绝对值。


def weighted_periodic_angle(
    angles: np.ndarray, weights: np.ndarray, period: float
) -> float:
    """周期域内的加权平均角：用复数向量求和避免 0/period 边界处的错误平均。"""

    radians = angles * (2.0 * math.pi / period)  # 把周期域角度映射到 0~2pi。
    vector = np.sum(weights * np.exp(1j * radians))  # 加权复数向量求和。
    if abs(vector) < 1e-8:  # 向量相互抵消，方向无意义。
        return float(np.mean(angles) % period)  # 退化为算术平均。
    return float((np.angle(vector) * period / (2.0 * math.pi)) % period)  # 取合向量方向并映射回周期域。


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


def choose_shape_candidate(
    candidates: list[ShapeDetection],
    template: ShapeTemplate | None,
    center: np.ndarray | None,
    confidence: float,
) -> ShapeDetection | None:
    """从白色候选里挑出最可能是被跟踪目标的那一个。"""

    if not candidates:  # 本帧没有任何白色候选。
        return None  # 返回空。
    if template is None or center is None:  # 还没学到模板，或没有位置先验。
        return candidates[0]  # 直接取排序第一（面积*置信度最大）的候选，用于初始化学习。
    plausible: list[tuple[float, ShapeDetection]] = []  # 收集通过尺度/距离窗口筛选的候选及其得分。
    allowed_distance = template.nominal_size * (1.5 if confidence >= 0.25 else 3.5)  # 置信度高时窗口收紧，丢失后放宽。
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
            - 0.18 * distance / max(template.nominal_size, 1.0)  # 归一化距离惩罚。
            - 0.16 * min(match, 2.0)  # Hu 矩形状惩罚，截断避免异常值主导。
            - 0.10 * shape_distance  # 循环对齐残差惩罚。
        )
        plausible.append((score, candidate))  # 记录候选。
    return max(plausible, key=lambda item: item[0])[1] if plausible else None  # 取得分最高者；全被筛掉则返回 None。


class DenseTemporalAligner:
    """用 DIS 稠密光流把历史帧对齐到当前帧，生成鲁棒的时序残差证据图。"""

    def __init__(  # 构造对齐器。
        self,
        lags: tuple[int, ...],  # 使用的时间基线（帧间隔），例如 (1, 2, 4)。
        play_height_ratio: float,  # 有效高度比例，其下部分在证据图里清零。
        preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,  # DIS 光流精度档位。
    ):
        self.lags = lags  # 保存时间基线。
        self.play_height_ratio = play_height_ratio  # 保存有效高度比例。
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(  # 定长历史队列，存 (彩色帧, 灰度帧)。
            maxlen=max(lags) + 1  # 只需要保留最大 lag 再加当前帧。
        )
        self.flow = cv2.DISOpticalFlow_create(preset)  # 创建 DIS 稠密光流实例。
        self.flow.setUseSpatialPropagation(True)  # 开启空间传播，提升弱纹理区域的稳定性。

    def reset(self) -> None:
        """清空历史帧队列；跨局/区域位移后必须调用，否则残差会被上一段画面污染。"""

        self.history.clear()  # 丢弃全部历史帧。

    def update(self, frame: np.ndarray) -> TemporalEvidence | None:
        """喂入一帧，返回多 lag 合并后的时序残差证据；历史不足时返回 None。"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 光流只需要灰度。
        self.history.append((frame.copy(), gray))  # 入队；彩色帧留一份拷贝用于重建背景。
        available = [lag for lag in self.lags if len(self.history) > lag]  # 筛出历史长度已经足够的 lag。
        if not available:  # 一个 lag 都算不了（刚启动）。
            return None  # 返回空证据。

        height, width = gray.shape  # 帧尺寸。
        play_height = int(height * self.play_height_ratio)  # 有效高度。
        xx, yy = np.meshgrid(  # 采样网格坐标，供 remap 使用。
            np.arange(width, dtype=np.float32),  # x 网格。
            np.arange(height, dtype=np.float32),  # y 网格。
        )
        normalized_maps: list[np.ndarray] = []  # 收集每个 lag 的归一化残差图。
        raw_means: list[float] = []  # 收集每个 lag 归一化前的残差均值。

        for lag in available:  # 逐个时间基线计算。
            previous, previous_gray = self.history[-1 - lag]  # 取出对应的历史帧（倒数第 lag+1 个）。
            # 光流把当前帧每个像素映射到它在旧帧里的来源位置，
            # 因此按该光流 remap 旧帧即可重建「当前帧的背景」。
            flow = self.flow.calc(gray, previous_gray, None)  # 计算当前帧到历史帧的稠密光流。
            aligned = cv2.remap(  # 按光流重建背景。
                previous,  # 源图像为历史彩色帧。
                xx + flow[:, :, 0],  # x 方向映射坐标。
                yy + flow[:, :, 1],  # y 方向映射坐标。
                cv2.INTER_LINEAR,  # 双线性插值。
                borderMode=cv2.BORDER_REFLECT,  # 边界反射填充，避免黑边。
            )
            delta = np.max(cv2.absdiff(frame, aligned), axis=2).astype(np.float32)  # 逐像素绝对差取三通道最大值。
            valid_delta = delta[:play_height]  # 只统计有效带内的残差。
            median = float(np.median(valid_delta))  # 残差中位数，作为鲁棒基线。
            mad = float(np.median(np.abs(valid_delta - median)))  # 中位数绝对偏差。
            sigma = max(1.0, 1.4826 * mad)  # 鲁棒标准差，下限 1.0 防止过度放大。
            normalized = np.clip((delta - median) / sigma, 0.0, 12.0)  # 鲁棒归一化并截断极端值。
            normalized[play_height:] = 0.0  # 有效带以下一律清零。
            normalized_maps.append(normalized)  # 收集该 lag 的归一化残差。
            raw_means.append(float(np.mean(valid_delta)))  # 收集该 lag 的原始残差均值。

        # 真实运动的边界只需要在某一个时间基线上清晰即可，因此多 lag 取最大。
        combined = np.maximum.reduce(normalized_maps)  # 逐像素取各 lag 的最大响应。
        combined = cv2.GaussianBlur(combined, (3, 3), 0)  # 轻度平滑，抑制单像素噪声。
        # remap 的边界填充会产生又长又笔直的假边界。只压制这条窄的不可靠带：
        # 贴到画面边缘的目标仍能靠它剩下的两三条边被找回来。
        edge_band = max(8, int(round(width * 0.014)))  # 边缘带宽度，随画面宽度自适应。
        combined[:edge_band] = 0.0  # 压制上边缘带。
        combined[max(0, play_height - edge_band) :] = 0.0  # 压制有效带下边缘。
        combined[:, :edge_band] = 0.0  # 压制左边缘带。
        combined[:, max(0, width - edge_band) :] = 0.0  # 压制右边缘带。
        return TemporalEvidence(combined, min(raw_means))  # 残差均值取最小，避免单一 lag 的抖动误判场景切换。


def sample_map(image: np.ndarray, x: np.ndarray, y: np.ndarray, backend=None) -> np.ndarray:
    """在单通道证据图上做最近邻批量采样，越界位置返回 0（双后端薄壳，实现在 gpu_shape_backend）。"""

    chosen = _resolve_backend(backend)  # 解析本次生效的后端。
    return _to_numpy_array(sample_map_core(chosen.xp, image, x, y))  # 后端采样后统一转回 NumPy。


def score_rotated_borders(
    evidence: np.ndarray, states: np.ndarray, side: float, play_height: int, backend=None
) -> BorderEvidence:
    """矩形模型专用：给一批位姿假设的四条薄边打分（双后端薄壳，实现在 gpu_shape_backend）。"""

    return _run_scoring(score_rotated_borders_core, BorderEvidence, backend, evidence, states, side, play_height)  # 统一调度与降级。


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

    return _run_scoring(score_shape_contours_core, ShapeEvidence, backend, evidence, states, template, scales, play_height)  # 统一调度与降级。


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
        backend=None,  # 打分后端（gpu_shape_backend.ShapeScoreBackend），None 时用 NumPy 后端。
    ):
        self.template = template  # 保存模板。
        self.backend = numpy_backend() if backend is None else backend  # 打分后端：会话装配时注入，缺省 NumPy（双后端结构：有 N 卡时为 CuPy）。
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
        self.scale = 1.0  # 当前估计的目标缩放，初值 1。
        self.states = np.zeros((particle_count, 6), dtype=np.float32)  # 粒子状态矩阵。
        self.states[:, :2] = detection.center + rng.normal(  # 位置围绕检测中心小幅扩散。
            0.0, template.nominal_size * 0.035, size=(particle_count, 2)  # 扩散标准差为标称尺寸的 3.5%。
        )
        self.states[:, 2:4] = rng.normal(0.0, 1.2, size=(particle_count, 2))  # 初速度零均值随机，标准差 1.2 像素/帧。
        self.states[:, 4] = rng.normal(0.0, 2.5, particle_count) % template.symmetry_period  # 初始角度围绕 0 小范围扰动后折回对称周期域。
        self.states[:, 5] = rng.normal(0.0, 0.6, particle_count)  # 初始角速度零均值随机。
        self.weights = np.full(particle_count, 1.0 / particle_count, dtype=np.float64)  # 权重均匀初始化。
        self.confidence = detection.confidence  # 初始置信度取白色检测置信度。
        self.last_reliable_center = detection.center.astype(np.float32).copy()  # 最后一个已确认可靠的中心。
        self.pending_reliable_center = self.last_reliable_center.copy()  # 正在累积命中次数的待定中心。
        self.pending_reliable_hits = 3  # 初始已有 3 次命中（白色检测本身很可靠）。
        self.frames_since_reliable = 0  # 距上次可靠估计的帧数。
        self.frames_since_relocation = relocation_cooldown  # 距上次重定位的帧数，初始即满允许首次立即重定位。
        self.last_border_score = 0.0  # 上一次边界加权得分，供诊断输出。
        self.last_border_snr = 0.0  # 上一次边界信噪比，供诊断输出。
        self.search_radius = template.nominal_size * 0.5  # 当前搜索半径，供诊断输出。

    def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """输出加权估计：中心、速度与周期域加权平均角度。"""

        center = np.sum(self.states[:, :2] * self.weights[:, None], axis=0)  # 位置加权均值。
        velocity = np.sum(self.states[:, 2:4] * self.weights[:, None], axis=0)  # 速度加权均值。
        angle = weighted_periodic_angle(  # 角度必须走周期域平均，不能直接算术均。
            self.states[:, 4], self.weights, self.template.symmetry_period  # 传入角度、权重与对称周期。
        )
        return center.astype(np.float32), velocity.astype(np.float32), angle  # 返回三个估计量。

    def _apply_bounds(self, states: np.ndarray) -> None:
        """对状态矩阵就地施加边界、限速、角度取模与角速度截断约束。"""

        margin = self.template.nominal_size * 0.50 * self.scale  # 边距为半个形状，避免中心贴到画面边缘。
        max_x = self.frame_width - margin  # x 上限。
        max_y = self.play_height - margin  # y 上限（以有效高度为准）。
        left = states[:, 0] < margin  # 越左界的粒子。
        right = states[:, 0] > max_x  # 越右界的粒子。
        top = states[:, 1] < margin  # 越上界的粒子。
        bottom = states[:, 1] > max_y  # 越下界的粒子。
        states[left, 0] = margin  # 钳回左边界。
        states[left, 2] = np.maximum(states[left, 2], 0.0)  # 左界处不允许继续向左的速度。
        states[right, 0] = max_x  # 钳回右边界。
        states[right, 2] = np.minimum(states[right, 2], 0.0)  # 右界处不允许继续向右的速度。
        states[top, 1] = margin  # 钳回上边界。
        states[top, 3] = np.maximum(states[top, 3], 0.0)  # 上界处不允许继续向上的速度。
        states[bottom, 1] = max_y  # 钳回下边界。
        states[bottom, 3] = np.minimum(states[bottom, 3], 0.0)  # 下界处不允许继续向下的速度。
        max_speed = self.template.nominal_size * self.max_speed_ratio  # 单帧最大位移。
        speed = np.linalg.norm(states[:, 2:4], axis=1)  # 当前速度大小。
        too_fast = speed > max_speed  # 超速的粒子。
        states[too_fast, 2:4] *= (max_speed / speed[too_fast])[:, None]  # 保留方向、按比例缩到限速。
        states[:, 4] %= self.template.symmetry_period  # 角度折回对称周期域。
        states[:, 5] = np.clip(states[:, 5], -6.0, 6.0)  # 角速度截断，避免无限累积。

    def propagate(self) -> None:
        """粒子推进一步：按当前速度平移，并按「1 - 置信度」放大扩散噪声。"""

        uncertainty = 1.0 - float(np.clip(self.confidence, 0.0, 1.0))  # 不确定度：置信度越低扩散越大。
        self.states[:, :2] += self.states[:, 2:4]  # 位置按速度推进。
        self.states[:, :2] += self.rng.normal(  # 位置扩散噪声。
            0.0, 0.45 + 2.8 * uncertainty, size=(self.count, 2)  # 基准 0.45，不确定度满时额外 2.8。
        )
        self.states[:, 2:4] += self.rng.normal(  # 速度随机游走。
            0.0, 0.18 + 0.85 * uncertainty, size=(self.count, 2)  # 基准 0.18。
        )
        self.states[:, 4] += self.states[:, 5]  # 角度按角速度推进。
        self.states[:, 4] += self.rng.normal(  # 角度扩散噪声。
            0.0, 0.35 + 2.0 * uncertainty, self.count  # 基准 0.35 度。
        )
        self.states[:, 5] += self.rng.normal(  # 角速度随机游走。
            0.0, 0.20 + 0.30 * uncertainty, self.count  # 基准 0.20。
        )
        self._apply_bounds(self.states)  # 推进后统一施加约束。
        self.frames_since_relocation += 1  # 重定位冷却计数 +1。

    def observe_color(self, detection: ShapeDetection) -> None:
        """强测量：白色轮廓仍然可见，直接用检测结果修正粒子群与尺度。"""

        distance = np.linalg.norm(self.states[:, :2] - detection.center, axis=1)  # 每个粒子到检测中心的距离。
        angle_distance = periodic_angle_difference(  # 周期域角度偏差。
            self.states[:, 4], detection.angle, self.template.symmetry_period  # 粒子角度 vs 检测角度。
        )
        position_sigma = max(5.0, self.template.nominal_size * 0.22)  # 位置似然标准差，随尺寸自适应。
        angle_sigma = max(8.0, self.template.symmetry_period * 0.15)  # 角度似然标准差，随对称周期自适应。
        likelihood = np.exp(-0.5 * (distance / position_sigma) ** 2)  # 位置高斯似然。
        # 对称/接近圆形的轮廓携带的角度信息很弱，必须降低角度项权重。
        angle_weight = 0.35 if self.template.symmetry_period <= 60.0 else 0.75  # 周期≤ 60 度时降权到 0.35。
        likelihood *= (1.0 - angle_weight) + angle_weight * np.exp(  # 角度高斯似然按权重混合进去。
            -0.5 * (angle_distance / angle_sigma) ** 2  # 角度高斯项。
        )
        self.weights *= likelihood + 1e-9  # 权重乘似然，加极小量避免全零。
        self._normalize_weights()  # 归一化。
        innovation = detection.center - self.states[:, :2]  # 位置新息（测量 - 预测）。
        self.states[:, 2:4] += 0.14 * innovation  # 速度弱修正，保留惯性。
        self.states[:, :2] += 0.52 * innovation  # 位置强修正，快速贴向测量。
        angular_innovation = (  # 角度新息，必须走周期域最短弧。
            detection.angle  # 测量角度。
            - self.states[:, 4]  # 减去粒子角度。
            + self.template.symmetry_period * 0.5  # 平移半周期。
        ) % self.template.symmetry_period - self.template.symmetry_period * 0.5  # 取模后折回，得到最短角差。
        self.states[:, 4] = (  # 角度按权重修正，并折回对称周期域。
            self.states[:, 4] + 0.28 * angle_weight * angular_innovation  # 修正量同时受对称性降权影响。
        ) % self.template.symmetry_period  # 取模保证角度始终在周期域内。
        measured_scale = detection.nominal_size / max(self.template.nominal_size, 1.0)  # 测量到的相对尺度。
        if 0.86 <= measured_scale <= 1.16:  # 只接受合理范围内的尺度测量。
            self.scale = float(np.clip(0.94 * self.scale + 0.06 * measured_scale, 0.86, 1.16))  # 低通滤波平滑尺度，并限制上下限。
        self._apply_bounds(self.states)  # 修正后重新施加约束。
        self.confidence = max(self.confidence * 0.55, detection.confidence)  # 置信度：旧值衰减后与测量值取大。
        self.last_reliable_center = detection.center.astype(np.float32).copy()  # 白色检测就是可靠中心。
        self.pending_reliable_center = self.last_reliable_center.copy()  # 待定中心同步。
        self.pending_reliable_hits = 3  # 重置为 3，再命中一次即达 4 可确认。
        self.frames_since_reliable = 0  # 可靠计数归零。
        self.last_border_score = 0.0  # 清零边界诊断值。
        self.last_border_snr = 0.0  # 清零信噪比诊断值。
        self.search_radius = self.template.nominal_size * 0.5  # 搜索半径回到默认值。
        self._resample_if_needed(force=False)  # 按需重采样，不强制。

    def _proposal_states(self, reference_center: np.ndarray) -> np.ndarray:
        """生成全局重定位候选位姿：30% 局部圆域 + 70% 全画面网格×角度离散。"""

        proposal_count = self.global_proposals  # 候选总数。
        lost = max(1, self.frames_since_reliable)  # 已丢失帧数，下限 1 避免除零。
        max_speed = self.template.nominal_size * self.max_speed_ratio  # 单帧最大位移。
        radius = min(  # 局部搜索半径：丢失越久半径越大，但不超过画面对角线。
            math.hypot(self.frame_width, self.play_height),  # 上限：画面对角线。
            self.template.nominal_size * 0.55 + lost * max_speed,  # 基础半径 + 丢失期间可走的最大距离。
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
        _, velocity, current_angle = self.estimate()  # 取当前速度与角度作为候选初值。
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
        self._apply_bounds(states)  # 统一施加边界与限速约束。
        return states  # 返回全部候选位姿。

    def _relocate(  # 两级粗到细全局重定位。
        self,
        temporal_map: np.ndarray,  # 时序残差证据图。
        particle_evidence: ShapeEvidence,  # 当前粒子打分结果，拉回的候选要同步写回。
        predicted_center: np.ndarray,  # 推进后的预测中心，作为运动先验参考点。
        control_median: float,  # 对照组得分中位数。
        control_sigma: float,  # 对照组鲁棒标准差。
    ) -> None:
        """第一级用有限候选粗扫全图（单尺度），第二级对 top-K 邻域拖尾细化（3 尺度）。
        替代旧版 2400×3 尺度一次性打分，把单次重定位从 ~240ms 降到 ~30ms。"""

        scale_factors = (0.90, 1.0, 1.10)  # 精扫跑三个尺度。
        coarse_states = self._proposal_states(predicted_center)  # 第一级粗扫候选，数量 = global_proposals（实时档 400）。
        coarse_result = score_shape_contours(  # 粗扫只在 1.0 尺度下打分。
            temporal_map, coarse_states, self.template, 1.0, self.play_height, self.backend  # 证据图、候选、模板、尺度、有效高度与打分后端。
        )
        prior_radius = self.template.nominal_size * (  # 距离先验半径：丢失越久容忍越宽。
            0.55 + 0.060 * min(self.frames_since_reliable, 10)  # 基础 0.55，每丢失一帧 +0.06，上限 10 帧。
        )
        coarse_distance = np.linalg.norm(  # 候选到预测中心的距离。
            coarse_states[:, :2] - predicted_center, axis=1  # 二维距离。
        )
        coarse_ranking = coarse_result.scores - 2.20 * (  # 排序得分 = 证据得分 - 平方距离惩罚。
            coarse_distance / max(prior_radius, 1.0)  # 归一化距离，下限防除零。
        ) ** 2
        top_k = min(self.coarse_top, len(coarse_states))  # 进入精扫的邻域数，不超过粗扫候选总数。
        top_indices = np.argsort(coarse_ranking)[::-1][:top_k]  # 粗扫排序得分最高的 top-K 索引。
        if top_k == 0:  # 没有任何候选（理论上不会出现），直接放弃重定位。
            return  # 保持冷却计数不变，下一帧再试。

        period = self.template.symmetry_period  # 对称周期。
        position_jitter = max(4.0, self.template.nominal_size * 0.14)  # 位置抖动标准差：标称尺寸的 14%。
        angle_jitter = max(6.0, period * 0.10)  # 角度抖动标准差：周期的 10%。
        refined_list: list[np.ndarray] = []  # 收集每个邻域的抖动候选组。
        for index in top_indices:  # 遍历粗扫 top-K 邻域。
            base = coarse_states[index]  # 邻域基准位姿。
            variants = np.tile(base, (self.refine_per_top, 1))  # 复制基准位姿成一组。
            variants[:, :2] += self.rng.normal(  # 位置抖动。
                0.0, position_jitter, size=(self.refine_per_top, 2)  # x/y 独立抖动。
            )
            variants[:, 4] = (  # 角度抖动后折回周期域。
                base[4] + self.rng.normal(0.0, angle_jitter, self.refine_per_top)  # 基准角度 + 抖动。
            ) % period  # 取模保证在周期域内。
            refined_list.append(variants)  # 收集本邻域候选。
        proposals = np.vstack(refined_list).astype(np.float32)  # 合并成精扫候选集。
        self._apply_bounds(proposals)  # 统一施加边界约束。
        # 三个尺度的精扫合并成一次打分：候选按「尺度段 × 候选」排列，与旧版逐尺度拼接顺序一致，
        # 后续 ranking/selected 索引到 all_states 的映射语义不变。
        all_states = np.concatenate([proposals] * len(scale_factors)).astype(np.float32)  # 同一批候选在三个尺度下各记一次。
        all_scales = np.concatenate(  # 逐候选尺度：每段 proposals 对应一个尺度因子。
            [np.full(len(proposals), float(np.clip(factor, 0.86, 1.16)), dtype=np.float32) for factor in scale_factors]  # 尺度限在合法区间内。
        )
        refine_evidence = score_shape_contours(  # 一次合批对全部尺度候选打分。
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
        distance_from_prediction = np.linalg.norm(  # 候选到预测中心的距离。
            all_states[:, :2] - predicted_center, axis=1  # 二维距离。
        )
        ranking_scores = all_scores - 2.20 * (  # 排序得分 = 证据得分 - 平方距离惩罚。
            distance_from_prediction / max(prior_radius, 1.0)  # 归一化距离，下限防除零。
        ) ** 2
        order = np.argsort(ranking_scores)[::-1]  # 排序得分降序索引。
        selected: list[int] = []  # 已选中的候选索引。
        minimum_separation = self.template.nominal_size * 0.24  # 候选之间的最小间距，避免拉回一堆重复位姿。
        for candidate in order:  # 按排序依次尝试选取。
            if all(  # 与所有已选候选的距离都达标准。
                np.linalg.norm(  # 两点距离。
                    all_states[candidate, :2] - all_states[chosen, :2]  # 当前候选 vs 已选候选。
                )
                >= minimum_separation  # 不小于最小间距。
                for chosen in selected  # 遍历已选集合。
            ):
                selected.append(int(candidate))  # 选入。
                if len(selected) >= top_count:  # 已经选够。
                    break  # 提前结束循环。
        if len(selected) < top_count:  # 去重后不够 top_count 个。
            used = set(selected)  # 已用索引集合。
            selected.extend(int(item) for item in order if int(item) not in used)  # 按排序补齐剩下的名额。
        top = np.asarray(selected[:top_count], dtype=np.int32)  # 最终拉回的候选索引。
        replace = np.argsort(self.weights)[:top_count]  # 权重最低的 top_count 个粒子被替换。
        self.states[replace] = all_states[top]  # 用高分候选替换低权重粒子的状态。
        particle_evidence.scores[replace] = all_scores[top]  # 同步替换得分。
        particle_evidence.coverage[replace] = all_coverages[top]  # 同步替换覆盖率。
        self.weights[replace] = np.median(self.weights)  # 新粒子给中位数权重，避免它们直接主导。
        best = int(np.argmax(ranking_scores))  # 精扫集内最优候选索引。
        best_snr = (float(all_scores[best]) - control_median) / control_sigma  # 最优候选的信噪比。
        if best_snr >= 4.0:  # 信噪比足够高才相信它的尺度。
            self.scale = float(  # 尺度低通滤波。
                np.clip(0.90 * self.scale + 0.10 * all_scales[best], 0.86, 1.16)  # 90% 保留旧值 + 10% 吸收测量，并限制上下限。
            )
            self._apply_bounds(self.states)  # 尺度变化后边界也变，重新施加约束。
        self.frames_since_relocation = 0  # 重定位完成，冷却计数归零。

    def observe_border(self, temporal_map: np.ndarray) -> bool:
        """弱测量：目标已透明，只能靠时序残差证据加权。返回本帧估计是否可靠。"""

        # 把推进后的状态先存一份作为回滚点。纹理丰富的场景里经常存在
        # 另一条更强的轮廓；不能让单独一帧把跟踪器「传送」过去。
        predicted_states = self.states.copy()  # 回滚用状态快照。
        predicted_weights = self.weights.copy()  # 回滚用权重快照。
        predicted_scale = self.scale  # 回滚用尺度快照。
        predicted_center, _, _ = self.estimate()  # 推进后的预测中心，作为运动先验的参考点。
        particle_evidence = score_shape_contours(  # 给当前全部粒子打分。
            temporal_map, self.states, self.template, self.scale, self.play_height, self.backend  # 证据图、状态、模板、尺度、有效高度与打分后端。
        )
        control_count = self.control_count  # 对照组数量：随机位姿，用于估计背景得分分布。
        controls = np.zeros((control_count, 6), dtype=np.float32)  # 对照组状态矩阵。
        margin = self.template.nominal_size * 0.50  # 与 _apply_bounds 一致的边距。
        controls[:, 0] = self.rng.uniform(margin, self.frame_width - margin, control_count)  # 随机 x。
        controls[:, 1] = self.rng.uniform(margin, self.play_height - margin, control_count)  # 随机 y。
        controls[:, 4] = self.rng.uniform(  # 随机角度，铺满对称周期域。
            0.0, self.template.symmetry_period, control_count  # 角度上下限。
        )
        scale_factors = (0.90, 1.0, 1.10)  # 对照组与重定位都跑三个尺度。
        # 粒子 + 对照组×3 尺度合并成一次打分：显卡上加大 batch 几乎免费，CPU 上也省掉多次函数调用与证据图重复处理。
        batch_states = np.concatenate(  # 合并状态：粒子在前、对照组按尺度重复三次在后。
            [self.states] + [controls] * len(scale_factors)
        ).astype(np.float32)  # 统一精度。
        batch_scales = np.concatenate(  # 合并尺度：粒子段用当前估计尺度，对照组三段逐尺度铺开。
            [np.full(len(self.states), self.scale, dtype=np.float32)]  # 粒子段。
            + [np.full(control_count, factor, dtype=np.float32) for factor in scale_factors]  # 对照组三段。
        )
        batch_evidence = score_shape_contours(  # 一次合批打分。
            temporal_map, batch_states, self.template, batch_scales, self.play_height, self.backend  # 合并后的状态与逐假设尺度。
        )
        particle_evidence = ShapeEvidence(  # 拆回粒子段得分，保持原有变量语义。
            batch_evidence.scores[: len(self.states)],  # 前 count 个是粒子。
            batch_evidence.coverage[: len(self.states)],  # 覆盖率同段。
        )
        control_scores = batch_evidence.scores[len(self.states):]  # 其余是三个尺度段的对照组得分（拼接顺序与旧版一致）。
        control_median = float(np.median(control_scores))  # 对照组得分中位数，作为背景基线。
        control_mad = float(np.median(np.abs(control_scores - control_median)))  # 中位数绝对偏差。
        control_sigma = max(0.12, 1.4826 * control_mad)  # 鲁棒标准差，下限 0.12 避免除零放大。

        if (  # 置信度偏低或已不是刚刚可靠，且冷却期已满：启动两级粗到细全局重定位。
            self.confidence < 0.56 or self.frames_since_reliable > 0
        ) and self.frames_since_relocation >= self.relocation_cooldown:  # 限频：避免透明阶段每帧都花几十毫秒重定位。
            self._relocate(  # 两级粗到细重定位。
                temporal_map, particle_evidence, predicted_center, control_median, control_sigma  # 证据、粒子得分、预测中心与对照统计。
            )

        robust_z = (particle_evidence.scores - control_median) / control_sigma  # 每个粒子的鲁棒 z 分数。
        likelihood = np.exp(np.clip(0.48 * robust_z, -3.5, 3.8))  # z 分数转似然，上下限截断避免指数爆炸/消失。
        likelihood *= 0.35 + 0.65 * np.clip(particle_evidence.coverage, 0.0, 1.0)  # 覆盖率调制似然：边界命中越多越可信。
        continuity_sigma = max(  # 运动连续性先验的标准差：丢失越久容忍越宽。
            10.0,  # 下限 10 像素。
            self.template.nominal_size  # 随形状尺度自适应。
            * (0.16 + 0.035 * min(self.frames_since_reliable, 12)),  # 基础 0.16，每丢失一帧 +0.035，上限 12 帧。
        )
        prior_distance = np.linalg.norm(  # 粒子到预测中心的距离。
            self.states[:, :2] - predicted_center, axis=1  # 二维距离。
        )
        motion_prior = np.exp(  # 运动先验：离预测越远可能性越低。
            -0.5 * (prior_distance / max(continuity_sigma, 1.0)) ** 2  # 高斯衰减。
        )
        prior_floor = min(  # 先验下限：丢失很久后必须允许「完全不看运动先验」的重定位。
            0.18, 0.015 + 0.012 * min(self.frames_since_reliable, 14)  # 从 0.015 逐步括到上限 0.18。
        )
        likelihood *= prior_floor + (1.0 - prior_floor) * motion_prior  # 先验按下限混合，保留一定的全局探索能力。
        self.weights *= likelihood + 1e-10  # 权重乘似然，加极小量避免全零。
        self._normalize_weights()  # 归一化。
        weighted_score = float(np.sum(self.weights * particle_evidence.scores))  # 加权证据得分。
        weighted_coverage = float(np.sum(self.weights * particle_evidence.coverage))  # 加权覆盖率。
        border_snr = (weighted_score - control_median) / control_sigma  # 相对对照组的信噪比。
        evidence_confidence = float(  # 证据置信度：信噪比与覆盖率两项相乘，缺一不可。
            np.clip((border_snr - 1.15) / 4.5, 0.0, 1.0)  # 信噪比项：1.15 起步，4.5 跨度。
            * np.clip((weighted_coverage - 0.16) / 0.50, 0.0, 1.0)  # 覆盖率项：0.16 起步，0.50 跨度。
        )
        self.last_border_score = weighted_score  # 记录得分供诊断。
        self.last_border_snr = border_snr  # 记录信噪比供诊断。
        center, _, _ = self.estimate()  # 加权后的新中心。
        correction = float(np.linalg.norm(center - predicted_center))  # 本帧修正幅度。
        maximum_correction = max(  # 允许的最大单帧修正。
            20.0, self.template.nominal_size * self.max_correction_ratio  # 下限 20 像素，否则按标称尺寸比例。
        )
        evidence_confidence *= math.exp(  # 修正幅度越大，置信度越高扣（软惩罚）。
            -0.5 * (correction / max(maximum_correction * 0.75, 1.0)) ** 2  # 高斯惩罚，标准差为上限的 75%。
        )
        if correction > maximum_correction:  # 单帧修正超限：认定这一帧证据不可信，整体回滚。
            self.states = predicted_states  # 恢复状态。
            self.weights = predicted_weights  # 恢复权重。
            self.scale = predicted_scale  # 恢复尺度。
            self.pending_reliable_hits = 0  # 清零待定命中。
            self.confidence *= 0.94  # 置信度衰减。
            self.frames_since_reliable += 1  # 不可靠帧数 +1。
            return False  # 返回不可靠。
        reliable = evidence_confidence >= 0.18  # 置信度过阈即认为本帧可靠。
        if reliable:  # 可靠分支。
            self.confidence = 0.40 * self.confidence + 0.60 * evidence_confidence  # 置信度低通吸收证据置信度。
            if evidence_confidence >= 0.72:  # 强可靠：才有资格推进「已确认可靠中心」。
                if (  # 与待定中心的距离足够近，说明连续多帧指向同一位置。
                    np.linalg.norm(center - self.pending_reliable_center)  # 两者距离。
                    <= self.template.nominal_size * 0.70  # 阈值为标称尺寸的 70%。
                ):
                    self.pending_reliable_hits += 1  # 命中次数 +1。
                else:
                    self.pending_reliable_hits = 1  # 位置跳了，重新开始计数。
                self.pending_reliable_center = center.copy()  # 更新待定中心。
                if self.pending_reliable_hits >= 4:  # 连续 4 帧强可靠且位置一致才正式确认。
                    self.last_reliable_center = center.copy()  # 更新已确认可靠中心。
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
        self._resample_if_needed(force=reliable)  # 可靠时强制重采样以集中粒子，否则按需。
        return reliable  # 返回本帧是否可靠。

    def _normalize_weights(self) -> None:
        """归一化权重；全部退化时重置为均匀分布，避免数值崩溃。"""

        total = float(np.sum(self.weights))  # 权重总和。
        if not np.isfinite(total) or total <= 1e-18:  # 出现 NaN/Inf 或权重全部衰减到 0。
            self.weights.fill(1.0 / self.count)  # 重置为均匀分布，等价于放弃当前假设重新开始。
        else:
            self.weights /= total  # 正常归一化。

    def _resample_if_needed(self, force: bool) -> None:
        """系统重采样：有效样本数过低或强制时重建粒子群，并对一部分粒子加抖动。"""

        effective = 1.0 / float(np.sum(self.weights**2))  # 有效样本数（Kish 有效样本量）。
        if not force and effective >= self.count * 0.56:  # 未强制且粒子多样性还够。
            return  # 不重采样，保留现有假设分布。
        positions = (self.rng.random() + np.arange(self.count)) / self.count  # 系统重采样的均匀分层位置。
        cumulative = np.cumsum(self.weights)  # 权重累积分布。
        indexes = np.searchsorted(cumulative, positions, side="right")  # 查找每个分层位置对应的粒子索引。
        indexes = np.clip(indexes, 0, self.count - 1)  # 防止浮点误差越界。
        self.states = self.states[indexes].copy()  # 按权重重建粒子群（拷贝避免视图共享）。
        self.weights.fill(1.0 / self.count)  # 重采样后权重重新均匀。
        jitter_count = max(10, self.count // 16)  # 需要抖动的粒子数，保留探索能力。
        chosen = self.rng.choice(self.count, jitter_count, replace=False)  # 不重复地选出抖动粒子。
        uncertainty = 1.0 - float(np.clip(self.confidence, 0.0, 1.0))  # 不确定度：置信度低时抖动更大。
        self.states[chosen, :2] += self.rng.normal(  # 位置抖动。
            0.0,  # 零均值。
            1.0 + self.template.nominal_size * 0.10 * uncertainty,  # 基准 1 像素 + 随不确定度放大的形状尺度项。
            size=(jitter_count, 2),  # 二维噪声。
        )
        self.states[chosen, 2:4] += self.rng.normal(  # 速度抖动。
            0.0, 0.8 + uncertainty, size=(jitter_count, 2)  # 基准 0.8。
        )
        self.states[chosen, 4] += self.rng.normal(  # 角度抖动。
            0.0, 2.0 + min(10.0, self.template.symmetry_period * 0.10) * uncertainty,  # 基准 2 度，周期项上限 10。
            jitter_count,  # 一维噪声。
        )
        self._apply_bounds(self.states)  # 抖动后重新施加约束。
