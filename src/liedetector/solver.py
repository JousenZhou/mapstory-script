# 透明图形求解器移植，逐段对照参考项目 solvers/shape.rs 的 TransparentShapeSolver：
# 初始锁定区域中心最近轨迹 -> 背景方向估计 -> 夹角评分+高斯距离惩罚 ->
# 连续 2 帧最高分才切换目标 -> 光标取卡尔曼中心+1 步速度，丢失时 1.5 倍速度外推。
import math

from src.liedetector.tracker import ByteTracker  # 导入移植的 ByteTrack 跟踪器。


def _norm(x, y):  # 向量模长。
    return math.sqrt(x * x + y * y)


def _unit(x, y):  # 归一化为单位向量，零向量返回 None，对应参考项目 unit。
    norm = _norm(x, y)
    if norm < 1e-3:
        return None
    return x / norm, y / norm


def _mid_point(rect):  # 矩形中心点，对应参考项目 mid_point。
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def _diag(rect):  # 矩形对角线长度，对应参考项目 diag。
    return math.sqrt(rect[2] ** 2 + rect[3] ** 2)


def _predicted_center(track):  # 光标预测：卡尔曼矩形中心 + 1 步卡尔曼速度，对应参考项目 predicted_center。
    vx, vy = track.kalman_velocity
    cx, cy = _mid_point(track.kalman_rect)
    return cx + _round_half_up(vx), cy + _round_half_up(vy)


def _round_half_up(value):  # 对齐 Rust f32::round 的四舍五入语义。
    return int(math.floor(value + 0.5))


def _track_background_degree(track, bg_direction):  # 轨迹速度方向与背景方向的夹角（度），对应参考项目 track_background_degree。
    unit_velocity = _unit(*track.kalman_velocity)
    if unit_velocity is None:
        return None
    dot = unit_velocity[0] * bg_direction[0] + unit_velocity[1] * bg_direction[1]
    det = unit_velocity[0] * bg_direction[1] - unit_velocity[1] * bg_direction[0]
    return abs(math.degrees(math.atan2(det, dot)))


def _track_background_score(track, last_cursor, bg_direction, region):  # 轨迹打分，对应参考项目 track_background_score。
    angle = _track_background_degree(track, bg_direction)
    if angle is None:
        return None
    if angle <= 45.0:  # 与背景方向夹角过小视为背景图形，直接淘汰。
        return None
    score = angle / 180.0
    if angle >= 60.0:  # 夹角足够大无需距离惩罚。
        distance_penalty = 1.0
    else:  # 可疑夹角区间：离上次光标越远惩罚越重，防止锁定远处斜飘的背景图形。
        cursor_dir = (
            _mid_point(track.rect)[0] - last_cursor[0],
            _mid_point(track.rect)[1] - last_cursor[1],
        )
        cursor_squared = cursor_dir[0] ** 2 + cursor_dir[1] ** 2
        sigma = 0.25 * _diag(region)
        distance_penalty = math.exp(-cursor_squared / (2.0 * sigma ** 2))
    if distance_penalty <= 0.3:  # 惩罚过重直接淘汰。
        return None
    return score * distance_penalty


def _estimate_background_direction(last_cursor, tracks):  # 背景方向估计，对应参考项目 estimate_background_direction。
    last_rect_contains_cursor = None  # 首个包含光标的轨迹框，其后重叠框一并排除。
    velocities = []
    for track in tracks:
        if track.tracklet_len < 5:  # 速度估计需要至少 5 帧连续命中。
            continue
        if last_rect_contains_cursor is not None:
            if _intersection_area(last_rect_contains_cursor, track.rect) > 0:
                continue
        if last_cursor is not None:
            rect = track.rect
            if _rect_contains(rect, last_cursor):  # 包含光标的轨迹是目标本身，不能作为背景样本。
                if last_rect_contains_cursor is None:
                    last_rect_contains_cursor = rect
                continue
            mid = _mid_point(rect)
            if _norm(mid[0] - last_cursor[0], mid[1] - last_cursor[1]) < _diag(rect):  # 离光标太近的轨迹同样排除。
                continue
        velocities.append(track.kalman_velocity)
    if len(velocities) < 3:  # 样本不足保留旧方向。
        return None
    sum_x = sum(v[0] for v in velocities)
    sum_y = sum(v[1] for v in velocities)
    return _unit(sum_x, sum_y)


def _rect_contains(rect, point):  # 点是否在矩形内（含边界）。
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def _intersection_area(a, b):  # 两个矩形的交集面积。
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


class TransparentShapeSolver:  # 透明图形求解器：跟踪 + 目标选择 + 光标预测，参数与参考完全一致。

    def __init__(self, fps=30):
        self.tracker = ByteTracker(fps=fps)  # 参考项目用 FPS=30 构造，丢失容忍 30 帧。
        self.current_track_id = None  # 当前锁定的目标轨迹 ID。
        self.candidate_track_id = None  # 候选切换轨迹 ID，用于连续 2 帧防抖。
        self.candidate_track_count = 0  # 候选轨迹连续占据最高分的帧数。
        self.last_cursor = None  # 上次输出的光标位置（区域局部坐标）。
        self.last_velocity = None  # 上次目标轨迹的卡尔曼速度。
        self.bg_direction = (0.0, 0.0)  # 平滑后的背景运动方向单位向量。

    def solve(self, region, detections):  # 输入区域 (x, y, w, h) 与区域局部检测 [(tlwh, score)]，返回 (绝对光标, 目标轨迹或 None)。
        rx, ry, rw, rh = region
        tracks = self.tracker.update(detections)
        self._update_initial_track_if_needed((0, 0, rw, rh), tracks)
        self._update_background_direction(tracks)
        best = self._update_and_find_best_track(tracks, (0, 0, rw, rh))
        if best is not None:
            next_cursor = _predicted_center(best)
            self.current_track_id = best.track_id
            self.last_cursor = next_cursor
            self.last_velocity = best.kalman_velocity
            return (rx + next_cursor[0], ry + next_cursor[1]), best
        if self.last_cursor is None:
            return None, None
        # 目标全部丢失：按上次速度的 1.5 倍外推，出界则停止输出。
        last_velocity = (self.last_velocity[0] * 1.5, self.last_velocity[1] * 1.5)
        next_cursor = (
            self.last_cursor[0] + _round_half_up(last_velocity[0]),
            self.last_cursor[1] + _round_half_up(last_velocity[1]),
        )
        absolute = (rx + next_cursor[0], ry + next_cursor[1])
        if not (rx <= absolute[0] < rx + rw and ry <= absolute[1] < ry + rh):
            return None, None
        self.last_cursor = next_cursor
        return absolute, None

    def _update_initial_track_if_needed(self, local_region, tracks):  # 尚无目标时锁定离区域中心最近的轨迹，对应参考项目同名函数。
        if self.current_track_id is not None:
            return
        region_mid = _mid_point(local_region)
        best_track = None
        best_distance = None
        for track in tracks:
            track_mid = _mid_point(track.rect)
            distance = _norm(region_mid[0] - track_mid[0], region_mid[1] - track_mid[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_track = track
        if best_track is not None:
            self.current_track_id = best_track.track_id
            self.last_cursor = _mid_point(best_track.rect)
            self.last_velocity = best_track.kalman_velocity

    def _update_background_direction(self, tracks):  # 背景方向估计 + 0.5/0.5 指数平滑，对应参考项目 update_background_direction。
        direction = _estimate_background_direction(self.last_cursor, tracks)
        if direction is None:
            return
        blended = (
            self.bg_direction[0] * 0.5 + direction[0] * 0.5,
            self.bg_direction[1] * 0.5 + direction[1] * 0.5,
        )
        unit_blended = _unit(*blended)
        if unit_blended is not None:
            self.bg_direction = unit_blended

    def _update_and_find_best_track(self, tracks, local_region):  # 打分选目标并做切换防抖，对应参考项目 update_and_find_best_track。
        if self.current_track_id is None or self.last_cursor is None:
            return None
        bg_direction = self.bg_direction
        best_track = None
        best_score = None
        for track in tracks:  # 当前目标或至少存活 1 帧的轨迹参与打分。
            if track.track_id != self.current_track_id and track.tracklet_len < 1:
                continue
            score = _track_background_score(track, self.last_cursor, bg_direction, local_region)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_track = track
        if best_track is not None:
            if best_track.track_id == self.current_track_id:  # 当前目标仍在榜首，清空候选。
                self.candidate_track_id = None
                self.candidate_track_count = 0
            if self.candidate_track_id == best_track.track_id:  # 候选轨迹再次占榜首，计数累加。
                self.candidate_track_count += 1
            else:  # 新的候选轨迹，计数归零重数。
                self.candidate_track_id = best_track.track_id
                self.candidate_track_count = 0
            if self.candidate_track_count >= 1:  # 连续 2 帧最高分才允许切换目标。
                self.candidate_track_id = None
                self.candidate_track_count = 0
                return best_track
        for track in tracks:  # 未切换时保持当前目标轨迹。
            if track.track_id == self.current_track_id:
                return track
        return None
