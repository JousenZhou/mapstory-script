# ByteTrack 多目标跟踪移植，逐段对照参考项目：
#   kalman_filter.rs -> KalmanXYAH
#   strack.rs        -> STrack
#   bytetracker.rs   -> ByteTracker
# 关联求解优先用 scipy 的 Jonker-Volgenant（与参考项目 lapjv 同族），
# 不可用时回退纯 numpy 的 LAPJV 短增广路径实现，结果与最优分配一致。
import numpy as np  # 导入 NumPy，用于卡尔曼滤波的矩阵运算与分配求解。

try:  # scipy 在本机可能被系统应用控制策略拦截其扩展模块，允许整体缺失。
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except Exception:
    _scipy_lsa = None

# 轨迹状态：与参考 TrackState 枚举一致，仅 Tracked / Lost 两种。
TRACKED = 1
LOST = 2

_next_track_id = 0  # 全局轨迹 ID 计数器，对应参考项目 STrack 的原子计数。


def _next_id():  # 分配新的轨迹 ID，语义等同 Rust 的 fetch_add。
    global _next_track_id
    _next_track_id += 1
    return _next_track_id


def tlwh_to_xyah(tlwh):  # 左上角+宽高 转 中心点+宽高比+高，对应参考项目同名函数。
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0, w / h if h > 0 else 0.0, h], dtype=np.float64)


def _iou_tlwh(a, b):  # 两个 tlwh 框的 IoU，对应参考项目 iou_tlwh。
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    inter_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    inter_area = inter_w * inter_h
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    return inter_area / (area_a + area_b - inter_area + 1e-6)


class KalmanXYAH:  # SORT 式卡尔曼滤波，8 维状态 [x, y, a, h, vx, vy, va, vh]，对应参考项目 KalmanXYAH。

    def __init__(self):  # 构造函数：建立运动矩阵与观测矩阵，噪声权重与参考一致。
        self.mean = np.zeros(8)  # 状态均值向量。
        self.covariance = np.eye(8)  # 状态协方差矩阵。
        self.motion_mat = np.eye(8)  # 状态转移矩阵：位置积分速度。
        for i in range(4):
            self.motion_mat[i, i + 4] = 1.0
        self.update_mat = np.zeros((4, 8))  # 观测矩阵：只观测前 4 维。
        for i in range(4):
            self.update_mat[i, i] = 1.0
        self.std_weight_pos = 1.0 / 20.0  # 位置噪声权重，与参考一致。
        self.std_weight_vel = 1.0 / 160.0  # 速度噪声权重，与参考一致。

    def initiate(self, measurement):  # 用首个观测初始化状态与协方差。
        self.mean = np.zeros(8)
        self.mean[:4] = measurement
        h = measurement[3]
        std = np.array([
            2.0 * self.std_weight_pos * h,
            2.0 * self.std_weight_pos * h,
            1e-2,
            2.0 * self.std_weight_pos * h,
            10.0 * self.std_weight_vel * h,
            10.0 * self.std_weight_vel * h,
            1e-5,
            10.0 * self.std_weight_vel * h,
        ])
        self.covariance = np.diag(std * std)

    def predict(self):  # 一步状态预测，过程噪声随目标高度缩放。
        h = self.mean[3]
        std = np.array([
            self.std_weight_pos * h,
            self.std_weight_pos * h,
            1e-2,
            self.std_weight_pos * h,
            self.std_weight_vel * h,
            self.std_weight_vel * h,
            1e-5,
            self.std_weight_vel * h,
        ])
        motion_cov = np.diag(std * std)
        self.mean = self.motion_mat @ self.mean
        self.covariance = self.motion_mat @ self.covariance @ self.motion_mat.T + motion_cov

    def project(self):  # 投影到观测空间，返回 (观测均值, 观测协方差)。
        h = self.mean[3]
        r_std = np.array([
            self.std_weight_pos * h,
            self.std_weight_pos * h,
            1e-1,
            self.std_weight_pos * h,
        ])
        r = np.diag(r_std * r_std)
        mean = self.update_mat @ self.mean
        cov = self.update_mat @ self.covariance @ self.update_mat.T + r
        return mean, cov

    def update(self, measurement):  # 用新观测做卡尔曼修正。
        projected_mean, projected_cov = self.project()
        ph_t = self.covariance @ self.update_mat.T
        kalman_gain = np.linalg.solve(projected_cov.T, ph_t.T).T  # 等价参考项目的 Cholesky 求解。
        innovation = measurement - projected_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ projected_cov @ kalman_gain.T

    def tlwh(self):  # 从卡尔曼状态还原左上角+宽高框。
        cx, cy, a, h = self.mean[0], self.mean[1], self.mean[2], self.mean[3]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h])


class STrack:  # 单条轨迹，对应参考项目 strack.rs。

    def __init__(self, tlwh, score):  # 用一帧检测框创建轨迹，初始为 Lost 待激活。
        self.track_id = 0  # 激活后分配的轨迹 ID。
        self.tracklet_len = 0  # 连续命中帧数。
        self.score = float(score)  # 最近一次关联的检测置信度。
        self.frame_id = 0  # 最近一次更新的帧号。
        self.start_frame_id = 0  # 激活帧号，去重时比较存活时长用。
        self.state = LOST  # 初始状态为丢失。
        self.kalman = KalmanXYAH()  # 每条轨迹独立的卡尔曼滤波器。
        self.tlwh = np.asarray(tlwh, dtype=np.float64)  # 最近一次观测框。

    def activate(self, frame_id, score):  # 激活为新轨迹并初始化卡尔曼。
        self.track_id = _next_id()
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.start_frame_id = frame_id
        self.state = TRACKED
        self.score = float(score)
        self.kalman.initiate(tlwh_to_xyah(self.tlwh))

    def reactivate(self, tlwh, frame_id, score):  # 丢失轨迹被重新关联，保留原 track_id。
        self.update(tlwh, frame_id, score)
        self.tracklet_len = 0

    def predict(self):  # 卡尔曼预测；非跟踪态清零宽高维速度，与参考一致。
        if self.state != TRACKED:
            self.kalman.mean[6] = 0.0
            self.kalman.mean[7] = 0.0
        self.kalman.predict()

    def update(self, tlwh, frame_id, score):  # 用新观测更新轨迹。
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.state = TRACKED
        self.score = float(score)
        self.kalman.update(tlwh_to_xyah(self.tlwh))

    def mark_lost(self):  # 标记为丢失。
        self.state = LOST

    @property
    def rect(self):  # 观测框取整，对应参考项目 rect()。
        return tuple(int(v) for v in self.tlwh)

    @property
    def kalman_tlwh(self):  # 卡尔曼平滑框，关联代价计算用。
        return self.kalman.tlwh()

    @property
    def kalman_rect(self):  # 卡尔曼平滑框取整，对应参考项目 kalman_rect()。
        return tuple(int(v) for v in self.kalman.tlwh())

    @property
    def kalman_velocity(self):  # 卡尔曼速度 (vx, vy)，单位像素/帧，对应参考项目 kalman_velocity()。
        return float(self.kalman.mean[4]), float(self.kalman.mean[5])


def _iou_distance(tracks, detections):  # 轨迹与检测的 1-IoU 代价矩阵；参考使用 IouGating::None，不做马氏门控。
    return [[1.0 - _iou_tlwh(t.kalman_tlwh, d.tlwh) for d in detections] for t in tracks]


def _hungarian_jv(cost):  # 纯 numpy 的 LAPJV 短增广路径求解，无外部扩展依赖，供 scipy 不可用时回退。
    n, m = cost.shape
    transposed = n > m  # 行数多于列数时转置求解再映射回来。
    c = cost.T if transposed else cost
    n, m = c.shape
    INF = np.inf
    u = np.zeros(n + 1)  # 行位势。
    v = np.zeros(m + 1)  # 列位势。
    p = np.zeros(m + 1, dtype=np.int64)  # 列->行匹配。
    way = np.zeros(m + 1, dtype=np.int64)  # 增广路径前驱。
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:  # 沿增广路径回溯翻转匹配。
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_to_row = p[1:]  # 每一列 c 匹配到的行（0 表示未匹配），对应参考项目 lapjv 的列分配结果。
    matched_cols = np.nonzero(col_to_row > 0)[0]  # 实际产生匹配的列下标。
    rows_c = col_to_row[matched_cols] - 1  # 匹配对中的行下标（矩阵 c 坐标系）。
    cols_c = matched_cols  # 匹配对中的列下标（矩阵 c 坐标系）。
    if transposed:  # 转置求解需映射回原坐标系：c 的行是原列，c 的列是原行。
        return cols_c, rows_c
    return rows_c, cols_c


def _solve_assignment(cost):  # 线性分配入口：优先 scipy，失败回退纯 numpy 实现。
    if _scipy_lsa is not None:
        try:
            try:  # scipy 1.14+ 提供与参考项目 lapjv 同族的 Jonker-Volgenant 求解。
                return _scipy_lsa(cost, method="lap")
            except TypeError:  # 旧版 scipy 退回匈牙利算法，最优解一致。
                return _scipy_lsa(cost)
        except Exception:  # 扩展模块被系统策略拦截等情况同样回退。
            pass
    return _hungarian_jv(np.asarray(cost, dtype=np.float64))


def _linear_assignment(costs, thresh):  # 线性分配：代价超过阈值视为未匹配，对应参考项目 linear_assignment。
    n = len(costs)
    m = len(costs[0]) if n > 0 else 0
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    cost = np.asarray(costs, dtype=np.float64)
    row, col = _solve_assignment(cost)
    matches, unmatched_a, unmatched_b = [], [], [True] * m
    for i, j in zip(row, col):
        if cost[i, j] <= thresh:
            matches.append((int(i), int(j)))
            unmatched_b[int(j)] = False
        else:
            unmatched_a.append(int(i))
    unmatched_b = [j for j, u in enumerate(unmatched_b) if u]
    return matches, unmatched_a, unmatched_b


def _remove_duplicate_stracks(a, b):  # 重叠轨迹去重：保留存活时间长的一条，对应参考项目同名函数。
    cost = _iou_distance(a, b)
    duplicate_a = [False] * len(a)
    duplicate_b = [False] * len(b)
    for i in range(len(a)):
        for j in range(len(b)):
            if cost[i][j] < 0.15:  # 代价小于 0.15 视为同一目标的两条轨迹。
                time_a = a[i].frame_id - a[i].start_frame_id
                time_b = b[j].frame_id - b[j].start_frame_id
                if time_a > time_b:
                    duplicate_b[j] = True
                else:
                    duplicate_a[i] = True
    return [t for i, t in enumerate(a) if not duplicate_a[i]], [t for j, t in enumerate(b) if not duplicate_b[j]]


class ByteTracker:  # ByteTrack 跟踪器，参数默认值与参考项目 TransparentShapeSolver 构造一致。

    def __init__(self, fps=30, high_match_score_threshold=0.25, low_match_score_threshold=0.1, new_track_score_threshold=0.25):
        self.initialized = False  # 首帧用全部检测初始化轨迹。
        self.tracked = []  # 当前跟踪中的轨迹。
        self.unconfirmed = []  # 未确认的新轨迹。
        self.lost = []  # 丢失待找回的轨迹。
        self.frame_id = 0  # 已处理帧数。
        self.max_time_lost = fps  # 丢失容忍帧数：参考项目传 FPS=30。
        self.high_match_score_threshold = high_match_score_threshold  # 高分档阈值。
        self.low_match_score_threshold = low_match_score_threshold  # 低分档阈值。
        self.new_track_score_threshold = new_track_score_threshold  # 新轨迹确认阈值。

    def update(self, detections):  # 输入 [(tlwh, score)]，返回当前跟踪中的轨迹列表。
        self.frame_id += 1
        for track in self.tracked:  # 全部轨迹先做卡尔曼预测。
            track.predict()
        for track in self.lost:
            track.predict()
        for track in self.unconfirmed:
            track.predict()
        low_tracks, high_tracks = [], []
        for bbox, score in detections:  # 过滤极低分检测后按高低分档。
            if score <= self.low_match_score_threshold:
                continue
            track = STrack(bbox, score)
            if self.low_match_score_threshold < track.score < self.high_match_score_threshold:
                low_tracks.append(track)
            else:
                high_tracks.append(track)
        if self.init(low_tracks, high_tracks):  # 首帧初始化后直接返回。
            return list(self.tracked)
        activated, unmatched_tracks, unmatched_detections = self._associate_high(high_tracks)
        lost = self._associate_low(unmatched_tracks, low_tracks, activated)
        unconfirmed = self._associate_unconfirmed(unmatched_detections, activated)
        tracked, lost = _remove_duplicate_stracks(activated, lost)
        self.tracked = tracked
        self.lost = lost
        self.unconfirmed = unconfirmed
        return list(self.tracked)

    def init(self, low_tracks, high_tracks):  # 首帧把全部检测激活为轨迹，对应参考项目 init。
        if self.initialized:
            return False
        self.initialized = True
        self.tracked = []
        for track in low_tracks + high_tracks:
            track.activate(self.frame_id, track.score)
            self.tracked.append(track)
        return True

    def _associate_high(self, detections):  # 高分检测与跟踪+丢失轨迹关联，阈值 0.5。
        current = self.tracked + self.lost
        cost = _iou_distance(current, detections)
        matches, unmatched_tracks, unmatched_detections = _linear_assignment(cost, 0.5)
        activated = []
        for ci, di in matches:
            track = current[ci]
            detection = detections[di]
            if track.state == TRACKED:
                track.update(detection.tlwh, self.frame_id, detection.score)
            else:
                track.reactivate(detection.tlwh, self.frame_id, detection.score)
            activated.append(track)
        return activated, [current[i] for i in unmatched_tracks], [detections[i] for i in unmatched_detections]

    def _associate_low(self, remain_tracks, detections, activated):  # 低分检测二次捞回，阈值 0.5；仍未匹配的按容忍时长转丢失。
        cost = _iou_distance(remain_tracks, detections)
        matches, unmatched_tracks, _ = _linear_assignment(cost, 0.5)
        lost = []
        for ci, di in matches:
            track = remain_tracks[ci]
            detection = detections[di]
            if track.state == TRACKED:
                track.update(detection.tlwh, self.frame_id, detection.score)
            else:
                track.reactivate(detection.tlwh, self.frame_id, detection.score)
            activated.append(track)
        for ci in unmatched_tracks:
            track = remain_tracks[ci]
            if self.frame_id - track.frame_id <= self.max_time_lost:  # 超过容忍时长的轨迹直接丢弃。
                track.mark_lost()
                lost.append(track)
        return lost

    def _associate_unconfirmed(self, detections, activated):  # 未确认轨迹与剩余高分检测关联，阈值 0.7。
        if not self.unconfirmed:  # 没有未确认轨迹时，高分新检测直接激活为未确认轨迹。
            result = []
            for track in detections:
                if track.score >= self.new_track_score_threshold:
                    track.activate(self.frame_id, track.score)
                    result.append(track)
            return result
        current_unconfirmed = self.unconfirmed
        cost = _iou_distance(current_unconfirmed, detections)
        matches, _, unmatched_detections = _linear_assignment(cost, 0.7)
        for ui, di in matches:
            track = current_unconfirmed[ui]
            detection = detections[di]
            track.update(detection.tlwh, self.frame_id, detection.score)
            activated.append(track)
        result = []  # 未匹配的未确认轨迹按参考实现直接丢弃。
        for di in unmatched_detections:
            track = detections[di]
            if track.score < self.new_track_score_threshold:
                continue
            track.activate(self.frame_id, track.score)
            result.append(track)
        return result
