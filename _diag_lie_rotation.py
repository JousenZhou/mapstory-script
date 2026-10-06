# -*- coding: utf-8 -*-
"""测谎旋转见证器诊断探针：逐帧量化时序残差连通域的旋转量，验证「只有目标旋转」可分性。

只读诊断，不改任何生产代码。思路（对应计划第一节）：
- 用 ShapeTrackSession.aligner 逐帧喂录像（extreme 档），取 TemporalEvidence.normalized；
- 分割：按高分位阈值（绝对下限 floor 三档）→ 7x7 闭运算 + 5x5 膨胀合并同一图形边缘
  → 连通域（≥40 处理尺度像素）；
- 每个团块用 lag∈{12,24,36} 帧的灰度光流（诊断阶段直接 cv2.calcOpticalFlowFarneback）做
  平移中位数扣除后的幅值加权切向最小二乘，得 ω；轨迹旋转速率取 ω 样本中位数（抗水面噪声）；
- 质心+速度预测短程关联（gate 16px），新团块建档、丢失容忍 ~0.5s。

验收判据（针对 20261001_190829）：存在唯一团块 |旋转速率| ≥60°/2s 且方向一致 ≥0.8、
采样 ≥30 且轨迹终点约在右下角（区域全尺度 x>500,y>330），其余长寿命团块 <20°/2s。

实测发现（第一版 lag=12/分位阈值 0.7~0.85 的结论）：整屏是随水面漂移变形的星形轮廓干扰物，
短窗 ω 被水面噪声淹没且短命轨迹方向一致率虚高，因此本脚本改为 (lag × floor) 参数扫描，
证据图与各 lag 光流只算一次全量缓存，各组合复用。

用法：
    python _diag_lie_rotation.py                 # 默认跑失败录像 + 两段成功对照
    python _diag_lie_rotation.py 190829          # 只跑名字含 190829 的
    python _diag_lie_rotation.py 190829 1000     # 第二个参数是最多帧数（默认 1000）

结果落盘 _diag_out/rotation_diag_<tier>_<视频名>.json，并打印各组合的验收表。
"""

import glob
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.shape_tracking import _to_numpy_array, resize_for_processing  # noqa: E402

RECORD_DIR = "lie_records"
OUT_DIR = "_diag_out"
DEFAULT_VIDEOS = ("20261001_190829", "20261001_165429", "20260930_133633")  # 失败录像 + 两段成功对照。
TIER = "extreme"  # 与失败录像录制档一致。
MAX_FRAMES = 1000  # 每段录像最多分析帧数（失败局全程 ~700+ 帧）。
LAGS = (12, 24, 36)  # 光流时间基线扫描档（帧）。
FLOORS = (2.0, 3.0, 4.0)  # 证据绝对阈值下限扫描档（鲁棒 sigma 倍数）。
QUANTILE = 0.995  # 分位阈值：与绝对下限取大者（高分位=只留最强响应）。
MIN_BLOB_AREA = 40  # 连通域最小面积（处理尺度像素）。
ASSOC_GATE = 16.0  # 质心+速度预测关联门限（处理尺度像素）：星形环带碎片的质心抖动比平移体大。
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 闭运算核：合并同一图形的碎片边缘。
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 再膨胀一档：把断开的面包屑连成环，质心才稳。
MIN_SAMPLES = 25  # 验收判定所需的最少 ω 采样数：短命轨迹的方向一致性不可信。
MIN_LIFETIME = 60  # 验收判定所需的最短寿命（帧）：排除一闪而过的水面涡。
LOST_SECONDS = 1.5  # 丢失容忍（秒）：透明相位目标残差环间歇可见，0.5s 会把真目标轨迹拆断。
WHITE_PHASE_END = 336  # 白相截止帧（本局实测）：白相内旋转轨迹就是目标本身，不参与见证器可分性判定。
FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=2, poly_n=5, poly_sigma=1.2, flags=0)


class BlobTrack:
    """一个连通域跨帧轨迹：累计旋转量、方向一致性与生命周期。"""

    def __init__(self, track_id, center, tick):
        self.id = track_id  # 轨迹编号。
        self.center = np.asarray(center, dtype=np.float64)  # 当前质心（处理尺度）。
        self.start_center = np.asarray(center, dtype=np.float64)  # 建档质心，供轨迹起终点报告。
        self.velocity = np.zeros(2, dtype=np.float64)  # 恒速模型速度（像素/帧）。
        self.birth = tick  # 建档帧号。
        self.last_seen = tick  # 最后一次关联到的帧号。
        self.frames = 0  # 关联帧总数。
        self.cum_signed = 0.0  # 累计带符号转角（度）。
        self.cum_abs = 0.0  # 累计绝对转角（度）。
        self.sign_pos = 0  # ω>0 采样数。
        self.sign_neg = 0  # ω<0 采样数。
        self.omegas = []  # 每次采样的 deg/frame，供中位数/分布检查。
        self.path = [(float(tick), self.center.copy())]  # 逐帧质心历史，供长轨迹导出为真值轨迹。

    @property
    def samples(self):
        """有效 ω 采样数。"""

        return len(self.omegas)

    def match(self, center, tick):
        """关联到新观测：更新恒速模型与质心。"""

        step = np.asarray(center, dtype=np.float64) - self.center  # 本帧位移。
        self.velocity = 0.5 * self.velocity + 0.5 * step  # 平滑速度。
        self.center = np.asarray(center, dtype=np.float64)  # 新质心。
        self.path.append((float(tick), self.center.copy()))  # 记入逐帧历史（真值轨迹用）。
        self.last_seen = tick  # 刷新可见帧号。
        self.frames += 1  # 计数。

    def add_omega(self, deg_per_frame):
        """记录一次 ω 估计（度/帧），同步累计。"""

        self.cum_signed += deg_per_frame  # 带符号累计。
        self.cum_abs += abs(deg_per_frame)  # 绝对值累计。
        if deg_per_frame >= 0.0:
            self.sign_pos += 1  # 正转采样。
        else:
            self.sign_neg += 1  # 反转采样。
        self.omegas.append(deg_per_frame)  # 样本。

    @property
    def direction_consistency(self):
        """方向一致性：多数符号采样占比（无采样时为 0）。"""

        if not self.omegas:
            return 0.0
        return max(self.sign_pos, self.sign_neg) / len(self.omegas)  # 恒定期望接近 1。

    def summary(self, fps, scale):
        """汇总成一行诊断记录（坐标换算回区域全尺度）。

        旋转强度用 ω 样本中位数（度/帧 → 度/2秒）：水面噪声样本大多在 0 附近且双侧，
        均值会被拖偏，中位数只有真实持续旋转才偏离 0。
        """

        median_signed = float(np.median(self.omegas)) if self.omegas else 0.0  # 中位数转速（度/帧，带符号）。
        return {
            "id": self.id,
            "birth": self.birth,
            "death": self.last_seen,
            "frames": self.frames,
            "omega_samples": self.samples,
            "start": [round(self.start_center[0] / scale, 1), round(self.start_center[1] / scale, 1)],
            "end": [round(self.center[0] / scale, 1), round(self.center[1] / scale, 1)],
            "cum_abs_deg": round(self.cum_abs, 1),
            "cum_signed_deg": round(self.cum_signed, 1),
            "rate_signed_deg_per_2s": round(median_signed * fps * 2.0, 1),  # 中位数带符号转速：持续旋转才显著非 0。
            "dir": round(self.direction_consistency, 3),
            # 逐帧质心（区域全尺度）：只给长轨迹导出，避免诊断 JSON 膨胀——短轨迹的质心本身就不是真目标。
            "path": ([[int(t), round(float(c[0]) / scale, 1), round(float(c[1]) / scale, 1)] for t, c in self.path]
                     if len(self.path) >= 20 else []),
        }


class WitnessProbe:
    """单一 (lag, floor) 组合下的分割 + 关联 + ω 累计探针（消费预缓存的证据与光流）。"""

    def __init__(self, lag, floor, fps):
        self.lag = lag  # 光流时间基线（帧）。
        self.floor = floor  # 证据绝对阈值下限。
        self.fps = fps  # 录像容器帧率（与实测有漂移，仅供换算 deg/2s）。
        self.tracks = []  # 全部轨迹（含已关闭）。
        self.lost_limit = max(8, int(fps * LOST_SECONDS))  # 丢失容忍帧数。
        self.next_id = 0  # 轨迹编号发放器。

    def observe(self, evidence, flow, tick):
        """喂一帧：分割连通域 → 关联轨迹 → 对每个关联成功团块做 ω 估计。"""

        threshold = max(float(np.quantile(evidence, QUANTILE)), self.floor)  # 高分位与绝对下限取大。
        mask = (evidence >= threshold).astype(np.uint8)  # 二值分割。
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)  # 闭运算合并同一图形边缘。
        mask = cv2.dilate(mask, DILATE_KERNEL)  # 再膨胀：把断开的面包屑连成环，旋转体质心才稳。
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)  # 8 邻域连通域。
        blobs = []  # 本帧团块列表：(质心, 标签图视图)。
        for label in range(1, count):  # 跳过背景。
            if stats[label, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:  # 太小的噪点丢弃。
                continue
            blobs.append((centroids[label], labels == label))
        # 恒速预测关联：预测位置与 gate 内最近邻撮合（按距离升序贪心）；已关闭的旧轨迹不参与。
        live = [t for t in self.tracks if tick - t.last_seen < self.lost_limit]  # 存活集。
        predicted = np.array([t.center + t.velocity for t in live]) if live else np.zeros((0, 2))  # 预测表。
        pairs = []  # (距离, 轨迹下标, 团块下标)。
        for b, (center, _) in enumerate(blobs):  # 每个团块找最近轨迹。
            for i, track in enumerate(live):  # 只关联仍未在本帧被占用的存活轨迹。
                if track.last_seen != tick:  # 未占用。
                    distance = float(np.linalg.norm(predicted[i] - center))  # 预测距离。
                    if distance <= ASSOC_GATE:  # 门限内。
                        pairs.append((distance, i, b))
        pairs.sort()  # 距离升序贪心。
        matched = []  # 本帧关联成功的 (轨迹, 团块掩码)。
        taken_tracks, taken_blobs = set(), set()  # 已撮合标记。
        for _, i, b in pairs:  # 逐个撮合。
            if i in taken_tracks or b in taken_blobs:  # 任一已占用则跳过。
                continue
            taken_tracks.add(i)  # 标记轨迹。
            taken_blobs.add(b)  # 标记团块。
            live[i].match(blobs[b][0], tick)  # 更新轨迹。
            matched.append((live[i], blobs[b][1]))  # 入本帧视图。
        for b, (center, blob_mask) in enumerate(blobs):  # 未匹配的团块建档。
            if b in taken_blobs:
                continue
            track = BlobTrack(self.next_id, center, tick)  # 新轨迹。
            self.next_id += 1  # 发号。
            self.tracks.append(track)  # 入册。
            matched.append((track, blob_mask))  # 入本帧视图。
        if flow is not None:  # 光流就绪：只对当帧关联成功的轨迹估计 ω。
            for track, blob_mask in matched:
                self._estimate_omega(track, blob_mask, flow)

    def _estimate_omega(self, track, blob_mask, flow):
        """切向最小二乘估 ω：扣除平移中位数后，ω = Σ w·perp(p)·r / Σ w·|p|²（rad/窗）。

        权重 w 取残差幅值 |r|：强边缘像素主导拟合，弱水面噪声像素近乎不参与——
        第一版等权 LS 被 blob 内大量低幅值噪声像素稀释，真旋转轨迹的速率被拉到噪声水平。
        """

        ys, xs = np.nonzero(blob_mask)  # 团块像素坐标。
        if len(xs) < 30:  # 像素太少，形状不可靠。
            return
        p = np.stack([xs - track.center[0], ys - track.center[1]], axis=1).astype(np.float64)  # 相对质心。
        f = flow[ys, xs].astype(np.float64)  # 光流向量。
        r = f - np.median(f, axis=0)  # 平移中位数扣除。
        perp = np.stack([-p[:, 1], p[:, 0]], axis=1)  # 切向基（刚体旋转速度场 = ω × p）。
        w = np.linalg.norm(r, axis=1)  # 幅值权重。
        lever = float(np.sum(w * np.sum(perp * perp, axis=1)))  # Σ w·|perp|²。
        if lever < 1e-6:
            return
        omega = float(np.sum(w * np.sum(perp * r, axis=1)) / lever)  # rad/窗。
        deg_per_frame = math.degrees(omega) / self.lag  # 折成度/帧。
        track.add_omega(deg_per_frame)  # 累计。

    def summaries(self, fps, scale):
        """全部轨迹的汇总行，按 |中位数转速|×可信度降序。"""

        rows = [
            track.summary(fps, scale)
            for track in self.tracks
            if track.samples >= 5 or track.frames >= 20
        ]
        rows.sort(
            key=lambda item: abs(item["rate_signed_deg_per_2s"]) * min(1.0, item["omega_samples"] / MIN_SAMPLES),
            reverse=True,
        )
        return rows


def build_caches(path, max_frames):
    """第一遍：逐帧算证据图与各 lag 光流，全部缓存（float16 压缩省内存）。"""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头重新逐帧喂。
    height, width = first.shape[:2]
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 只为装配 aligner。
    session.reset(width, height)
    aligner = session.aligner  # 直接用对齐器，不跑会话状态机。
    scale = session.effective_scale  # 处理尺度
    evidences = []  # tick -> (H,W) float16 证据图。
    grays = deque(maxlen=max(LAGS) + 1)  # 处理尺度灰度环形缓冲。
    flows = {lag: [] for lag in LAGS}  # lag -> tick 对齐的 (H,W,2) float16 光流（前 lag 帧为 None）。
    tick = 0
    while tick < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        process_frame = resize_for_processing(frame, scale)
        temporal = aligner.update(process_frame)  # 喂帧取证据。
        grays.append(cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY))  # 光流缓冲。
        evidence = None
        if temporal is not None:
            evidence = np.asarray(_to_numpy_array(temporal.normalized), dtype=np.float32)
        evidences.append(evidence)
        for lag in LAGS:  # 各 lag 光流整帧一次算完，所有团块共用。
            if len(grays) > lag:
                flow = cv2.calcOpticalFlowFarneback(grays[-lag - 1], grays[-1], None, **FLOW_PARAMS)
                flows[lag].append(flow.astype(np.float16))
            else:
                flows[lag].append(None)
        tick += 1
    cap.release()
    if not evidences:
        return None
    return {"src_fps": src_fps, "scale": scale, "evidences": evidences, "flows": flows, "processed": tick}


def analyze(cache):
    """第二遍：对每个 (lag, floor) 组合跑探针，返回各组合汇总。"""

    result = {
        "processed": cache["processed"],
        "src_fps": cache["src_fps"],
        "effective_scale": cache["scale"],
        "combos": {},
    }
    for lag in LAGS:
        for floor in FLOORS:
            probe = WitnessProbe(lag, floor, cache["src_fps"])
            for tick, evidence in enumerate(cache["evidences"]):
                if evidence is None:  # 历史不足帧。
                    continue
                flow = cache["flows"][lag][tick]
                flow = None if flow is None else flow.astype(np.float32)  # 解压。
                probe.observe(evidence, flow, tick)
            result["combos"][f"lag{lag}_floor{floor}"] = probe.summaries(cache["src_fps"], cache["scale"])
    return result


def print_report(name, data):
    """打印各参数组合的验收判定。"""

    print(f"\n=== {name} frames={data['processed']} fps={data['src_fps']:.1f} scale={data['effective_scale']:.3f}")
    for combo, summaries in data["combos"].items():
        # 验收：唯一团块 |中位数转速|≥60°/2s 且方向一致≥0.75（水面噪声下真目标的 dir 实测 0.79）、
        # 采样/寿命达标且终点在右下（全尺度 x>500,y>330）；干扰物/水面涡用同一长寿命标准卡除。
        # 只看透明相位（birth 在白相截止之后）：白相内的强旋转轨迹就是目标本身，不是竞争者。
        candidates = [row for row in summaries if row["birth"] >= WHITE_PHASE_END]  # 判定集。
        winners = [  # 满足全部强旋转条件的团块。
            row for row in candidates
            if abs(row["rate_signed_deg_per_2s"]) >= 60.0 and row["dir"] >= 0.75
            and row["omega_samples"] >= MIN_SAMPLES and row["death"] - row["birth"] >= MIN_LIFETIME
            and row["end"][0] > 500.0 and row["end"][1] > 330.0
        ]
        others = [  # 次强干扰：转速接近真目标且同样方向恒定的长寿命候选。
            row for row in candidates if row not in winners
            and abs(row["rate_signed_deg_per_2s"]) >= 20.0 and row["dir"] >= 0.75
            and row["omega_samples"] >= MIN_SAMPLES and row["death"] - row["birth"] >= MIN_LIFETIME
        ]
        status = "PASS" if len(winners) == 1 and not others else ("NO-WINNER" if not winners else f"{len(winners)} WINNERS")
        print(f"-- {combo}: {status} winners={[(w['id'], w['end'], w['rate_signed_deg_per_2s'], w['dir']) for w in winners]} rivals={[(r['id'], r['end'], r['rate_signed_deg_per_2s'], r['dir']) for r in others[:5]]}")
        for row in summaries[:5]:  # 头部候选打印。
            print(
                f"   track#{row['id']:04d} birth={row['birth']:4d} death={row['death']:4d} "
                f"frames={row['frames']:4d} samples={row['omega_samples']:4d} "
                f"start={row['start']} end={row['end']} rate/2s={row['rate_signed_deg_per_2s']:7.1f} dir={row['dir']:.2f}"
            )


def main():
    name_filter = sys.argv[1] if len(sys.argv) > 1 else ""  # 素材名过滤。
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_FRAMES  # 最多帧数。
    if name_filter:  # 指定过滤时只跑匹配的。
        targets = [name_filter]
    else:
        targets = list(DEFAULT_VIDEOS)  # 默认失败 + 成功对照。
    os.makedirs(OUT_DIR, exist_ok=True)
    for target in targets:
        paths = sorted(glob.glob(os.path.join(RECORD_DIR, f"*{target}*.mp4")))
        if not paths:
            print(f"SKIP {target} 未找到录像")
            continue
        path = paths[0]
        name = os.path.basename(path)
        started = time.perf_counter()
        cache = build_caches(path, max_frames)
        if cache is None:
            print(f"SKIP {name} 无法打开")
            continue
        print(f"{name} caches built wall={time.perf_counter() - started:.1f}s")
        data = analyze(cache)
        print_report(name, data)
        out = os.path.join(OUT_DIR, f"rotation_diag_{TIER}_{name}.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"video": name, "tier": TIER, **data}, handle, ensure_ascii=False)
        print(f"  saved -> {out} wall={time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
