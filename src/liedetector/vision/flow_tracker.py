# -*- coding: utf-8 -*-
"""稀疏 LK 光流单目标追踪器：身份锁定的 bbox 由像素逐帧携带。

流程（每次 update）：
1. 特征点：播种时在 bbox 内 goodFeaturesToTrack（≤FLOW_MAX_POINTS，天然落在轮廓带）；
2. LK：calcOpticalFlowPyrLK 金字塔前向追踪；
3. 前向-后向校验：回track距离 > FLOW_FB_THRESHOLD 的点剔除；
4. 运动门控：点位移偏离中位数 > FLOW_MOTION_GATE_SIGMA×MAD 的离群点剔除
   （目标穿越干扰物时，被拉到干扰物轮廓上的点被当作离群点丢弃）；
5. 存活点 < FLOW_MIN_POINTS 判死亡；
6. bbox 整体平移中位位移；与种子 patch 做 TM_CCOEFF_NORMED，低于 FLOW_MIN_NCC 判死亡
   （防缓慢漂移与外观崩坏累积）。

死亡语义：调用方（FlowSession）负责转入 YOLO 重获流程。重获成功后重新 start() 重播种。
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision.logger import get_logger

log = get_logger("flow")

BBox = Tuple[float, float, float, float]

_LK_PARAMS_TEMPLATE = dict(winSize=(21, 11), maxLevel=3,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _clamp_box(box: BBox, w: int, h: int, margin: int = 2) -> Optional[BBox]:
    """bbox 完全出帧（留 margin 容差）返回 None；部分越界夹回帧内。"""
    x1, y1, x2, y2 = box
    if x2 < margin or y2 < margin or x1 > w - margin or y1 > h - margin:
        return None
    return (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))


def _ring_band_mask(box: BBox, shape) -> Optional[np.ndarray]:
    """框边内向内 18% 的环带掩膜：透明图形唯一可追踪的像素是描边，而描边贴着
    检测框边缘；框内其余区域是透出地形（静止）——把点撒在全框会形成
    「地形 0 位移 vs 星体位移」的双峰分布，中位数跟随多数派必然跟错。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2), min(shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    band = max(6, int(0.18 * min(x2 - x1, y2 - y1)))
    mask = np.zeros(shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    ix1, iy1 = min(x2, x1 + band), min(y2, y1 + band)
    ix2, iy2 = max(x1, x2 - band), max(y1, y2 - band)
    mask[iy1:iy2, ix1:ix2] = 0  # 挖掉内部。
    return mask


def _ring_grid_points(box: BBox, shape) -> np.ndarray:
    """环带上的均匀网格点（描边带兜底/补足）。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2), min(shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return np.empty((0, 2), np.float32)
    band = max(6, int(0.18 * min(x2 - x1, y2 - y1)))
    gx = np.linspace(x1 + 2, x2 - 2, num=8, dtype=np.float32)
    gy = np.linspace(y1 + 2, y2 - 2, num=6, dtype=np.float32)
    pts = []
    for y in gy:
        for x in gx:
            # 只留落在环带内的网格点（贴四边）。
            if (x < x1 + band or x > x2 - band or y < y1 + band or y > y2 - band):
                pts.append((x, y))
    return np.array(pts, np.float32) if pts else np.empty((0, 2), np.float32)


def _seed_points(gray: np.ndarray, box: BBox, max_points: int) -> np.ndarray:
    """环带内取角点；不足用环带网格补足（绝不取框内平原——那是透出的静止地形）。"""
    mask = _ring_band_mask(box, gray.shape)
    if mask is None:
        return np.empty((0, 2), np.float32)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=0.01,
                                  minDistance=3, mask=mask)
    out = []
    if pts is not None and len(pts) > 0:
        out.append(pts.reshape(-1, 2))
    if len(out) == 0 or len(out[0]) < config.FLOW_MIN_POINTS:
        out.append(_ring_grid_points(box, gray.shape))
    merged = np.concatenate(out, axis=0)[:max_points]
    return merged.astype(np.float32).reshape(-1, 1, 2)


def _patch(gray: np.ndarray, box: BBox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return gray[y1:y2, x1:x2]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """TM_CCOEFF_NORMED 单值：两 patch 尺寸不一致时缩放 b 到 a。"""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return -1.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])


class FlowTracker:
    """单目标 LK 光流载波。start 播种 → update 逐帧推进 → None/alive=False 死亡。"""

    def __init__(self):
        self._prev_gray = None  # 上一帧灰度图（LK 的前一拍）。
        self._seed_patch = None  # 播种 patch：NCC 防漂移的基准。
        self._points = None  # 当前特征点 (N,1,2) float32。
        self._bbox = None  # 当前携带的 bbox。
        self._age = 0  # 已携带帧数（重播种时清零）。
        self.alive = False

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    @property
    def age(self) -> int:
        return self._age

    def start(self, gray: np.ndarray, box: BBox) -> None:
        """播种/重播种：清年龄，取特征点，存种子 patch。"""
        h, w = gray.shape[:2]
        box = _clamp_box(tuple(float(v) for v in box), w, h)
        if box is None:
            self.alive = False
            return
        self._points = _seed_points(gray, box, config.FLOW_MAX_POINTS)
        self._seed_patch = _patch(gray, box)
        self._prev_gray = gray
        self._bbox = box
        self._age = 0
        self.alive = bool(self._seed_patch is not None and len(self._points) > 0)

    def reset(self) -> None:
        self._prev_gray = None
        self._seed_patch = None
        self._points = None
        self._bbox = None
        self._age = 0
        self.alive = False

    def update(self, gray: np.ndarray) -> Optional[Tuple[BBox, Tuple[float, float]]]:
        """推进一帧。存活返回 (新bbox, (dx,dy))；死亡返回 None 且 alive=False。"""
        if not self.alive or self._prev_gray is None or self._points is None:
            return None
        h, w = gray.shape[:2]
        prev = self._prev_gray
        # 前向追踪（返回顺序：nextPts, status, err）。
        pts_f, st_f, _ = cv2.calcOpticalFlowPyrLK(prev, gray, self._points, None, **_LK_PARAMS_TEMPLATE)
        if pts_f is None:
            self._die("lk failed")
            return None
        p0 = self._points.reshape(-1, 2)
        pf = pts_f.reshape(-1, 1, 2).reshape(-1, 2) if pts_f.ndim == 3 else pts_f.reshape(-1, 2)
        good = st_f.reshape(-1) == 1
        finite = np.isfinite(pf).all(axis=1)  # 前向失败的点可能是 inf/NaN：OpenCV5 断言拒绝。
        if not finite.any():
            self._die("all points non-finite")
            return None
        # 后向回track做往返校验（仅喂有限且前向成功的点）。
        valid = good & finite
        if valid.sum() < 1:
            self._die("no valid forward points")
            return None
        pts_b, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev,
            pf[valid].reshape(-1, 1, 2).astype(np.float32), None, **_LK_PARAMS_TEMPLATE)
        if pts_b is None:
            self._die("lk backward failed")
            return None
        pb = pts_b.reshape(-1, 2)
        pf_v = pf[valid]
        # FB 误差 = 原始点与后向回落点的距离（一致性校验），不是前向点与回落点的距离。
        fb_err = np.linalg.norm(p0[valid] - pb, axis=1)
        keep_valid = fb_err <= config.FLOW_FB_THRESHOLD
        disp_all = pf_v - p0[valid]
        if keep_valid.sum() >= 3:  # 运动门控：偏离中位位移的离群点（穿越时被干扰物轮廓拉走的点）。
            d = disp_all[keep_valid]
            med = np.median(d, axis=0)
            mad = np.median(np.linalg.norm(d - med, axis=1)) + 1e-6
            gate = (np.linalg.norm(d - med, axis=1) <= config.FLOW_MOTION_GATE_SIGMA * mad)
            idx = np.nonzero(keep_valid)[0][gate]  # 相对 valid 子集的下标。
        else:
            idx = np.nonzero(keep_valid)[0]
        if len(idx) < config.FLOW_MIN_POINTS:
            self._die(f"points {len(idx)} < {config.FLOW_MIN_POINTS}")
            return None
        displacement = tuple(float(v) for v in np.median(disp_all[idx], axis=0))
        new_box = (self._bbox[0] + displacement[0], self._bbox[1] + displacement[1],
                   self._bbox[2] + displacement[0], self._bbox[3] + displacement[1])
        new_box = _clamp_box(new_box, w, h)
        if new_box is None:
            self._die("box left frame")
            return None
        # NCC 双门：帧间参考（上一帧位置 patch vs 当前 patch）抓遮挡/闪变/外观崩坏；
        # 种子参考仅作粗筛（半透明轮廓下的地形随移动变化，与原始种子的相关性天然
        # 衰减，阈值必须很低——长时漂移由上层周期校验吸附兜底，不靠这里）。
        cur_patch = _patch(gray, new_box)
        ref_patch = _patch(self._prev_gray, self._bbox)
        for p in (cur_patch, ref_patch, self._seed_patch):
            if p is None or float(p.std()) < 2.0:  # 无结构（纯平）不可追踪。
                self._die("patch no structure")
                return None
        if _ncc(ref_patch, cur_patch) < config.FLOW_MIN_NCC:
            self._die("ncc frame-to-frame")
            return None
        if _ncc(self._seed_patch, cur_patch) < config.FLOW_SEED_NCC_FLOOR:
            self._die("ncc seed gross change")
            return None
        self._points = pf_v[idx].reshape(-1, 1, 2).astype(np.float32)
        self._bbox = new_box
        self._prev_gray = gray
        self._age += 1
        # 点位补充：存活点不足一半时在当前框内重新取特征点补足（平坦区的弱点会逐帧
        # 流失，不补充则长carrier必然枯竭）；新点下一帧才参与追踪，不影响本帧位移。
        if len(self._points) < config.FLOW_MAX_POINTS // 2:
            fresh = _seed_points(gray, new_box, config.FLOW_MAX_POINTS - len(self._points))
            if len(fresh) > 0:
                self._points = np.concatenate([self._points, fresh], axis=0)
        if self._age >= config.FLOW_CARRIER_MAX_FRAMES:  # 寿命上限：校验吸附会重播种清零。
            self._die("carrier max age")
            return None
        return new_box, displacement

    def _die(self, why: str) -> None:
        log.debug("flow tracker died: %s (age=%s)", why, self._age)
        self.reset()
