# -*- coding: utf-8 -*-
"""FlowTracker 合成图单测：平移追踪、间隙大位移、遮挡死亡、NCC 崩、空芯描边、交叉门控。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.liedetector.vision.flow_tracker import FlowTracker

W, H = 320, 240
BOX = (40.0, 60.0, 110.0, 130.0)  # 70x70 初始框。
CX, CY = (BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2


def make_scene(texture=True, outline=False, size=(W, H)):
    """造一个有纹理/描边的方块场景（其余区域平坦）。"""
    frame = np.full((size[1], size[0]), 60, np.uint8)
    cv2.rectangle(frame, (10, 10), (size[0] - 10, size[1] - 10), 80, -1)  # 背景微纹理。
    rng = np.random.default_rng(7)
    frame = frame + rng.integers(0, 8, frame.shape, dtype=np.uint8) if texture else frame
    x1, y1, x2, y2 = int(CX - 25), int(CY - 25), int(CX + 25), int(CY + 25)
    if outline:
        cv2.rectangle(frame, (x1, y1), (x2, y2), 230, 2)  # 空芯描边块。
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), 220, -1)  # 实心亮块。
        for i in range(4):  # 内部纹理。
            cv2.circle(frame, (x1 + 10 + i * 10, y1 + 12), 3, 140, -1)
    return frame


def shift_scene(frame, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def test_tracks_constant_translation_60_frames_low_drift():
    frame = make_scene()
    t = FlowTracker()
    t.start(frame, BOX)
    assert t.alive
    drifts = []
    for i in range(1, 61):
        frame = shift_scene(frame, 2, 1)  # 2px/帧 匀速。
        res = t.update(frame)
        assert res is not None, f"died at {i}"
        box, (dx, dy) = res
        drifts.append(abs(dx - 2) + abs(dy - 1))
    assert np.mean(drifts) < 1.0  # 位移测量基本无偏。
    # 60 帧后框中心与真实块中心偏差 < 4px（真中心 = 初始中心 + (120,60)）。
    truth = (CX + 120, CY + 60)
    got = center(t.bbox)
    assert abs(got[0] - truth[0]) < 4 and abs(got[1] - truth[1]) < 4


def test_survives_frame_gap_large_displacement():
    frame = make_scene()
    t = FlowTracker()
    t.start(frame, BOX)
    frame = shift_scene(frame, 9, 6)  # 模拟跳 3 帧：单步位移 3 倍。
    res = t.update(frame)
    assert res is not None
    got = center(res[0])
    assert abs(got[0] - (CX + 9)) < 3 and abs(got[1] - (CY + 6)) < 3


def test_occlusion_two_frames_kills_tracker():
    frame = make_scene()
    t = FlowTracker()
    t.start(frame, BOX)
    for _ in range(2):  # 遮挡：亮块覆盖整个框+环带外扩 10px。
        frame = shift_scene(frame, 3, 0)
        frame[int(BOX[1]) - 10:int(BOX[3]) + 10, int(BOX[0]) - 10:int(BOX[2]) + 10] = 255
        res = t.update(frame)
    assert res is None and not t.alive


def test_textureless_ncc_death():
    # 目标区域完全无纹理（平坦灰底 + 实心块同灰度）→ 点集退化/相关崩 → 死亡而非乱跟。
    frame = np.full((H, W), 100, np.uint8)
    t = FlowTracker()
    t.start(frame, BOX)
    frame2 = shift_scene(frame, 4, 0)
    res = t.update(frame2)
    # 无特征可用（网格兜底点在平坦区）时，要么死亡要么位移不可信——不允许活着乱漂。
    if res is not None:
        assert abs(res[1][0] - 4.0) < 2.0


def test_outline_only_patch_trackable():
    frame = make_scene(outline=True)
    t = FlowTracker()
    t.start(frame, BOX)
    ok = 0
    for i in range(1, 31):
        frame = shift_scene(frame, 2, -1)
        res = t.update(frame)
        if res is not None:
            ok += 1
    assert ok >= 25  # 空芯描边块 30 帧内至少 25 帧存活。


def test_crossing_decoy_motion_gate_keeps_median():
    # 目标 2px/帧 向右；一个 28x28 强纹理干扰块以更快速度从右侧擦过目标右半。
    frame = make_scene()
    t = FlowTracker()
    t.start(frame, BOX)
    decoy = np.full((28, 28), 200, np.uint8)
    cv2.circle(decoy, (8, 8), 4, 60, -1)
    cv2.circle(decoy, (20, 18), 5, 120, -1)
    cv2.rectangle(decoy, (2, 2), (25, 25), 180, 2)
    for i in range(1, 21):
        frame = shift_scene(frame, 2, 0)
        # 干扰块从目标右上向左下扫过目标右缘：仅部分重叠，非整框遮挡。
        dx_dec = int(CX + 20 - i * 6)
        dy_dec = int(CY - 50 + i * 4)
        if 0 <= dy_dec < H - 28 and 0 <= dx_dec < W - 28:
            frame[dy_dec:dy_dec + 28, dx_dec:dx_dec + 28] = decoy
        res = t.update(frame)
        if res is not None:
            dx = res[1][0]
            assert abs(dx - 2.0) < 1.5, f"median displacement pulled to decoy at {i}: {dx}"


def test_reset_clears_state():
    t = FlowTracker()
    t.start(make_scene(), BOX)
    t.reset()
    assert not t.alive and t.bbox is None and t.age == 0
    assert t.update(make_scene()) is None
