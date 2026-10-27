# -*- coding: utf-8 -*-
"""FlowSession 三相编排测试：交接、光流跟随、校验吸附/滑移、重获、超时、工厂回退。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.liedetector.vision import config
from src.liedetector.vision.detector import Detection
from src.liedetector.vision.flow_session import FlowSession, create_session
from src.liedetector.vision.pipeline import Pipeline
from src.liedetector.vision.state_machine import GameState

W, H = 545, 427  # 面板裁剪尺寸量级。


def det(cx, cy, tid, half=30, conf=0.9):
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=conf,
        class_id=3,
        class_name="pentagram",
        centroid=(cx, cy),
        track_id=tid,
        shape="star",
    )


class ScriptedDetector:
    """按「检测器调用序号」给检测的桩：phase-1 期间调用序号==帧号，可精确编排。"""

    def __init__(self, script=None, default=None):
        self.script = dict(script or {})  # call_idx -> [Detection]
        self.default = default or []
        self.calls = 0

    def classes_for_shape(self, shape):
        return [3]

    def track(self, frame):
        dets = self.script.get(self.calls, self.default)
        self.calls += 1
        return list(dets)

    def detect(self, frame):
        return self.track(frame)


def white_frame():
    f = np.zeros((H, W, 3), np.uint8)
    cv2.rectangle(f, (170, 170), (230, 230), (255, 255, 255), -1)
    return f


def moving_texture_frame(cx, cy):
    """暗色空心描边块（贴合框缘，模拟褪色后的透明目标）：光流追的对象。"""
    f = np.full((H, W, 3), 50, np.uint8)
    x1, y1 = int(cx - 30), int(cy - 30)
    cv2.rectangle(f, (x1 + 1, y1 + 1), (x1 + 59, y1 + 59), (130, 130, 130), 3)  # 描边贴框缘。
    cv2.rectangle(f, (x1 + 10, y1 + 10), (x1 + 50, y1 + 50), (110, 110, 110), 2)
    for i in range(3):
        cv2.circle(f, (x1 + 6 + i * 24, y1 + 4), 3, (60, 60, 60), -1)  # 环带上的纹理点。
    return f


@pytest.fixture
def fast_cfg(monkeypatch):
    # 加速：倒计时 10 帧、褪色 2 帧、重获超时 20 帧。
    monkeypatch.setattr(config, "COUNTDOWN_DURATION_SEC", 10 / 30.0)
    monkeypatch.setattr(config, "FADE_CONFIRM_FRAMES", 2)
    monkeypatch.setattr(config, "TRANSITION_TIMEOUT_FRAMES", 60)
    monkeypatch.setattr(config, "RECOVERY_TIMEOUT_FRAMES", 20)
    monkeypatch.setattr(config, "FLOW_VERIFY_INTERVAL", 5)
    monkeypatch.setattr(config, "FLOW_HANDOFF_STABLE", 2)


def build_round():
    """构造 0..30 白帧 + 31..70 暗块移动帧（(2,1)/帧），及 phase-1 检测脚本。

    检测脚本：帧 0..30 白块检测（倒计时锁定）；帧 31..36 跟随暗块真实位置
    （褪色确认需要锁定轨迹持续有测量）；帧 37 起走 default（各用例自行设定）。
    返回 (frames, detector, block_pos_fn)。
    """
    script = {}
    for i in range(31):
        script[i] = [det(200, 200, 5)]
    frames = [white_frame() for _ in range(31)]
    for i in range(40):
        cx, cy = 200 + 2 * (i + 1), 200 + 1 * (i + 1)
        frames.append(moving_texture_frame(cx, cy))
        if i < 6:  # 褪色确认窗口：检测跟随暗块。
            script[31 + i] = [det(cx, cy, 5)]

    def pos(frame_idx):
        """帧号 >=31 的暗块中心。"""
        k = frame_idx - 30
        return 200 + 2 * k, 200 + 1 * k

    return frames, ScriptedDetector(script), pos


def test_handoff_after_fade_and_flow_carries(fast_cfg):
    """褪色完成交接后，画面块继续移动：follow_bbox 持续跟随块中心。

    校验帧持续提供块位置检测（真实面板上 YOLO 总能看到形状），吸附无害。
    """
    frames, detector, pos = build_round()
    session = FlowSession(detector, 30.0)
    outs = []
    for i, f in enumerate(frames):
        if i >= 31:
            detector.default = [det(*pos(i), tid=9)]
        outs.append(session.process(f, i))
    assert session._phase == "flow", "healthy tracking with matching verify stays in flow"
    assert outs[-1].follow_bbox is not None
    got = ((outs[-1].follow_bbox[0] + outs[-1].follow_bbox[2]) / 2,
           (outs[-1].follow_bbox[1] + outs[-1].follow_bbox[3]) / 2)
    truth = pos(len(frames) - 1)
    assert abs(got[0] - truth[0]) < 12 and abs(got[1] - truth[1]) < 12
    assert outs[-1].state == "TRACKING"


def test_verify_misses_lead_to_reacquire(fast_cfg):
    """连续 2 次校验无匹配（画面上找不到目标形状）→ 判滑移进入重获。"""
    frames, detector, pos = build_round()
    session = FlowSession(detector, 30.0)
    for i, f in enumerate(frames):
        if i >= 31:
            detector.default = [det(*pos(i), tid=9)]
        session.process(f, i)
    assert session._phase == "flow"
    detector.default = []  # 校验再也找不到目标形状。
    for j in range(12):
        session.process(moving_texture_frame(*pos(len(frames) + j)), len(frames) + j)
    assert session._phase == "reacquire"


def test_verify_snap_pulls_flow_back(fast_cfg):
    """校验检测恒偏 +6px：吸附后光流框带 6px 偏移。"""
    frames, detector, pos = build_round()
    session = FlowSession(detector, 30.0)
    for i, f in enumerate(frames):
        if i >= 31:
            detector.default = [det(*pos(i), tid=9)]
        session.process(f, i)
    assert session._phase == "flow"
    cx, cy = pos(len(frames) - 1)
    for j in range(6):
        cx, cy = cx + 2, cy + 1
        detector.default = [det(cx + 6, cy, tid=9)]
        out = session.process(moving_texture_frame(cx, cy), len(frames) + j)
    assert session._flow.bbox is not None
    assert abs(session._flow.bbox[0] - (cx + 6 - 30)) < 4  # 框左缘吸附到检测框。


def test_reacquire_adopts_candidate_near_flow_anchor(fast_cfg):
    """校验连续无匹配 → 重获；光流锚附近出现候选 → 采纳回 phase-2。"""
    frames, detector, pos = build_round()
    session = FlowSession(detector, 30.0)
    for i, f in enumerate(frames):
        if i >= 31:
            detector.default = [det(*pos(i), tid=9)]
        session.process(f, i)
    assert session._phase == "flow"
    detector.default = []  # 校验失败 → 滑移 → 重获。
    for j in range(12):
        session.process(moving_texture_frame(*pos(len(frames) + j)), len(frames) + j)
    assert session._phase == "reacquire"
    anchor = session._loss_bbox
    ac = ((anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2)
    detector.default = [det(ac[0] + 8, ac[1], tid=42)]  # 锚附近的候选（全帧率 track）。
    outs = []
    for j in range(8):
        outs.append(session.process(moving_texture_frame(ac[0], ac[1]),
                                    len(frames) + 12 + j))
    assert session._phase == "flow", "candidate near flow anchor should be adopted"
    assert session._target_id == 42
    assert outs[-1].follow_bbox is not None


def test_reacquire_timeout_reaches_game_over(fast_cfg):
    frames, detector, pos = build_round()
    session = FlowSession(detector, 30.0)
    for i, f in enumerate(frames):
        if i >= 31:
            detector.default = [det(*pos(i), tid=9)]
        session.process(f, i)
    detector.default = []  # 校验失败进入重获，且此后永远无候选。
    idx = len(frames)
    outs = []
    for j in range(40):  # ≤12 帧进重获 + RECOVERY_TIMEOUT=20 → 40 帧必超时。
        outs.append(session.process(moving_texture_frame(*pos(idx + j)), idx + j))
    assert session._phase == "game_over"
    assert outs[-1].state == GameState.GAME_OVER.value


def test_factory_returns_pipeline_when_disabled(monkeypatch, fast_cfg):
    monkeypatch.setattr(config, "FLOW_PRIMARY", False)
    session = create_session(ScriptedDetector(), 30.0)
    assert isinstance(session, Pipeline)
    assert session.sm.state == GameState.COUNTDOWN


def test_factory_default_is_pipeline(monkeypatch):
    """默认（FLOW_PRIMARY=False）走原 d/lie Pipeline；显式开启才走光流会话。"""
    monkeypatch.setattr(config, "FLOW_PRIMARY", True)
    session = create_session(ScriptedDetector(), 30.0)
    assert isinstance(session, FlowSession)
