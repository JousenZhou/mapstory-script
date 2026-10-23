# -*- coding: utf-8 -*-
"""解题器测试：鼠标直跟映射、区域重置、退出条件与 3 秒结算协议（假 host + 假管线）。"""
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.liedetector.solver as solver_mod
from src.liedetector.solver import LieSolver

REGION = (10, 20, 300, 200)
FRAME = np.zeros((480, 860, 3), np.uint8)


class FakePipeline:
    """脚本化管线：按帧序返回 follow_bbox / state；记录构造次数验证区域重置。"""

    built = 0
    script = []  # [(follow_bbox, state), ...] 全局脚本，每个用例重置。

    def __init__(self, detector, fps):
        self.idx = 0
        self.calls = 0
        type(self).built += 1
        self.sm = SimpleNamespace(start=lambda idx: None)

    def process(self, crop, frame_idx):
        i = min(self.idx, len(type(self).script) - 1)
        self.idx += 1
        self.calls += 1
        bbox, state = type(self).script[i]
        return SimpleNamespace(follow_bbox=bbox, state=state, tracks=[], target_id=1 if bbox else None,
                               frame=crop, detect_count=0)


class FakeHost:
    """框架胶水假件：记录鼠标/点击/推送，按脚本回答标注匹配。"""

    def __init__(self, trigger=True, region=REGION, success_script=None, ok_script=None):
        self.trigger = trigger
        self.region = region
        self.moves = []
        self.clicks = []
        self.visions = 0
        self.front_calls = 0
        self._success_iter = iter(success_script or [])
        self._ok_iter = iter(ok_script or [])

    def find_feature(self, frame, name, threshold):
        if name == solver_mod.LIE_SUCCESS_TEMPLATE:
            return next(self._success_iter, None)
        if name == solver_mod.LIE_SUCCESS_OK_TEMPLATE:
            return next(self._ok_iter, None)
        return SimpleNamespace(x=1, y=1, width=8, height=8) if self.trigger else None

    def get_region(self, frame, name):
        if self.region is None:
            return None
        x, y, w, h = self.region
        return SimpleNamespace(x=x, y=y, width=w, height=h)

    def capture(self):
        return FRAME.copy()

    def start_pump(self):
        return None, None

    def move_mouse(self, x, y):
        self.moves.append((x, y))

    def click_window(self, x, y):
        self.clicks.append((x, y))

    def is_background_mouse(self):
        return False

    def ensure_in_front(self):
        self.front_calls += 1

    def push_vision(self, frame):
        self.visions += 1


def _solver(host):
    return LieSolver(host, threading.Event(), threading.Event())


def _run_solve(monkeypatch, script, host=None):
    FakePipeline.built = 0
    FakePipeline.script = script
    monkeypatch.setattr(solver_mod, "Pipeline", FakePipeline)
    monkeypatch.setattr(solver_mod, "VISION_PUSH_INTERVAL", 0)  # 每帧都推送，便于断言。
    host = host or FakeHost()
    solver = _solver(host)
    outcome = solver.solve(SimpleNamespace(), FRAME, "测谎触发", "测谎坐标框", 0.75)
    return outcome, host


def test_mouse_follows_follow_bbox_center_with_region_offset(monkeypatch):
    # follow_bbox 是区域局部坐标；鼠标目标 = 区域原点 + 框中心（画面绝对坐标）。
    script = [((20, 30, 60, 90), "TRACKING"), ((40, 10, 80, 50), "TRACKING"),
              (None, "GAME_OVER")]
    outcome, host = _run_solve(monkeypatch, script)
    assert outcome == "solved"
    assert host.moves == [
        pytest.approx((REGION[0] + 40.0, REGION[1] + 60.0)),
        pytest.approx((REGION[0] + 60.0, REGION[1] + 30.0)),
    ]  # 第三帧无 follow_bbox 不移动。
    assert host.visions >= 2


def test_trigger_lost_returns_gone(monkeypatch):
    # 触发标注从一开始就消失：超过容忍帧数后以 gone 退出。
    script = [(None, "IDLE")] * 40
    host = FakeHost(trigger=False)
    outcome, _ = _run_solve(monkeypatch, script, host)
    assert outcome == "gone"


def test_region_shift_rebuilds_pipeline(monkeypatch):
    # 区域中途位移超阈值 → 重建管线会话。
    FakePipeline.built = 0
    FakePipeline.script = [((10, 10, 50, 50), "TRACKING")] * 6 + [(None, "GAME_OVER")]
    monkeypatch.setattr(solver_mod, "Pipeline", FakePipeline)
    host = FakeHost(region=REGION)
    solver = _solver(host)
    solver.solve(SimpleNamespace(), FRAME, "测谎触发", "测谎坐标框", 0.75)
    assert FakePipeline.built == 1
    host.region = (REGION[0] + 100, REGION[1], REGION[2], REGION[3])  # 位移超 4px。
    solver.solve(SimpleNamespace(), FRAME, "测谎触发", "测谎坐标框", 0.75)
    assert FakePipeline.built == 2  # 新会话重建了管线。


def test_abort_event_stops_solve(monkeypatch):
    FakePipeline.built = 0
    FakePipeline.script = [((10, 10, 50, 50), "TRACKING")] * 100
    monkeypatch.setattr(solver_mod, "Pipeline", FakePipeline)
    abort = threading.Event()
    abort.set()
    solver = LieSolver(FakeHost(), threading.Event(), abort)
    outcome = solver.solve(SimpleNamespace(), FRAME, "测谎触发", "测谎坐标框", 0.75)
    assert outcome == "aborted"


def test_settle_success_clicks_confirm_until_gone(monkeypatch):
    # 出现【测谎成功】→ 点击【测谎成功确定】→ 【测谎成功】消失 = success。
    box = SimpleNamespace(x=5, y=5, width=10, height=10)
    host = FakeHost(success_script=[box, box, None], ok_script=[box, box, None])
    solver = _solver(host)
    outcome = solver.settle(0.75)
    assert outcome == "success"
    assert len(host.clicks) >= 1  # 至少点过一次确定。


def test_settle_failure_when_no_success_within_window(monkeypatch):
    # 3 秒窗口内【测谎成功】始终不出现 = failure（窗口调小加速测试）。
    monkeypatch.setattr(solver_mod, "SETTLE_WINDOW", 0.05)
    monkeypatch.setattr(solver_mod, "SETTLE_POLL_INTERVAL", 0.005)
    host = FakeHost(success_script=[])
    solver = _solver(host)
    assert solver.settle(0.75) == "failure"


def test_settle_aborted(monkeypatch):
    abort = threading.Event()
    abort.set()
    solver = LieSolver(FakeHost(), threading.Event(), abort)
    assert solver.settle(0.75) == "aborted"


def test_run_routes_normal_outcomes_to_settle(monkeypatch):
    # run() 对 gone/solved 先结算再返回最终结果。
    FakePipeline.built = 0
    FakePipeline.script = [(None, "GAME_OVER")]
    monkeypatch.setattr(solver_mod, "Pipeline", FakePipeline)
    host = FakeHost(success_script=[SimpleNamespace(x=1, y=1, width=4, height=4)],
                    ok_script=[SimpleNamespace(x=1, y=1, width=4, height=4), None])
    solver = _solver(host)
    assert solver.run(SimpleNamespace(), FRAME, "测谎触发", "测谎坐标框", 0.75) == "success"
