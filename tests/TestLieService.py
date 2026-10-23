# -*- coding: utf-8 -*-
"""测谎服务测试：配置解析、触发编排（报警/暂停/录像/恢复顺序）、急停、后端描述。全部用假件，不依赖真实框架。"""
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.liedetector.service as service_mod
from src.liedetector.service import LieDetectorService


def _service():
    return LieDetectorService(threading.Event())


def test_read_config_parses_and_defaults():
    svc = _service()
    injected = {
        "Lie Detector Auto Solve": True,
        "Lie Detector Trigger Feature": "测谎触发",
        "Lie Detector Region Feature": "测谎坐标框",
        "Lie Detector Threshold": "bad",  # 非法阈值回退 0.75。
        "Lie Alarm Sound": "alarm.mp3",
    }
    svc._get_config = lambda: injected  # 绕过磁盘刷新。
    auto, trigger, region, threshold, alarm = svc._read_config()
    assert auto is True
    assert trigger == "测谎触发" and region == "测谎坐标框"
    assert threshold == 0.75
    assert alarm == "alarm.mp3"


def test_handle_trigger_orchestration_order(monkeypatch, tmp_path):
    """触发处理顺序：报警 → 暂停任务 → 起录 → 解题+结算 → 停录 → 恢复任务。"""
    svc = _service()
    calls = []
    monkeypatch.setattr(svc, "_read_config",
                        lambda: (True, "测谎触发", "测谎坐标框", 0.75, "alarm.mp3"))
    monkeypatch.setattr(svc, "_play_alarm", lambda p: calls.append(("alarm", p)) or True)
    monkeypatch.setattr(service_mod, "pause_current_task", lambda owner=None: calls.append(("pause", owner)) or "TASK")
    monkeypatch.setattr(service_mod, "resume_task", lambda task, owner=None: calls.append(("resume", task)))
    monkeypatch.setattr(svc, "_detector_for_solve", lambda: calls.append(("detector",)) or SimpleNamespace(backend_label="CPU", conf=0.3))
    monkeypatch.setattr(svc, "get_region", lambda frame, name: SimpleNamespace(x=1, y=2, width=300, height=200))
    monkeypatch.setattr(svc, "_start_recorder", lambda shape, crop, det: calls.append(("record_start", crop)))
    monkeypatch.setattr(svc, "_stop_recorder", lambda outcome: calls.append(("record_stop", outcome)))

    class FakeSolver:
        def __init__(self, host, exit_event, abort_event):
            calls.append(("solver_ctor",))

        def run(self, detector, frame, trig, region, threshold):
            calls.append(("solve",))
            return "success"

    monkeypatch.setattr(service_mod, "LieSolver", FakeSolver)
    monkeypatch.setattr(service_mod, "POST_SOLVE_SLEEP", 0)
    monkeypatch.setattr(service_mod, "SOLVE_COOLDOWN", 0)
    frame = SimpleNamespace(shape=(480, 640))
    svc._handle_trigger(frame, SimpleNamespace(), "测谎触发", "测谎坐标框", 0.75)
    order = [c[0] for c in calls]
    assert order == ["alarm", "pause", "detector", "record_start", "solver_ctor",
                     "solve", "record_stop", "resume"]
    assert calls[order.index("record_stop")][1] == "success"  # 结算结果写入录像。
    assert svc.state == service_mod.STATE_IDLE  # 回到空闲态。
    assert svc._cooldown_until > 0  # 进入冷却。


def test_handle_trigger_abandoned_when_no_detector(monkeypatch):
    svc = _service()
    calls = []
    monkeypatch.setattr(svc, "_read_config", lambda: (True, "t", "r", 0.75, ""))
    monkeypatch.setattr(svc, "_play_alarm", lambda p: False)
    monkeypatch.setattr(service_mod, "pause_current_task", lambda owner=None: "TASK")
    monkeypatch.setattr(service_mod, "resume_task", lambda task, owner=None: calls.append("resume"))
    monkeypatch.setattr(svc, "_detector_for_solve", lambda: None)  # 后端不可用。
    monkeypatch.setattr(service_mod, "POST_SOLVE_SLEEP", 0)
    svc._handle_trigger(SimpleNamespace(shape=(480, 640)), None, "t", "r", 0.75)
    assert calls == ["resume"]  # 不解题但仍恢复任务。
    assert svc._recorder is None  # 未起录。


def test_abort_writes_config_off_and_notice(monkeypatch, tmp_path):
    svc = _service()
    saved = {}
    monkeypatch.setattr(service_mod, "load_dashboard_config", lambda: {"Lie Detector Auto Solve": True})
    monkeypatch.setattr(service_mod, "save_dashboard_config", lambda cfg: saved.update(cfg))
    svc._trigger_abort()
    assert saved["Lie Detector Auto Solve"] is False  # 开关写回为关。
    assert svc._abort_event.is_set()
    assert svc.consume_abort_notice() is True  # 一次性通知。
    assert svc.consume_abort_notice() is False  # 取后即清。
    svc._trigger_abort()  # 幂等：重复急停不再触发。


def test_backend_desc_states(monkeypatch):
    svc = _service()
    assert "装配中" in svc.backend_desc()  # 未装配。
    svc._detector = SimpleNamespace(backend_label="GPU · TensorRT INT8（缓存引擎）")
    assert svc.backend_desc() == "GPU · TensorRT INT8（缓存引擎）"
    assert svc._backend_desc() == svc.backend_desc()  # 旧名别名。


def test_recorder_coordination(monkeypatch, tmp_path):
    """起录失败不影响解题；停录把 outcome 交给录像器。"""
    import numpy as np

    import src.liedetector.recorder as recorder_mod
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    svc = _service()
    tick = {"n": 0}

    def fake_capture():  # 独立采集假件：每次返回一帧互异画面。
        tick["n"] += 1
        return np.full((480, 640, 3), tick["n"] % 200, np.uint8)

    monkeypatch.setattr(service_mod, "make_independent_capture",
                        lambda fallback: (fake_capture, lambda: None))
    detector = SimpleNamespace(backend_label="CPU", conf=0.3)
    svc._start_recorder((480, 640), (0, 0, 300, 200), detector)
    assert svc._recorder is not None
    import time as _time
    _time.sleep(0.3)  # 让自采集线程攒帧。
    svc._stop_recorder("success")
    assert svc._recorder is None
    records = recorder_mod.list_records(str(tmp_path))
    assert records and records[0]["outcome"] == "success"
    assert records[0]["meta"]["backend"] == "CPU"
