# -*- coding: utf-8 -*-
"""测谎检验页签测试（offscreen）：控件接线、加载视频、历史下拉、拖动预览、启停按钮状态。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = QApplication.instance() or QApplication([])

import src.liedetector.recorder as recorder_mod
import src.ui.LieDetectorTab as tab_mod
from src.liedetector.recorder import LieRecorder
from src.ui.LieDetectorTab import LieDetectorTab


def _make_video(path, frames=12, w=64, h=48, fps=30.0):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        writer.write(np.full((h, w, 3), 20 * i, np.uint8))
    writer.release()
    return path


class FakeThread(QThread):
    """DetectThread 假件：不加载真实模型，仅驱动按钮状态机。"""

    frame_ready = Signal(object, dict)
    status = Signal(str)
    finished_run = Signal()

    def __init__(self, *a, **k):
        super().__init__()
        self.show_boxes = self.show_trails = True
        self.realtime = False
        self.paused = False
        self.step_once = False

    def start(self):
        pass  # 不真正起线程。

    def stop(self):
        pass

    def wait(self, ms):
        pass


@pytest.fixture
def tab(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_mod, "LIE_RECORD_DIR", str(tmp_path))
    monkeypatch.setattr(tab_mod, "LIE_RECORD_DIR", str(tmp_path))
    widget = LieDetectorTab()
    yield widget
    widget.on_stop()
    widget._scrub_cap and widget._scrub_cap.release()


def test_initial_state(tab):
    # d/lie 交互：开始/暂停/单帧/停止初始禁用，状态标签 IDLE。
    assert not tab.btn_start.isEnabled()
    assert not tab.btn_pause.isEnabled()
    assert not tab.btn_step.isEnabled()
    assert not tab.btn_stop.isEnabled()
    assert tab.lbl_state.text() == "● IDLE"


def test_load_sets_slider_and_buttons(tab, tmp_path):
    video = _make_video(str(tmp_path / "v.mp4"))
    tab.load(video)
    assert tab.btn_start.isEnabled() and tab.btn_stop.isEnabled()
    assert tab.slider.maximum() == 11  # 12 帧 → 0..11。
    assert tab.lbl_pos.text().startswith("0 / 12")
    assert tab.video_label.text() == "v.mp4"


def test_scrub_previews_raw_frame_and_sets_start(tab, tmp_path):
    video = _make_video(str(tmp_path / "v.mp4"))
    tab.load(video)
    tab._on_scrub(5)
    tab._flush_scrub()
    assert tab.start_frame == 5  # 分析起点移动到拖动处。
    assert "预览（未分析）" in tab.lbl_pos.text()
    assert tab.video_label.pixmap() is not None  # 显示了原始帧。


def test_stop_resets_position(tab, tmp_path):
    video = _make_video(str(tmp_path / "v.mp4"))
    tab.load(video)
    tab.start_frame = 7
    tab.on_stop()
    assert tab.start_frame == 0  # 停止即回到开头。


def test_history_reload_and_select(tab, tmp_path):
    # 造两条历史记录。
    for i in range(2):
        rec = LieRecorder()
        rec.start((480, 640), meta={"conf": 0.4}, crop=None)
        for j in range(10):
            rec.write(np.full((480, 640, 3), 30 * i + j, np.uint8))
        rec.stop("success" if i == 0 else "failure")
    tab._reload_history()
    # 占位项 + 2 条记录。
    assert tab.history_combo.count() == 3
    video = tab.history_combo.itemData(1)
    assert video and video.endswith(".mp4")
    # 选中第一条：加载视频并回填实战置信度。
    tab.history_combo.setCurrentIndex(1)
    tab._on_history_selected(1)
    assert tab.video_path and tab.video_path.endswith(video)
    assert tab.spn_conf.value() == pytest.approx(0.4)


def test_start_stop_button_states(tab, tmp_path, monkeypatch):
    video = _make_video(str(tmp_path / "v.mp4"))
    tab.load(video)
    monkeypatch.setattr(tab_mod, "DetectThread", FakeThread)
    tab.on_start()
    assert tab.btn_pause.isEnabled() and tab.btn_step.isEnabled()
    assert not tab.cmb_device.isEnabled()  # 运行中设备锁定。
    assert not tab.spn_conf.isEnabled()
    tab.on_pause()
    assert tab.btn_pause.text() == "继续"  # d/lie 交互：暂停按钮切换文案。
    tab.on_pause()
    assert tab.btn_pause.text() == "暂停"
    tab.on_stop()
    assert not tab.btn_pause.isEnabled()
    assert tab.cmb_device.isEnabled() and tab.spn_conf.isEnabled()


def test_verdict_after_finished_run(tab, tmp_path, monkeypatch):
    video = _make_video(str(tmp_path / "v.mp4"))
    tab.load(video)
    monkeypatch.setattr(tab_mod, "DetectThread", FakeThread)
    tab.on_start()
    tab.ever_locked = True
    tab._on_finished()
    assert "锁定成功" in tab.debug.toPlainText()
