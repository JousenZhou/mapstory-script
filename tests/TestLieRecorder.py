# -*- coding: utf-8 -*-
"""录像器测试：去重、标定开流、边车字段、滚动保留、历史列举/删除（全部落在临时目录）。"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.liedetector.recorder as recorder_mod
from src.liedetector.recorder import LieRecorder, delete_record, list_records


def _frame(value=128, w=64, h=48):
    return np.full((h, w, 3), value, np.uint8)


def _record(tmp_path, frames, crop=None, outcome="success", meta=None):
    """在 tmp 目录录一段互异帧并收尾，返回边车 dict。"""
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    rec = LieRecorder()
    rec.start((480, 640), meta=meta or {"backend": "CPU", "conf": 0.3}, crop=crop)
    for i, f in enumerate(frames):
        rec.write(f)
    rec.stop(outcome)
    records = list_records(str(tmp_path))
    assert records, "应产出一条历史记录"
    return records[0]


def test_records_distinct_frames_and_sidecar(tmp_path):
    frames = [_frame(10 + i) for i in range(30)] + [_frame(39)] * 10  # 末尾 10 帧重复。
    record = _record(tmp_path, frames, crop=(10, 20, 64, 48), outcome="success")
    assert record["outcome"] == "success"
    assert record["frames"] == 30  # 重复帧被去重。
    assert record["region"] == [10, 20, 64, 48]
    assert record["meta"]["backend"] == "CPU"
    assert record["width"] == 64 and record["height"] == 48  # 输出尺寸=裁剪区域尺寸。
    assert record["fps"] > 0 and record["duration"] > 0
    assert os.path.exists(os.path.join(str(tmp_path), record["video"]))


def test_zero_frames_produces_nothing(tmp_path):
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    rec = LieRecorder()
    rec.start((480, 640))
    rec.write(None)  # 无有效帧。
    rec.stop("abandoned")
    assert list_records(str(tmp_path)) == []
    assert os.listdir(str(tmp_path)) == []  # 0 帧不落地任何文件。


def test_list_records_newest_first_and_delete(tmp_path):
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    names = []
    for i in range(3):
        rec = LieRecorder()
        rec.start((480, 640))
        for j in range(10):
            rec.write(_frame(i * 50 + j))
        rec.stop("failure")
        names.append(os.listdir(str(tmp_path)))
    # 三条记录都在，按 mtime 倒序。
    records = list_records(str(tmp_path))
    assert len(records) == 3
    mtimes = [r["_mtime"] for r in records]
    assert mtimes == sorted(mtimes, reverse=True)
    # 删除中间一条：mp4+json 成对删除。
    victim = records[1]["video"]
    assert delete_record(victim, folder=str(tmp_path)) is True
    assert len(list_records(str(tmp_path))) == 2
    assert not os.path.exists(os.path.join(str(tmp_path), victim))
    stem = os.path.splitext(victim)[0]
    assert not os.path.exists(os.path.join(str(tmp_path), stem + ".json"))


def test_rolling_keep_prunes_oldest(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_mod, "LIE_RECORD_KEEP", 2)
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    for i in range(4):
        rec = LieRecorder()
        rec.start((480, 640))
        for j in range(10):
            rec.write(_frame(i * 30 + j))
        rec.stop("timeout")
        os.utime(rec._path, (1000000 + i, 1000000 + i))  # 显式拉开 mtime 顺序。
    assert len(list_records(str(tmp_path))) == 2  # 只留最近 2 组。


def test_invalid_shape_skipped(tmp_path):
    recorder_mod.LIE_RECORD_DIR = str(tmp_path)
    rec = LieRecorder()
    rec.start((0, 0))  # 非法分辨率。
    rec.stop("abandoned")
    assert list_records(str(tmp_path)) == []


def test_crop_fully_outside_falls_back_to_full_frame(tmp_path):
    # 裁剪区域完全越界 → 回退录整帧。
    record = _record(tmp_path, [_frame(i) for i in range(20)], crop=(9000, 9000, 10, 10))
    assert record["region"] is None
    assert record["width"] == 640 and record["height"] == 480
