"""Device gating and TensorRT engine helpers: no GPU or model needed to test the logic."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.liedetector.vision import config
import src.liedetector.vision.device as device
from src.liedetector.vision.tensorrt_backend import TRTBuildError, engine_cache_path, prepare_calibration


def _set_probes(monkeypatch, driver=True, cuda=True, trt=True, name="Fake RTX"):
    monkeypatch.setattr(device, "nvidia_driver_present", lambda: (driver, name))
    monkeypatch.setattr(device, "torch_cuda_available", lambda: cuda)
    monkeypatch.setattr(device, "tensorrt_available", lambda: trt)


def test_no_driver_disables_gpu(monkeypatch):
    _set_probes(monkeypatch, driver=False)
    opt = device.gpu_option_state()
    assert not opt.enabled
    assert "nvidia" in opt.reason.lower() or "NVIDIA" in opt.reason


def test_cpu_only_torch_disables_gpu(monkeypatch):
    _set_probes(monkeypatch, cuda=False)
    opt = device.gpu_option_state()
    assert not opt.enabled
    assert "PyTorch" in opt.reason and "CUDA" in opt.reason


def test_missing_tensorrt_disables_gpu(monkeypatch):
    _set_probes(monkeypatch, trt=False)
    opt = device.gpu_option_state()
    assert not opt.enabled
    assert "TensorRT" in opt.reason


def test_full_stack_enables_gpu(monkeypatch):
    _set_probes(monkeypatch)
    opt = device.gpu_option_state()
    assert opt.enabled
    assert "Fake RTX" in opt.reason


def test_auto_follows_availability(monkeypatch):
    _set_probes(monkeypatch)
    assert device.resolve_request(device.REQUEST_AUTO) == device.REQUEST_GPU
    _set_probes(monkeypatch, driver=False)
    assert device.resolve_request(device.REQUEST_AUTO) == device.REQUEST_CPU


def test_explicit_request_passes_through(monkeypatch):
    _set_probes(monkeypatch, driver=False)
    assert device.resolve_request(device.REQUEST_CPU) == device.REQUEST_CPU
    assert device.resolve_request(device.REQUEST_GPU) == device.REQUEST_GPU


def test_backend_labels():
    assert "TensorRT INT8" in device.backend_label(device.BACKEND_TRT)
    assert "缓存" in device.backend_label(device.BACKEND_TRT_CACHE)
    assert "回退" in device.backend_label(device.BACKEND_CUDA)
    assert device.backend_label(device.BACKEND_CPU) == "CPU"


def test_engine_cache_tracks_checkpoint_and_imgsz(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"x")
    a = engine_cache_path(str(pt), 416)
    assert "best" in a.name and "int8416" in a.name and a.suffix == ".engine"
    # same content signature -> same cache name
    assert engine_cache_path(str(pt), 416) == a
    # a different imgsz or a swapped checkpoint must not hit the old engine
    assert engine_cache_path(str(pt), 320) != a
    import os

    os.utime(pt, ns=(0, 1_000_000))
    assert engine_cache_path(str(pt), 416) != a


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "game.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(40):
        frame = np.full((48, 64, 3), i * 3, dtype=np.uint8)
        w.write(frame)
    w.release()
    return str(path)


def test_prepare_calibration_extracts_and_reuses(tmp_path, tiny_video, monkeypatch):
    monkeypatch.setattr(config, "TRT_CALIB_DIR", str(tmp_path / "calib"))
    names = {0: "circle", 1: "pentagram"}
    yaml_path = prepare_calibration(tiny_video, names)
    text = yaml_path.read_text(encoding="utf-8")
    assert "train: images" in text and "val: images" in text
    assert "pentagram" in text
    # plain-style yaml scalars cannot carry backslashes
    assert "\\" not in text.splitlines()[0]
    img_dir = yaml_path.parent / "images"
    lbl_dir = yaml_path.parent / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    assert len(images) == 40  # 40 frames, step = 40 // 256 -> every frame
    assert all((lbl_dir / (p.stem + ".txt")).exists() for p in images)
    assert all((lbl_dir / (p.stem + ".txt")).stat().st_size == 0 for p in images)
    # second call without force reuses the cache instead of re-decoding
    mtime = images[0].stat().st_mtime_ns
    assert prepare_calibration(tiny_video, names) == yaml_path
    assert images[0].stat().st_mtime_ns == mtime


def test_prepare_calibration_rejects_sparse_video(tmp_path, monkeypatch):
    path = tmp_path / "short.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(10):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    monkeypatch.setattr(config, "TRT_CALIB_DIR", str(tmp_path / "calib"))
    monkeypatch.setattr(config, "TRT_CALIB_MIN_IMAGES", 32)
    with pytest.raises(TRTBuildError):
        prepare_calibration(str(path), {0: "circle"})
