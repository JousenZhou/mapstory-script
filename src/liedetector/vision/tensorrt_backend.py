"""TensorRT INT8 engine: build once, cache, load in milliseconds.

The engine is keyed by (checkpoint content, imgsz) so a stale cache can never be loaded
against a different input shape. INT8 needs a calibration set; instead of shipping one,
frames are sampled evenly from the video being analysed - the quantization ranges then
reflect the game's actual pixel distribution (white fills, dim outlines), which is what
the detector's confidence depends on.
"""

import hashlib
import os
import shutil
from pathlib import Path

import cv2

from src.liedetector.vision import config
from src.liedetector.vision.logger import get_logger

log = get_logger("trt")


class TRTBuildError(Exception):
    pass


def engine_cache_path(model_path: str, imgsz: int) -> Path:
    """<stem>_int8<imgsz>_<checkpoint-hash>.engine next to the checkpoint. The hash is from
    size+mtime: swapping best.pt invalidates the old engine without a manual cleanup."""
    p = Path(model_path)
    st = p.stat()
    digest = hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:8]
    return p.with_name(f"{p.stem}_int8{imgsz}_{digest}.engine")


def prepare_calibration(video_path: str, names: dict, force: bool = False) -> Path:
    """Extract evenly spaced frames + empty labels, write the calibration yaml. Returns the
    yaml path. An existing folder with enough frames is reused - re-extracting costs nothing
    correctness-wise but seconds of decode each engine build."""
    stem = Path(video_path).stem or "calib"
    root = Path(config.TRT_CALIB_DIR) / stem
    img_dir, lbl_dir = root / "images", root / "labels"
    existing = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    if not force and len(existing) >= config.TRT_CALIB_MIN_IMAGES:
        log.info("calibration cache reused: %d frames in %s", len(existing), root)
        return _write_calib_yaml(root, names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TRTBuildError(f"cannot open {video_path} for INT8 calibration frames")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise TRTBuildError(f"{video_path} reports 0 frames; cannot calibrate")
        step = max(1, total // config.TRT_CALIB_MAX_IMAGES)
        if img_dir.is_dir():
            shutil.rmtree(root)
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n = 0
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            name = f"img_{idx:06d}.jpg"
            cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"img_{idx:06d}.txt").touch()  # empty label = background image
            n += 1
        if n < config.TRT_CALIB_MIN_IMAGES:
            raise TRTBuildError(f"only {n} calibration frames extractable from {video_path}")
        log.info("calibration set: %d frames from %s -> %s", n, os.path.basename(video_path), root)
    finally:
        cap.release()
    return _write_calib_yaml(root, names)


def _write_calib_yaml(root: Path, names: dict) -> Path:
    yaml_path = root / "calib.yaml"
    # forward slashes: a yaml plain-style scalar treats \\ as an escape
    base = str(root).replace("\\", "/")
    lines = [f"path: {base}", "train: images", "val: images", "names:"]
    lines += [f"  {int(k)}: {v}" for k, v in sorted(names.items())]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def build_int8_engine(model_path: str, imgsz: int, calib_video: str, names: dict) -> Path:
    """Export best.pt -> TensorRT INT8 engine (static batch=1 shape: lowest latency), move it
    to the cache path and return it. Raises TRTBuildError with the cause on any failure."""
    if not calib_video:
        raise TRTBuildError("no video available for INT8 calibration frames")
    yaml_path = prepare_calibration(calib_video, names)

    from ultralytics import YOLO

    src_pt = Path(model_path)
    produced = src_pt.with_suffix(".engine")
    if produced.exists():
        produced.unlink()
    log.info("building TensorRT INT8 engine (imgsz=%d, batch=1); first build takes minutes", imgsz)
    try:
        model = YOLO(model_path)
        out = model.export(
            format="engine",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            simplify=True,
            int8=True,
            half=False,
            data=str(yaml_path),
            device=0,
            workspace=config.TRT_WORKSPACE_GB,
            verbose=False,
        )
    except Exception as exc:
        produced.unlink(missing_ok=True)
        raise TRTBuildError(f"TensorRT INT8 export failed: {exc}") from exc

    engine = Path(out) if out else produced
    if not engine.exists():
        raise TRTBuildError("export reported success but no engine file was produced")
    cache = engine_cache_path(model_path, imgsz)
    if engine.resolve() != cache.resolve():
        # engine is often produced itself; deleting it here would orphan the rename
        os.replace(str(engine), str(cache))
    # intermediates are not reused across builds and cost ~100MB: best.onnx + best.int8.onnx
    for leftover in src_pt.parent.glob(f"{src_pt.stem}*.onnx"):
        leftover.unlink(missing_ok=True)
    log.info("engine ready: %s", cache)
    return cache


def load_engine(engine_path: Path):
    from ultralytics import YOLO

    return YOLO(str(engine_path), task="detect")
