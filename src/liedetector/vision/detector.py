"""YOLO detection wrapper with runtime class-name verification."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.liedetector.vision import config
from src.liedetector.vision import tensorrt_backend
from src.liedetector.vision.geometry import bbox_center
from src.liedetector.vision.logger import get_logger

log = get_logger("detector")


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[float, float]
    track_id: Optional[int] = None
    shape: str = ""  # logical shape name, may differ from the model's label

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ClassNameMismatch(Exception):
    pass


def resolve_class_map(model_names: Dict[int, str]) -> Dict[str, int]:
    """
    Map logical shape names to the model's actual class ids via config.SHAPE_ALIASES.
    The doc's id table is only an assumption - trust model.names over it.
    """
    # SHAPE_ALIASES is keyed by the model's real label -> logical shape name
    alias = {str(k).strip().lower(): v for k, v in config.SHAPE_ALIASES.items()}
    resolved: Dict[str, int] = {}
    for cid, cname in model_names.items():
        logical = alias.get(str(cname).strip().lower())
        if logical:
            resolved.setdefault(logical, int(cid))

    missing = [n for n in config.LOGICAL_SHAPES if n not in resolved]
    if missing:
        raise ClassNameMismatch(
            f"model.names={dict(sorted(model_names.items()))} has no alias for {missing}; "
            f"add them to config.SHAPE_ALIASES"
        )
    log.info("class map resolved: %s", resolved)
    return resolved


class Detector:
    """
    device: "cpu" | "gpu" | "auto" (None -> config, which is auto).
    GPU means TensorRT INT8 (lowest latency): a cached engine loads in ms, a missing one is
    built from the checkpoint with calibration frames from `calib_video` (minutes, once).
    Degradation is explicit and logged: TensorRT INT8 -> torch CUDA -> CPU.
    """

    def __init__(self, model_path: str = None, conf: float = None, device: str = None,
                 calib_video: str = None):
        import src.liedetector.vision.device as device_mod

        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.conf = conf if conf is not None else config.YOLO_CONF_THRESHOLD
        requested = device or config.YOLO_DEVICE or device_mod.REQUEST_AUTO
        self.requested = device_mod.resolve_request(requested)
        self.backend = device_mod.BACKEND_CPU
        self.device = "cpu"
        base = None  # loaded lazily: on the GPU path the checkpoint is only needed as the
        self.model = self._init_model(device_mod, calib_video, base)
        self.class_map = resolve_class_map(self.model.names)
        log.info("model %s on %s (%s)", os.path.basename(self.model_path),
                 self.device, device_mod.backend_label(self.backend))
        self.warmup()

    def _base_model(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def _init_model(self, device_mod, calib_video: str, base):
        if self.requested == device_mod.REQUEST_CPU:
            return self._base_model()

        cache = tensorrt_backend.engine_cache_path(self.model_path, config.YOLO_INPUT_SIZE[0])
        if cache.exists():
            try:
                engine = tensorrt_backend.load_engine(cache)
                self.backend, self.device = device_mod.BACKEND_TRT_CACHE, 0
                log.info("loaded cached TensorRT engine %s", cache.name)
                return engine
            except Exception as exc:
                log.warning("cached engine %s unusable (%s), rebuilding", cache.name, exc)
                cache.unlink(missing_ok=True)

        cuda_ok = device_mod.torch_cuda_available()
        trt_ok = device_mod.tensorrt_available()
        if not cuda_ok or not trt_ok:
            why = "PyTorch 无 CUDA 支持" if not cuda_ok else "未安装 TensorRT"
            log.warning("GPU 请求被降级为 CPU：%s（完整 GPU 方案为 TensorRT INT8）", why)
            return self._base_model()

        try:
            names = (base := self._base_model()).names
            cache = tensorrt_backend.build_int8_engine(
                self.model_path, config.YOLO_INPUT_SIZE[0], calib_video, names,
            )
            engine = tensorrt_backend.load_engine(cache)
        except tensorrt_backend.TRTBuildError as exc:
            log.warning("TensorRT INT8 构建失败：%s", exc)
            if cuda_ok:
                log.warning("回退到 torch CUDA；修复 TensorRT 后删除 %s 可重建 INT8 引擎", cache.name)
                self.backend, self.device = device_mod.BACKEND_CUDA, 0
                return base if base is not None else self._base_model()
            log.warning("回退到 CPU")
            return self._base_model()

        self.backend, self.device = device_mod.BACKEND_TRT, 0
        return engine

    @property
    def backend_label(self) -> str:
        import src.liedetector.vision.device as device_mod

        return device_mod.backend_label(self.backend)

    def warmup(self) -> None:
        """GPU context init and engine deserialization land on the first inference; pay that
        once here so the first real frame is not measured (or dropped) because of it."""
        if self.device == "cpu":
            return
        import numpy as np

        self.detect(np.zeros((*config.YOLO_INPUT_SIZE, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=config.YOLO_NMS_THRESHOLD,
            imgsz=config.YOLO_INPUT_SIZE[0],
            device=self.device,
            verbose=False,
        )
        return self._to_detections(results[0])

    def reset_tracking(self) -> None:
        """宿主适配：下一次 track() 以 persist=False 调用，清掉 ultralytics OC-SORT 内部
        残留的上一解题会话轨迹簿（warm Detector 跨会话复用时由新 Pipeline 构造触发）。"""
        self._reset_next_track = True

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        if getattr(self, "_reset_next_track", False):
            persist = False
            self._reset_next_track = False
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=config.YOLO_NMS_THRESHOLD,
            imgsz=config.YOLO_INPUT_SIZE[0],
            device=self.device,
            persist=persist,
            tracker=config.TRACKER_CONFIG_PATH,
            verbose=False,
        )
        dets = self._to_detections(results[0])
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int).tolist()
            for det, tid in zip(dets, ids):
                det.track_id = int(tid)
        return dets

    def _to_detections(self, result) -> List[Detection]:
        out: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        id_to_logical = {v: k for k, v in self.class_map.items()}
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            bbox = (float(x1), float(y1), float(x2), float(y2))
            out.append(
                Detection(
                    bbox=bbox,
                    confidence=float(c),
                    class_id=int(k),
                    class_name=str(self.model.names.get(int(k), k)),
                    centroid=bbox_center(bbox),
                    shape=id_to_logical.get(int(k), ""),
                )
            )
        return out

    def classes_for_shape(self, target_shape: str) -> List[int]:
        """Doc 4.2: basic shapes track with their own class, combos with all combo classes."""
        if target_shape in config.BASIC_SHAPES:
            return [self.class_map[target_shape]]
        return [self.class_map[n] for n in config.COMBO_SHAPES]
