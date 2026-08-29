# 透明图形检测与弹窗定位移植，对照参考项目 detect.rs：
#   detect_transparent_shapes / preprocess_for_yolo / remap_from_yolo
#   detect_lie_detector_shape / detect_lie_detector_shape_preparing
# 推理引擎与参考项目一致：ONNX Runtime，优先 CUDA EP，模型为同一份
# transparent_shape_nms.onnx（YOLOv12n 内嵌 NMS，输出 [1,300,6]）。
import math
import os
from pathlib import Path

import cv2  # 导入 OpenCV，用于 letterbox 预处理与模板匹配。
import numpy as np  # 导入 NumPy，用于张量构造与框裁剪。

MODULE_DIR = Path(__file__).resolve().parent  # 本模块目录，模板图片随模块存放。
TEMPLATE_DIR = MODULE_DIR / "templates"  # 弹窗标题模板目录。
MODEL_PATH = MODULE_DIR.parent.parent / "assets" / "models" / "transparent_shape_nms.onnx"  # 检测模型路径。

TEMPLATE_THRESHOLD = 0.6  # 模板匹配阈值，与参考项目 detect_lie_detector_shape 一致。
REGION_OFFSET = (0, 20)  # 图形区域相对标题左上角的偏移，与参考项目 solve_shape.rs 一致。
REGION_SIZE = (755, 505)  # 图形区域尺寸（Ideal Ratio 分辨率标定），与参考项目一致。

_YOLO_SIZE = 640  # YOLO 输入边长。


def _round_half_up(value):  # 四舍五入（0.5 向上取整），对齐 Rust f32::round 语义。
    return int(math.floor(value + 0.5))


def _preload_nvidia_dlls():  # 预加载 pip 安装的 NVIDIA 运行时 DLL（含 cuDNN 子模块）。
    # ORT 以受限搜索加载 CUDA 提供器，cuDNN 的壳 DLL 需子模块已加载才能解析符号，
    # 因此这里用绝对路径显式加载全部 DLL，之后按模块名复用即可。
    try:
        import ctypes
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            return []
        nvidia_root = Path(spec.submodule_search_locations[0])
        loaded = []
        for sub in ("nvjitlink", "cuda_runtime", "cublas", "cufft", "cudnn"):
            bin_dir = nvidia_root / sub / "bin"
            if not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))  # 绝对路径要求，供其余依赖搜索。
            except OSError:
                pass
            for dll in sorted(bin_dir.glob("*.dll")):  # 按名排序保证依赖顺序稳定。
                try:
                    ctypes.WinDLL(str(dll))
                    loaded.append(dll.name)
                except OSError:
                    continue
        return loaded
    except Exception:
        return []


def preprocess_for_yolo(bgr):  # 对照参考项目 preprocess_for_yolo：BGR→RGB、min-ratio letterbox 到 640、灰度 114 填充、/255。
    height, width = bgr.shape[:2]
    w_ratio = _YOLO_SIZE / width
    h_ratio = _YOLO_SIZE / height
    min_ratio = min(w_ratio, h_ratio)
    new_w = round(width * min_ratio)
    new_h = round(height * min_ratio)
    pad_w = (_YOLO_SIZE - new_w) / 2.0
    pad_h = (_YOLO_SIZE - new_h) / 2.0
    top = _round_half_up(pad_h - 0.1)
    bottom = _round_half_up(pad_h + 0.1)
    left = _round_half_up(pad_w - 0.1)
    right = _round_half_up(pad_w + 0.1)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = image.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])  # HWC→NCHW 并加批次维。
    return blob, min_ratio, min_ratio, left, top


def remap_from_yolo(pred, size, w_ratio, h_ratio, left, top):  # 对照参考项目 remap_from_yolo：网络坐标逆映射回原图并裁剪边界。
    width, height = size
    tl_x = min(max((pred[0] - left) / w_ratio, 0.0), float(width))
    tl_y = min(max((pred[1] - top) / h_ratio, 0.0), float(height))
    br_x = min(max((pred[2] - left) / w_ratio, 0.0), float(width))
    br_y = min(max((pred[3] - top) / h_ratio, 0.0), float(height))
    return tl_x, tl_y, br_x - tl_x, br_y - tl_y


class TransparentShapeDetector:  # 透明图形检测器：懒加载 ONNX 会话，detect 输入 BGR 区域、输出 [(tlwh, score)]。

    def __init__(self, model_path=None, logger=None):  # 构造函数：记录模型路径与日志器，会话首次推理时创建。
        self.model_path = str(model_path or MODEL_PATH)
        self.logger = logger
        self.session = None  # ONNX 会话，首次使用时创建。
        self.provider = ""  # 实际生效的执行提供器名称，供界面展示。

    def _log(self, message):  # 有日志器时输出，否则静默。
        if self.logger is not None:
            self.logger.info(message)

    def _ensure_session(self):  # 创建推理会话：优先 CUDA EP，失败回退 CPU；模型缺失时报错。
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime not installed, run: pip install -e '.[inference]'") from exc
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"transparent shape model not found: {self.model_path}")
        _preload_nvidia_dlls()  # 先预加载 NVIDIA pip 轮子的全部 DLL 再创建会话。
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            try:
                self.session = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as exc:
                self._log(f"CUDAExecutionProvider init failed, fallback to CPU: {exc}")
                self.session = None
        if self.session is None:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self._log(f"transparent shape model loaded, provider: {self.provider}")

    def detect(self, bgr):  # 检测一张 BGR 区域内的全部透明图形，返回 [(x, y, w, h, score)]。
        self._ensure_session()
        blob, w_ratio, h_ratio, left, top = preprocess_for_yolo(bgr)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        rows = outputs[0][0]  # 输出形状 [1,300,6]：x1,y1,x2,y2,score,class，类别列忽略。
        height, width = bgr.shape[:2]
        results = []
        for pred in rows:
            score = float(pred[4])
            if score <= 0:  # 内嵌 NMS 用零分行补齐到 300 行，直接跳过。
                continue
            x, y, w, h = remap_from_yolo(pred, (width, height), w_ratio, h_ratio, left, top)
            if w <= 0 or h <= 0:
                continue
            results.append(((x, y, w, h), score))
        return results


def _match_template(frame, template, threshold=TEMPLATE_THRESHOLD):  # TM_CCOEFF_NORMED 单点匹配，命中返回左上角坐标。
    if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return max_loc
    return None


_MULTI_SCALES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.9, 0.8, 0.7, 0.6, 0.5)  # 多尺度金字塔，兼容不同分辨率录像。


def _match_template_multiscale(frame, template, threshold=TEMPLATE_THRESHOLD):
    # 多尺度模板匹配：优先原始尺寸，再逐个缩放重试，命中即返回 (左上角, 缩放比例)，未命中返回 (None, 1.0)。
    for scale in _MULTI_SCALES:
        if scale == 1.0:
            scaled = template
        else:
            new_w = int(round(template.shape[1] * scale))
            new_h = int(round(template.shape[0] * scale))
            if new_w < 8 or new_h < 8:  # 缩得过小会失去判别力，跳过。
                continue
            scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if frame.shape[0] < scaled.shape[0] or frame.shape[1] < scaled.shape[1]:
            continue
        result = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return max_loc, scale
    return None, 1.0


class MultiScaleTemplate:  # 预缩放多尺度模板：构造时一次性生成全部尺度，高频匹配时免去每帧重复 resize。

    def __init__(self, template, threshold=TEMPLATE_THRESHOLD):
        self.threshold = threshold
        self.scales = []  # [(缩放比例, 缩放后模板)]，顺序与 _MULTI_SCALES 一致。
        for scale in _MULTI_SCALES:
            if scale == 1.0:
                self.scales.append((scale, template))
                continue
            new_w = int(round(template.shape[1] * scale))
            new_h = int(round(template.shape[0] * scale))
            if new_w < 8 or new_h < 8:
                continue
            self.scales.append((scale, cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)))

    def match(self, frame):  # 逐尺度匹配，命中即返回 (左上角, 缩放比例)，未命中返回 (None, 1.0)。
        for scale, scaled in self.scales:
            if frame.shape[0] < scaled.shape[0] or frame.shape[1] < scaled.shape[1]:
                continue
            result = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= self.threshold:
                return max_loc, scale
        return None, 1.0


def _template_best_score(frame, template):  # 多尺度最高分（不过阈值），供裁剪模式探测诊断用。
    frame_h, frame_w = frame.shape[:2]
    best = -1.0
    for scale in _MULTI_SCALES:
        if scale == 1.0:
            scaled = template
        else:
            new_w = int(round(template.shape[1] * scale))
            new_h = int(round(template.shape[0] * scale))
            if new_w < 8 or new_h < 8:
                continue
            scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if frame_h < scaled.shape[0] or frame_w < scaled.shape[1]:
            continue
        result = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(result.max()))
    return best


def _imread_template(name):  # 读取模块内置模板图片，缺失时抛出明确错误。
    path = TEMPLATE_DIR / name
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"lie detector template not found: {path}")
    return template


class LieDetectorRegion:  # 弹窗定位：模板匹配标题并计算图形活动区域，对照参考项目 detect_lie_detector_shape 系列。

    def __init__(self, threshold=TEMPLATE_THRESHOLD):
        self.threshold = threshold
        self._title_template = None  # 谎言检测器标题模板，懒加载。
        self._prepare_template = None  # “准备中”画面模板，懒加载。

    @property
    def title_template(self):
        if self._title_template is None:
            self._title_template = _imread_template("lie_detector_new.png")
        return self._title_template

    @property
    def prepare_template(self):
        if self._prepare_template is None:
            self._prepare_template = _imread_template("lie_detector_shape_prepare.png")
        return self._prepare_template

    def is_preparing(self, frame):  # 检测“准备中”画面，准备阶段不应开始求解，与参考一致（多尺度兼容不同分辨率）。
        location, _ = _match_template_multiscale(frame, self.prepare_template, self.threshold)
        return location is not None

    def find_title(self, frame):  # 多尺度匹配弹窗标题，命中返回 (x, y, w, h)，否则 None。
        location, scale = _match_template_multiscale(frame, self.title_template, self.threshold)
        if location is None:
            return None
        height = int(round(self.title_template.shape[0] * scale))
        width = int(round(self.title_template.shape[1] * scale))
        return location[0], location[1], width, height

    def find_region(self, frame):  # 由标题与命中缩放比例推算图形区域（偏移与尺寸同比例缩放），按帧边界钳制。
        title = self.find_title(frame)
        if title is None:
            return None
        scale = title[2] / self.title_template.shape[1]  # 标题实测宽/模板原始宽，即命中缩放。
        frame_h, frame_w = frame.shape[:2]
        x = title[0] + int(round(REGION_OFFSET[0] * scale))
        y = title[1] + int(round(REGION_OFFSET[1] * scale))
        w = min(int(round(REGION_SIZE[0] * scale)), frame_w - x)
        h = min(int(round(REGION_SIZE[1] * scale)), frame_h - y)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
