# 弹窗定位：多尺度模板匹配谎言检测器弹窗标题并推算图形活动区域，
# 对照参考项目 detect.rs 的 detect_lie_detector_shape / detect_lie_detector_shape_preparing。
# 透明图形求解已迁移到 shape_session（DIS 光流，无神经网络），本文件不再包含 ONNX/YOLO 检测。
from pathlib import Path

import cv2  # 导入 OpenCV，用于多尺度模板匹配。

MODULE_DIR = Path(__file__).resolve().parent  # 本模块目录，模板图片随模块存放。
TEMPLATE_DIR = MODULE_DIR / "templates"  # 弹窗标题模板目录。

TEMPLATE_THRESHOLD = 0.6  # 模板匹配阈值，与参考项目 detect_lie_detector_shape 一致。
REGION_OFFSET = (0, 20)  # 图形区域相对标题左上角的偏移，与参考项目 solve_shape.rs 一致。
REGION_SIZE = (755, 505)  # 图形区域尺寸（Ideal Ratio 分辨率标定），与参考项目一致。

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
