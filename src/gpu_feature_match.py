# 按标注分类名的显卡全屏模板匹配，提供两条等价于 CPU 路径的入口：
#   - prepare(names, frame)：模板取自框架 FeatureSet（已按当前画面尺寸缩放），结果等价于
#     FeatureSet.find_feature(frame, name, 1, 1, threshold, True, limit=1)；
#   - prepare_templates(named_mats, frame, fingerprint)：模板由调用方直接给出（原生尺度、不随画面缩放），
#     一个分类可给多张模板，结果等价于「逐张 cv2.matchTemplate 取最高分」，供全桌面重登匹配器使用。
# 一帧画面只上传一次显存、只做一次帧 FFT，多个标注分类共用同一次变换，
# 适合「同一帧连查多个分类」与「同一分类多张模板」这类每帧多次匹配的场景。
#
# 无法显卡化的情况自动排除，由调用方按 has(name) 回退框架 CPU 匹配：
#   - 未安装 CuPy 或没有可用 NVIDIA 显卡（gpu_match_available() 为 False）；
#   - 分类名未在模板页标注；
#   - 标注带 mask：OpenCV 的掩码匹配无法用 FFT 互相关等价复现；
#   - 模板比画面还大：CPU 路径会跳过该模板，这里同样跳过；
#   - 运行期显卡异常：由调用方捕获后关闭加速，本模块不做静默降级以免掩盖故障。
#
# 唯一已知的得分差异：纯色（零方差）模板或纯色窗口下，CPU 的 TM_CCOEFF_NORMED 得分图会退化成整张常数
# （OpenCV 5.0 实测这个常数值还依尺寸/内部代码路径而变，大尺寸给 1.0 并在左上角 (0,0) 误报命中，小尺寸给 0.0），
# 本模块把分母退化的位置一律记 0 分，不会误报也不会产生无意义的 (0,0) 点击。这类模板本身就是无效标注。
#
# 位置精度的边界：同一模板在画面里出现多份像素相同的副本时，底层按 TIE_EPS 容差取最靠前的那份，
# 与 OpenCV minMaxLoc 的首个最大值规则一致（不做平局判定时显卡的 float32 舍入会选中靠后的副本）。
# 但若得分面本身就是一大片平台（低对比度或重复图案，实测某分类在 5360x1440 全桌面上有 1700 多个位置
# 与最高分相差不超 1e-4），最高分位置在数值上就不可判定，两条路径可能选中相隔一个重复周期的不同位置。
# 此时得分差仍在 1e-4 量级，对阈值判定（普遍 0.7 以上）没有影响，只影响重复图案上的点击落点。
import os  # 导入标准库 os，用于取标注文件指纹判断是否需要重建模板。

import numpy as np  # 导入 NumPy，用于画面通道裁剪与内存连续性整理。

from ok.feature.Box import Box  # 导入框架匹配框类型，返回值与 CPU 路径保持同一结构，调用方无需区分。

from src.gpu_match import GpuTemplateMatcher, gpu_available  # 导入 CuPy FFT 模板匹配器与显卡可用性探测。

_GPU_AVAILABLE = None  # 显卡可用性探测结果缓存：CuPy 首次探测要初始化驱动，进程内只探一次。


def gpu_match_available():  # 显卡模板匹配是否可用（结果缓存，避免每帧重复初始化探测）。
    global _GPU_AVAILABLE  # 需要写模块级缓存。
    if _GPU_AVAILABLE is None:  # 尚未探测过。
        _GPU_AVAILABLE = bool(gpu_available())  # 探测一次并缓存结果。
    return _GPU_AVAILABLE  # 返回缓存结果。


def _clean_names(names):  # 清洗分类名列表：去空白、去空项、去重且保序（顺序参与重建键，避免同一组名字因顺序不同反复重建）。
    unique = []  # 结果列表。
    for name in names or []:  # 逐个清洗。
        cleaned = str(name or '').strip()  # 去空白。
        if cleaned and cleaned not in unique:  # 有效且不重复。
            unique.append(cleaned)  # 收集。
    return unique  # 返回去重保序的分类名列表。


class GpuFeatureMatcher:  # 把「按分类名找最高分框」搬到显卡上，对外返回与 CPU 路径同结构的 Box 或 (x, y, w, h, score) 元组。

    def __init__(self, feature_set=None, gray=True, logger=None):  # 构造函数：feature_set 仅 prepare 路径需要，显式模板路径可省略。
        self.feature_set = feature_set  # 框架特征集：标注模板与画面尺寸都由它提供；prepare_templates 路径不使用它。
        self.gray = bool(gray)  # 记录匹配色彩模式。
        self.logger = logger  # 可选日志器，仅用于记录跳过显卡化的分类名，None 时静默。
        self.matcher = GpuTemplateMatcher(gray=self.gray)  # 底层 CuPy FFT 匹配器。
        self.skipped = []  # 无法显卡化的分类名（未标注/带 mask/无可用模板），调用方对这些名字回退 CPU。
        self._key = None  # prepare 路径上次成功准备的重建键，None 表示尚未准备好。
        self._groups = {}  # 分类名 -> 显卡模板 key 列表：一个分类可标注在多处，匹配时取最高分的那处。
        self._shapes = {}  # 分类名 -> 已注册模板的形状元组，供 prepare_templates 判断能否增量复用。
        self._stamp = None  # prepare_templates 路径的标注指纹，变化时整体重建模板。

    def prepare(self, names, frame):  # 按分类名列表与画面尺寸准备模板（框架特征集路径），返回是否至少有一个分类可显卡匹配。
        unique = _clean_names(names)  # 去重且保序的分类名列表。
        if not unique or frame is None or getattr(frame, 'ndim', 0) < 2:  # 没有名字或没有画面，无从匹配。
            return False  # 准备失败。
        height, width = frame.shape[:2]  # 当前画面尺寸：框架按它缩放模板，尺寸变了模板必须重建。
        key = (tuple(unique), self.gray, int(height), int(width), self._fingerprint())  # 重建键：名字集合 + 色彩模式 + 画面尺寸 + 标注文件指纹。
        if key == self._key and self.matcher.templates:  # 与上次准备完全一致。
            return True  # 直接复用，不重复上传模板。
        self._reset()  # 丢弃旧模板与核 FFT 缓存；准备失败时也不留半套模板，调用方按 has() 回退 CPU。
        try:  # check_size 会在画面尺寸变化时清空框架特征字典，必须先于取模板调用，保证模板与画面同尺度。
            self.feature_set.check_size(frame)  # 框架内部加锁，线程安全。
        except Exception:  # 特征集异常（标注文件损坏等）。
            self._log(f"GPU match check_size failed: {self.feature_set}")  # 记录，交由调用方按 CPU 处理。
            return False  # 准备失败。
        for name in unique:  # 逐个分类名取模板并上传显存。
            feature = self._load_feature(name)  # 取框架特征对象。
            mat = getattr(feature, 'mat', None) if feature is not None else None  # 模板原图（框架已按当前画面尺寸缩放）。
            if mat is None or getattr(feature, 'mask', None) is not None:  # 未标注，或标注带掩码无法用 FFT 等价复现。
                self.skipped.append(name)  # 记入跳过名单，调用方对该名字回退 CPU 匹配。
                continue  # 下一个分类名。
            self._groups[name] = self._register(name, [mat])  # 框架特征集每个分类只有一张模板图。
        if not self.matcher.templates:  # 一个模板都没注册上（全部未标注或全部带掩码）。
            self._log(f"GPU match has no usable template: skipped={self.skipped}")  # 记录原因，便于排查为何没走显卡。
            return False  # 准备失败，调用方整体回退 CPU。
        self._key = key  # 准备成功，记录重建键供下一帧复用。
        return True  # 至少一个分类可显卡匹配。

    def prepare_templates(self, named_mats, frame, fingerprint=None):  # 显式模板路径：模板由调用方直接给出（原生尺度、不随画面缩放），一个分类可给多张。
        if frame is None or getattr(frame, 'ndim', 0) < 2:  # 没有画面，无从匹配。
            return False  # 准备失败。
        named = []  # 清洗后的 [(分类名, [模板, ...]), ...]。
        for name, mats in named_mats or []:  # 逐个分类清洗。
            cleaned = str(name or '').strip()  # 去空白。
            if not cleaned:  # 空名字无从注册。
                continue  # 跳过。
            usable = [m for m in (mats or []) if getattr(m, 'ndim', 0) >= 2 and m.size > 0]  # 丢掉非法维度与空模板。
            named.append((cleaned, usable))  # 收集（usable 为空表示该分类没有可显卡化的模板）。
        if not named:  # 一个有效分类都没有。
            return False  # 准备失败。
        if fingerprint != self._stamp:  # 标注来源变化（或首次准备）：整体重建，丢弃旧模板与核 FFT 缓存。
            self._reset()  # 清空底层匹配器与分组索引。
            self._stamp = fingerprint  # 记录本次标注指纹。
        for name, mats in named:  # 逐分类增量注册：已按同样形状注册过的直接复用，避免反复上传与重算核 FFT。
            shapes = tuple(tuple(m.shape) for m in mats)  # 本分类全部模板的形状。
            if self._shapes.get(name) == shapes:  # 形状一致说明模板没变。
                continue  # 复用已上传的模板。
            self._drop(name)  # 形状变了（重新标注）：先摘掉旧模板，避免残留 key 干扰。
            keys = self._register(name, mats)  # 上传新模板。
            if keys:  # 至少注册上一张。
                self._groups[name] = keys  # 记录分组，供 has/best 查询。
                self._shapes[name] = shapes  # 记录形状供下次比对。
            elif name not in self.skipped:  # 该分类一张可用模板都没有。
                self.skipped.append(name)  # 记入跳过名单，调用方对该名字回退 CPU。
        return bool(self._groups)  # 至少一个分类可显卡匹配才算准备好。

    def _reset(self):  # 丢弃全部已注册模板与缓存，回到未准备状态。
        self.matcher = GpuTemplateMatcher(gray=self.gray)  # 重建底层匹配器，核 FFT 缓存随之作废。
        self.skipped = []  # 清空跳过名单，本轮重新判定。
        self._key = None  # 清空 prepare 路径的重建键，下一帧会重试。
        self._groups = {}  # 清空分类名到显卡模板 key 的分组索引。
        self._shapes = {}  # 清空模板形状索引。

    def _register(self, name, mats):  # 把一个分类的全部模板上传显存，返回注册上的 key 列表。
        keys = []  # 本分类注册成功的显卡模板 key。
        total = len(mats)  # 模板张数：单张直接用分类名做 key，多张追加序号区分。
        for index, mat in enumerate(mats):  # 逐张上传（同一分类标注在多处时全部注册，匹配时取最高分）。
            key = name if total == 1 else f"{name}#{index}"  # 显卡模板 key。
            self.matcher.add_template(key, self._to_bgr(mat))  # 注册：零均值化、翻转核与统计量在底层一次算好。
            keys.append(key)  # 收集。
        return keys  # 返回本分类的 key 列表。

    def _drop(self, name):  # 摘掉某分类已注册的显卡模板（重新标注后形状变化时用）。
        for key in self._groups.pop(name, []):  # 逐个 key 从底层匹配器移除。
            self.matcher.templates.pop(key, None)  # 底层模板字典是普通 dict，直接弹出即可。
        self._shapes.pop(name, None)  # 形状索引一并清掉。
        if name in self.skipped:  # 之前被跳过的分类这次可能变得可用。
            self.skipped.remove(name)  # 从跳过名单移除，避免残留。

    def has(self, name):  # 指定分类名是否已注册到显卡匹配器（未注册的名字调用方应回退 CPU）。
        return bool(self._groups.get(str(name or '').strip()))  # 以清洗后的名字查分组索引。

    def frame(self, frame_bgr):  # 上传一帧画面并返回匹配句柄，句柄内的帧 FFT 由全部模板复用。
        return self.matcher.match_frame(self._to_bgr(frame_bgr))  # 裁掉 alpha 通道后交给底层匹配器。

    def best(self, handle, name):  # 取指定分类的最高分匹配 (x, y, w, h, score)，不做阈值判定；无可用模板返回 None。
        hit = self._best_hit(handle, name)  # 显卡上求该分类全部模板里的最高分。
        if hit is None:  # 未注册或模板都比画面大，与 CPU 路径一致地无结果。
            return None  # 无匹配。
        key, x, y, score = hit  # 拆包胜出模板的结果。
        td = self.matcher.templates[key]  # 取胜出模板的尺寸，框大小与 CPU 路径一致（模板宽高）。
        return int(x), int(y), int(td['w']), int(td['h']), float(score)  # 返回与 cv2 匹配同结构的五元组。

    def best_box(self, handle, name, threshold):  # 取指定分类的最高分框，低于阈值返回 None，语义与框架 limit=1 一致。
        hit = self._best_hit(handle, name)  # 显卡上求最高分。
        if hit is None:  # 句柄缺失、该分类未注册或模板都比画面大。
            return None  # 按未匹配处理。
        key, x, y, score = hit  # 拆包胜出模板的结果。
        if score < threshold:  # 最高分未达阈值。
            return None  # 未匹配。
        td = self.matcher.templates[key]  # 取模板尺寸，框大小与 CPU 路径一致（模板宽高）。
        return Box(int(x), int(y), td['w'], td['h'], confidence=float(score), name=str(name or '').strip())  # 组装框架 Box 返回。

    def best_score(self, handle, name):  # 取指定分类的实际最高分（不做阈值判定），供诊断日志使用。
        hit = self._best_hit(handle, name)  # 显卡上求最高分。
        return 0.0 if hit is None else float(hit[3])  # 无可用模板时报 0 分。

    def _best_hit(self, handle, name):  # 在该分类注册的全部模板里取最高分，返回 (key, x, y, score)；无可用模板返回 None。
        keys = self._groups.get(str(name or '').strip())  # 取该分类的显卡模板 key 列表。
        if handle is None or not keys:  # 句柄缺失或分类未注册。
            return None  # 无结果。
        best = None  # 当前最高分记录。
        for key in keys:  # 逐张模板求最高分，与 CPU 路径「逐模板 matchTemplate 再取最大」一致。
            td = self.matcher.templates.get(key)  # 模板缓存。
            if td is None:  # 已被摘掉（重新标注中）。
                continue  # 跳过。
            if td['h'] > handle.H or td['w'] > handle.W:  # 模板比画面还大：CPU 路径会跳过该模板，显卡也必须跳过，否则得分图切片会越界。
                continue  # 跳过。
            x, y, score = handle.best(key)  # 显卡上取最高分位置与分数。
            if best is None or score > best[3]:  # 刷新最高分（严格大于：同分时保留先注册的模板，与 CPU 判定一致）。
                best = (key, x, y, score)  # 记录胜出模板。
        return best  # 返回胜出结果或 None。

    def _load_feature(self, name):  # 从框架特征集取分类对应的特征对象，未标注或异常返回 None。
        try:  # ensure_feature 首次调用才读盘解析标注，之后走字典命中，开销可忽略。
            self.feature_set.ensure_feature(name)  # 确保该分类已加载。
            return self.feature_set.feature_dict.get(name)  # 返回特征对象或 None。
        except Exception as e:  # 标注缺失/文件损坏等异常。
            self._log(f"GPU match load feature {name} failed: {e}")  # 记录原因。
            return None  # 按未标注处理，调用方回退 CPU。

    def _fingerprint(self):  # 取标注文件指纹（修改时间与大小），文件不可读时返回 None。
        path = getattr(self.feature_set, 'coco_json', None)  # 框架已解析为绝对路径。
        if not path:  # 特征集未提供路径。
            return None  # 无指纹，重建键退化为「画面尺寸变化才重建」。
        try:  # 只读文件元数据，比整文件读取便宜得多，可以每帧调用。
            stat = os.stat(path)  # 取文件元信息。
            return stat.st_mtime_ns, stat.st_size  # 修改时间与大小共同构成指纹，重新标注后必定变化。
        except OSError:  # 文件不存在或不可读。
            return None  # 无指纹。

    @staticmethod
    def _to_bgr(mat):  # 把画面/模板整理成显卡匹配器可用的三通道连续数组（框架截图可能是 BGRA）。
        if getattr(mat, 'ndim', 0) == 3 and mat.shape[2] == 4:  # 四通道带 alpha。
            mat = mat[:, :, :3]  # 裁掉 alpha，与框架 search_area[..., :3] 的处理一致。
        return np.ascontiguousarray(mat)  # 保证内存连续，CuPy 上传与 cvtColor 都要求连续数组。

    def _log(self, message):  # 有日志器才输出，避免无日志场景下报错。
        if self.logger is not None:  # 调用方注入了日志器。
            self.logger.info(message)  # 记录显卡匹配的诊断信息。
