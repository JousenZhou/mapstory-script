# 显卡标注匹配封装（GpuFeatureMatcher）回归测试：验证它按标注分类名在一帧上取最高分框的结果
# 与框架 CPU 路径（cv2.matchTemplate + TM_CCOEFF_NORMED + 灰度 + limit=1）等价，
# 并覆盖无法显卡化的降级场景：未标注、标注带 mask、四通道画面、画面尺寸变化后重建模板。
# 无 CuPy 或无 NVIDIA 显卡的环境下自动跳过显卡用例，不阻断 CI。
import unittest

import cv2  # CPU 参考实现：与显卡结果逐分类对比。
import numpy as np  # 构造合成画面与模板。

from src.gpu_feature_match import GpuFeatureMatcher, gpu_match_available  # 被测封装与显卡可用性探测。

GPU_READY = gpu_match_available()  # 本机是否有可用显卡：无则全部显卡用例跳过。


class FakeFeature:  # 假框架特征对象：只保留 GpuFeatureMatcher 用到的 mat 与 mask 两个属性。

    def __init__(self, mat, mask=None):  # mat 为模板原图，mask 非 None 表示该标注带掩码。
        self.mat = mat
        self.mask = mask


class FakeFeatureSet:  # 假框架特征集：模拟 check_size 按画面尺寸重载标注与 ensure_feature 的字典查询。

    def __init__(self, features, coco_json=''):  # features 为 {分类名: FakeFeature}。
        self._source = dict(features)  # 标注源数据，尺寸变化时用它模拟框架重读标注。
        self.feature_dict = dict(features)  # 框架对外的特征字典。
        self.coco_json = coco_json  # 标注文件路径，供指纹计算；空串表示无指纹。
        self.width = 0  # 当前标注尺度的画面宽。
        self.height = 0  # 当前标注尺度的画面高。
        self.size_checks = 0  # check_size 调用次数，用于验证封装确实先同步了画面尺寸。

    def check_size(self, frame):  # 画面尺寸变化时清空并按新尺寸重读标注（这里直接还原源数据模拟重读）。
        self.size_checks += 1
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height) and height > 0 and width > 0:
            self.width, self.height = width, height
            self.feature_dict = dict(self._source)
        return True

    def ensure_feature(self, feature_name):  # 假实现：特征已在字典里，不做任何读盘。
        return None


def cpu_best(frame, template):  # CPU 参考：全屏灰度 TM_CCOEFF_NORMED 的最高分位置与分数，与框架 limit=1 语义一致。
    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    tpl = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], float(max_val)


def build_scene(height=480, width=640):  # 造一帧带结构的合成画面，并从中裁出两个模板作为「已标注」分类。
    frame = (np.random.RandomState(7).rand(height, width, 3) * 255).astype(np.uint8)
    cv2.rectangle(frame, (450, 350), (620, 460), (40, 60, 90), -1)  # 右下角加块纯色结构，不与下方模板裁剪区重叠。
    cv2.circle(frame, (520, 120), 40, (200, 30, 30), -1)  # 再加个圆形，让画面更接近真实游戏截图。
    features = {
        '测谎触发': FakeFeature(frame[100:130, 200:260].copy()),  # 画面上确实存在的模板，应能高分命中。
        '掉线2': FakeFeature(frame[200:240, 300:360].copy()),  # 第二个模板，验证同帧多模板复用。
        '带掩码': FakeFeature(frame[50:80, 100:140].copy(), mask=np.zeros((30, 40), dtype=np.uint8)),  # 带 mask 的标注不能显卡化。
    }
    return frame, features


class TestGpuFeatureMatcherCpuSide(unittest.TestCase):
    """不依赖显卡的部分：通道裁剪、可用性缓存、无画面/无名字的准备失败。"""

    def test_to_bgr_crops_alpha_and_keeps_three_channels(self):
        # 框架截图可能是 BGRA，必须裁掉 alpha 才能喂给 cvtColor/CuPy。
        bgra = np.zeros((10, 12, 4), dtype=np.uint8)
        cropped = GpuFeatureMatcher._to_bgr(bgra)
        self.assertEqual((10, 12, 3), cropped.shape)  # 四通道被裁成三通道。
        self.assertTrue(cropped.flags['C_CONTIGUOUS'])  # 裁剪后必须重新整理成连续内存。
        bgr = np.zeros((10, 12, 3), dtype=np.uint8)
        self.assertEqual((10, 12, 3), GpuFeatureMatcher._to_bgr(bgr).shape)  # 三通道原样保留。
        gray = np.zeros((10, 12), dtype=np.uint8)
        self.assertEqual((10, 12), GpuFeatureMatcher._to_bgr(gray).shape)  # 二维灰度图不做裁剪。

    def test_gpu_match_available_is_cached(self):
        # 探测结果按进程缓存，避免每帧重复初始化 CuPy 驱动。
        import src.gpu_feature_match as module
        first = module.gpu_match_available()
        self.assertEqual(first, module.gpu_match_available())  # 两次调用结果一致。
        self.assertIsNotNone(module._GPU_AVAILABLE)  # 结果已写入缓存。
        self.assertEqual(bool(first), first)  # 返回值必须是布尔量。

    def test_prepare_requires_names_and_frame(self):
        # 没有分类名或没有画面时准备失败，调用方据此整体回退 CPU。
        frame, features = build_scene()
        matcher = GpuFeatureMatcher(FakeFeatureSet(features))
        self.assertFalse(matcher.prepare([], frame))  # 空名字列表。
        self.assertFalse(matcher.prepare(['测谎触发'], None))  # 无画面。
        self.assertFalse(matcher.prepare(['  ', ''], frame))  # 全是空白名字，清洗后为空。
        self.assertIsNone(matcher._key)  # 准备失败不写重建键，下一帧会重试。

    def test_templates_have_variance(self):
        # 前提自检：测试模板必须是带纹理的，纯色模板在两条路径上都退化为 0 分，对比就没意义了。
        _, features = build_scene()
        for name in ('测谎触发', '掉线2'):
            gray = cv2.cvtColor(features[name].mat, cv2.COLOR_BGR2GRAY)
            self.assertGreater(float(gray.std()), 1.0, name)  # 标准差足够大，说明不是纯色块。


@unittest.skipUnless(GPU_READY, "no available NVIDIA GPU or cupy not installed")
class TestGpuFeatureMatcherOnGpu(unittest.TestCase):
    """需要显卡的部分：与 CPU 结果对比、降级名单、重建时机。"""

    def setUp(self):
        self.frame, self.features = build_scene()
        self.feature_set = FakeFeatureSet(self.features)
        self.matcher = GpuFeatureMatcher(self.feature_set, gray=True)

    def test_prepare_registers_only_matchable_names(self):
        # 带 mask 与未标注的分类被排除，其余正常注册。
        self.assertTrue(self.matcher.prepare(['测谎触发', '掉线2', '带掩码', '没标注过'], self.frame))
        self.assertTrue(self.matcher.has('测谎触发'))  # 已标注且无掩码 -> 可显卡匹配。
        self.assertTrue(self.matcher.has('掉线2'))
        self.assertFalse(self.matcher.has('带掩码'))  # 带掩码无法用 FFT 等价复现。
        self.assertFalse(self.matcher.has('没标注过'))  # 未标注。
        self.assertEqual(['带掩码', '没标注过'], sorted(self.matcher.skipped))  # 两个都被记入跳过名单。
        self.assertGreaterEqual(self.feature_set.size_checks, 1)  # 准备前确实同步了画面尺寸。

    def test_prepare_returns_false_when_nothing_matchable(self):
        # 请求的分类全都不能显卡化时返回 False，调用方整体回退 CPU。
        self.assertFalse(self.matcher.prepare(['带掩码', '没标注过'], self.frame))
        self.assertEqual(0, len(self.matcher.matcher.templates))  # 一个模板都没注册。

    def test_best_box_matches_cpu_result(self):
        # 显卡最高分框与 CPU 结果一致：位置容差 3 像素、分数容差 0.02，框尺寸等于模板尺寸。
        self.assertTrue(self.matcher.prepare(['测谎触发', '掉线2'], self.frame))
        handle = self.matcher.frame(self.frame)
        for name in ('测谎触发', '掉线2'):
            cx, cy, cscore = cpu_best(self.frame, self.features[name].mat)
            box = self.matcher.best_box(handle, name, 0.0)
            self.assertIsNotNone(box, name)  # 阈值 0 时必然命中。
            self.assertLessEqual(abs(box.x - cx), 3, name)
            self.assertLessEqual(abs(box.y - cy), 3, name)
            self.assertLess(abs(box.confidence - cscore), 0.02, name)
            self.assertEqual((self.features[name].mat.shape[1], self.features[name].mat.shape[0]),
                             (box.width, box.height), name)  # 框尺寸与 CPU 路径一致（模板宽高）。
            self.assertEqual(name, box.name)  # 分类名原样带回，调用方可直接用于日志与绘制。
            self.assertGreater(box.confidence, 0.99)  # 模板就取自本帧，理应接近满分。

    def test_best_box_respects_threshold(self):
        # 最高分低于阈值时返回 None，与框架 limit=1 的判定一致。
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))
        handle = self.matcher.frame(self.frame)
        self.assertIsNotNone(self.matcher.best_box(handle, '测谎触发', 0.5))  # 阈值合理时命中。
        self.assertIsNone(self.matcher.best_box(handle, '测谎触发', 1.01))  # 阈值不可能达到时未命中。
        self.assertIsNone(self.matcher.best_box(handle, '带掩码', 0.0))  # 未注册的分类一律未命中。
        self.assertIsNone(self.matcher.best_box(None, '测谎触发', 0.0))  # 句柄缺失时安全返回 None。

    def test_best_score_reports_actual_maximum(self):
        # 诊断用的最高分不受阈值影响，未注册分类报 0 分。
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))
        handle = self.matcher.frame(self.frame)
        _, _, cscore = cpu_best(self.frame, self.features['测谎触发'].mat)
        self.assertLess(abs(self.matcher.best_score(handle, '测谎触发') - cscore), 0.02)
        self.assertEqual(0.0, self.matcher.best_score(handle, '带掩码'))  # 未注册分类。

    def test_identical_duplicates_pick_same_copy_as_cpu(self):
        # 同一模板在画面里出现多份像素完全相同的副本时，各处得分在数学上完全相等，
        # 而显卡的 float32 舍入会让靠后的副本成为严格最大值（实测不做平局判定时显卡会选第二份而 CPU 选第一份）。
        # 底层 best() 在 TIE_EPS 容差内取最靠前的位置，与 OpenCV minMaxLoc 的首个最大值规则对齐。
        frame = cv2.GaussianBlur((np.random.RandomState(7).rand(480, 640, 3) * 255).astype(np.uint8), (5, 5), 0)
        patch = frame[100:140, 120:180].copy()  # 取一块带纹理的区域做模板。
        for y, x in ((100, 400), (300, 120), (300, 400)):  # 原样复制到另外三处，形成四个像素相同的副本。
            frame[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
        cx, cy, cscore = cpu_best(frame, patch)
        self.assertEqual((120, 100), (cx, cy))  # CPU 取第一个最大值，即最靠上最靠左的那份。
        matcher = GpuFeatureMatcher(gray=True)  # 显式模板路径，不经过框架特征集。
        self.assertTrue(matcher.prepare_templates([('副本', [patch])], frame, fingerprint=1))
        x, y, w, h, score = matcher.best(matcher.frame(frame), '副本')
        self.assertEqual((cx, cy), (x, y))  # 显卡选中同一份副本，而不是靠后的那份。
        self.assertEqual((patch.shape[1], patch.shape[0]), (w, h))  # 框尺寸仍为模板原生宽高。
        self.assertLess(abs(score - cscore), 1e-4)  # 得分也在容差内一致。

    def test_frame_accepts_bgra_capture(self):
        # 框架 WGC 截图可能带 alpha 通道，封装要能直接吃下四通道画面。
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))
        bgra = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)
        box = self.matcher.best_box(self.matcher.frame(bgra), '测谎触发', 0.5)
        self.assertIsNotNone(box)  # 四通道画面同样能匹配。
        cx, cy, _ = cpu_best(self.frame, self.features['测谎触发'].mat)
        self.assertLessEqual(abs(box.x - cx), 3)
        self.assertLessEqual(abs(box.y - cy), 3)

    def test_prepare_reuses_cache_until_frame_size_changes(self):
        # 画面尺寸与标注文件都没变时复用同一批模板；尺寸变化后必须按新尺度重建。
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))
        first_key = self.matcher._key
        first_matcher = self.matcher.matcher
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))  # 同尺寸同名字。
        self.assertIs(first_matcher, self.matcher.matcher)  # 底层匹配器未被重建，核 FFT 缓存得以保留。
        self.assertEqual(first_key, self.matcher._key)
        small = self.frame[:240, :320]  # 换一帧更小的画面（模拟窗口分辨率变化）。
        self.assertTrue(self.matcher.prepare(['测谎触发'], small))
        self.assertIsNot(first_matcher, self.matcher.matcher)  # 尺寸变化触发重建，模板按新尺度重新上传。
        self.assertNotEqual(first_key, self.matcher._key)

    def test_prepare_rebuilds_when_name_set_changes(self):
        # 值守名单变化（如测谎触发标注名改了）时重建，多出来的分类必须可查。
        self.assertTrue(self.matcher.prepare(['测谎触发'], self.frame))
        self.assertFalse(self.matcher.has('掉线2'))
        self.assertTrue(self.matcher.prepare(['测谎触发', '掉线2'], self.frame))
        self.assertTrue(self.matcher.has('掉线2'))  # 新分类已注册。

    def test_one_frame_handle_serves_all_templates(self):
        # 一个句柄服务同一帧上的多个模板：这正是掉线检测与测谎检测复用同一帧的关键。
        names = ['测谎触发', '掉线2']
        self.assertTrue(self.matcher.prepare(names, self.frame))
        handle = self.matcher.frame(self.frame)
        boxes = {name: self.matcher.best_box(handle, name, 0.0) for name in names}
        for name, box in boxes.items():  # 每个模板都能从同一句柄拿到正确结果。
            self.assertIsNotNone(box, name)
            cx, cy, _ = cpu_best(self.frame, self.features[name].mat)
            self.assertLessEqual(abs(box.x - cx), 3, name)
            self.assertLessEqual(abs(box.y - cy), 3, name)


class TestGpuFeatureMatcherExplicitTemplatesCpuSide(unittest.TestCase):
    """显式模板路径（prepare_templates）不依赖显卡的部分：入参校验与无框架特征集构造。"""

    def test_construct_without_feature_set(self):
        # 全桌面重登匹配器不提供框架特征集，封装必须能在 feature_set 缺省时构造。
        matcher = GpuFeatureMatcher(gray=True)
        self.assertIsNone(matcher.feature_set)  # 不绑定特征集。
        self.assertEqual({}, matcher._groups)  # 尚未注册任何分类。
        self.assertFalse(matcher.has('连接'))  # 未注册时 has 返回 False。
        self.assertEqual(0.0, matcher.best_score(None, '连接'))  # 句柄缺失也不报错。

    def test_prepare_templates_requires_frame_and_names(self):
        # 没有画面或没有有效分类名时准备失败，调用方据此回退 CPU。
        frame, features = build_scene()
        matcher = GpuFeatureMatcher(gray=True)
        self.assertFalse(matcher.prepare_templates([], frame))  # 空列表。
        self.assertFalse(matcher.prepare_templates([('连接', [features['测谎触发'].mat])], None))  # 无画面。
        self.assertFalse(matcher.prepare_templates([('  ', [features['测谎触发'].mat])], frame))  # 名字全空白。
        self.assertFalse(matcher.has('连接'))  # 失败后不会留下半套模板。

    def test_prepare_templates_drops_empty_mats(self):
        # 模板列表里混入空数组/None 时应被过滤，全空时记入跳过名单。
        frame, _ = build_scene()
        matcher = GpuFeatureMatcher(gray=True)
        self.assertFalse(matcher.prepare_templates([('连接', [None, np.zeros((0, 0, 3), dtype=np.uint8)])], frame))
        self.assertIn('连接', matcher.skipped)  # 无可显卡化模板，调用方对它回退 CPU。
        self.assertFalse(matcher.has('连接'))


@unittest.skipUnless(GPU_READY, "no available NVIDIA GPU or cupy not installed")
class TestGpuFeatureMatcherExplicitTemplatesOnGpu(unittest.TestCase):
    """需要显卡的部分：显式模板路径与 CPU 逐张 matchTemplate 取最高分的结果对比。"""

    def setUp(self):
        self.frame, self.features = build_scene()
        self.matcher = GpuFeatureMatcher(gray=True)  # 显式模板路径不需要框架特征集。

    def test_prepare_templates_matches_cpu_result(self):
        # 单模板分类：显卡结果与 CPU 全屏灰度 TM_CCOEFF_NORMED 一致。
        mats = {'测谎触发': [self.features['测谎触发'].mat], '掉线2': [self.features['掉线2'].mat]}
        self.assertTrue(self.matcher.prepare_templates(list(mats.items()), self.frame, fingerprint=1))
        handle = self.matcher.frame(self.frame)
        for name, templates in mats.items():
            self.assertTrue(self.matcher.has(name), name)
            result = self.matcher.best(handle, name)  # 五元组 (x, y, w, h, score)。
            self.assertIsNotNone(result, name)
            cx, cy, cscore = cpu_best(self.frame, templates[0])
            self.assertLessEqual(abs(result[0] - cx), 3, name)  # 位置容差 3 像素。
            self.assertLessEqual(abs(result[1] - cy), 3, name)
            self.assertLess(abs(result[4] - cscore), 0.02, name)  # 分数容差 0.02。
            self.assertEqual((templates[0].shape[1], templates[0].shape[0]), (result[2], result[3]), name)  # 框尺寸=模板尺寸。
            box = self.matcher.best_box(handle, name, 0.0)  # Box 与五元组必须是同一个结果。
            self.assertEqual((box.x, box.y, box.width, box.height), result[:4], name)
            self.assertLess(abs(box.confidence - result[4]), 1e-9, name)

    def test_prepare_templates_group_picks_best_of_many(self):
        # 一个分类多张模板：显卡必须逐张求最高分再取最大，与 CPU 一致。
        good = self.features['测谎触发'].mat  # 取自本帧，应接近满分。
        noise = (np.random.RandomState(99).rand(30, 60, 3) * 255).astype(np.uint8)  # 与画面无关的噪声模板，得分低。
        self.assertTrue(self.matcher.prepare_templates([('连接', [noise, good])], self.frame, fingerprint=1))
        handle = self.matcher.frame(self.frame)
        cx, cy, cscore = cpu_best(self.frame, good)
        result = self.matcher.best(handle, '连接')
        self.assertIsNotNone(result)
        self.assertLessEqual(abs(result[0] - cx), 3)  # 胜出的是高分模板的位置。
        self.assertLessEqual(abs(result[1] - cy), 3)
        self.assertLess(abs(result[4] - cscore), 0.02)  # 分数也来自高分模板。
        self.assertEqual(2, len(self.matcher._groups['连接']))  # 两张模板都已注册。

    def test_prepare_templates_incremental_keeps_underlying_matcher(self):
        # 分类名交替变化时增量注册，不重建底层匹配器（核 FFT 缓存得以保留）。
        first = self.features['测谎触发'].mat
        second = self.features['掉线2'].mat
        self.assertTrue(self.matcher.prepare_templates([('测谎触发', [first])], self.frame, fingerprint=1))
        underlying = self.matcher.matcher
        self.assertTrue(self.matcher.prepare_templates([('掉线2', [second])], self.frame, fingerprint=1))
        self.assertIs(underlying, self.matcher.matcher)  # 未重建，仅追加注册。
        self.assertTrue(self.matcher.has('测谎触发'))  # 旧分类仍在。
        self.assertTrue(self.matcher.has('掉线2'))  # 新分类已注册。
        handle = self.matcher.frame(self.frame)
        cx, cy, _ = cpu_best(self.frame, second)
        result = self.matcher.best(handle, '掉线2')
        self.assertLessEqual(abs(result[0] - cx), 3)  # 两个分类都能从同一句柄拿到正确结果。
        self.assertLessEqual(abs(result[1] - cy), 3)

    def test_prepare_templates_rebuilds_on_fingerprint_change(self):
        # 标注指纹变化（重新标注）时整体重建，旧模板不得残留。
        self.assertTrue(self.matcher.prepare_templates([('测谎触发', [self.features['测谎触发'].mat])], self.frame, fingerprint=1))
        underlying = self.matcher.matcher
        self.assertTrue(self.matcher.prepare_templates([('掉线2', [self.features['掉线2'].mat])], self.frame, fingerprint=2))
        self.assertIsNot(underlying, self.matcher.matcher)  # 指纹变了 -> 重建。
        self.assertFalse(self.matcher.has('测谎触发'))  # 旧分类已丢弃。
        self.assertTrue(self.matcher.has('掉线2'))

    def test_prepare_templates_reregisters_when_shape_changes(self):
        # 同一分类模板尺寸变了（改了标注框）时必须重新上传，结果跟着新模板走。
        self.assertTrue(self.matcher.prepare_templates([('测谎触发', [self.frame[100:130, 200:260]])], self.frame, fingerprint=1))
        resized = cv2.resize(self.frame[100:130, 200:260], (40, 20))  # 换一个尺寸的模板。
        self.assertTrue(self.matcher.prepare_templates([('测谎触发', [resized])], self.frame, fingerprint=1))
        result = self.matcher.best(self.matcher.frame(self.frame), '测谎触发')
        self.assertEqual((resized.shape[1], resized.shape[0]), (result[2], result[3]))  # 框尺寸跟新模板一致。
        self.assertEqual(1, len(self.matcher._groups['测谎触发']))  # 旧模板已摘掉，没有残留 key。
        self.assertEqual(1, len(self.matcher.matcher.templates))

    def test_oversized_template_is_skipped_like_cpu(self):
        # 模板比画面还大：CPU 会跳过该模板，显卡也必须跳过而不是报错或给出假结果。
        big = self.frame  # 与画面同尺寸，对小帧而言就是超大模板。
        small = self.frame[:100, :120]  # 小帧。
        self.assertTrue(self.matcher.prepare_templates([('大模板', [big])], small, fingerprint=1))
        self.assertTrue(self.matcher.has('大模板'))  # 注册本身不报错。
        handle = self.matcher.frame(small)
        self.assertIsNone(self.matcher.best(handle, '大模板'))  # 无结果，与 CPU 一致。
        self.assertEqual(0.0, self.matcher.best_score(handle, '大模板'))
        self.assertIsNone(self.matcher.best_box(handle, '大模板', 0.0))

    def test_bgra_frame_accepted_on_explicit_path(self):
        # 框架窗口采集可能是 BGRA，显式模板路径同样要能吃下。
        self.assertTrue(self.matcher.prepare_templates([('测谎触发', [self.features['测谎触发'].mat])], self.frame, fingerprint=1))
        bgra = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)
        result = self.matcher.best(self.matcher.frame(bgra), '测谎触发')
        self.assertIsNotNone(result)
        cx, cy, _ = cpu_best(self.frame, self.features['测谎触发'].mat)
        self.assertLessEqual(abs(result[0] - cx), 3)
        self.assertLessEqual(abs(result[1] - cy), 3)


if __name__ == '__main__':
    unittest.main()
