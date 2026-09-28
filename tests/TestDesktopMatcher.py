# 全桌面原生尺度模板匹配器（DesktopTemplateMatcher）回归测试。
#
# 重点是「显卡与 CPU 结果一致」：同一份标注、同一帧画面，分别用 gpu_enabled=True/False 两个匹配器跑，
# 逐分类对比框位置（容差 3 像素）、框尺寸（必须完全相等，都等于模板原生宽高）与最高分（容差 0.02）。
# 另外覆盖降级链路：看板开关关闭、本机无显卡、显卡运行期异常（永久关闭且只记一次日志）、
# 模板比画面大、四通道 BGRA 帧、重新标注后模板重建，以及 AutoLoginFlow 从看板配置读开关。
# 无 CuPy 或无 NVIDIA 显卡的环境下自动跳过显卡用例，不阻断 CI。
import json  # 写测试用的 COCO 标注文件。
import os  # 临时目录与标注文件路径拼接。
import shutil  # 清理临时目录。
import tempfile  # 建临时目录存放源图与标注。
import time  # 重新标注用例里把 mtime 推到未来，确保指纹变化可被检测到。
import unittest
from unittest.mock import patch  # 打桩显卡可用性与显卡匹配异常。

import cv2  # 造合成画面、四通道转换，同时作为 CPU 参考实现。
import numpy as np  # 合成画面与模板数组。

from src.autologin.desktop_matcher import DesktopTemplateMatcher  # 被测匹配器。
from src.gpu_feature_match import gpu_match_available  # 显卡可用性探测，决定是否跳过显卡用例。

GPU_READY = gpu_match_available()  # 本机是否有可用显卡：无则全部显卡用例跳过。

SCENE_H, SCENE_W = 480, 640  # 合成画面尺寸，模拟一帧全桌面截图。

# 标注定义：(分类名, 源图, bbox[x, y, w, h])。
# scene 就是查询帧本身（其上的标注应接近满分命中），other 是一张无关图（其上的标注在查询帧上得分很低）。
ANNOTATIONS = [
    ('连接', 'scene', (100, 80, 60, 30)),        # 启动器【连接】按钮，单模板。
    ('服务区', 'scene', (300, 200, 50, 40)),      # 单模板，验证换分类时增量注册。
    ('频道', 'scene', (450, 350, 70, 25)),        # 单模板，扁宽比例接近真实按钮。
    ('双模板', 'other', (20, 20, 40, 40)),        # 同一分类标注在无关图上（低分）。
    ('双模板', 'scene', (100, 80, 60, 30)),       # 同一分类标注在查询帧上（高分），验证取最高分那张。
    ('不在画面', 'other', (200, 100, 50, 30)),     # 只标注在无关图上：查询帧上必然低分但仍有最佳位置。
]
ALL_NAMES = ('连接', '服务区', '频道', '双模板', '不在画面')  # 参与逐分类对比的全部分类名。


def build_scene(seed):  # 造一帧带结构的合成桌面：随机噪声打底保证模板有纹理，再叠几个几何图形接近真实截图。
    frame = (np.random.RandomState(seed).rand(SCENE_H, SCENE_W, 3) * 255).astype(np.uint8)
    cv2.rectangle(frame, (60, 400), (200, 470), (30, 90, 160), -1)  # 左下角色块：必须避开全部标注裁剪区，否则纯色填充会让模板零方差。
    cv2.circle(frame, (400, 300), 60, (200, 40, 40), -1)  # 右侧圆形，与标注区只部分重叠，不会抽干纹理。
    cv2.line(frame, (0, 430), (SCENE_W - 1, 200), (250, 250, 250), 3)  # 斜线，增加方向性纹理。
    return frame


def write_coco(path, annotations):  # 按 COCO 结构写标注文件，源图与标注同目录（与项目 ok_templates 布局一致）。
    categories = []  # 分类条目。
    category_ids = {}  # 分类名 -> 分类 id。
    anns = []  # 标注明细。
    for index, (name, _image, bbox) in enumerate(annotations, 1):  # 逐条转成 COCO 记录。
        if name not in category_ids:  # 首次出现的分类名。
            category_ids[name] = len(category_ids) + 1  # 顺序分配 id。
            categories.append({'id': category_ids[name], 'name': name})  # 收集分类条目。
        anns.append({  # 一条标注。
            'id': index,
            'image_id': 1 if _image == 'scene' else 2,  # scene=1，other=2。
            'category_id': category_ids[name],
            'bbox': list(bbox),
        })
    data = {  # 完整 COCO 文档。
        'images': [
            {'id': 1, 'file_name': 'scene.png', 'width': SCENE_W, 'height': SCENE_H},
            {'id': 2, 'file_name': 'other.png', 'width': SCENE_W, 'height': SCENE_H},
        ],
        'categories': categories,
        'annotations': anns,
    }
    with open(path, 'w', encoding='utf-8') as f:  # 写盘。
        json.dump(data, f, ensure_ascii=False)


class RecordingLogger:  # 假日志器：收集 info/warning 文本，供断言「显卡已启用」「已回退 CPU」等日志只记一次。

    def __init__(self):
        self.infos = []  # info 级别消息。
        self.warnings = []  # warning 级别消息。

    def info(self, message):  # 记录 info。
        self.infos.append(str(message))

    def warning(self, message):  # 记录 warning。
        self.warnings.append(str(message))

    def count(self, keyword, level='warnings'):  # 统计含关键字的消息条数。
        return len([m for m in getattr(self, level) if keyword in m])


class DesktopMatcherCase(unittest.TestCase):
    """公共装置：临时目录里写好 scene.png / other.png / coco_annotations.json，并给出便捷构造方法。"""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix='ok_desktop_matcher_')  # 临时目录，用例结束后整体删除。
        self.addCleanup(shutil.rmtree, self.folder, True)  # 注册清理，失败也不残留。
        self.scene = build_scene(11)  # 查询帧（同时作为标注源图 scene.png 的内容）。
        self.other = build_scene(77)  # 另一张无关源图，用于「同一分类多张模板」与「不在画面」两个场景。
        cv2.imwrite(os.path.join(self.folder, 'scene.png'), self.scene)  # PNG 无损，读回来与内存数组完全一致。
        cv2.imwrite(os.path.join(self.folder, 'other.png'), self.other)
        self.coco = os.path.join(self.folder, 'coco_annotations.json')  # 标注文件路径。
        write_coco(self.coco, ANNOTATIONS)  # 写标注。
        self.logger = RecordingLogger()  # 收集日志。

    def make(self, gpu_enabled=True):  # 构造一个匹配器，gpu_enabled=False 即纯 CPU 参考实现。
        return DesktopTemplateMatcher(self.coco, self.logger, gpu_enabled=gpu_enabled)


class TestDesktopMatcherCpu(DesktopMatcherCase):
    """不依赖显卡的部分：开关语义、CPU 原生匹配行为与辅助方法。"""

    def test_default_switch_is_on(self):
        # 显卡加速隐藏式启用：不传参默认开启，仅测试可显式关闭构造纯 CPU 参考实现。
        self.assertTrue(self.make()._gpu_enabled)  # 默认开启。
        self.assertFalse(self.make(gpu_enabled=False)._gpu_enabled)  # 显式关闭（仅测试）。

    def test_switch_off_never_creates_gpu_matcher(self):
        # 开关关闭时完全不碰显卡：结果正常，且显卡匹配器始终为 None。
        matcher = self.make(gpu_enabled=False)
        box, conf = matcher.find_best(self.scene, '连接')
        self.assertIsNotNone(box)  # CPU 原生匹配命中。
        self.assertGreater(conf, 0.99)  # 模板就取自本帧，理应接近满分。
        self.assertEqual((100, 80, 60, 30), box[:4])  # 位置与尺寸都是标注框原值。
        self.assertIsNone(matcher._gpu)  # 从未创建显卡匹配器。
        self.assertFalse(matcher._gpu_off)  # 也不是异常降级。

    def test_multi_template_prefers_high_score_on_cpu(self):
        # 同一分类标注在多处时取最高分那张，CPU 路径的既有语义。
        box, conf = self.make(gpu_enabled=False).find_best(self.scene, '双模板')
        self.assertGreater(conf, 0.99)  # 命中 scene 上那张（other 上那张在查询帧里得分很低）。
        self.assertEqual((100, 80, 60, 30), box[:4])  # 位置与尺寸来自高分模板。

    def test_oversized_template_skipped_on_cpu(self):
        # 模板比查询帧还大：matchTemplate 会报错，CPU 路径必须跳过并返回无结果。
        small = self.scene[:20, :20]  # 20x20 的帧，装不下 60x30 的模板。
        self.assertEqual((None, 0.0), self.make(gpu_enabled=False).find_best(small, '连接'))

    def test_bgra_frame_supported_on_cpu(self):
        # 框架 WGC 截图可能是 BGRA，CPU 路径也要能吃下且结果与 BGR 一致。
        matcher = self.make(gpu_enabled=False)
        bgra = cv2.cvtColor(self.scene, cv2.COLOR_BGR2BGRA)
        self.assertEqual(matcher.find_best(self.scene, '连接'), matcher.find_best(bgra, '连接'))

    def test_helpers_and_threshold(self):
        # has / template_size / find 的既有契约：供流程判断是否跳过未标注步骤、日志确认未被缩放。
        matcher = self.make(gpu_enabled=False)
        self.assertTrue(matcher.has('连接'))  # 已标注。
        self.assertFalse(matcher.has('没标注过'))  # 未标注。
        self.assertEqual((60, 30), matcher.template_size('连接'))  # 原生尺寸 (w, h)。
        self.assertIsNone(matcher.template_size('没标注过'))  # 无模板返回 None。
        self.assertIsNotNone(matcher.find(self.scene, '连接', 0.5))  # 阈值合理时命中。
        self.assertIsNone(matcher.find(self.scene, '连接', 1.01))  # 阈值不可能达到时未命中。
        self.assertEqual((None, 0.0), matcher.find_best(None, '连接'))  # 无画面。
        self.assertEqual((None, 0.0), matcher.find_best(self.scene, '没标注过'))  # 无模板。

    def test_fingerprint_follows_annotation_file(self):
        # 指纹取标注文件 mtime，供显卡匹配器判断是否需要整体重建。
        matcher = self.make(gpu_enabled=False)
        matcher.has('连接')  # 触发一次解析。
        self.assertIsNotNone(matcher.fingerprint())  # 标注可读时有指纹。

    def test_all_annotated_templates_have_variance(self):
        # 前提自检：全部标注模板都必须带纹理。纯色模板会让 CPU 的得分图退化成整张常数（含无意义的左上角命中），
        # 显卡上按 0 分处理（见 test_constant_template_divergence_is_documented），两者对比就没意义了。
        matcher = self.make(gpu_enabled=False)
        for name in ALL_NAMES:  # 逐分类逐张模板检查。
            templates = matcher._get_templates(name)
            self.assertTrue(templates, name)  # 标注确实解析出了模板。
            for tpl in templates:
                gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                self.assertGreater(float(gray.std()), 1.0, name)  # 标准差足够大，说明不是纯色块。


@unittest.skipUnless(GPU_READY, "no available NVIDIA GPU or cupy not installed")
class TestDesktopMatcherOnGpu(DesktopMatcherCase):
    """需要显卡的部分：显卡结果必须与 CPU 一致，且各级降级都能正确回退 CPU。"""

    def setUp(self):
        super().setUp()
        self.cpu = self.make(gpu_enabled=False)  # CPU 参考实现。
        self.gpu = self.make(gpu_enabled=True)  # 被测显卡实现。

    def assert_same_result(self, frame, name, pos_tolerance=3, score_tolerance=0.02):  # 断言两条路径结果一致。
        cbox, cconf = self.cpu.find_best(frame, name)  # CPU 参考结果。
        gbox, gconf = self.gpu.find_best(frame, name)  # 显卡结果。
        if cbox is None:  # CPU 无结果时显卡也必须无结果。
            self.assertIsNone(gbox, name)
            self.assertEqual(cconf, gconf, name)
            return cbox, gbox
        self.assertIsNotNone(gbox, name)  # CPU 有结果时显卡也必须有。
        self.assertEqual(cbox[2:4], gbox[2:4], name)  # 模板宽高完全相等（都是原生尺寸，未被缩放）；置信度单独用容差比。
        self.assertLessEqual(abs(cbox[0] - gbox[0]), pos_tolerance, name)  # 位置容差。
        self.assertLessEqual(abs(cbox[1] - gbox[1]), pos_tolerance, name)
        self.assertLess(abs(cconf - gconf), score_tolerance, name)  # 最高分容差。
        self.assertEqual(gconf, gbox[4], name)  # 返回的分数与框内分数一致。
        return cbox, gbox

    def test_gpu_result_matches_cpu_for_every_category(self):
        # 逐分类对比：单模板、多模板、以及只标注在无关图上的低分分类都要一致。
        for name in ALL_NAMES:
            self.assert_same_result(self.scene, name)
        self.assertIsNotNone(self.gpu._gpu)  # 确实走了显卡。
        self.assertFalse(self.gpu._gpu_off)  # 且没有异常降级。
        self.assertEqual(1, self.logger.count('GPU template match enabled', 'infos'))  # 启用日志只记一次。

    def test_gpu_matches_cpu_on_smaller_frame(self):
        # 窗口后端帧比桌面帧小：原生模板不缩放，两条路径同样要一致。
        window = self.scene[:400, :500]  # 模拟游戏窗口客户区帧。
        for name in ('连接', '服务区', '频道'):
            self.assert_same_result(window, name)

    def test_gpu_matches_cpu_on_bgra_frame(self):
        # 框架窗口采集可能是 BGRA，两条路径都要能处理且结果一致。
        bgra = cv2.cvtColor(self.scene, cv2.COLOR_BGR2BGRA)
        self.assert_same_result(bgra, '连接')

    def test_gpu_oversized_template_matches_cpu(self):
        # 模板比帧大：显卡也必须跳过该模板，返回与 CPU 相同的「无结果」。
        small = self.scene[:20, :20]
        self.assertEqual((None, 0.0), self.gpu.find_best(small, '连接'))
        self.assert_same_result(small, '连接')

    def test_gpu_reuses_cache_across_calls_and_names(self):
        # 同分类重复匹配、以及换分类匹配，都不重建底层匹配器（核 FFT 缓存得以保留）。
        self.gpu.find_best(self.scene, '连接')
        underlying = self.gpu._gpu.matcher  # 底层 CuPy 匹配器。
        self.gpu.find_best(self.scene, '连接')  # 同分类再来一次。
        self.assertIs(underlying, self.gpu._gpu.matcher)  # 未重建。
        self.gpu.find_best(self.scene, '服务区')  # 换分类（重登流程下一步）。
        self.assertIs(underlying, self.gpu._gpu.matcher)  # 仅增量注册，仍未重建。
        self.assertTrue(self.gpu._gpu.has('连接'))  # 旧分类还在。
        self.assertTrue(self.gpu._gpu.has('服务区'))  # 新分类已注册。

    def test_gpu_keeps_cache_when_frame_size_changes(self):
        # 桌面后端与窗口后端交替时帧尺寸会变：原生模板不随帧缩放，因此不该触发重建。
        self.gpu.find_best(self.scene, '连接')
        underlying = self.gpu._gpu.matcher
        self.assert_same_result(self.scene[:400, :500], '连接')  # 换一帧更小的画面。
        self.assertIs(underlying, self.gpu._gpu.matcher)  # 未重建，fk_cache 保留。

    def test_gpu_falls_back_to_cpu_on_runtime_error(self):
        # 显卡运行期异常：本帧立刻用 CPU 得出正确结果，并永久关闭加速、只记一次日志。
        self.gpu.find_best(self.scene, '连接')  # 先正常跑一次，确保显卡匹配器已创建。
        with patch.object(self.gpu._gpu, 'best', side_effect=RuntimeError('显存不足')):
            box, conf = self.gpu.find_best(self.scene, '连接')  # 本帧显卡抛异常。
        cbox, cconf = self.cpu.find_best(self.scene, '连接')  # CPU 参考。
        self.assertIsNotNone(box)  # 异常当帧仍拿到了正确结果。
        self.assertEqual(cbox[2:4], box[2:4])  # 尺寸一致。
        self.assertLess(abs(cconf - conf), 0.02)  # 分数一致。
        self.assertTrue(self.gpu._gpu_off)  # 已置永久关闭标志。
        self.assertIsNone(self.gpu._gpu)  # 显卡匹配器已释放。
        self.assertEqual(1, self.logger.count('fallback to CPU'))  # 记了一次降级告警。
        self.assertIsNotNone(self.gpu.find_best(self.scene, '服务区')[0])  # 之后走 CPU 仍正常工作。
        self.assertIsNone(self.gpu._gpu)  # 且不再重建显卡匹配器。
        self.assertEqual(1, self.logger.count('fallback to CPU'))  # 不重复刷日志。

    def test_gpu_unavailable_falls_back_silently(self):
        # 本机无 CuPy/无显卡：不创建显卡匹配器、不算异常降级，结果与 CPU 一致。
        with patch('src.gpu_feature_match.gpu_match_available', return_value=False):
            self.assert_same_result(self.scene, '连接')
        self.assertIsNone(self.gpu._gpu)  # 从未创建。
        self.assertFalse(self.gpu._gpu_off)  # 不是异常降级。
        self.assertEqual(0, self.logger.count('fallback to CPU'))  # 不该刷降级告警。

    def test_reannotation_rebuilds_gpu_templates(self):
        # 重新标注后：指纹变化触发显卡模板整体重建，结果跟着新标注走且仍与 CPU 一致。
        self.gpu.find_best(self.scene, '连接')  # 先按旧标注注册一次。
        first = self.gpu.fingerprint()  # 旧指纹。
        moved = [('连接', 'scene', (300, 200, 50, 40))]  # 把【连接】的标注框挪到别处并改尺寸。
        moved += [a for a in ANNOTATIONS if a[0] != '连接']  # 其余分类保持不变。
        write_coco(self.coco, moved)  # 重写标注文件。
        future = time.time() + 10  # 把 mtime 推到未来，确保与旧值不同。
        os.utime(self.coco, (future, future))
        self.gpu._last_check = 0.0  # 绕开 mtime 检查限频，让本次调用立刻重读标注。
        self.cpu._last_check = 0.0
        self.assertNotEqual(first, self.gpu.fingerprint())  # 指纹已变。
        cbox, gbox = self.assert_same_result(self.scene, '连接')  # 两条路径都跟上了新标注。
        self.assertEqual((50, 40), gbox[2:4])  # 框尺寸是新标注的原生宽高。
        self.assertNotEqual(cbox[2:4], (60, 30))  # 确认不再是旧标注的尺寸。

    def test_constant_template_divergence_is_documented(self):
        # 已知且刻意的差异：标注成纯色块时模板零方差，CPU 的 TM_CCOEFF_NORMED 得分图会退化成整张常数——
        # OpenCV 5.0 实测这个常数值还依尺寸/内部代码路径而变：真实几何（480x640 帧 + 30x60 模板）走 DFT 路径
        # 给处处 1.0、minMaxLoc 落在左上角 (0,0) 误报命中；小尺寸（120x160 帧 + 20x30 模板）走直接路径给处处 0.0。
        # 无论哪种常数都不含位置信息。显卡路径把分母退化的位置一律记 0 分，既不会误报也不会去点 (0,0)，
        # 对重登流程更安全；这类标注本身就是无效的，故此处只把「显卡恒 0 分、CPU 得分图退化」这两个稳定结论钉住。
        from src.gpu_feature_match import GpuFeatureMatcher  # 直接用底层封装验证，不经过标注文件。
        solid_frame = np.full((SCENE_H, SCENE_W, 3), 90, dtype=np.uint8)  # 纯色帧，尺寸与真实场景一致。
        solid_tpl = np.full((30, 60, 3), 90, dtype=np.uint8)  # 纯色模板，与真实按钮标注同量级。
        result = cv2.matchTemplate(cv2.cvtColor(solid_frame, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(solid_tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        self.assertEqual(1, np.unique(result).size)  # 整张得分图只有一个值：CPU 已退化成无意义的常数。
        matcher = GpuFeatureMatcher(gray=True)
        self.assertTrue(matcher.prepare_templates([('纯色', [solid_tpl])], solid_frame, fingerprint=1))
        self.assertEqual(0.0, matcher.best_score(matcher.frame(solid_frame), '纯色'))  # 显卡按 0 分处理，不会误报。
        self.assertIsNone(matcher.best_box(matcher.frame(solid_frame), '纯色', 0.75))  # 因此也不会产生点击。


class TestAutoLoginFlowGpuHidden(DesktopMatcherCase):
    """显卡加速隐藏式启用：AutoLoginFlow 不再读看板开关，构造的匹配器默认启用显卡。"""

    def test_flow_always_enables_gpu(self):
        from src.autologin.flow import AutoLoginFlow  # 延迟导入：flow 依赖 pynput。
        self.assertTrue(AutoLoginFlow(self.coco, {}, self.logger)._matcher._gpu_enabled)  # 无配置也默认开启。
        self.assertTrue(AutoLoginFlow(self.coco, None, self.logger)._matcher._gpu_enabled)  # 配置缺失也不崩，按默认开启。


if __name__ == '__main__':
    unittest.main()
