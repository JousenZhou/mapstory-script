# TestLieShapeSolver：用合成帧验证测谎检验「找目标」算法内核与在线编排。
# 不依赖任何录像文件，全部用合成图像驱动，对齐外部 tests/test_shape_tracker.py 的断言思路。
import math
import unittest

import cv2
import numpy as np

from src.liedetector.shape_session import (  # 在线编排与回看插值。
    ShapeFrameResult, ShapeTrackParams, ShapeTrackSession,
    SOURCE_BORDER, SOURCE_COLOR, SOURCE_INTERPOLATED, SOURCE_PREDICTION,
    SOURCE_SCENE_ENDED, SOURCE_WAITING,
    repair_shape_gaps)
from src.liedetector.shape_tracking import (  # 算法内核。
    detect_white_shapes, build_shape_template, choose_template_detection,
    choose_shape_candidate, ParticleShapeTracker, ShapeDetection)


def _make_white_circle_frame(size: int, cx: int, cy: int, radius: int,
                             bg: int = 60, fg: int = 255) -> np.ndarray:
    """合成一帧带白色圆的暗底画面，用于模拟学习/跟踪场景。"""

    frame = np.full((size, size, 3), bg, dtype=np.uint8)  # 暗底。
    cv2.circle(frame, (cx, cy), radius, (fg, fg, fg), -1)  # 白色实心圆。
    return frame


def _make_white_square_frame(size: int, x1: int, y1: int, side: int,
                             bg: int = 60, fg: int = 255) -> np.ndarray:
    """合成一帧带白色正方形的暗底画面。"""

    frame = np.full((size, size, 3), bg, dtype=np.uint8)  # 暗底。
    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (fg, fg, fg), -1)  # 白色实心正方形。
    return frame


def _halo_bgr() -> tuple:
    """倒计时蓝灰阴影的合成色（HSV H=100,S=60,V=160 转 BGR），保证落在光环阈值区间内。"""

    bgr = cv2.cvtColor(np.uint8([[[100, 60, 160]]]), cv2.COLOR_HSV2BGR)[0, 0]  # 单像素转换。
    return int(bgr[0]), int(bgr[1]), int(bgr[2])  # 解包成 BGR 三元组。


def _make_countdown_frame(size: int = 320, with_square: bool = True,
                          with_halo: bool = True, touch: bool = False,
                          overlap: bool = False) -> np.ndarray:
    """合成模拟测谎弹窗画面：tan 底 + 白色方块（可选）+ 白色倒计时数字（可选冷色光环）。

    数字先画成 glyph 掩码，膨胀一圈涂冷色光环再盖白色本体，光环宽度确定可控，与生产录像一致；
    touch=True 时把数字下移到光环压住方块上边，模拟录像里数字与图形粘连的帧；
    overlap=True 时把数字本体下压进方块顶部，粗笔画数字与图形融成单一厚桥轮廓（失败局复现）。
    """

    frame = np.full((size, size, 3), 95, dtype=np.uint8)  # tan 底 BGR 初值。
    frame[:, :] = (95, 151, 172)  # 录像实测的 tan 底色。
    square_center = (size // 2, int(size * 0.625))  # 方块中心。
    side = 60  # 方块边长。
    if with_square:  # 画白色目标方块。
        cv2.rectangle(frame, (square_center[0] - side // 2, square_center[1] - side // 2),
                      (square_center[0] + side // 2, square_center[1] + side // 2),
                      (255, 255, 255), -1)
    if overlap:  # 数字本体底边压进方块上部，粗笔画与方块融成单一厚桥连通域。
        baseline_y = square_center[1] - 5
    elif touch:
        baseline_y = square_center[1] - side // 2 - 4
    else:
        baseline_y = int(size * 0.31)  # 数字基线 y。
    origin = (size // 2 - 20, baseline_y)  # putText 左下角原点。
    glyph = np.zeros((size, size), np.uint8)  # 数字 glyph 掩码。
    cv2.putText(glyph, "5", origin, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 10)
    if with_halo:  # glyph 外胀 6px 涂冷色光环，本体白色盖回中间后露出一圈阴影。
        halo_mask = cv2.dilate(glyph, np.ones((13, 13), np.uint8))
        frame[halo_mask > 0] = _halo_bgr()
    frame[glyph > 0] = (255, 255, 255)  # 白色数字本体。
    return frame


class TestLieShapeSolver(unittest.TestCase):
    """六项合成帧测试，覆盖学习、矩形模型、检测过滤、区域重置、回看插值与场景结束。"""

    def test_circle_learning_and_tracking(self):
        """圆形学习：合成「白色圆 → 逐帧降低对比度渐隐 → 匀速平移」序列，
        断言学到模板、symmetry_period == 30.0、use_rectangle_model is False、
        source 出现过 color 与 border、渐隐后跟踪误差 < nominal_size * 1.5。"""

        params = ShapeTrackParams(  # 中等粒子数，加速测试但保留足够精度；选窗缩到 0.15s（≥5 条即可）适配本用例短白相。
            process_scale=1.0, particle_count=150, global_proposals=200,
            temporal_lags=(1, 2), template_learn_window=0.15)
        session = ShapeTrackSession(params=params)
        session.reset(200, 200)  # 初始化 200x200 区域。

        sources_seen = set()  # 收集出现过的 source 类型。
        last_error = 0.0  # 渐隐后最近一帧的跟踪误差。
        nominal = 0.0  # 学到的标称尺寸。
        cx_start, cy_start, radius = 100, 100, 28  # 起始圆心与半径。
        total_frames = 50  # 总帧数（给渐隐和跟踪更多时间）。
        fade_start = 3  # 从第 3 帧开始渐隐。
        move_start = 10  # 从第 10 帧开始匀速平移（渐隐后）。

        for i in range(total_frames):
            # 白色圆逐帧降低对比度（前 3 帧纯白，之后线性降低到消失）。
            fg = max(60, 255 - max(0, i - fade_start) * 18)  # 前景亮度逐渐趋近背景（更慢渐隐）。
            # 第 10 帧起匀速向右平移。
            cx = cx_start + max(0, i - move_start) * 1  # 每帧右移 1 像素（更慢平移）。
            cy = cy_start  # y 不变。
            frame = _make_white_circle_frame(200, cx, cy, radius, bg=60, fg=fg)
            result = session.update(frame, 30)
            sources_seen.add(result.source)
            if i >= move_start and result.center is not None:  # 渐隐后记录跟踪误差。
                error = float(np.linalg.norm(result.center - np.array([cx, cy], dtype=np.float32)))
                last_error = error
                if session.template is not None:
                    nominal = session.template.nominal_size

        self.assertIsNotNone(session.template, "应学到模板")  # 模板非空。
        self.assertEqual(session.template.symmetry_period, 30.0, "圆形对称周期应为 30 度")  # 圆的旋转对称。
        self.assertFalse(session.template.use_rectangle_model, "圆形不应启用矩形模型")  # 非矩形。
        self.assertIn(SOURCE_COLOR, sources_seen, "应出现过 color 源")  # 白色阶段。
        self.assertIn(SOURCE_BORDER, sources_seen, "应出现过 border 源")  # 渐隐后。
        if nominal > 0:  # 跟踪误差应在合理范围内（放宽到 1.5 倍 nominal）。
            self.assertLess(last_error, nominal * 1.5,
                            f"渐隐后跟踪误差 {last_error:.1f} 应小于 nominal*1.5={nominal * 1.5:.1f}")

    def test_rectangle_model_switch(self):
        """矩形模型切换：合成白色正方形逐帧喂满最佳帧选窗（fps=30×0.4s=12），断言 use_rectangle_model is True 且 symmetry_period == 90.0。"""

        params = ShapeTrackParams(process_scale=1.0, particle_count=40, global_proposals=60)
        session = ShapeTrackSession(params=params)
        session.reset(200, 200)
        frame = _make_white_square_frame(200, 70, 70, 60, bg=60, fg=255)  # 60x60 白色正方形。
        result = None  # 取选窗攒满那一帧的结果（第 12 帧学习）。
        for _ in range(12):
            result = session.update(frame, 30)

        self.assertIsNotNone(session.template, "应学到模板")
        self.assertEqual(session.template.symmetry_period, 90.0, "正方形对称周期应为 90 度")
        self.assertTrue(session.template.use_rectangle_model, "正方形应启用矩形模型")
        self.assertGreater(session.template.rectangularity, 0.95, "正方形矩形度应接近 1")
        self.assertEqual(result.source, SOURCE_COLOR, "选窗攒满帧应为 color 源")

    def test_white_detection_filtering(self):
        """白色检测过滤：断言过小块（最短边<14）、过长条（长宽比>4.2）、过暗块被丢弃。"""

        # 过小块：10x10 白色正方形（最短边 < 14）。
        frame_small = _make_white_square_frame(200, 90, 90, 10, bg=60, fg=255)
        dets_small = detect_white_shapes(frame_small, 1.0)
        self.assertEqual(len(dets_small), 0, "10x10 白色块应被过滤（最短边 < 14）")

        # 过长条：4x40 白色长条（长宽比 > 4.2）。
        frame_long = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame_long, (90, 80), (94, 120), (255, 255, 255), -1)
        dets_long = detect_white_shapes(frame_long, 1.0)
        self.assertEqual(len(dets_long), 0, "4x40 长条应被过滤（长宽比 > 4.2）")

        # 过暗块：亮度 150 < 188 阈值。
        frame_dark = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.circle(frame_dark, (100, 100), 30, (150, 150, 150), -1)
        dets_dark = detect_white_shapes(frame_dark, 1.0)
        self.assertEqual(len(dets_dark), 0, "亮度 150 的圆应被过滤（V < 188）")

        # 正常白色圆：应被检测到。
        frame_ok = _make_white_circle_frame(200, 100, 100, 30, bg=60, fg=255)
        dets_ok = detect_white_shapes(frame_ok, 1.0)
        self.assertGreater(len(dets_ok), 0, "正常白色圆应被检测到")

    def test_countdown_digit_with_halo_rejected(self):
        """倒计时剔除：带冷色光环的白色数字与方块同帧时，只返回方块一个候选且中心不偏。"""

        frame = _make_countdown_frame(with_square=True, with_halo=True)
        dets = detect_white_shapes(frame, 1.0)
        self.assertEqual(len(dets), 1, "带光环倒计时数字应被剔除，只剩方块候选")
        np.testing.assert_allclose(dets[0].center, [160.0, 200.0], atol=2.0,
                                   err_msg="唯一候选应是方块且中心不偏")

    def test_countdown_digit_with_halo_alone_rejected(self):
        """倒计时剔除：只有带光环数字的帧应零候选（防开局把模板学成数字）。"""

        frame = _make_countdown_frame(with_square=False, with_halo=True)
        self.assertEqual(len(detect_white_shapes(frame, 1.0)), 0,
                         "带光环倒计时数字单独出现时不应产生任何候选")

    def test_countdown_digit_without_halo_still_detected(self):
        """判别守卫：无光环的白色数字仍是普通白色候选，证明剔除凭据是光环而非字形。"""

        frame = _make_countdown_frame(with_square=False, with_halo=False)
        self.assertGreater(len(detect_white_shapes(frame, 1.0)), 0,
                           "无光环白色数字应照常检成候选")

    def test_halo_touching_square_keeps_center(self):
        """粘连帧：光环压住方块上边时，方块候选中心仍不偏（防线一切断粘连桥）。"""

        frame = _make_countdown_frame(with_square=True, with_halo=True, touch=True)
        dets = detect_white_shapes(frame, 1.0)
        self.assertEqual(len(dets), 1, "粘连帧应只返回方块一个候选")
        np.testing.assert_allclose(dets[0].center, [160.0, 200.0], atol=3.0,
                                   err_msg="粘连帧方块中心偏移应 ≤3px")

    def test_countdown_digit_body_overlap_recovers_square(self):
        """失败局回归：数字本体与图形融成单一厚桥轮廓。旧版仅向白像素吃入 2px、够不到粗笔画核心，
        粘连大轮廓要么质心被数字拽入排除区被整块丢弃（丢图形→解不出来），要么学成数字+图形混合大轮廓；
        防线零用光环凸包填成数字足迹、整块删掉数字本体，恢复出干净的方块候选（面积远小于混合轮廓）。"""

        frame = _make_countdown_frame(with_square=True, with_halo=True, overlap=True)
        dets = detect_white_shapes(frame, 1.0)
        self.assertEqual(len(dets), 1, "数字本体粘连时应恢复出方块，而非丢弃或混成一大块")
        self.assertLess(dets[0].area, 4000.0,
                        "候选面积应是方块量级（~3600），不应含数字本体（数字 glyph 面积上千）")
        self.assertLess(abs(dets[0].center[0] - 160.0), 12.0, "方块水平中心不应被数字拉偏")

    def test_choose_template_detection_prefers_median_clean(self):
        """最佳帧挑选：面积中位数挡住双向离群（粘连虚高/削顶虚低），组内再取置信度最高。"""

        def det(area, conf):  # 构造只用于挑选的最小检测对象。
            return ShapeDetection(center=np.zeros(2, np.float32), contour=np.zeros((9, 1, 2), np.float32),
                                  area=area, nominal_size=math.sqrt(area), confidence=conf)
        dets = [det(5000.0, 0.90), det(3600.0, 0.82), det(3580.0, 0.75), det(3620.0, 0.78), det(2400.0, 0.85)]  # 中位 3600，正常带 [2880,4500]。
        best = choose_template_detection(dets)  # 虚高粘连帧（面积 5000、conf 0.9 最高但在带外）应被排除。
        self.assertEqual(3600.0, best.area, "应选中中位邻域内（带内 conf 最高）的正常帧")

    def test_contaminated_first_frames_not_learned(self):
        """最佳帧学习回归：前 3 帧目标粘连白条（面积虚高 +1/3），之后干净；
        旧版首帧就把粘连放大轮廓学成模板，新版应从窗口内的干净轮廓学习。"""

        params = ShapeTrackParams(process_scale=1.0, particle_count=40, global_proposals=60)
        session = ShapeTrackSession(params=params)
        session.reset(200, 200)
        for i in range(14):  # fps=30 → 选窗 12 条；前 3 帧污染。
            frame = np.full((200, 200, 3), 60, dtype=np.uint8)  # 暗底。
            cv2.rectangle(frame, (70, 70), (130, 130), (255, 255, 255), -1)  # 60x60 目标方块。
            if i < 3:  # 顶边粘连 30x40 白条，面积 +1200（>1.25×中位，落在正常带外）。
                cv2.rectangle(frame, (85, 31), (115, 71), (255, 255, 255), -1)
            session.update(frame, 30)
        self.assertIsNotNone(session.template, "选窗后应学到模板")
        self.assertLess(session.template.area, 3600.0 * 1.15, "模板应学自干净轮廓而非首帧粘连放大版")
        self.assertGreater(session.template.area, 3600.0 * 0.70, "模板面积应是方块量级（不是被削碎的残块）")

    def test_small_target_distance_window_floor(self):
        """方案 B 距离窗下限：nominal=20 的小模板按 1.5x 窗口只有 30px，40px 外的同一目标被丢；
        传 size_floor=36 后窗口抬到 54px，应正常采信；超出新窗口的仍被拒。"""

        small = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], np.int32)  # 20x20 方块，nominal=20。
        template = build_shape_template(small)
        self.assertAlmostEqual(20.0, float(template.nominal_size), places=1, msg="前置条件：小模板 nominal 应≈ 20")
        moved = small + np.array([[[40, 0]], [[40, 0]], [[40, 0]], [[40, 0]]], np.int32)  # 候选右移 40px。
        hint = np.array([10.0, 10.0], np.float32)  # 位置先验：模板方块中心。
        cand = ShapeDetection(center=np.array([50.0, 10.0], np.float32), contour=moved,
                              area=400.0, nominal_size=20.0, confidence=0.8)
        picked_narrow = choose_shape_candidate([cand], template, hint, confidence=0.8)  # 无下限：窗口 30 < 40。
        self.assertIsNone(picked_narrow, "旧窗口应把 40px 外的小目标候选拒掉")
        picked_floor = choose_shape_candidate([cand], template, hint, confidence=0.8, size_floor=36.0)  # 下限 36：窗口 54 > 40。
        self.assertIsNotNone(picked_floor, "抬下限后同一候选应被采信")
        far = ShapeDetection(center=np.array([120.0, 10.0], np.float32), contour=moved,
                             area=400.0, nominal_size=20.0, confidence=0.8)  # 距先验 110px，超出新窗口。
        self.assertIsNone(choose_shape_candidate([far], template, hint, confidence=0.8, size_floor=36.0),
                          "超出抬升后窗口的候选仍应被拒，下限不是无限放宽")

    def test_small_tracker_size_ref_floor(self):
        """方案 B 跟踪器阈值基准：min_size_floor 只抬阈值基准 size_ref，不抬模板几何尺寸；大目标不受影响。"""

        small = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], np.int32)  # nominal=20 小模板。
        template = build_shape_template(small)
        det = ShapeDetection(center=np.array([100.0, 100.0], np.float32), contour=small,
                             area=400.0, nominal_size=20.0, confidence=0.8)
        common = (det, template, 20, 200, 200, 30)  # 构造器公共位置参数（小粒子数加速）。
        floored = ParticleShapeTracker(*common, np.random.default_rng(1), min_size_floor=36.0)  # 带下限。
        plain = ParticleShapeTracker(*common, np.random.default_rng(1))  # 无下限（默认 0）。
        self.assertEqual(36.0, floored.size_ref, "小目标阈值基准应抬到下限")
        self.assertAlmostEqual(20.0, float(plain.size_ref), places=1, msg="无下限时阈值基准仍等于模板尺寸")
        big_contour = np.array([[[0, 0]], [[80, 0]], [[80, 80]], [[0, 80]]], np.int32)  # nominal≈80 大模板。
        big_template = build_shape_template(big_contour)
        big_det = ShapeDetection(center=np.array([100.0, 100.0], np.float32), contour=big_contour,
                                 area=6400.0, nominal_size=80.0, confidence=0.8)
        big = ParticleShapeTracker(big_det, big_template, 20, 200, 200, 30, np.random.default_rng(1), min_size_floor=36.0)
        self.assertAlmostEqual(80.0, float(big.size_ref), places=1, msg="大目标不被下限影响，行为零变化")

    def test_region_reset(self):
        """区域重置：连续喂帧后调用 reset，断言光流历史被清空（下一次 update 因历史不足返回 waiting）。"""

        params = ShapeTrackParams(process_scale=1.0, particle_count=40, global_proposals=60,
                                  temporal_lags=(1, 2, 4))
        session = ShapeTrackSession(params=params)
        session.reset(200, 200)

        # 连续喂 14 帧：建立光流历史并攒满最佳帧选窗（fps=30×0.4s=12 条检测）。
        for i in range(14):
            frame = _make_white_circle_frame(200, 100 + i, 100, 28, bg=60, fg=255)
            result = session.update(frame, 30)
        self.assertTrue(session.template is not None, "选窗攒满后应已学到模板")

        self.assertGreater(session.frame_index, 0, "帧计数应 > 0")
        self.assertIsNotNone(session.template, "重置前应有模板")

        session.reset(200, 200)  # 调用 reset。

        self.assertIsNone(session.template, "重置后模板应为 None")
        self.assertIsNone(session.tracker, "重置后跟踪器应为 None")
        self.assertEqual(session.frame_index, 0, "重置后帧计数应为 0")
        self.assertEqual(len(session.aligner.history), 0, "重置后光流历史应为空")

        # 下一次 update 因光流历史不足应返回 flow_residual=None（无证据图）。
        frame_after = _make_white_circle_frame(200, 100, 100, 28, bg=60, fg=255)
        result_after = session.update(frame_after, 30)
        self.assertIsNone(result_after.flow_residual, "重置后首帧应无光流残差")

    def test_repair_shape_gaps(self):
        """回看插值：构造两端 confidence=0.9、中间 5 帧 confidence=0.3 的 ShapeFrameResult 列表，
        断言 repair_shape_gaps 返回 5、中间帧 source=='interpolated'、center 为两端线性中值、
        angle_deg 走最短弧。"""

        results: list[ShapeFrameResult] = []
        # 帧 0：左端可靠帧。
        r0 = ShapeFrameResult(
            source=SOURCE_COLOR, tracker_alive=True,
            center=np.array([10.0, 20.0], dtype=np.float32),
            velocity=np.array([1.0, 0.0], dtype=np.float32),
            scale=1.0, angle_deg=10.0, confidence=0.9)
        results.append(r0)
        # 帧 1~5：低置信区间（5 帧）。
        for i in range(5):
            ri = ShapeFrameResult(
                source=SOURCE_PREDICTION, tracker_alive=True,
                center=np.array([15.0 + i, 20.0], dtype=np.float32),
                velocity=np.array([1.0, 0.0], dtype=np.float32),
                scale=1.0, angle_deg=20.0, confidence=0.3)
            results.append(ri)
        # 帧 6：右端可靠帧。
        r6 = ShapeFrameResult(
            source=SOURCE_COLOR, tracker_alive=True,
            center=np.array([40.0, 20.0], dtype=np.float32),
            velocity=np.array([2.0, 0.0], dtype=np.float32),
            scale=1.2, angle_deg=80.0, confidence=0.9)
        results.append(r6)

        repaired = repair_shape_gaps(results, threshold=0.75, max_gap_frames=10, angle_period=360.0)
        self.assertEqual(repaired, 5, "应回填 5 帧")
        for i in range(1, 6):  # 检查中间帧。
            self.assertEqual(results[i].source, SOURCE_INTERPOLATED, f"帧 {i} 应为 interpolated")
            self.assertTrue(results[i].smoothed, f"帧 {i} 应标记 smoothed")
        # 帧 3 是正中间，center 应为两端线性中值。
        expected_center = r0.center + 0.5 * (r6.center - r0.center)  # (10+40)/2=25, (20+20)/2=20。
        np.testing.assert_allclose(results[3].center, expected_center, atol=1e-3,
                                   err_msg="帧 3 中心应为两端线性中值")
        # angle_deg 走最短弧：10° → 80°，中间应为 45°。
        self.assertAlmostEqual(results[3].angle_deg, 45.0, places=1,
                               msg="帧 3 角度应为两端最短弧中值 45°")
        # scale 线性插值：1.0 → 1.2，中间应为 1.1。
        self.assertAlmostEqual(results[3].scale, 1.1, places=2,
                               msg="帧 3 缩放应为两端线性中值 1.1")

    def test_scene_ended(self):
        """场景结束：构造 raw_mean 持续 >24 的序列，
        断言超过 max(4, int(fps*0.16)) 帧后 source == 'scene-ended'。
        
        注意：必须在跳过首秒检查后（frame_index > fps）再制造大残差，
        且前面的帧必须各不相同（避免 stagnant_scene_run 提前触发）。
        """

        params = ShapeTrackParams(process_scale=1.0, particle_count=40, global_proposals=60,
                                  temporal_lags=(1,))
        session = ShapeTrackSession(params=params)
        session.reset(200, 200)
        fps = 30
        rng = np.random.default_rng(42)  # 用于生成帧间微小差异。

        # 先喂 fps+5 帧各不相同的帧（让 frame_index > fps 且避免 stagnant 计数器触发）。
        for i in range(fps + 5):
            # 每帧加少量高斯噪声（均值 60，标准差 3），保证帧间残差 > 0.22。
            bg_val = 60
            frame = np.full((200, 200, 3), bg_val, dtype=np.uint8)
            noise = rng.normal(0, 3, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # 前 3 帧带白色圆，用于学习模板。
            if i < 3:
                cv2.circle(frame, (100, 100), 28, (255, 255, 255), -1)
            result = session.update(frame, fps)

        self.assertIsNotNone(session.template, "应已学到模板")
        self.assertFalse(session.scene_ended, "场景不应已结束（噪声帧残差 > 0.22 避免 stagnant 触发）")

        # 连续喂入与前帧差异极大的帧（交替黑/白，每帧残差远 > 24）。
        scene_ended_tick = -1
        run_limit = max(4, int(fps * 0.16))  # = 4。
        for i in range(run_limit + 5):
            # 交替黑/白帧，保证每帧残差远 > 24。
            val = 0 if i % 2 == 0 else 255
            extreme = np.full((200, 200, 3), val, dtype=np.uint8)
            result = session.update(extreme, fps)
            if result.source == SOURCE_SCENE_ENDED:
                scene_ended_tick = i
                break

        self.assertGreaterEqual(scene_ended_tick, 0, "应在连续大残差后触发场景结束")
        self.assertLess(scene_ended_tick, run_limit + 2,
                        f"场景结束应在 {run_limit} 帧左右触发，实际在第 {scene_ended_tick} 帧")


if __name__ == '__main__':
    unittest.main()
