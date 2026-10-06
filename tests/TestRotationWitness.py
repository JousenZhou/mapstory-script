# TestRotationWitness：验证测谎「旋转见证器」的核心判别力与重锚机制，全部合成数据驱动，不依赖录像。
#
# 设计要点（对应实现计划第三节单测）：
#   - _estimate_omega 的符号/量级：直接给定「绕质心的刚体旋转位移场」与「纯平移位移场」，
#     断言前者估出正确符号与每帧角速度、后者折算窗内转角≈0（平移中位数扣除后无切向分量）。
#   - 分割→关联→累计：注入固定长基线光流场的 flow_fn，喂固定位置团块多帧，验证旋转团块被累计成
#     高分候选、平移团块不形成旋转证据；rotation_at 只在团块所在半径内返回非零。
#   - 重锚（observe_rotation）：以候选中心重撒粒子后，加权估计落到新位置、置信度复位到温和值、
#     frames_since_reliable 归零（后续由正常 color/border 观测接管）。
#   - 会话层节奏（滞回与冷却）：ShapeTrackSession 的两条重锚规则只在检查点评估，低置信需连续
#     witness_recover_frames 帧才重锚一次，重锚后 interval×6 帧内不得重复触发（回归：早先规则逐帧
#     评估而计数按检查点 +=interval 记账，滞回被压缩 5 倍、冷却只剩 6 帧）。
#
# 运行：按仓库规范逐模块起进程：.venv\Scripts\python.exe tests\TestRotationWitness.py
import math
import unittest
from types import SimpleNamespace  # 构造假见证器/假候选/假时序证据。

import numpy as np

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # 会话层重锚规则（滞回/冷却）。
from src.liedetector.shape_tracking import (  # 见证器与跟踪器内核，纯计算无 I/O。
    ParticleShapeTracker,
    RotationWitness,
    ShapeDetection,
    _WitnessTrack,
    build_shape_template,
)

FRAME = 120  # 合成证据/光流的处理尺度边长。
CX, CY = 60.0, 60.0  # 团块中心（处理尺度），也是注入旋转场的旋转中心。


def _disk_evidence(center=(CX, CY), radius=15.0, value=5.0, floor=0.1):
    """构造一张含单个圆形高响应团块的 float32 证据图（其余为低幅底）。"""

    ev = np.full((FRAME, FRAME), floor, dtype=np.float32)  # 低幅底。
    ys, xs = np.ogrid[:FRAME, :FRAME]  # 像素坐标网格。
    dist = (xs - center[0]) ** 2 + (ys - center[1]) ** 2  # 到圆心距离平方。
    ev[dist <= radius * radius] = value  # 圆内高响应，形成稳定团块。
    return ev


def _rotation_flow(omega_deg_per_frame, span):
    """绕 (CX,CY) 的刚体旋转在 span 帧基线上的总位移场 (FRAME,FRAME,2)。

    位移 = 总角度(rad) × perp(p)，perp=(-(y-cy),(x-cx))。_estimate_omega 会除以 span，
    因此期望返回值恰为 omega_deg_per_frame（度/帧），且符号与 omega 一致。
    """

    theta = math.radians(omega_deg_per_frame * span)  # 基线内总旋转角（rad）。
    ys, xs = np.mgrid[:FRAME, :FRAME]  # 完整网格（同形，便于堆叠成场）。
    perp = np.stack([-(ys - CY), xs - CX], axis=-1).astype(np.float32)  # 切向基场。
    return theta * perp  # (H,W,2) 总位移。


def _translation_flow(vector=(2.0, -1.5)):
    """纯平移位移场（无旋转）：平移中位数扣除后应几乎为零切向分量。"""

    flow = np.zeros((FRAME, FRAME, 2), dtype=np.float32)  # 常矢量场。
    flow[..., 0] = vector[0]  # x 平移。
    flow[..., 1] = vector[1]  # y 平移。
    return flow


class TestEstimateOmega(unittest.TestCase):
    """切向最小二乘 ω 估计的符号与量级（直接调用静态方法，绕开分割/关联）。"""

    def test_rotation_sign_and_magnitude(self):
        """给定绕质心 ω=2°/帧、span=15 的旋转位移场，应估回约 +2 度/帧。"""

        span = 15.0  # 与实现里 dt=5、flow_span=12 稳态基线一致的示范值。
        mask = np.zeros((FRAME, FRAME), bool)  # 布尔团块掩码。
        ys, xs = np.ogrid[:FRAME, :FRAME]  # 网格。
        mask[(xs - CX) ** 2 + (ys - CY) ** 2 <= 15 ** 2] = True  # 圆形团块。
        track = _WitnessTrack(0, (CX, CY), 0.0)  # 质心在旋转中心。
        flow = _rotation_flow(2.0, span)  # 总位移场。
        deg_per_frame = RotationWitness._estimate_omega(track, mask, flow, span)  # 估计。
        self.assertIsNotNone(deg_per_frame)  # 像素足够，应返回估计值。
        self.assertGreater(deg_per_frame, 1.5, "正向旋转应估出正角速度")
        self.assertLess(deg_per_frame, 2.5, "量级应接近注入的 2 度/帧")

    def test_rotation_reverse_sign(self):
        """反向旋转（ω=-3°/帧）应估出负号，验证符号可判别（单一定向假设的对照）。"""

        span = 15.0  # 基线。
        mask = np.zeros((FRAME, FRAME), bool)  # 掩码。
        ys, xs = np.ogrid[:FRAME, :FRAME]  # 网格。
        mask[(xs - CX) ** 2 + (ys - CY) ** 2 <= 15 ** 2] = True  # 圆团块。
        track = _WitnessTrack(0, (CX, CY), 0.0)  # 轨迹。
        flow = _rotation_flow(-3.0, span)  # 反向总位移。
        deg_per_frame = RotationWitness._estimate_omega(track, mask, flow, span)  # 估计。
        self.assertLess(deg_per_frame, -1.5, "反向旋转应估出负角速度")

    def test_pure_translation_is_zero_rotation(self):
        """纯平移场扣除中位数后无切向分量，估出的 ω 应≈0（干扰物只平移的判据）。"""

        span = 15.0  # 基线。
        mask = np.zeros((FRAME, FRAME), bool)  # 掩码。
        ys, xs = np.ogrid[:FRAME, :FRAME]  # 网格。
        mask[(xs - CX) ** 2 + (ys - CY) ** 2 <= 15 ** 2] = True  # 圆团块。
        track = _WitnessTrack(0, (CX, CY), 0.0)  # 轨迹。
        flow = _translation_flow()  # 常平移场。
        deg_per_frame = RotationWitness._estimate_omega(track, mask, flow, span)  # 估计。
        # 纯平移扣除中位数后无切向信号，幅值权重全 0 使 lever 退化为 0：应返回 None（或极小）。
        self.assertTrue(deg_per_frame is None or abs(deg_per_frame) < 0.2, "纯平移折算转角应接近 0 或被判为退化")

    def test_too_few_pixels_returns_none(self):
        """团块像素不足 30 时不可靠，应拒绝估计（返回 None）。"""

        mask = np.zeros((FRAME, FRAME), bool)  # 空掩码。
        mask[10:13, 10:13] = True  # 仅 9 像素。
        track = _WitnessTrack(0, (CX, CY), 0.0)  # 轨迹。
        flow = _rotation_flow(2.0, 15.0)  # 任意场。
        self.assertIsNone(RotationWitness._estimate_omega(track, mask, flow, 15.0))


class TestWitnessPipeline(unittest.TestCase):
    """分割→连通域→关联→累计→查询 的端到端（注入固定光流场，绕过 Farneback）。"""

    def _drive(self, flow_fn, updates=20, **kwargs):
        """喂固定圆形团块与注入光流场若干检查点，返回见证器；kwargs 可覆盖可信度门槛（供 cluster_at 口径取证）。"""

        witness = RotationWitness(threshold=2.0, check_interval=5, flow_fn=flow_fn, **kwargs)  # 注入场。
        evidence = _disk_evidence()  # 固定位置团块。
        gray = np.zeros((FRAME, FRAME), dtype=np.uint8)  # 灰度仅用于占位（flow_fn 不消费）。
        for _ in range(updates):  # 多检查点累计。
            witness.update(evidence, gray, dt=5)  # 每检查点推进 5 帧。
        return witness

    def test_rotating_blob_forms_candidate(self):
        """持续旋转团块应累计成方向恒定、折算窗内转角显著的非零候选。"""

        witness = self._drive(lambda a, b: _rotation_flow(2.0, 15.0))  # 注入恒定旋转场。
        candidates = witness.query()  # 取候选。
        self.assertTrue(candidates, "旋转团块应形成候选")
        best = candidates[0]  # 最优。
        self.assertGreater(best.windowed_deg, 20.0, "折算窗内转角应显著非零")
        self.assertGreater(best.direction_consistency, 0.99, "单一定向旋转方向一致性≈1")
        self.assertGreater(best.signed_angle, 0.0, "累计符号应与注入旋转同向")

    def test_translating_blob_no_rotation(self):
        """纯平移团块不应形成旋转证据（折算窗内转角≈0），验证与旋转目标的可分性。"""

        witness = self._drive(lambda a, b: _translation_flow())  # 注入恒定平移场。
        candidates = witness.query()  # 候选。
        max_windowed = max([c.windowed_deg for c in candidates], default=0.0)  # 最大窗内转角。
        self.assertLess(max_windowed, 5.0, "平移团块的旋转证据应接近 0")

    def test_rotation_at_localizes(self):
        """rotation_at 应在团块所在半径内返回非零窗内转角，远处返回 0（供防假锁复查）。"""

        witness = self._drive(lambda a, b: _rotation_flow(2.0, 15.0))  # 旋转团块。
        on_target = witness.rotation_at((CX, CY), 20.0)  # 团块中心查询。
        off_target = witness.rotation_at((10.0, 10.0), 20.0)  # 远处查询。
        self.assertGreater(on_target, 0.0, "团块位置应查到旋转量")
        self.assertEqual(off_target, 0.0, "远处无团块应返回 0")

    def test_cluster_at_matches_rotation_at_bit_for_bit(self):
        """cluster_at 返回簇对象本身、rotation_at 是它的薄包装：两者读数必须逐位同口径。

        会话侧规则③读的是簇对象（samples/age/windowed_deg），规则①②读的是标量转角；
        一旦两者分叉，同一个脚下团块会在三条规则里得出矛盾结论。
        返回对象而非标量的原因：0.0 既可能是「旋转相位衰减」也可能是「簇已碎裂消失」，
        而规则③对两者的处置完全相反（前者上锁定、后者丢锚解锁）。
        """

        witness = self._drive(lambda a, b: _rotation_flow(2.0, 15.0))  # 旋转团块。
        cluster = witness.cluster_at((CX, CY), 20.0)  # 团块中心查询。
        self.assertIsNotNone(cluster, "团块位置应取到簇对象")
        self.assertGreater(cluster.samples, 0, "对象应带 ω 采样数（规则③的活度判据要读）")
        self.assertGreater(cluster.age, 0.0, "对象应带年龄（区分新建碎片与稳定轨迹）")
        self.assertEqual(
            float(cluster.windowed_deg), float(witness.rotation_at((CX, CY), 20.0)), "包装后读数不变"
        )
        self.assertIsNone(witness.cluster_at((10.0, 10.0), 20.0), "远处无团块应给 None 而不是 0.0")

    def test_cluster_at_respects_sample_gate(self):
        """半径内有团块但采样不足→None：规则③「脚下读数达标且簇可信」的确认口径依赖这一闸门。"""

        witness = self._drive(  # 把单轨迹采样下限抬到不可能达标，并关掉簇聚合口径。
            lambda a, b: _rotation_flow(2.0, 15.0), cluster_enabled=False, min_track_samples=10 ** 6
        )
        self.assertIsNone(witness.cluster_at((CX, CY), 20.0), "采样不达标的簇不可信")
        self.assertEqual(0.0, witness.rotation_at((CX, CY), 20.0), "读数与簇对象同步归零")

    def test_cluster_at_respects_age_gate(self):
        """年龄不足（刚建档的单点碎片）→None：与采样下限配套，压制新出现就抢锚的碎片。"""

        witness = self._drive(lambda a, b: _rotation_flow(2.0, 15.0), min_track_age=10 ** 6)  # 年龄门槛抬到无穷。
        self.assertIsNone(witness.cluster_at((CX, CY), 20.0), "年龄不达标的簇不可信")

    def test_reset_clears_state(self):
        """reset 应清空轨迹、灰度缓冲与时钟（跨局隔离）。"""

        witness = self._drive(lambda a, b: _rotation_flow(2.0, 15.0))  # 先跑出状态。
        witness.reset()  # 复位。
        self.assertEqual(witness.tracks, [], "轨迹应清空")
        self.assertEqual(len(witness._grays), 0, "灰度缓冲应清空")
        self.assertEqual(witness.tick, 0.0, "时钟应归零")
        self.assertFalse(witness.query(), "复位后无候选")


class TestObserveRotation(unittest.TestCase):
    """ParticleShapeTracker.observe_rotation：重锚到新位置并复位置信/可靠计数。"""

    def _tracker(self):
        small = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], np.int32)  # 20x20 方块模板。
        template = build_shape_template(small)  # 构模板。
        det = ShapeDetection(
            center=np.array([100.0, 100.0], np.float32), contour=small,
            area=400.0, nominal_size=20.0, confidence=0.8,
        )  # 首检。
        return ParticleShapeTracker(det, template, 60, 200, 200, 30, np.random.default_rng(1))  # 小粒子加速。

    def test_recenters_and_resets_confidence(self):
        """重锚后加权估计落到候选中心附近、置信度回温和值、frames_since_reliable 归零。"""

        tracker = self._tracker()  # 建跟踪器。
        tracker.confidence = 0.95  # 人为抬高置信（模拟假锁后高置信）。
        tracker.frames_since_reliable = 50  # 抬高不可靠计数。
        new_center = np.array([60.0, 40.0], np.float32)  # 见证候选中心（处理尺度）。
        tracker.observe_rotation(new_center, confidence=0.7)  # 重锚。
        center = tracker.estimate()[0]  # 加权估计位置。
        distance = float(np.linalg.norm(center - new_center))  # 距候选中心。
        self.assertLess(distance, 8.0, "粒子应围绕候选中心重撒，估计落到其附近")
        self.assertAlmostEqual(0.7, tracker.confidence, places=6, msg="置信度应复位为温和值")
        self.assertEqual(0, tracker.frames_since_reliable, "可靠计数应复位由后续观测接管")


class TestSessionWitnessCadence(unittest.TestCase):
    """会话层两条重锚规则的节奏：检查点才评估、滞回按帧、冷却按帧。

    用假见证器与假跟踪器固定住「存在一个强旋转候选、不在当前跟踪位置且转角明显高于当前位置」，
    只观察 _apply_rotation_witness 在哪些帧返回 True，把 interval/滞回/冷却三个口径钉死。

    两个参面分开调：`current_cum`（当前位置团块的窗内转角）决定规则②能不能走——
    静态阈值 max(12, min_cum*0.25)=15，给 30 则 static_now=False，只剩规则①（低置信滞回）；
    给 0 则“自己不转、别处在转”，走规则②（防假锁，连续 3 个检查点）。
    """

    INTERVAL = 5  # 检查间隔（帧）。
    RECOVER_FRAMES = 20  # 低置信滞回帧数（= 4 个检查点）。
    ROTATING_HERE = 30.0  # 当前位置也在转（>静态阈值 15、与候选差 50 度仍达 margin）：只可能走规则①。

    def _session(self, confidence=0.5, current_cum=ROTATING_HERE):
        """造一个只装假件的会话：规则代码路径真实，见证器/跟踪器换成可控桩。"""

        params = ShapeTrackParams(  # 只关心见证器节奏，其余参数取默认。
            witness_check_interval=self.INTERVAL,
            witness_recover_frames=self.RECOVER_FRAMES,
        )
        session = ShapeTrackSession(params=params, logger=None)  # 真实会话。
        session.reset(60, 60)  # 建区域（会造出真实见证器，下面替换为桩）。
        reanchors = []  # 重锚调用记录（先声明，供下面 lambda 在调用时查阅）。
        candidate = SimpleNamespace(  # 固定强旋转候选：远离当前跟踪位置、转角与方向一致性都达标。
            center=np.array([40.0, 40.0], np.float32),
            windowed_deg=params.witness_min_cum_deg + 20.0,
            direction_consistency=1.0,
            samples=10,  # 合计 ω 采样数（诊断快照与簇口径门槛要读）。
            age=60.0,  # 有效旋转时长（帧）。
        )
        def fake_cluster(center, radius):  # 假证据簇：把查询点当作簇质心，读数与 rotation_at 同口径。
            if current_cum <= 0.0:  # 无团块：与真实 cluster_at 一致返回 None。
                return None
            return SimpleNamespace(
                center=np.asarray(center, np.float32),
                windowed_deg=float(current_cum),
                direction_consistency=1.0,
                samples=10,
                age=60.0,
            )

        session.witness = SimpleNamespace(  # 假见证器：只记更新次数，查询恒返回同一候选。
            update=lambda evidence, gray, dt: None,
            query=lambda: [candidate],
            cluster_at=fake_cluster,
            rotation_at=lambda center, radius: current_cum,
        )
        session.tracker = SimpleNamespace(  # 假跟踪器：位置固定在 (10,10)，记录重锚调用。
            confidence=confidence,
            size_ref=20.0,
            estimate=lambda: (np.array([10.0, 10.0], np.float32), 0.0),
            observe_rotation=lambda center, confidence=0.7: reanchors.append(tuple(center)),
        )
        session._reanchors = reanchors  # 挂到会话上便于用例读取。
        return session

    def _step(self, session):  # 喂一帧：时序证据与处理尺度画面都取最小假件。
        temporal = SimpleNamespace(normalized=np.zeros((60, 60), np.float32))  # 归一化证据（numpy 直接可用）。
        frame = np.zeros((60, 60, 3), np.uint8)  # 处理尺度 BGR 帧。
        return session._apply_rotation_witness(temporal, frame)  # 走真实规则。

    def _drive(self, session, from_tick, to_tick):  # 逐帧推进，返回其中触发重锚的帧号列表。
        return [tick for tick in range(from_tick, to_tick + 1) if self._step(session)]

    def test_rules_only_evaluated_at_checkpoints(self):
        """未到检查点不得评估：前 4 帧不重分割也不累加滞回计数，第 5 帧才进检查点。"""

        session = self._session()  # 假件会话。
        applied = [self._step(session) for _ in range(self.INTERVAL - 1)]  # 检查点之前的帧。
        self.assertFalse(any(applied), "非检查点帧不应评估规则")
        self.assertEqual(0.0, session._witness_low_run, "非检查点帧不应累加滞回计数")
        self.assertFalse(self._step(session), "首个检查点只攒 5 帧，不满滞回不应重锚")
        self.assertEqual(float(self.INTERVAL), session._witness_low_run, "检查点按帧长累加滞回")

    def test_hysteresis_needs_full_low_confidence_window(self):
        """滞回按帧（规则①）：低置信要连续满 witness_recover_frames 帧（=4 个检查点）才重锚。"""

        session = self._session()  # 恒低置信。
        applied = self._drive(session, 1, 3 * self.INTERVAL)  # 跑 3 个检查点（15 帧）。
        self.assertEqual([], applied, "15 帧不满 20 帧滞回，不应重锚")
        self.assertEqual(self.INTERVAL * 3, session._witness_low_run, "滞回应已累计 15 帧")
        applied = self._drive(session, 3 * self.INTERVAL + 1, 4 * self.INTERVAL)  # 第 4 个检查点。
        self.assertEqual([4 * self.INTERVAL], applied, "累计满 20 帧的那个检查点才重锚")
        self.assertEqual(1, len(session._reanchors), "重锚只发生一次")

    def test_cooldown_blocks_reanchor_for_six_intervals(self):
        """冷却按帧：重锚后 interval×6=30 帧内不得再抢，之后才允许第二次。"""

        session = self._session()  # 恒低置信、候选恒强。
        first_window = self._drive(session, 1, 4 * self.INTERVAL)  # 跑到首次重锚。
        self.assertEqual([self.RECOVER_FRAMES], first_window, "首次重锚应落在累计满 20 帧的检查点")
        gap = self._drive(session, self.RECOVER_FRAMES + 1, self.RECOVER_FRAMES + 29)  # 冷却窗口内再跑 29 帧。
        self.assertEqual([], gap, "冷却未到期不应重复重锚")
        again = self._drive(session, self.RECOVER_FRAMES + 30, self.RECOVER_FRAMES + 30)  # 第 30 帧的检查点。
        self.assertEqual([self.RECOVER_FRAMES + 30], again, "满 interval×6 帧后同一检查点应可再次重锚")
        self.assertEqual(2, len(session._reanchors), "共两次重锚，中间隔 30 帧")

    def test_high_confidence_resets_hysteresis(self):
        """滞回要求“连续”：中途回到高置信要清零计数，断续低置信不应攒够 20 帧。"""

        session = self._session()  # 假件会话。
        self.assertEqual([], self._drive(session, 1, 3 * self.INTERVAL), "前 15 帧不应重锚")
        self.assertEqual(15.0, session._witness_low_run, "前置：已攒 15 帧低置信")
        session.tracker.confidence = 0.95  # 回到高置信。
        self._drive(session, 16, 20)  # 跳过一个检查点。
        self.assertEqual(0.0, session._witness_low_run, "高置信一个检查点就把滞回计数清零（滞回语义）")
        session.tracker.confidence = 0.5  # 重新低置信。
        applied = self._drive(session, 21, 8 * self.INTERVAL)  # 再跑 4 个检查点。
        self.assertEqual([8 * self.INTERVAL], applied, "只有重新攒满 20 帧的那个检查点才重锚")

    def test_anti_false_lock_needs_three_checks(self):
        """规则②（防假锁）：高置信但自己不转、别处在转，连续 3 个检查点才重锚，且重锚后降信。"""

        session = self._session(confidence=0.95, current_cum=0.0)  # 高置信假锁在静止团块上。
        applied = self._drive(session, 1, 2 * self.INTERVAL)  # 前两个检查点。
        self.assertEqual([], applied, "只持续 2 次检查不应抢锚")
        applied = self._drive(session, 2 * self.INTERVAL + 1, 3 * self.INTERVAL)  # 第三个检查点。
        self.assertEqual([3 * self.INTERVAL], applied, "连续 3 次检查后应重锚")
        self.assertLess(session.tracker.confidence, 0.8, "重锚前先降信，不让假锁保持高置信")


class _FakeWitness:
    """脚本化的见证器：每个检查点给出「哪几处有可信旋转簇」，用于锚点约束的取证式单测。

    只实现会话真正调的三个接口（update/query/cluster_at），blob 四元组是
    ``(x, y, windowed_deg, samples)``；direction_consistency 默认 0.9（低于会话里的门槛 1.0），
    让三个锚点用例只跑规则③、不被规则①②干扰（真答案簇 C2 的一致性本来就只有 0.67~0.75，
    达不到 0.8 的老门槛，这一设定与实况一致）；需要激活规则①②的用例把该值调成 1.0。
    """

    def __init__(self, steps, direction_consistency=0.9):  # steps: 每个检查点的 blob 列表。
        self.steps = steps  # 逐步回放。
        self.count = 0  # 已走过的检查点数。
        self.direction_consistency = direction_consistency  # 方向一致性（决定规则①②能不能走）。

    def update(self, evidence, gray, dt=None):  # 生产里先 update 再读取数。
        self.count = min(self.count + 1, len(self.steps))  # 末步重复，便于只写关键检查点。

    def _blobs(self):  # 当前检查点的 blob。
        return self.steps[max(0, min(self.count, len(self.steps)) - 1)]

    def _candidate(self, blob):  # blob 四元组 -> 与 RotationCandidate 同字段的假簇。
        return SimpleNamespace(
            center=np.array([blob[0], blob[1]], np.float32),
            windowed_deg=float(blob[2]),
            direction_consistency=float(self.direction_consistency),
            samples=int(blob[3]),
            age=60.0,
        )

    def cluster_at(self, center, radius):  # 半径内最近簇（与生产同口径：不可信簇不参与）。
        target = np.asarray(center, dtype=np.float64)  # 查询点。
        best = None  # 最近簇。
        best_distance = float("inf")  # 最近距离。
        for blob in self._blobs():  # 遍历本检查点的团块。
            if blob[3] < 1:  # 采样为零的当作无可信簇。
                continue
            distance = float(np.hypot(blob[0] - target[0], blob[1] - target[1]))  # 质心距。
            if distance <= radius and distance < best_distance:  # 落半径内且更近。
                best_distance = distance  # 更新。
                best = blob  # 记录。
        return None if best is None else self._candidate(best)  # 簇对象或 None。

    def query(self):  # 按窗内转角降序的候选列表（真实实现按质量分，这里两者单调一致）。
        candidates = [self._candidate(blob) for blob in self._blobs() if blob[3] >= 1]  # 过采样下限的簇。
        candidates.sort(key=lambda item: item.windowed_deg, reverse=True)  # 转角降序。
        return candidates  # 交会话侧筛选。


class _FakeTracker:
    """假跟踪器：中心可手动移开（模拟弱测量拖跑），observe_rotation 把中心落回锚点。"""

    def __init__(self, center, size_ref=40.0, confidence=0.95):
        self.center = np.asarray(center, np.float32)  # 当前中心（处理尺度）。
        self.size_ref = size_ref  # 目标尺寸基准：查询半径 0.6×、锚点容差 0.25×（都被下限 12 抬到可区分区间）。
        self.confidence = confidence  # 置信度。
        self.observations = []  # 重锚/拉回调用记录 (center, confidence)。

    def estimate(self):  # 与真实跟踪器同形状：(center, velocity, angle)。
        return self.center, np.zeros(2, np.float32), 0.0

    def observe_rotation(self, center, confidence: float = 0.7):  # 重摊粒子：中心落到先验处。
        self.observations.append((tuple(float(v) for v in center), float(confidence)))  # 记账。
        self.center = np.asarray(center, dtype=np.float32)  # 拉回。


class TestSessionAnchorRein(unittest.TestCase):
    """会话层规则③（旋转锚点约束）：自确认才建锚、活簇才计滞回、衰减到位才收紧。

    三个门槛各自对应实测踩过的坑（取证数据见 _diag_out/anchor_trace2.log）：
      - 远处强候选不得建锚：第一版拿 query() 最优候选当确认，被画面顶部干扰簇抢锚，17s 偏差 96→233px；
      - 采样不涨的死簇不得建锚：干扰簇 C1 自确认时 samples 恒为 6、age 恒为 35.0，靠 lost_frames 苟活；
      - 旋转相位未结束时不得收紧：真答案簇读数还在峰值（75.8/78.5/75.7）时钉死会拦住本该跟随目标的正确漂移。
    """

    INTERVAL = 5  # 检查间隔（帧）。

    def _session(self, steps, center=(0.0, 0.0), confidence=0.95, direction_consistency=0.9):
        """造只装假件的会话：规则代码真实，见证器/跟踪器换成脚本桩。"""

        params = ShapeTrackParams(
            witness_check_interval=self.INTERVAL,
            witness_min_cum_deg=60.0,
            witness_anchor_enabled=True,
            witness_anchor_lock_enabled=True,
            witness_anchor_min_checks=2,
            witness_anchor_decay=0.5,
            witness_anchor_dead_checks=3,
            witness_anchor_conf=0.85,
            witness_anchor_tol_ratio=0.25,
            witness_dir_consistency=1.0,  # 把规则①②的一致性门槛顶到只能命中 1.0 的簇：默认假簇 0.9 一律不过，规则③孤立可测。
        )
        session = ShapeTrackSession(params=params, logger=None)  # 真实会话。
        session.reset(60, 60)  # 建区域（会造出真实见证器，下面替换为桩）。
        session.witness = _FakeWitness(steps, direction_consistency=direction_consistency)  # 脚本化证据簇。
        session.tracker = _FakeTracker(center, size_ref=40.0, confidence=confidence)  # 假跟踪器。
        return session

    def _step(self, session):  # 推进一个检查点（喂假证据+假画面）。
        temporal = SimpleNamespace(normalized=np.zeros((60, 60), np.float32))  # 归一化证据。
        frame = np.zeros((60, 60, 3), np.uint8)  # 处理尺度 BGR 帧。
        for _ in range(self.INTERVAL):  # 逐帧喂到检查点。
            applied = session._apply_rotation_witness(temporal, frame)  # 非检查点直接返回 False。
        return applied  # 检查点上的返回值。

    def _drive(self, session, times):  # 连续推进若干个检查点，返回其中触发拉回的那些次数。
        return [index for index in range(len(times)) if self._step(session)]

    def test_distant_candidate_does_not_set_anchor(self):
        """远处强旋转簇不得建锚：脚下没读数、全场最优在 40px 外（干扰簇 C1 情形）。"""

        session = self._session(  # 脚下始终无簇，远处一团持续旋转且采样增长。
            [
                [(40.0, 0.0, 80.0, 6)],
                [(40.0, 0.0, 80.0, 7)],
                [(40.0, 0.0, 80.0, 8)],
                [(40.0, 0.0, 80.0, 9)],
            ],
        )  # 一致性取默认 0.9：本用例只测「建锚不靠远处候选」，规则①②（门槛 1.0）保持惰性，
        # 否则规则②会在第三个检查点合法地抢一次锚，把 observations 弄脏，测不出规则③的行为。
        for _ in range(4):  # 跑满四个检查点。
            self._step(session)
        self.assertIsNone(session._witness_anchor, "锚点只能来自自确认，远处候选不算")
        self.assertEqual([], session.tracker.observations, "无锚点不得拉回中心")

    def test_stale_cluster_does_not_set_anchor(self):
        """采样不涨的死簇不得建锚：脚下读数达标，但 samples 恒为 6（靠 lost_frames 苟活的残影）。"""

        session = self._session(
            [
                [(4.0, 0.0, 80.0, 6)],
                [(4.0, 0.0, 80.0, 6)],
                [(4.0, 0.0, 80.0, 6)],
                [(4.0, 0.0, 80.0, 6)],
            ]
        )
        for _ in range(4):  # 连续四个检查点都自确认但采样零增长。
            self._step(session)
        self.assertIsNone(session._witness_anchor, "死簇外推不得升格为锚点")
        self.assertEqual(1, session._witness_anchor_run, "首次自确认记 1，此后采样不涨就不该再累加（够不到 min_checks=2）")

    def test_rein_engages_only_after_decay(self):
        """真答案簇（C2 情形）：活簇自确认建锚；峰值读数时不收紧，读数衰减后才拉回并锁定。"""

        session = self._session(
            [
                [(4.0, 0.0, 80.0, 6)],  # 第一次自确认（滞回 1）。
                [(4.0, 0.0, 80.0, 7)],  # 采样在涨：滞回 2 → 建锚。
                [(4.0, 0.0, 76.0, 8)],  # 仍在峰值：旋转相位未结束，不得钉死。
                [(4.0, 0.0, 24.0, 9)],  # 跌到峰值一半以下：相位结束 → 收紧。
            ]
        )
        self._step(session)  # 检查点 1。
        self._step(session)  # 检查点 2：建锚。
        self.assertIsNotNone(session._witness_anchor, "活簇两次连续自确认应建锚")
        self.assertEqual((4.0, 0.0), tuple(float(v) for v in session._witness_anchor), "锚点应落在簇质心")
        self.assertAlmostEqual(80.0, session._witness_anchor_peak, places=3, msg="峰值读数应等于首次达标转角")
        session.tracker.center = np.array([40.0, 0.0], np.float32)  # 模拟弱测量把中心拖跑（漂移 36 > 容差 12）。
        self.assertFalse(self._step(session), "旋转相位还在跑时不得收紧")
        self.assertEqual([], session.tracker.observations, "未衰减不得拉回")
        self.assertTrue(self._step(session), "读数衰减到峰值一半以下且漂移超容差应拉回")
        self.assertEqual(1, len(session.tracker.observations), "只拉回一次")
        self.assertEqual((4.0, 0.0), session.tracker.observations[0][0], "拉回目标应是锚点")
        self.assertAlmostEqual(0.85, session.tracker.observations[0][1], places=3, msg="拉回应赋予 witness_anchor_conf")
        self.assertTrue(session._witness_anchor_lock, "拉回后进入锁定（暂停 border 弱测量）")

    def test_lock_blocks_other_rules_and_releases_on_revival(self):
        """锁定期间规则①②不得把刚钉好的中心抢走；锚点处连续读不到簇则丢锚并解锁。"""

        session = self._session(
            [
                [(4.0, 0.0, 80.0, 6)],
                [(4.0, 0.0, 80.0, 7)],  # 建锚 (4,0)，峰值 80。
                [(4.0, 0.0, 24.0, 8), (80.0, 0.0, 160.0, 9)],  # 读数衰减 + 远处强候选：应拉回并锁定，不得被抢。
                [(4.0, 0.0, 24.0, 9), (80.0, 0.0, 160.0, 10)],  # 锁定中：低置信攒够也不得重锚。
                [(80.0, 0.0, 160.0, 11)],  # 锚点处读不到簇 1。
                [(80.0, 0.0, 160.0, 12)],  # 2。
                [(80.0, 0.0, 160.0, 13)],  # 3 → 达阈丢锚、解锁。
            ],
            direction_consistency=1.0,  # 让规则①②的判据真的成立，才能验证「锁定」把它们挡住。
        )
        session.tracker.confidence = 0.5  # 恒低置信：一旦解锁，规则①的滞回就会攒起来。
        self._step(session)  # 检查点 1：首次自确认（滞回 1）。
        self._step(session)  # 检查点 2：建锚。
        session.tracker.center = np.array([40.0, 0.0], np.float32)  # 弱测量把中心拖跑（漂移 36 > 容差 12）。
        self.assertTrue(self._step(session), "读数衰减且漂移超容差应拉回")  # 检查点 3。
        self.assertEqual(1, len(session.tracker.observations), "只应拉回一次，远处强候选不得抢锚")
        self.assertTrue(session._witness_anchor_lock, "拉回后进入锁定")
        self.assertFalse(self._step(session), "锁定中且中心已在锚点：既不该重复拉回，也不该被规则①抢走")  # 检查点 4。
        self.assertEqual(1, len(session.tracker.observations), "规则①的滞回（20 帧）已被锁定挡下")
        self.assertTrue(session._witness_anchor_lock, "锚点簇读数仍在（未衰减到丢锚条件）：锁定应保持")
        self._step(session)  # 检查点 5：锚点处读不到簇 1。
        self._step(session)  # 检查点 6：2。
        self.assertIsNotNone(session._witness_anchor, "未达 dead_checks 前应容忍读数抖动")
        self._step(session)  # 检查点 7：3 → 丢锚。
        self.assertIsNone(session._witness_anchor, "锚点处连续读不到可信簇应丢弃锚点")
        self.assertFalse(session._witness_anchor_lock, "丢锚同时解锁")

    def test_anchor_switch_resets_decay_baseline(self):
        """确认位置跳到了另一处：衰减基准（峰值）重新起算，不得拿旧目标的高读数把新锚点当成已衰减。"""

        session = self._session(
            [
                [(4.0, 0.0, 80.0, 6)],
                [(4.0, 0.0, 80.0, 7)],  # 建锚 A，峰值 80。
                [(20.0, 0.0, 64.0, 6)],  # 跳到 16px 外的簇 B（仍在查询半径内）：重新起算滞回。
                [(20.0, 0.0, 66.0, 7)],  # 簇 B 采样增长 → 换锚，峰值应以 66 重算而非沿用 80。
            ]
        )
        self._step(session)  # 检查点 1。
        self._step(session)  # 检查点 2：锚点 A。
        anchor_a = tuple(float(v) for v in session._witness_anchor)  # 记下 A。
        self._step(session)  # 检查点 3：确认跳到 B。
        self.assertEqual(anchor_a, tuple(float(v) for v in session._witness_anchor), "单次跳变不应立即换锚")
        self._step(session)  # 检查点 4：B 连续两次活确认 → 换锚。
        self.assertEqual((20.0, 0.0), tuple(float(v) for v in session._witness_anchor), "应换到簇 B 质心")
        self.assertAlmostEqual(66.0, session._witness_anchor_peak, places=3, msg="换锚后衰减基准重新起算")


if __name__ == "__main__":
    unittest.main()
