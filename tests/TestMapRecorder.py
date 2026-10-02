# 路线录制工具（src/map_recorder.py）单元测试。
# 分三层：
#   1) 纯函数：parse_rect_percent / compute_map_rect / detect_yellow_dot（黄点 HSV 取色与区域换算）；
#   2) 拼接器 MapStitcher：首帧播种、逐帧 SQDIFF 跟踪贴图为增量拼接、描线、停止按覆盖外接矩形裁剪，
#      断言 map 与 route 尺寸一致；
#   3) 键盘捕获链路：把真实按住/点按注入捕获器后驱动录制一拍（_trace_tick），断言自动推导指令、
#      色表缺色时自动扩色、事件流 sidecar 与自动终点都随资产落盘；
#   4) 端到端几何一致性（录制→PNG 落盘→消费）：用合成世界喂录制流程产出资产，存成真实 PNG 读回后，
#      用消费端同款 locate_on_global_map + nearest_color 还原角色全局坐标并解码指令色，
#      证明录出来的图一定可被 MapleRouteTask 消费（复用同一套定位/查表函数与同一通道约定），
#      并证明同一事件流离线重绘得到的路线与录制当时描出的路线逐像素一致（改粗细/改色表无需重走地图）。
# 全程不依赖 Qt/执行器/真实设备，仅构造 numpy 图像与临时目录断言。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 离屏平台设置，避免任何 Qt 初始化触碰屏幕。

import shutil
import tempfile
import unittest

import cv2  # 绘制测试黄点。
import numpy as np  # 构造测试图像。

import src.map_recorder as rec  # 被测录制模块。
import src.map_route as mr  # 消费端同款纯算法，用于端到端一致性断言。
import src.map_store as map_store  # 地图资产层，PNG 无损落盘/读回。
import src.route_palette as rp  # 录制端同款色表反查，用于重绘断言。
from src.map_store import MAP_META_DEFAULTS  # 默认 meta（含内置指令色表）。


class TestRecorderPureHelpers(unittest.TestCase):  # 区域换算与黄点检测纯函数。

    def test_parse_rect_percent(self):  # 空→None、合法→四元组、缺数/越界→抛错。
        self.assertIsNone(rec.parse_rect_percent(''))  # 留空表示整框。
        self.assertIsNone(rec.parse_rect_percent(None))  # None 同空。
        self.assertEqual((5.0, 20.0, 90.0, 75.0), rec.parse_rect_percent('5,20,90,75'))  # 合法。
        self.assertEqual((5.0, 20.0, 90.0, 75.0), rec.parse_rect_percent('5%,20%,90%,75%'))  # 带百分号也能解析。
        with self.assertRaises(ValueError):
            rec.parse_rect_percent('1,2,3')  # 数量不对。
        with self.assertRaises(ValueError):
            rec.parse_rect_percent('1,2,3,120')  # 百分比越界。

    def test_compute_map_rect(self):  # 留空取整框，配置百分比换算为画面坐标。
        self.assertEqual((10, 20, 100, 50), rec.compute_map_rect(10, 20, 100, 50, ''))  # 整框。
        # 框 (100,80,200,100)，区域 10%,10%,80%,80% → 左上各 +20/+10，尺寸 160x80。
        rx, ry, rw, rh = rec.compute_map_rect(100, 80, 200, 100, '10,10,80,80')
        self.assertEqual((120, 90, 160, 80), (rx, ry, rw, rh))

    def test_detect_yellow_dot(self):  # 区域内合成黄点应检出质心，色相不匹配时返回 None。
        frame = np.full((120, 160, 3), 20, dtype=np.uint8)  # 深灰底。
        cv2.circle(frame, (90, 60), 4, (0, 255, 255), -1)  # 画面坐标 (90,60) 画黄色块。
        rect = (40, 20, 100, 80)  # 地图区域。
        dot = rec.detect_yellow_dot(frame, rect, 18, 38, 4)  # 黄色相窗口。
        self.assertIsNotNone(dot)  # 命中。
        self.assertEqual((90, 60), (dot[0], dot[1]))  # 质心回到画黄色块中心。
        self.assertIsNone(rec.detect_yellow_dot(frame, rect, 90, 120, 4))  # 色相窗口不含黄色→无结果。

    def test_yellow_dot_prefers_bright_marker_over_terrain(self):  # 实机地形是大片暗橙黄，角色标记是小纯黄十字：必须选中标记而不是地形大块。
        frame = np.full((97, 171, 3), (43, 51, 67), dtype=np.uint8)  # 深蓝灰底（贴近小地图背景色）。
        cv2.rectangle(frame, (62, 69), (104, 77), (26, 83, 92), -1)  # 大片暗黄绿地形（旧阈值下会被当成黄点）。
        cv2.rectangle(frame, (12, 78), (36, 93), (26, 83, 92), -1)  # 再来一块地形。
        cv2.rectangle(frame, (112, 78), (146, 93), (26, 83, 92), -1)  # 第三块地形。
        cv2.line(frame, (51, 34), (58, 34), (0, 255, 255), 2)  # 角色标记：纯黄十字横笔。
        cv2.line(frame, (54, 31), (54, 38), (0, 255, 255), 2)  # 纯黄十字竖笔。
        dot = rec.detect_yellow_dot(frame, (0, 0, 171, 97), 18, 38, 4)  # 默认严格阈值。
        self.assertIsNotNone(dot)  # 命中。
        self.assertEqual((54, 34), (dot[0], dot[1]))  # 质心回到十字中心，而不是任何一块地形。

    def test_yellow_dot_ignores_dull_terrain(self):  # 只有暗黄地形、没有角色标记时应返回 None（旧实现会锁到地形上导致全程画不出线）。
        frame = np.full((97, 171, 3), (43, 51, 67), dtype=np.uint8)  # 同上的底图。
        cv2.rectangle(frame, (62, 69), (104, 77), (26, 83, 92), -1)  # 暗黄绿地形大块。
        cv2.rectangle(frame, (12, 78), (36, 93), (26, 83, 92), -1)  # 再来一块。
        self.assertIsNone(rec.detect_yellow_dot(frame, (0, 0, 171, 97), 18, 38, 4))  # 两档阈值都不应命中。

    def test_yellow_dot_detail_reports_degraded_tier(self):  # 标记偏暗时降级命中，detail 回显实际生效的阈值档供「试定位」展示。
        frame = np.full((60, 60, 3), (43, 51, 67), dtype=np.uint8)  # 小底图。
        cv2.circle(frame, (30, 30), 4, (0, 170, 170), -1)  # 偏暗的黄点（亮度 170 低于严格档 180）。
        detail = rec.detect_yellow_dot_detail(frame, (0, 0, 60, 60), 18, 38, 4)  # 带阈值档的详情版。
        self.assertIsNotNone(detail)  # 降级后命中。
        self.assertEqual((150, 150), detail[3])  # 生效的是降级档。
        self.assertEqual((30, 30), (detail[0], detail[1]))  # 坐标仍为黄点圆心。


class TestMapStitcher(unittest.TestCase):  # 增量拼接 + 描线 + 覆盖裁剪。

    def _build_world_and_path(self):  # 造一张唯一纹理的"世界小地图"与一条右下走向的相机路径。
        rng = np.random.default_rng(2026)  # 固定种子保证纹理唯一、匹配尖锐。
        world = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)  # 合成世界底图。
        path = [(50 + 3 * i, 60 + i) for i in range(12)]  # 每步右下移动，保证覆盖区在两个方向上都大于裁剪框。
        return world, path

    def test_seed_track_paste_and_crop(self):  # 首帧播种到中心，逐帧跟踪贴图，停止裁出与覆盖区同尺寸且 map==route。
        world, path = self._build_world_and_path()  # 合成场景。
        cw, ch = 40, 30  # 实时小地图裁剪尺寸。
        stitcher = rec.MapStitcher(300, 300, cw, ch)  # 预分配画布远大于路径覆盖。
        for (wx, wy) in path:  # 逐帧喂入。
            live = world[wy:wy + ch, wx:wx + cw]  # 该步实时小地图内容。
            cam, score = stitcher.locate_cam(live)  # 定位相机左上角。
            self.assertIsNotNone(cam, f'cam lost at step score={score}')  # 录制跟踪不应失败。
            placed = stitcher.place_and_paste(cam, live)  # 贴回画布。
            dot_offset = (cw // 2, ch // 2)  # 假设角色在裁剪框中心。
            stitcher.trace((placed[0] + dot_offset[0], placed[1] + dot_offset[1]), (0, 0, 255), 2)  # 蓝色描点。
        self.assertGreaterEqual(stitcher.pasted, len(path))  # 全部帧都已贴合。
        self.assertGreaterEqual(stitcher.located, len(path))  # 全部帧都描了轨迹。
        map_img, route_img = stitcher.finalize()  # 裁到覆盖外接矩形。
        self.assertIsNotNone(map_img)  # 有资产。
        self.assertEqual(map_img.shape, route_img.shape)  # 底图与路线图尺寸严格一致（可被 validate_routes 接受）。
        # 覆盖外接应至少比单块裁剪大一个步进的累计量（横向 3*11、纵向 1*11），验证确实长出了地图。
        self.assertGreaterEqual(map_img.shape[1], cw + 3 * (len(path) - 1))  # 宽度覆盖住全程。
        self.assertGreaterEqual(map_img.shape[0], ch + 1 * (len(path) - 1))  # 高度覆盖住全程。

    def test_finalize_empty_returns_none(self):  # 从没贴过任何帧时无资产。
        stitcher = rec.MapStitcher(200, 200, 20, 20)  # 新建空拼接器。
        map_img, route_img = stitcher.finalize()  # 直接收尾。
        self.assertIsNone(map_img)  # 无底图。
        self.assertIsNone(route_img)  # 无路线。

    def test_trace_out_of_bounds_is_skipped(self):  # 落点越出预分配画布时被安全跳过并计入丢失，不抛异常。
        stitcher = rec.MapStitcher(60, 60, 20, 20)  # 小画布。
        stitcher.trace((999, 999), (255, 0, 0), 2)  # 越界落点。
        self.assertEqual(1, stitcher.lost)  # 计一次丢失。
        self.assertIsNone(stitcher.prev_point)  # 未更新上一点。


class TestCommandFromKeys(unittest.TestCase):  # 实时按键 -> 三元指令的组合矩阵。

    def setUp(self):  # 默认键位（WASD + 方向键同义，跳空格，瞬移 z）。
        self.bindings = {'left': ['left', 'a'], 'right': ['right', 'd'], 'up': ['up', 'w'],
                        'down': ['down', 's'], 'jump': ['space'], 'teleport': ['z']}
        self.window = 0.25  # 动作窗。
        self.now = 1000.0  # 固定参考时刻。

    def cmd(self, held, taps):
        return rec.command_from_keys(held, taps, self.bindings, self.window, self.now)

    def test_single_direction(self):  # 只按一个方向键。
        self.assertEqual(('left', 'none', 'none'), self.cmd(['left'], {}))  # 向左。
        self.assertEqual(('none', 'up', 'none'), self.cmd(['up'], {}))  # 只爬梯向上。

    def test_alternate_binding_names(self):  # 同义键位（WASD）与规范名一样能推出指令。
        self.assertEqual(('right', 'none', 'none'), self.cmd(['d'], {}))  # d 绑在右。

    def test_opposite_directions_take_later_press(self):  # 左右同按取后按下的（与游戏后按优先一致）。
        self.assertEqual(('right', 'none', 'none'), self.cmd(['left', 'right'], {}))  # held 升序，右在后。
        self.assertEqual(('left', 'none', 'none'), self.cmd(['right', 'left'], {}))  # 左在后则取左。

    def test_diagonal_combination(self):  # 方向+跳组合（如左上斜跳）。
        self.assertEqual(('left', 'up', 'none'), self.cmd(['left', 'up'], {}))  # 两个方向共存。

    def test_jump_within_window(self):  # 点按落在窗内算 jump，松开也不影响（taps 保留）。
        self.assertEqual(('right', 'none', 'jump'), self.cmd(['right'], {'right': 999.9, 'space': 999.9}))  # 刚点过跳。
        self.assertEqual(('right', 'none', 'none'), self.cmd(['right'], {'right': 1000.0, 'space': 999.0}))  # 跳已过期（>0.25s）。

    def test_teleport_beats_jump(self):  # 窗内同时有跳与瞬移时，瞬移优先（与 apply_cmd 一致）。
        self.assertEqual(('none', 'none', 'teleport'), self.cmd([], {'space': 999.9, 'z': 999.95}))  # z 更晚但仍都有效。
        self.assertEqual(('none', 'none', 'teleport'), self.cmd([], {'space': 999.9, 'z': 999.85}))  # teleport 优先不看时序。

    def test_all_none_returns_none(self):  # 站着不动返回 None，调用方按 gap 处理。
        self.assertIsNone(self.cmd([], {}))  # 无任何键。
        self.assertIsNone(self.cmd([], {'space': 900.0}))  # 很久以前的跳。

    def test_unbound_keys_ignored(self):  # 没绑定的键（如 f1）不影响推导。
        self.assertEqual(('left', 'none', 'none'), rec.command_from_keys(['left', 'f1'], {'f1': 1000.0},
                                                                         self.bindings, self.window, self.now))  # f1 被忽略。

    def test_empty_bindings_yield_none(self):  # 键位全空时推不出任何指令。
        self.assertIsNone(rec.command_from_keys(['left'], {'left': 1000.0}, {}, self.window, self.now))


class TestBindingsFromMeta(unittest.TestCase):  # 键位 meta 解析。

    def test_defaults(self):  # 默认 meta 能解析出方向与跳，瞬移为空。
        bindings = rec.bindings_from_meta(dict(MAP_META_DEFAULTS))  # 默认值。
        self.assertEqual(['left'], bindings['left'])  # 左。
        self.assertEqual(['space'], bindings['jump'])  # 跳。
        self.assertEqual([], bindings['teleport'])  # 未绑定。

    def test_multi_key_and_normalization(self):  # 逗号分隔多键与键名归一（大写/LShift）。
        bindings = rec.bindings_from_meta({'Record Left Keys': 'A, left', 'Record Jump Keys': 'SPACE'})  # 自定义。
        self.assertEqual(['a', 'left'], bindings['left'])  # 保序去重并转小写。
        self.assertEqual(['space'], bindings['jump'])  # 大写归一。

    def test_tap_window_default_and_invalid(self):  # 动作窗读取与非法回退。
        self.assertAlmostEqual(0.25, rec.tap_window_from_meta(dict(MAP_META_DEFAULTS)))  # 默认值。
        self.assertAlmostEqual(0.25, rec.tap_window_from_meta({'Record Action Tap Window': 'abc'}))  # 非数字。
        self.assertAlmostEqual(0.25, rec.tap_window_from_meta({'Record Action Tap Window': 0}))  # 非正数。
        self.assertAlmostEqual(0.5, rec.tap_window_from_meta({'Record Action Tap Window': 0.5}))  # 正常值。


class TestRenderRouteFromEvents(unittest.TestCase):  # 事件流离线重绘。

    def setUp(self):  # 常用查找表：红=向左、蓝=向右、品红=原地上跳。
        self.lookup = {'left none none': (255, 0, 0), 'right none none': (0, 0, 255), 'none none jump': (255, 0, 255)}

    def test_lines_are_drawn_with_given_thickness(self):  # 连续同色事件连成线段，粗细可变粗。
        events = [[0.0, 10, 10, 'right none none'], [0.1, 20, 10, 'right none none']]  # 两点一段。
        img, warnings = rec.render_route_from_events(events, 60, 60, 3, self.lookup)  # 粗细 3。
        self.assertEqual([], warnings)  # 无告警。
        self.assertEqual((60, 60, 3), img.shape)  # 尺寸与画布一致。
        self.assertEqual((0, 0, 255), tuple(int(v) for v in img[10, 15]))  # 中点也被线宽覆盖（不是只有端点）。

    def test_gap_breaks_the_polyline(self):  # gap(null) 断开折线，不跨空隙拉直线。
        events = [[0.0, 5, 5, 'left none none'], [0.1, 50, 50, None], [0.2, 55, 55, 'right none none']]  # 中间断点。
        img, _ = rec.render_route_from_events(events, 80, 80, 1, self.lookup)  # 重绘。
        self.assertEqual((255, 0, 0), tuple(int(v) for v in img[5, 5]))  # 段一起点是红。
        self.assertEqual((0, 0, 255), tuple(int(v) for v in img[55, 55]))  # 段二起点是蓝。
        self.assertEqual((0, 0, 0), tuple(int(v) for v in img[30, 30]))  # 两点之间全黑：未被直线连起来。

    def test_single_event_segment_draws_dot(self):  # 孤立一拍（jump 按得很快）也要留下像素。
        img, _ = rec.render_route_from_events([[0.0, 20, 20, 'none none jump']], 40, 40, 3, self.lookup)  # 单事件。
        self.assertEqual((255, 0, 255), tuple(int(v) for v in img[20, 20]))  # 实心点中心有颜色。

    def test_unknown_command_and_out_of_bounds_warn(self):  # 色表缺色/越界事件跳过并告警（鼓止静默丢数据）。
        events = [[0.0, 10, 10, 'left none teleport'], [0.1, 500, 500, 'right none none']]  # 缺色 + 越界。
        img, warnings = rec.render_route_from_events(events, 60, 60, 2, self.lookup)  # 重绘。
        self.assertEqual(2, len(warnings))  # 两条告警。
        self.assertTrue((img == 0).all())  # 一个像素都没画。

    def test_malformed_rows_skipped(self):  # 行结构不合法时不抛异常。
        img, warnings = rec.render_route_from_events([[0.0, 1, 1], None], 10, 10, 2, self.lookup)  # 缺列 + None 行。
        self.assertEqual(2, len(warnings))  # 全部告警。

    def test_command_format_is_tolerant(self):  # 事件里的指令大小写/多余空格不影响落色。
        events = [[0.0, 5, 5, ' RIGHT  none  none '], [0.1, 15, 5, 'right none none']]  # 脏串 + 规范串。
        img, warnings = rec.render_route_from_events(events, 40, 40, 2, self.lookup)  # 重绘。
        self.assertEqual([], warnings)  # 脏串被规整后命中。
        self.assertEqual((0, 0, 255), tuple(int(v) for v in img[5, 10]))  # 两点连成同一蓝线。


class TestRecordThenConsume(unittest.TestCase):  # 端到端：录制资产→真实 PNG 落盘→消费端定位与解码一致。

    def setUp(self):  # 临时 maps 根目录隔离，绝不触碰真实 maps/。
        self.temp_dir = tempfile.mkdtemp(prefix='map_rec_test_')
        self._orig_root = map_store.MAPS_ROOT
        self._orig_index = map_store.MAP_INDEX_FILE
        map_store.MAPS_ROOT = self.temp_dir
        map_store.MAP_INDEX_FILE = os.path.join(self.temp_dir, '_index.json')
        self.main_map, self.ud_map = mr.build_color_map(dict(MAP_META_DEFAULTS))  # 默认色表查表字典。

    def tearDown(self):  # 还原常量并清理临时目录。
        map_store.MAPS_ROOT = self._orig_root
        map_store.MAP_INDEX_FILE = self._orig_index
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recorded_assets_are_consumable(self):  # 走一遍录制，存成 map.png/route.png 后消费端能定位并解码出正确指令。
        rng = np.random.default_rng(99)  # 固定种子。
        world = rng.integers(0, 256, size=(220, 220, 3), dtype=np.uint8)  # 唯一纹理合成世界。
        cw, ch = 40, 30  # 裁剪尺寸。
        dot_offset = (cw // 2, ch // 2)  # 角色在裁剪框中心。
        path = [(50 + 3 * i, 70 + i) for i in range(14)]  # 相机路径。
        stitcher = rec.MapStitcher(320, 320, cw, ch)  # 拼接器。
        for (wx, wy) in path:  # 喂录制。
            live = world[wy:wy + ch, wx:wx + cw]  # 实时小地图。
            cam, _ = stitcher.locate_cam(live)  # 定位。
            placed = stitcher.place_and_paste(cam, live)  # 贴图。
            stitcher.trace((placed[0] + dot_offset[0], placed[1] + dot_offset[1]), (0, 0, 255), 2)  # 全程蓝色=向右指令。
        map_img, route_img = stitcher.finalize()  # 收尾裁剪。
        # 落盘为真实地图资产（PNG 无损往返），复用 map_store 与运行期完全一致的读写路径。
        map_store.create_map('rec_map', map_img)  # 建图并写 map.png。
        route_file = map_store.add_route('rec_map')  # 建一条路线文件。
        map_store.save_route_image('rec_map', route_file, route_img)  # 覆盖写路线图。
        # 消费端从磁盘读回，验证尺寸校验通过。
        reloaded_map = map_store.load_map_image('rec_map')  # 读底图。
        reloaded_route = map_store.load_route_image('rec_map', route_file)  # 读路线。
        self.assertEqual(reloaded_map.shape, reloaded_route.shape)  # 尺寸一致。
        self.assertEqual([], [e for _, e in map_store.validate_routes('rec_map') if e])  # 无尺寸不一致错误。
        # 取路径中段的一帧"实时小地图"模拟运行期：用消费端同款定位还原相机位置与角色全局坐标。
        mid = len(path) // 2
        wx, wy = path[mid]
        live_mid = world[wy:wy + ch, wx:wx + cw]  # 该步实时小地图。
        cam_c, score = mr.locate_on_global_map(reloaded_map, live_mid, last_loc=None, radius=80, score_max=0.5)  # 全局定位。
        self.assertIsNotNone(cam_c)  # 命中。
        self.assertLess(score, 0.5)  # 得分达标。
        loc_c = (cam_c[0] + dot_offset[0], cam_c[1] + dot_offset[1])  # 角色全局坐标（与录制端同公式）。
        nearest, nearest_ud = mr.nearest_color(reloaded_route, loc_c, 8, self.main_map, self.ud_map)  # 最近色点解码。
        self.assertIsNotNone(nearest)  # 找到主指令色点。
        self.assertEqual('right none none', nearest['command'])  # 解码为录制时描的蓝色=向右。
        self.assertLessEqual(nearest['distance'], 8)  # 落点就在角色附近。

    def _record_with_keys(self, name, auto_goal, tap_window=5.0):  # 用合成世界 + 注入按键跑完一轮键盘捕获录制，返回 (recorder, 落盘路线图)。
        rng = np.random.default_rng(7)  # 固定种子保证纹理唯一。
        world = rng.integers(0, 256, size=(220, 220, 3), dtype=np.uint8)  # 合成世界底图。
        cw, ch = 40, 30  # 实时小地图裁剪尺寸。
        dot_offset = (cw // 2, ch // 2)  # 角色在裁剪框中心。
        path = [(50 + 3 * i, 70) for i in range(14)]  # 一路向右，模拟走图。
        map_store.create_map(name, np.zeros((60, 60, 3), dtype=np.uint8))  # 先建图目录，落盘时会被覆盖为实际尺寸。
        route_file = map_store.add_route(name)
        meta = dict(MAP_META_DEFAULTS)  # 默认 meta（含内置色表）。
        meta.update({'Record Left Keys': 'a', 'Record Right Keys': 'd', 'Record Up Keys': 'w',
                     'Record Jump Keys': 'space', 'Record Teleport Keys': '', 'Record Auto Goal': auto_goal})
        recorder = rec.MapRecorder(name, route_file, meta, lambda: None,  # 键盘捕获模式，画笔色回调不会被用到。
                                   bindings=rec.bindings_from_meta(meta), use_keys=True,
                                   auto_goal=auto_goal, tap_window=tap_window)
        recorder.stitcher = rec.MapStitcher(320, 320, cw, ch)  # 直接装配拼接器，绕开截图/模板匹配。
        for index, (wx, wy) in enumerate(path):  # 逐拍喂帧。
            live = world[wy:wy + ch, wx:wx + cw]  # 实时小地图内容。
            cam, score = recorder.stitcher.locate_cam(live)  # 定位相机。
            self.assertIsNotNone(cam, f'cam lost at step {index} score={score}')  # 合成纹理应稳定命中。
            placed = recorder.stitcher.place_and_paste(cam, live)  # 增量拼接。
            recorder.capture.press('d')  # 一直按住右。
            if 4 <= index < 8:  # 中段同时按住上：right up none 色表没定义，应触发自动扩色。
                recorder.capture.press('w')
            else:
                recorder.capture.release('w')
            recorder._trace_tick((placed[0] + dot_offset[0], placed[1] + dot_offset[1]))  # 走录制线程同款单拍逻辑。
        recorder._finish_and_save()  # 收尾裁剪落盘。
        return recorder, map_store.load_route_image(name, route_file)

    def test_keyboard_recording_writes_events_and_extends_palette(self):  # 键盘捕获：自动分色 + 扩色写回 meta + 事件流与自动终点落盘。
        recorder, route_img = self._record_with_keys('kb_auto', auto_goal=True)  # 跑一轮录制。
        self.assertTrue(recorder.saved)  # 已落盘。
        self.assertEqual([], recorder.errors)  # 无非致命问题。
        self.assertIn('right up none', [a[2] for a in recorder.color_additions])  # 中段组合被自动扩色。
        self.assertEqual('right none none', rp.command_text(*recorder.current_cmd))  # 末拍回到纯向右。
        meta = map_store.load_meta('kb_auto')  # 重读 meta。
        self.assertIn('right up none', meta['Color Code'].values())  # 新增色已写回主表（含左右 → 主表）。
        self.assertTrue(map_store.has_route_events('kb_auto', recorder.route_file))  # sidecar 已生成。
        payload = map_store.load_route_events('kb_auto', recorder.route_file)  # 读事件流。
        self.assertEqual(tuple(reversed(route_img.shape[:2])), tuple(payload['canvas']))  # canvas 记的是 [w, h]。
        self.assertEqual(recorder.thickness, payload['thickness'])  # 粗细随事件流保存，重绘据此复现。
        self.assertTrue(payload['events'])  # 有事件。
        self.assertEqual('none none goal', payload['events'][-1][3])  # 末事件为自动补的终点。
        self.assertEqual('right none none', payload['events'][0][3])  # 首事件为向右。
        goal_rgb = rp.build_reverse(meta['Color Code'], meta['Color Code Up Down'])['none none goal']  # 终点色。
        x, y = int(payload['events'][-1][1]), int(payload['events'][-1][2])  # 终点坐标（已平移到裁剪坐标）。
        self.assertEqual(tuple(goal_rgb), tuple(int(v) for v in route_img[y, x]))  # 路线图末点确实是终点色。

    def test_events_redraw_matches_recorded_pixels(self):  # 同一事件流离线重绘与录制当时描出的路线逐像素一致（关掉自动终点）。
        recorder, route_img = self._record_with_keys('kb_redraw', auto_goal=False)  # 跑一轮录制（不补终点）。
        payload = map_store.load_route_events('kb_redraw', recorder.route_file)  # 事件流。
        meta = map_store.load_meta('kb_redraw')  # 含自动新增色的 meta。
        lookup = rp.build_reverse(meta['Color Code'], meta['Color Code Up Down'])  # 反查色表。
        redraw, warnings = rec.render_route_from_events(payload['events'], payload['canvas'][0],
                                                       payload['canvas'][1], payload['thickness'], lookup)  # 重绘。
        self.assertEqual([], warnings)  # 无缺色/越界告警。
        self.assertEqual(route_img.shape, redraw.shape)  # 尺寸一致。
        self.assertTrue((route_img == redraw).all())  # 逐像素一致：重绘可完全复现录制结果。

    def test_stationary_tick_records_gap(self):  # 站着不动的一拍不描线但记下断点，重绘时在此断开（避免原地糊团与跨空隙直线）。
        map_store.create_map('kb_gap', np.zeros((60, 60, 3), dtype=np.uint8))  # 建图。
        route_file = map_store.add_route('kb_gap')
        meta = dict(MAP_META_DEFAULTS)  # 默认键位。
        recorder = rec.MapRecorder('kb_gap', route_file, meta, lambda: None,
                                   bindings=rec.bindings_from_meta(meta), use_keys=True,
                                   auto_goal=False, tap_window=5.0)  # 键盘捕获、不补终点。
        recorder.stitcher = rec.MapStitcher(300, 300, 40, 30)  # 装配拼接器。
        world = np.random.default_rng(11).integers(0, 256, size=(200, 400, 3), dtype=np.uint8)  # 合成世界（只用于让底图有覆盖区，横向宽到能容下远距离贴块）。

        def tick(x, y, cam):  # 贴一块实时小地图后描一拍，保证落点落在覆盖区内。
            recorder.stitcher.place_and_paste(cam, world[cam[1]:cam[1] + 30, cam[0]:cam[0] + 40])  # 拼接。
            recorder._trace_tick((x, y))  # 描线/记 gap。

        recorder.capture.press('right')  # 先向右走两拍（默认键位：方向右）。
        tick(100, 100, (80, 85))
        tick(110, 100, (90, 85))
        recorder.capture.release('right')  # 松手站住。
        tick(110, 100, (90, 85))  # 全 none 的一拍。
        self.assertIsNone(recorder.stitcher.prev_point)  # 连笔已断开，下一拍不会跨空隙连线。
        recorder.capture.press('right')  # 继续向右，但位置隔了一段空隙。
        tick(250, 100, (230, 85))
        self.assertEqual(['right none none', 'right none none', None, 'right none none'],
                         [item[3] for item in recorder.events])  # 站住那拍记为 gap（None）。
        recorder._finish_and_save()  # 落盘。
        payload = map_store.load_route_events('kb_gap', route_file)  # 读回事件流。
        self.assertEqual(4, len(payload['events']))  # gap 也写进 sidecar。
        route_img = map_store.load_route_image('kb_gap', route_file)  # 录制当时描出的路线。
        lookup = rp.build_reverse(meta['Color Code'], meta['Color Code Up Down'])  # 反查色。
        redraw, warnings = rec.render_route_from_events(payload['events'], payload['canvas'][0],
                                                       payload['canvas'][1], payload['thickness'], lookup)  # 重绘。
        self.assertEqual([], warnings)  # 无告警。
        self.assertTrue((route_img == redraw).all())  # 含 gap 的事件流重绘仍逐像素复现。

    def test_manual_brush_mode_keeps_old_behaviour(self):  # 关掉键盘捕获时沿用画笔色，且不产出事件流 sidecar。
        world = np.random.default_rng(3).integers(0, 256, size=(200, 200, 3), dtype=np.uint8)  # 合成世界。
        cw, ch = 40, 30  # 裁剪尺寸。
        map_store.create_map('kb_off', np.zeros((60, 60, 3), dtype=np.uint8))  # 建图。
        route_file = map_store.add_route('kb_off')
        recorder = rec.MapRecorder('kb_off', route_file, dict(MAP_META_DEFAULTS), lambda: (255, 0, 0),
                                   use_keys=False, auto_goal=True)  # 手点画笔模式。
        self.assertIsNone(recorder.capture)  # 不装键盘钩子。
        recorder.stitcher = rec.MapStitcher(300, 300, cw, ch)  # 装配拼接器。
        for index in range(8):  # 向右走 8 拍。
            live = world[70:70 + ch, 50 + 3 * index:50 + 3 * index + cw]
            cam, _ = recorder.stitcher.locate_cam(live)  # 定位。
            placed = recorder.stitcher.place_and_paste(cam, live)  # 贴合。
            recorder._trace_tick((placed[0] + cw // 2, placed[1] + ch // 2))  # 描一笔画笔色。
        recorder._finish_and_save()  # 落盘。
        self.assertTrue(recorder.saved)  # 成功。
        self.assertEqual([], recorder.events)  # 手点模式不记事件流。
        self.assertFalse(recorder.color_additions)  # 手点模式不扩色。
        self.assertFalse(map_store.has_route_events('kb_off', route_file))  # 不写 sidecar。
        route_img = map_store.load_route_image('kb_off', route_file)  # 读回路线图。
        self.assertTrue((route_img == (0, 0, 0)).any())  # 有空白处（旧图不受影响）。

    def _record_same_dot_every_tick(self, name, press, ticks=16):  # 落点全程不动的一轮录制（实机表现为黄点锁到地形），返回 (recorder, 路线图)。
        cw, ch = 40, 30  # 实时小地图裁剪尺寸。
        live = np.random.default_rng(13).integers(0, 256, size=(ch, cw, 3), dtype=np.uint8)  # 固定不变的小地图内容。
        map_store.create_map(name, np.zeros((60, 60, 3), dtype=np.uint8))  # 先建图。
        route_file = map_store.add_route(name)  # 路线文件。
        meta = dict(MAP_META_DEFAULTS)  # 默认 meta。
        recorder = rec.MapRecorder(name, route_file, meta, lambda: None,  # 键盘捕获模式。
                                   bindings=rec.bindings_from_meta(meta), use_keys=True,
                                   auto_goal=True, tap_window=5.0)
        recorder.stitcher = rec.MapStitcher(320, 320, cw, ch)  # 装配拼接器。
        for _ in range(ticks):  # 每拍同一块 live 与同一个落点。
            cam, _ = recorder.stitcher.locate_cam(live)  # 定位（永远同一处）。
            placed = recorder.stitcher.place_and_paste(cam, live)  # 贴图。
            if press:  # 按住了右：有指令但位置不动，旧行为会静默交出一张空图。
                recorder.capture.press('right')
            recorder._trace_tick((placed[0] + cw // 2, placed[1] + ch // 2))  # 落点恒定。
        recorder._finish_and_save()  # 收尾落盘。
        return recorder, map_store.load_route_image(name, route_file)

    def test_static_dot_finish_reports_actionable_warning(self):  # 落点全程没动：仍要落盘，但 errors 必须给出可执行原因而不是静默交空图。
        recorder, route_img = self._record_same_dot_every_tick('kb_static', press=True)  # 跑一轮“黄点锁到地形”的录制。
        self.assertTrue(recorder.saved)  # 底图仍有效，会落盘。
        self.assertEqual(16, recorder.trace_ticks)  # 每拍都自认为描了线。
        self.assertEqual(recorder.trace_box[0], recorder.trace_box[2])  # 外接矩形退化：x 方向从未展开。
        self.assertEqual(recorder.trace_box[1], recorder.trace_box[3])  # y 方向同样没展开。
        self.assertIn('落点全程未移动', ' '.join(recorder.errors))  # 明确告警。
        self.assertLessEqual(int(np.count_nonzero(route_img.any(axis=2))), 60)  # 除了自动终点几乎没像素，即用户看到的“没有线”。

    def test_no_key_recording_warns_no_trace(self):  # 全程没按键：一个指令都没推导出，应报“未描出任何路线”。
        recorder, _ = self._record_same_dot_every_tick('kb_nokey', press=False)  # 不按任何键跑一轮。
        self.assertEqual(0, recorder.trace_ticks)  # 一次都没描线。
        self.assertIn('未描出任何路线', ' '.join(recorder.errors))  # 告警可直接展示给用户。
        self.assertTrue(recorder.saved)  # 底图仍落盘。


if __name__ == '__main__':
    unittest.main()
