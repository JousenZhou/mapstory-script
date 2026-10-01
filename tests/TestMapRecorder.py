# 路线录制工具（src/map_recorder.py）单元测试。
# 分三层：
#   1) 纯函数：parse_rect_percent / compute_map_rect / detect_yellow_dot（黄点 HSV 取色与区域换算）；
#   2) 拼接器 MapStitcher：首帧播种、逐帧 SQDIFF 跟踪贴图为增量拼接、描线、停止按覆盖外接矩形裁剪，
#      断言 map 与 route 尺寸一致；
#   3) 端到端几何一致性（录制→PNG 落盘→消费）：用合成世界喂录制流程产出资产，存成真实 PNG 读回后，
#      用消费端同款 locate_on_global_map + nearest_color 还原角色全局坐标并解码指令色，
#      证明录出来的图一定可被 MapleRouteTask 消费（复用同一套定位/查表函数与同一通道约定）。
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


if __name__ == '__main__':
    unittest.main()
