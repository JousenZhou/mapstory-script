# 路线跟随纯算法层（src/map_route.py）单元测试：
# 覆盖全局定位 SQDIFF 命中、最近色点曼哈顿搜索、主/上下双色互补合成、平台边缘保护判定，
# 以及与地图资产层 map_store 的 PNG 无损往返一致性（证明"录制端写入通道顺序 == 解码端反查键"自洽）。
# 这些函数无状态、不依赖 Qt/执行器，可直接构造 numpy 图像断言。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 与仓库其它测试一致的离屏平台设置（本测试其实不依赖 Qt）。

import shutil
import tempfile
import unittest

import numpy as np  # 构造测试图像矩阵。

import src.map_route as mr  # 被测纯算法模块。
import src.map_store as map_store  # 地图资产层，用于 PNG 往返一致性集成断言。
from src.map_store import MAP_META_DEFAULTS  # 默认 meta（含内置指令色表）。


class TestMapRoute(unittest.TestCase):

    def setUp(self):  # 路线色点解码与合成用例的前置：构造默认可用的主/上下色表。
        self.main_map, self.ud_map = mr.build_color_map(dict(MAP_META_DEFAULTS))  # 由默认 meta 生成查表字典。

    def test_build_color_map_keys_are_channel_tuples(self):
        self.assertIn((255, 0, 0), self.main_map)  # 主表红→键为通道三元组。
        self.assertEqual('left none none', self.main_map[(255, 0, 0)])  # 红色指令为向左走。
        self.assertIn((127, 127, 127), self.ud_map)  # 上下表灰色→爬梯向上。
        self.assertEqual('none up none', self.ud_map[(127, 127, 127)])

    def test_nearest_color_finds_matching_command(self):  # 在角色处放一个蓝点，应解码为"向右走"。
        route = np.zeros((40, 40, 3), dtype=np.uint8)  # 全黑无指令。
        route[20, 10] = (0, 0, 255)  # 通道值 (0,0,255) → 键 (0,0,255) → 蓝：right none none。
        nearest, nearest_ud = mr.nearest_color(route, (10, 20), 5, self.main_map, self.ud_map)  # 角色全局坐标 (10,20)。
        self.assertIsNotNone(nearest)  # 找到主色点。
        self.assertEqual('right none none', nearest['command'])  # 解码为向右。
        self.assertEqual((10, 20), nearest['pixel'])  # 命中像素即角色处。
        self.assertIsNone(nearest_ud)  # 无上下色。

    def test_nearest_color_skips_black_and_picks_closest(self):  # 黑色被忽略，多个色点取曼哈顿最近。
        route = np.zeros((40, 40, 3), dtype=np.uint8)
        route[20, 18] = (255, 0, 0)  # 远一点的红（向左）。
        route[20, 12] = (0, 0, 255)  # 更近的蓝（向右）。
        nearest, _ = mr.nearest_color(route, (10, 20), 12, self.main_map, self.ud_map)  # 搜索半径覆盖两者。
        self.assertEqual('right none none', nearest['command'])  # 采信更近的蓝。
        self.assertEqual(2, nearest['distance'])  # |12-10|+|20-20| = 2。

    def test_combine_cmd_main_nearer_complements_y(self):  # 主色更近且无上下分量时，用上下色补 up。
        nearest = {'pixel': (0, 0), 'command': 'left none none', 'distance': 2}
        nearest_ud = {'pixel': (0, 0), 'command': 'none up none', 'distance': 5}
        mx, my, action = mr.combine_cmd(nearest, nearest_ud)  # 合成。
        self.assertEqual('left', mx)  # 主色左右保留。
        self.assertEqual('up', my)  # 上下被互补为爬梯。
        self.assertEqual('none', action)

    def test_combine_cmd_updown_nearer_complements_x(self):  # 上下色更近且无左右分量时，用主色补 left。
        nearest = {'pixel': (0, 0), 'command': 'left none none', 'distance': 5}
        nearest_ud = {'pixel': (0, 0), 'command': 'none down jump', 'distance': 2}
        mx, my, action = mr.combine_cmd(nearest, nearest_ud)  # 合成。
        self.assertEqual('left', mx)  # 左右由主色互补。
        self.assertEqual('down', my)  # 上下取主导的上下色。
        self.assertEqual('jump', action)  # 动作取主导上下色。

    def test_combine_cmd_only_main_or_none(self):  # 仅主色直接采用；两者皆无保持 none。
        self.assertEqual(('right', 'none', 'none'), mr.combine_cmd({'command': 'right none none', 'distance': 1}, None))
        self.assertEqual(('none', 'none', 'none'), mr.combine_cmd(None, None))

    def test_locate_on_global_map_finds_patch_origin(self):  # 从大图裁一块作实时小地图，应匹配回裁剪原点。
        rng = np.random.default_rng(7)  # 固定种子保证唯一纹理。
        map_img = rng.integers(0, 256, size=(200, 220, 3), dtype=np.uint8)  # 全局拼接底图（H=200, W=220）。
        live = map_img[50:90, 60:120].copy()  # 相机左上角真实位置 (x=60,y=50)，裁剪 40x60。
        loc, score = mr.locate_on_global_map(map_img, live, last_loc=None, score_max=0.5)  # 全局匹配。
        self.assertIsNotNone(loc)  # 命中。
        self.assertEqual((60, 50), loc)  # 返回相机左上角坐标。
        self.assertLess(score, 0.05)  # 完美匹配得分极低。

    def test_locate_local_search_uses_last_loc(self):  # 提供上一帧位置时走局部搜索，仍应命中附近真实位置。
        rng = np.random.default_rng(11)
        map_img = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
        live = map_img[40:70, 45:80].copy()  # 真实相机左上 (45,40)。
        loc, score = mr.locate_on_global_map(map_img, live, last_loc=(46, 41), radius=30, score_max=0.5)  # 局部窗覆盖真值。
        self.assertEqual((45, 40), loc)  # 局部命中真值。
        self.assertLess(score, 0.05)

    def test_locate_returns_none_when_template_too_big(self):  # 实时裁剪不小于全局图时无法匹配，返回 None。
        map_img = np.zeros((20, 20, 3), dtype=np.uint8)
        live = np.zeros((30, 30, 3), dtype=np.uint8)
        loc, score = mr.locate_on_global_map(map_img, live)  # 模板比搜索图大。
        self.assertIsNone(loc)  # 定位失败。
        self.assertEqual(1.0, score)  # 最差得分。

    def test_is_near_edge_true_and_false(self):  # 角色附近存在边缘标记色则告警，远离则不告警。
        route = np.zeros((40, 40, 3), dtype=np.uint8)
        route[20, 20] = (255, 127, 127)  # 边缘色通道值。
        self.assertTrue(mr.is_near_edge(route, (21, 20), '255,127,127', box_w=20, box_h=10))  # 近邻命中。
        self.assertFalse(mr.is_near_edge(route, (2, 2), '255,127,127', box_w=20, box_h=10))  # 远离不命中。
        self.assertFalse(mr.is_near_edge(route, (20, 20), ''))  # 未配置边缘色则恒不告警。


class TestMapRoutePngRoundTrip(unittest.TestCase):  # 与地图资产层真实 PNG 往回一致性集成测试。

    def setUp(self):  # 临时 maps 根目录隔离，绝不触碰真实 maps/。
        self.temp_dir = tempfile.mkdtemp(prefix='map_route_test_')
        self._orig_root = map_store.MAPS_ROOT
        self._orig_index = map_store.MAP_INDEX_FILE
        map_store.MAPS_ROOT = self.temp_dir
        map_store.MAP_INDEX_FILE = os.path.join(self.temp_dir, '_index.json')
        self.main_map, self.ud_map = mr.build_color_map(dict(MAP_META_DEFAULTS))  # 默认色表查表字典。

    def tearDown(self):  # 还原常量并清理临时目录。
        map_store.MAPS_ROOT = self._orig_root
        map_store.MAP_INDEX_FILE = self._orig_index
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_drawn_pixel_decodes_after_png_roundtrip(self):  # 画布写入通道值→存 PNG→读回→解码指令一致。
        map_store.create_map('rt_map', np.full((50, 50, 3), 40, dtype=np.uint8))  # 建地图与底图。
        route_file = map_store.add_route('rt_map')  # 新建全黑路线图。
        route = map_store.load_route_image('rt_map', route_file)  # 读回（全黑）。
        route[30, 12] = (255, 0, 0)  # 模拟录制端把某指令色原样写入像素通道。
        map_store.save_route_image('rt_map', route_file, route)  # 保存（PNG 无损往返）。
        reloaded = map_store.load_route_image('rt_map', route_file)  # 再次读回。
        nearest, _ = mr.nearest_color(reloaded, (12, 30), 5, self.main_map, self.ud_map)  # 在该像素处解码。
        self.assertIsNotNone(nearest)  # 真实 PNG 往返后仍能精确命中指令色。
        self.assertEqual('left none none', nearest['command'])  # 与写入的色键对应的指令一致。


if __name__ == '__main__':
    unittest.main()
