# 地图资产存取层（src/map_store.py）单元测试：
# 覆盖新建地图落盘、路线增删、尺寸一致性校验、重命名与删除维护索引、meta 默认值补齐、
# 以及中文/非 ASCII 路径的图像读写。MAPS_ROOT 与索引路径重定向到临时目录，绝不触碰真实 maps/。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 与 GUI 测试一致的离屏平台设置（本测试其实不依赖 Qt）。

import shutil
import tempfile
import unittest

import numpy as np  # 构造测试图像矩阵。

import src.map_store as map_store  # 被测模块。


class TestMapStore(unittest.TestCase):

    def setUp(self):  # 每个用例独享临时 maps 根目录，通过改写模块级常量实现隔离。
        self.temp_dir = tempfile.mkdtemp(prefix='map_store_test_')
        self._orig_root = map_store.MAPS_ROOT
        self._orig_index = map_store.MAP_INDEX_FILE
        map_store.MAPS_ROOT = self.temp_dir  # 所有路径由 MAPS_ROOT 派生，改写后自动指向临时目录。
        map_store.MAP_INDEX_FILE = os.path.join(self.temp_dir, '_index.json')

    def tearDown(self):  # 还原全局常量并清理临时目录。
        map_store.MAPS_ROOT = self._orig_root
        map_store.MAP_INDEX_FILE = self._orig_index
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _blank_map(self, w=120, h=80):  # 生成一张纯色底图。
        return np.full((h, w, 3), 40, dtype=np.uint8)

    def test_create_map_writes_files_and_index(self):
        created = map_store.create_map('fog_forest', self._blank_map())
        self.assertEqual('fog_forest', created)  # 无重名时目录名即输入名。
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'fog_forest', 'map.png')))  # 底图落盘。
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'fog_forest', 'meta.json')))  # 配置落盘。
        self.assertEqual(['fog_forest'], map_store.list_maps())  # 列表可见。
        self.assertEqual('fog_forest', map_store.get_default_map())  # 首个地图自动成为默认。

    def test_duplicate_name_gets_suffix(self):
        map_store.create_map('cave', self._blank_map())
        second = map_store.create_map('cave', self._blank_map())
        self.assertEqual('cave_2', second)  # 重名自动追加序号，不覆盖已有地图。

    def test_add_route_matches_map_size_and_lists(self):
        map_store.create_map('hill', self._blank_map(120, 80))
        route = map_store.add_route('hill')
        self.assertEqual('route1.png', route)  # 首条命名 route1.png。
        img = map_store.load_route_image('hill', route)
        self.assertEqual((80, 120), img.shape[:2])  # 与底图同尺寸。
        self.assertTrue((img == 0).all())  # 初始全黑（无指令）。
        self.assertEqual(['route1.png'], map_store.list_routes('hill'))

    def test_validate_routes_detects_mismatch(self):
        map_store.create_map('ridge', self._blank_map(120, 80))
        route = map_store.add_route('ridge')
        bad = np.zeros((50, 50, 3), dtype=np.uint8)  # 尺寸与底图不符。
        map_store.save_route_image('ridge', route, bad)
        problems = {r: err for r, err in map_store.validate_routes('ridge')}
        self.assertIsNotNone(problems[route])  # 校验标记为不一致。

    def test_rename_map_updates_index_and_default(self):
        map_store.create_map('old_name', self._blank_map())
        map_store.set_default_map('old_name')
        new_name = map_store.rename_map('old_name', 'new_name')
        self.assertEqual('new_name', new_name)
        self.assertIn('new_name', map_store.list_maps())
        self.assertNotIn('old_name', map_store.list_maps())
        self.assertEqual('new_name', map_store.get_default_map())  # 默认地图跟随改名。

    def test_delete_map_removes_dir_and_index(self):
        map_store.create_map('doomed', self._blank_map())
        map_store.delete_map('doomed')
        self.assertEqual([], map_store.list_maps())
        self.assertFalse(os.path.isdir(os.path.join(self.temp_dir, 'doomed')))

    def test_meta_defaults_merge_and_color_code_isolated(self):
        map_store.create_map('mine', self._blank_map())
        meta = map_store.load_meta('mine')
        self.assertEqual(map_store.MAP_META_DEFAULTS['Search Range'], meta['Search Range'])  # 缺省键补齐。
        meta['Color Code']['9,9,9'] = 'left none none'  # 修改本地图色表。
        self.assertNotIn('9,9,9', map_store.DEFAULT_COLOR_CODE)  # 不得污染全局默认常量。

    def test_unicode_path_image_roundtrip(self):
        map_store.create_map('迷雾森林 一号', self._blank_map(60, 40))  # 中文加空格目录名。
        img = map_store.load_map_image('迷雾森林 一号')
        self.assertIsNotNone(img)
        self.assertEqual((40, 60), img.shape[:2])  # 非 ASCII 路径读写正常。

    def test_save_meta_roundtrip(self):
        map_store.create_map('tune', self._blank_map())
        meta = map_store.load_meta('tune')
        meta['Search Range'] = 22
        meta['Minimap Feature'] = '我的地图'
        map_store.save_meta('tune', meta)
        reloaded = map_store.load_meta('tune')
        self.assertEqual(22, reloaded['Search Range'])
        self.assertEqual('我的地图', reloaded['Minimap Feature'])


if __name__ == '__main__':
    unittest.main()
