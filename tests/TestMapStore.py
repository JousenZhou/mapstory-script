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

    def test_recording_meta_defaults_present(self):  # 键盘捕获录制的默认参数齐全（录制与界面都要能直接读到）。
        map_store.create_map('keys', self._blank_map())
        meta = map_store.load_meta('keys')
        self.assertTrue(meta['Record Use Keys'])  # 默认开键盘捕获。
        self.assertTrue(meta['Record Auto Goal'])  # 默认自动补终点。
        self.assertEqual('left', meta['Record Left Keys'])  # 默认方向键。
        self.assertEqual('', meta['Record Teleport Keys'])  # 默认不用瞬移。
        self.assertAlmostEqual(0.25, meta['Record Action Tap Window'])  # 默认动作窗。

    def test_route_events_sidecar_roundtrip(self):  # 事件流写入/读回：坐标、时刻、指令与 gap(null) 都能原样取回。
        map_store.create_map('ev', self._blank_map(120, 80))
        route = map_store.add_route('ev')
        self.assertEqual(os.path.join(self.temp_dir, 'ev', 'route1.keys.json'),  # sidecar 与 PNG 同目录同前缀。
                         map_store.route_events_path('ev', route))
        events = [[0.0, 10, 20, 'right none none'], [0.5, 30, 20, None], [1.0, 40, 25, 'none none jump']]  # 含 gap。
        map_store.save_route_events('ev', route, events, canvas=(120, 80), thickness=3)  # 落盘。
        data = map_store.load_route_events('ev', route)  # 读回。
        self.assertEqual(1, data['version'])  # 格式版本。
        self.assertEqual([120, 80], data['canvas'])  # 画布尺寸。
        self.assertEqual(3, data['thickness'])  # 录制时粗细。
        self.assertEqual(events, data['events'])  # 事件逐字段一致（gap 仍是 None）。
        self.assertTrue(map_store.has_route_events('ev', route))  # 存在性可查。

    def test_load_route_events_missing_or_corrupt(self):  # 无 sidecar 或内容损坏时返回 None，不抛异常。
        map_store.create_map('none_ev', self._blank_map())
        route = map_store.add_route('none_ev')
        self.assertIsNone(map_store.load_route_events('none_ev', route))  # 从未写过。
        self.assertFalse(map_store.has_route_events('none_ev', route))  # 不存在。
        with open(map_store.route_events_path('none_ev', route), 'w', encoding='utf-8') as f:  # 写脏数据。
            f.write('{not json')
        self.assertIsNone(map_store.load_route_events('none_ev', route))  # 损坏→None。

    def test_route_events_short_rows_are_padded(self):  # 旧数据只有三列时补 None，调用方可安全解包。
        map_store.create_map('short', self._blank_map())
        route = map_store.add_route('short')
        with open(map_store.route_events_path('short', route), 'w', encoding='utf-8') as f:  # 直接伪造 JSON。
            f.write('{"version": 1, "events": [[1.0, 2, 3]]}')
        data = map_store.load_route_events('short', route)  # 读回。
        self.assertEqual([[1.0, 2, 3, None]], data['events'])  # 末尾补 null。

    def test_list_routes_ignores_sidecar_and_backup(self):  # sidecar 与 .bak 不得混进路线列表（否则回放会读到假路线）。
        map_store.create_map('polluted', self._blank_map())
        route = map_store.add_route('polluted')
        map_store.save_route_events('polluted', route, [[0.0, 1, 1, 'left none none']], canvas=(120, 80), thickness=2)  # 写 sidecar。
        map_store.backup_route_image('polluted', route)  # 写 route1.png.bak。
        self.assertEqual(['route1.png'], map_store.list_routes('polluted'))  # 列表仍只有 PNG。
        self.assertTrue(os.path.isfile(map_store.route_path('polluted', route) + '.bak'))  # 备份确实存在。

    def test_delete_route_removes_image_sidecar_and_backup(self):  # 删路线时三个文件一起消失。
        map_store.create_map('gone', self._blank_map())
        route = map_store.add_route('gone')
        map_store.save_route_events('gone', route, [[0.0, 1, 1, 'left none none']], canvas=(120, 80), thickness=2)  # sidecar。
        map_store.backup_route_image('gone', route)  # 备份。
        map_store.delete_route('gone', route)  # 删除。
        self.assertFalse(os.path.isfile(map_store.route_path('gone', route)))  # PNG 没了。
        self.assertFalse(os.path.isfile(map_store.route_events_path('gone', route)))  # sidecar 没了。
        self.assertFalse(os.path.isfile(map_store.route_path('gone', route) + '.bak'))  # 备份也没了。

    def test_backup_route_image_without_source_returns_none(self):  # 原图不存在时不造假备份。
        map_store.create_map('nobak', self._blank_map())
        self.assertIsNone(map_store.backup_route_image('nobak', 'route9.png'))  # 无原图。


if __name__ == '__main__':
    unittest.main()
