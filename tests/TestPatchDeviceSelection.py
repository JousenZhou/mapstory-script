# src.patch_device_selection 补丁测试：preferred 句柄失效时按窗口缓存回选、兜底排除系统窗口，防解测谎鼠标绑错窗口不跟随。
import os
import unittest  # 导入单元测试框架。

import src.patch_device_selection as patch_mod  # 导入被测补丁模块（导入即完成猴子补丁）。


def _dev(exe, nick, full_path, connected=True):  # 构造一条设备表条目（与框架 refresh 后的 dict 结构一致）。
    return {'imei': '', 'exe': exe, 'nick': nick, 'full_path': full_path, 'connected': connected, 'device': 'windows'}


class TestMatchCachedDevice(unittest.TestCase):  # 缓存回选匹配测试。

    def test_exe_and_title_double_match_wins(self):
        # exe+标题双匹配的条目优先返回，即使它在表中不是第一个。
        devices = {'pc_1': _dev('thunder.exe', '下载', r'C:\Program Files\x\t.exe'),
                   'pc_2': _dev('Moonlight.exe', 'JouJo - Moonlight', r'C:\Program Files\Moonlight Game Streaming\Moonlight.exe')}
        cache = {'exe_path': 'Moonlight.exe', 'title': 'JouJo - Moonlight'}
        self.assertEqual('pc_2', patch_mod.match_cached_device(devices, cache))

    def test_exe_only_unique_match_accepted(self):
        # 标题变了（如游戏改名）但 exe 全表唯一：仍可回选。
        devices = {'pc_a': _dev('Moonlight.exe', 'JouJo - Moonlight (2)', r'C:\p\Moonlight.exe')}
        self.assertEqual('pc_a', patch_mod.match_cached_device(devices, {'exe_path': r'C:\p\Moonlight.exe', 'title': '旧标题'}))

    def test_exe_ambiguous_returns_none(self):
        # 同 exe 多个窗口且标题都对不上：宁缺毋滥返回 None 交给兜底。
        devices = {'pc_a': _dev('Moonlight.exe', 'W1', r'C:\p\M.exe'), 'pc_b': _dev('Moonlight.exe', 'W2', r'C:\p\M.exe')}
        self.assertIsNone(patch_mod.match_cached_device(devices, {'exe_path': 'Moonlight.exe', 'title': '不存在'}))

    def test_empty_cache_returns_none(self):
        # 无缓存/空缓存：不猜测。
        self.assertIsNone(patch_mod.match_cached_device({}, None))
        self.assertIsNone(patch_mod.match_cached_device({'p': _dev('a.exe', 't', r'C:\a.exe')}, {'exe_path': '', 'title': ''}))


class TestFallbackPick(unittest.TestCase):  # 兜底选设备测试。

    def test_skips_system_and_unconnected(self):
        # 复现场景：TextInputHost（SystemRoot 下）排最后且 connected，框架原逻辑会选中它；兜底必须跳过系统窗口。
        root = os.environ.get('SystemRoot', r'C:\Windows')
        devices = {'pc_t': _dev('thunder.exe', 'suspension-renderer', r'C:\Program Files\t.exe', connected=False),
                   'pc_m': _dev('Moonlight.exe', 'JouJo - Moonlight', r'C:\Program Files\Moonlight Game Streaming\Moonlight.exe'),
                   'pc_sys': _dev('TextInputHost.exe', 'Windows 输入体验', os.path.join(root, 'SystemApps', 'TextInputHost.exe'))}
        self.assertEqual('pc_m', patch_mod.pick_fallback_device(devices))

    def test_all_system_returns_none(self):
        # 只剩系统窗口：返回 None 让框架走原兜底（不改行为）。
        root = os.environ.get('SystemRoot', r'C:\Windows')
        devices = {'pc_sys': _dev('TextInputHost.exe', 'Windows 输入体验', os.path.join(root, 'x.exe'))}
        self.assertIsNone(patch_mod.pick_fallback_device(devices))


class TestWrapperLogic(unittest.TestCase):  # set_preferred_device 包装逻辑测试。

    def setUp(self):
        self.calls = []  # 记录透传给原方法的调用参数。
        self._orig = patch_mod._original_set_preferred_device
        self._cache = patch_mod._load_selection_cache
        patch_mod._original_set_preferred_device = lambda self, imei=None, index=-1: self.calls.append((imei, index))
        self.addCleanup(lambda: setattr(patch_mod, '_original_set_preferred_device', self._orig))

    def _fake_self(self, preferred, devices):  # 构造最小 DeviceManager 替身。
        import types
        return types.SimpleNamespace(config={'preferred': preferred}, device_dict=dict(devices), calls=self.calls)

    def test_stale_preferred_restores_from_cache(self):
        # preferred 失效 + 缓存命中：显式传缓存 imei 走原方法正常绑定路径。
        devices = {'pc_m': _dev('Moonlight.exe', 'JouJo - Moonlight', r'C:\p\Moonlight.exe')}
        patch_mod._load_selection_cache = lambda: {'exe_path': 'Moonlight.exe', 'title': 'JouJo - Moonlight'}
        self.addCleanup(lambda: setattr(patch_mod, '_load_selection_cache', self._cache))
        fake = self._fake_self('pc_dead', devices)
        patch_mod._patched_set_preferred_device(fake)
        self.assertEqual([('pc_m', -1)], fake.calls)

    def test_stale_preferred_fallback_filters_system(self):
        # preferred 失效 + 无缓存：兜底跳过系统窗口选真实窗口。
        root = os.environ.get('SystemRoot', r'C:\Windows')
        devices = {'pc_sys': _dev('TextInputHost.exe', 'Windows 输入体验', os.path.join(root, 'x.exe')),
                   'pc_m': _dev('Moonlight.exe', 'JouJo', r'C:\p\Moonlight.exe')}
        patch_mod._load_selection_cache = lambda: None
        self.addCleanup(lambda: setattr(patch_mod, '_load_selection_cache', self._cache))
        fake = self._fake_self('pc_dead', devices)
        patch_mod._patched_set_preferred_device(fake)
        self.assertEqual([('pc_m', -1)], fake.calls)

    def test_valid_preferred_passthrough(self):
        # preferred 仍有效：不干预，按原签名透传。
        devices = {'pc_m': _dev('Moonlight.exe', 'JouJo', r'C:\p\Moonlight.exe')}
        fake = self._fake_self('pc_m', devices)
        patch_mod._patched_set_preferred_device(fake)
        self.assertEqual([(None, -1)], fake.calls)

    def test_explicit_args_passthrough(self):
        # GUI 显式指定 imei/index：绝不干预。
        fake = self._fake_self('pc_dead', {})
        patch_mod._patched_set_preferred_device(fake, index=2)
        self.assertEqual([(None, 2)], fake.calls)

    def test_patch_installed_on_class(self):
        # 补丁确实替换了框架方法（防重构后失配）。
        from ok.device.DeviceManager import DeviceManager
        self.assertIs(DeviceManager.set_preferred_device, patch_mod._patched_set_preferred_device)


if __name__ == '__main__':
    unittest.main()
