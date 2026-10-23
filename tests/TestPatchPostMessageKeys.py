# tests/TestPatchPostMessageKeys.py - PostMessage 扩展键标志补丁单元测试
import unittest

import win32con

import src.patch_post_message_keys as patch_mod
from ok.device.interaction_methods.post_message import PostMessageInteraction

KF = 1 << 24


class TestMakeLParam(unittest.TestCase):
    """_patched_make_lparam：扩展键补 bit24，普通键不动。"""

    def setUp(self):
        # make_lparam 不依赖实例状态，直接以 None 作 self 调用补丁函数。
        self.fn = patch_mod._patched_make_lparam

    def test_extended_keys_have_bit24(self):
        for vk in (win32con.VK_LEFT, win32con.VK_UP, win32con.VK_RIGHT, win32con.VK_DOWN,
                   win32con.VK_HOME, win32con.VK_END, win32con.VK_PRIOR, win32con.VK_NEXT,
                   win32con.VK_INSERT, win32con.VK_DELETE, win32con.VK_RCONTROL, win32con.VK_RMENU):
            lp = self.fn(None, vk, False)
            self.assertTrue(lp & KF, f"VK {vk} 应带 KF_EXTENDED")

    def test_normal_keys_no_bit24(self):
        for vk in (win32con.VK_SHIFT, win32con.VK_CONTROL, win32con.VK_MENU,
                   win32con.VK_RETURN, ord('A'), ord('1')):
            lp = self.fn(None, vk, False)
            self.assertFalse(lp & KF, f"VK {vk} 不应带 KF_EXTENDED")

    def test_scan_code_bits_preserved(self):
        # 原逻辑 (scan<<16)|1 的低位置与扫描码域不被破坏。
        vk = win32con.VK_RIGHT
        orig = patch_mod._original_make_lparam(None, vk, False)
        patched = self.fn(None, vk, False)
        self.assertEqual(patched & ~KF, orig)
        self.assertEqual(patched >> 16 & 0xFF, orig >> 16 & 0xFF)

    def test_keyup_flags_combined_with_extended(self):
        # WM_KEYUP 的 bit30/31 与扩展位共存。
        lp = self.fn(None, win32con.VK_LEFT, True)
        self.assertTrue(lp & KF)
        self.assertTrue(lp & (1 << 30))
        self.assertTrue(lp & (1 << 31))


class TestPatchInstalled(unittest.TestCase):
    def test_class_method_replaced(self):
        self.assertIs(PostMessageInteraction.make_lparam, patch_mod._patched_make_lparam)

    def test_extended_set_covers_arrows(self):
        self.assertTrue({win32con.VK_LEFT, win32con.VK_UP, win32con.VK_RIGHT, win32con.VK_DOWN}
                        <= patch_mod.EXTENDED_VKS)


if __name__ == '__main__':
    unittest.main()
