"""PostMessage 扩展键标志补丁：修复后台按键方向键不生效。

根因：ok 框架 PostMessageInteraction.make_lparam 构造 lParam 时只放 MapVirtualKey 扫描码，
缺 bit24 KF_EXTENDED。方向键等扩展键真实硬件扫描码带 E0 前缀，SDL(Moonlight 窗口) 收到无
扩展标志的扫描码会按非扩展表映射成小键盘键，转发到串流主机后游戏收错键 → 后台方向键失效。

修复方式：import 即替换 PostMessageInteraction.make_lparam，对扩展 VK 集合补置 bit24，
其余键位保持原逻辑（含 WM_KEYUP 的 bit30/31）。与 patch_device_selection 同款猴子补丁模式。
"""

import win32con

from ok.device.interaction_methods.post_message import PostMessageInteraction

KF_EXTENDED = 1 << 24  # lParam bit24：扩展键标志，对应扫描码 E0 前缀。

# Windows 平台扩展键 VK 集合：方向键/导航键区/右侧修饰键/PrintScreen/小键盘除号。
EXTENDED_VKS = frozenset({
    win32con.VK_LEFT, win32con.VK_UP, win32con.VK_RIGHT, win32con.VK_DOWN,
    win32con.VK_INSERT, win32con.VK_DELETE, win32con.VK_HOME, win32con.VK_END,
    win32con.VK_PRIOR, win32con.VK_NEXT,
    win32con.VK_RCONTROL, win32con.VK_RMENU,
    win32con.VK_SNAPSHOT, win32con.VK_DIVIDE,
})

_original_make_lparam = PostMessageInteraction.make_lparam


def _patched_make_lparam(self, vk_code, is_up=False):
    lparam = _original_make_lparam(self, vk_code, is_up)  # 原逻辑：扫描码<<16|1，UP 时带 bit30/31。
    if vk_code in EXTENDED_VKS:  # 扩展键补置 KF_EXTENDED，SDL 才会映射出正确的方向键扫描码。
        lparam |= KF_EXTENDED
    return lparam


PostMessageInteraction.make_lparam = _patched_make_lparam
