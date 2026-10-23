# 修复设备句柄失效时 DeviceManager 回退选错窗口的缺陷：优先按窗口选择缓存回选，回退兜底排除系统窗口。
#
# 背景（2026-10-18 测谎失败复盘）：configs/devices.json 的 preferred 存的是临时窗口句柄（pc_<hwnd>），
# 游戏/串流窗口重开后句柄变化，preferred 匹配不上；框架 set_preferred_device 的兜底是
# 「取设备列表里最后一个 connected 设备」，枚举顺序不保证，结果选中了 TextInputHost.exe
# （Windows 输入体验，系统隐形全屏窗口）——采集看着正常但 interaction 绑定错位，
# 解测谎每帧发的鼠标移动全落错坐标空间，表现为「鼠标不跟随、地图不转（ω=0）」，整局白打。
#
# 本补丁在兜底前做两层干预（运行时猴子补丁，不改 site-packages，与 patch_featureset_encoding 同模式）：
#   1. 按 configs/window_selection.json 缓存的 exe+标题在设备表里找回上次选择的真实窗口；
#   2. 缓存不可用时，兜底只从「非系统目录窗口」里挑第一个 connected 设备。
import json  # 读取窗口选择缓存。
import os  # 路径解析与 SystemRoot 判定。

from ok.device.DeviceManager import DeviceManager  # 被补丁的目标类。
from ok.util.logger import Logger  # 框架日志器。

logger = Logger.get_logger(__name__)

SELECTION_CACHE_FILE = os.path.join('configs', 'window_selection.json')  # 与 patch_window_selector 共用的缓存文件。

_original_set_preferred_device = DeviceManager.set_preferred_device


def _load_selection_cache():  # 读窗口选择缓存（exe_path+title），读失败按无缓存处理。
    try:
        with open(SELECTION_CACHE_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _device_exe_name(device):  # 取设备 exe 的文件名（小写）：exe 字段可能是完整路径或列表。
    exe = device.get('exe')
    if isinstance(exe, (list, tuple)):
        exe = exe[0] if exe else ""
    exe = str(exe or device.get('full_path') or "")
    return os.path.basename(exe).lower()


def _is_system_device(device):  # 判定系统窗口设备：exe 完整路径位于 %SystemRoot%（C:\Windows）下，TextInputHost/ShellExperienceHost 等全在这。
    full_path = str(device.get('full_path') or "")
    root = (os.environ.get('SystemRoot') or r'C:\Windows').lower()
    return full_path.lower().startswith(root)


def match_cached_device(device_dict, cache):  # 按缓存的 exe 文件名 + 窗口标题在设备表里找回目标，返回 imei；找不到返回 None。纯函数便于单测。
    if not cache:
        return None
    exe_name = os.path.basename(str(cache.get('exe_path') or "")).lower()
    title = str(cache.get('title') or "")
    if not exe_name and not title:
        return None
    exe_hits = []
    for imei, device in device_dict.items():
        if not isinstance(device, dict):
            continue
        if exe_name and _device_exe_name(device) == exe_name:
            if title and str(device.get('nick') or "") == title:
                return imei  # exe+标题双匹配：最可信，直接采用。
            exe_hits.append(imei)
    if len(exe_hits) == 1:
        return exe_hits[0]  # 仅 exe 匹配且全表唯一（如标题变了的游戏窗口）：也可放心回选。
    return None


def pick_fallback_device(device_dict):  # 兜底选设备：取第一个 connected 且非系统目录的窗口，跳过 TextInputHost 等隐形系统窗口。纯函数便于单测。
    for imei, device in device_dict.items():
        if not isinstance(device, dict):
            continue
        if device.get('connected') and not _is_system_device(device):
            return imei
    return None


def _patched_set_preferred_device(self, imei=None, index=-1):  # 仅在「无显式指定且持久化 preferred 已失效」时接管兜底，其余路径原样透传。
    if imei is None and index == -1:
        preferred = self.config.get("preferred")
        if preferred is None or self.device_dict.get(preferred) is None:  # 句柄失效（或从未选过）：框架即将走随机兜底。
            target = match_cached_device(self.device_dict, _load_selection_cache())  # 先按缓存回选上次窗口。
            if target is None:
                target = pick_fallback_device(self.device_dict)  # 缓存不可用：过滤系统窗口后取首个 connected。
                if target is not None:
                    logger.info(f"preferred '{preferred}' stale, fallback to non-system device {target}. 持久化窗口句柄已失效，缓存也未匹配上，兜底改选非系统窗口设备。")
            else:
                logger.info(f"preferred '{preferred}' stale, restored from window cache {target}. 持久化窗口句柄已失效，按窗口选择缓存回选上次窗口。")
            if target is not None:
                return _original_set_preferred_device(self, imei=target)  # 显式传 imei 走原方法的正常绑定路径（含 select_hwnd+start）。
    return _original_set_preferred_device(self, imei=imei, index=index)


DeviceManager.set_preferred_device = _patched_set_preferred_device
