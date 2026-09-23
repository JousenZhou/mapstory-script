# 全桌面截图工具：游戏掉线后窗口关闭，框架 capture_method 不可用，
# 改用 PIL.ImageGrab 截取整个桌面供模板匹配定位启动器按钮。
import ctypes  # 读取虚拟屏原点，用于多显示器坐标换算。
import numpy as np  # 导入 NumPy，用于图像数组转换。
import cv2  # 导入 OpenCV，用于 RGB→BGR 色彩空间转换。

from PIL import ImageGrab  # 导入 PIL 截屏模块（pillow 已在项目依赖中）。

SM_XVIRTUALSCREEN = 76  # GetSystemMetrics：虚拟屏左上角 X（主屏左上为 0,0，副屏在左则为负）。
SM_YVIRTUALSCREEN = 77  # GetSystemMetrics：虚拟屏左上角 Y。


def virtual_screen_origin():
    """返回 all_screens=True 全屏截图的图像坐标换算到 pynput/SetCursorPos 坐标需加的原点偏移 (ox, oy)。

    all_screens 截图的像素 (0,0) 对应虚拟屏左上角，而 pynput(SetCursorPos) 以主屏左上为 (0,0)；
    副屏在主屏左侧时虚拟屏原点为负（如 -1920），故 pynput 坐标 = 图像坐标 + (ox, oy)。
    多显示器/分辨率动态变化时该偏移随之变化，保证点击坐标始终落在启动器所在屏幕。
    """
    try:
        user32 = ctypes.windll.user32  # Windows user32。
        return int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN)), int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    except Exception:  # 非 Windows 或调用失败：按单屏原点处理。
        return 0, 0


def capture_desktop(all_screens=False):
    """截取全桌面，返回 BGR numpy 数组（与 OpenCV / feature_set.find_feature 兼容）。

    Args:
        all_screens: True 时截取全部显示器拼接画面，False 仅主显示器。

    Returns:
        BGR numpy ndarray，截图失败返回 None。
    """
    try:
        img = ImageGrab.grab(all_screens=all_screens)  # 截取屏幕，返回 PIL.Image (RGB)。
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL RGB → OpenCV BGR。
        return frame
    except Exception:
        return None
