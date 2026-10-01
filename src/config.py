import os

import numpy as np
from ok import ConfigOption

version = "dev"
#不需要修改version, Github Action打包会自动修改

app_profile = os.environ.get("PYAPPIFY_APP_PROFILE", "")
gui_config = {
    'type': 'web' if app_profile.casefold() == 'web' else 'qt',
    'window_size': {
        'width': 1200,
        'height': 800,
        'min_width': 600,
        'min_height': 450,
    },
}
if gui_config['type'] == 'web':
    gui_config['launch_mode'] = 'pywebview'

key_config_option = ConfigOption('Game Hotkey Config', { #全局配置示例
    'Echo Key': 'q',
    'Liberation Key': 'r',
    'Resonance Key': 'e',
    'Tool Key': 't',
}, description='In Game Hotkey for Skills')


def make_bottom_right_black(frame): #可选. 某些游戏截图时遮挡UID使用
    """
    Changes a portion of the frame's pixels at the bottom right to black.

    Args:
        frame: The input frame (NumPy array) from OpenCV.

    Returns:
        The modified frame with the bottom-right corner blackened.  Returns the original frame
        if there's an error (e.g., invalid frame).
    """
    try:
        height, width = frame.shape[:2]  # Get height and width

        # Calculate the size of the black rectangle
        black_width = int(0.13 * width)
        black_height = int(0.025 * height)

        # Calculate the starting coordinates of the rectangle
        start_x = width - black_width
        start_y = height - black_height

        # Create a black rectangle (NumPy array of zeros)
        black_rect = np.zeros((black_height, black_width, frame.shape[2]), dtype=frame.dtype)  # Ensure same dtype

        # Replace the bottom-right portion of the frame with the black rectangle
        frame[start_y:height, start_x:width] = black_rect

        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

config = {
    'custom_tasks':True, # enable creating and editing custom tasks
    'debug': False,  # Optional, default: False
    'gui': gui_config,
    'config_folder': 'configs', #最好不要修改
    'global_configs': [key_config_option],
    'screenshot_processor': make_bottom_right_black, # 在截图的时候对frame进行修改, 可选
    'gui_icon': 'icons/icon.png', #窗口图标, 最好不需要修改文件名
    'wait_until_before_delay': 0,
    'wait_until_check_delay': 0,
    'wait_until_settle_time': 0, #调用 wait_until时候, 在第一次满足条件的时候, 会等待再次检测, 以避免某些滑动动画没到预定位置就在动画路径中被检测到
    'ocr': { #可选, 使用的OCR库
        'lib': 'onnxocr',
        'auto_simplify': True, #自动繁体转简体, 需要ppocrv5等可以识别繁体的库
        'params': {
            'use_openvino': True,
        }
    },
    'windows': {  # Windows游戏请填写此设置
        # 不锁定 exe：枚举全部可见窗口，交由「选择窗口」的可搜索下拉框选择（见 src/ui/patch_window_selector.py）。
        # 如需重新锁定为只搜索冒险岛客户端，取消下面这行注释即可。
        # 'exe': ['Maplestory_Classic.exe'],
        # optional, if set, will search the exe only
        # 'hwnd_class': 'UnrealWindow', #增加重名检查准确度
        'interaction': ['Pynput', 'PostMessage', 'Genshin', 'PyDirect','ForegroundPostMessage'], # Genshin:某些操作可以后台, 部分游戏支持 PostMessage:可后台点击, 极少游戏支持 ForegroundPostMessage:前台使用PostMessage Pynput/PyDirect:仅支持前台使用
        # 采集方式优先级。这里把 BitBlt_RenderFull 放在 WGC 前面，原因：播放器类窗口（如 QQPlayer 的 TXGuiFoundation）
        # 视频画面走独立渲染层，WGC 的 FrameArrived 回调不触发，会陷入 "no frame for 10 sec, try to restart" 死循环，
        # 且该自愈路径只重启 WGC 自身、永远不会降级到 BitBlt（降级只发生在刷新设备时的 1.5s 探测阶段）。
        # 注意：configs/devices.json 的 capture 字段优先级更高，会被提到本列表首位，需一并保持为 BitBlt_RenderFull。
        'capture_method': ['BitBlt_RenderFull', 'WGC', 'BitBlt'],  # 支持的capture有 BitBlt, WGC, BitBlt_RenderFull, DXGI
        'check_hdr': False, #当用户开启AutoHDR时候提示用户, 但不禁止使用
        'force_no_hdr': False, #True=当用户开启AutoHDR时候禁止使用
        'require_bg': True # 要求使用后台截图
    },
    'adb': {  # 模拟器或Android设备请填写此设置, mumu模拟器使用原生截图和input,速度极快. 其他模拟器和真机使用adb,截图速度较慢
        # optional, if set, will start the pacakge and ensure installed
        #'packages': ['com.abc.efg1', 'com.abc.efg1']
    },
    # 'browser': {  # 浏览器游戏请填写此设置；windows、adb、browser 至少配置一个，也可以同时配置多个
    #     'url': 'https://example.com/game',
    #     'nick': 'Browser',
    #     'resolution': (1280, 720),
    # },
    'start_timeout': 120,  # default 60
    'supported_resolution': {
        'ratio': '16:9', #支持的游戏分辨率
        'min_size': (1280, 720), #支持的最低游戏分辨率
        'resize_to': [(2560, 1440), (1920, 1080), (1600, 900), (1280, 720)], #可选, 如果非16:9自动缩放为 resize_to
    },
    'links': { # 关于里显示的链接, 可选
            'default': {
                'github': 'https://github.com/ok-oldking/ok-script-app',
                'discord': 'https://discord.gg/vVyCatEBgA',
                'share': 'Download from https://github.com/ok-oldking/ok-script-app',
                'qq_group':'https://qm.qq.com/q/3Gq4VLvQe',
                'qq_channel': 'https://pd.qq.com/s/djmm6l44y',
                'faq': 'https://github.com/ok-oldking/ok-script-app'
            }
        },
    'screenshots_folder': "screenshots", #截图存放目录, 每次重新启动会清空目录
    'gui_title': 'goodgoodstudydaydayup',  #窗口名（无特征, 与 exe/进程名一致, 避免游戏扫描识别）
    'template_matching': { # 可选, 如使用OpenCV的模板匹配
        'coco_feature_json': os.path.join('ok_templates', 'coco_annotations.json'), #coco格式标记, 与 GUI 模板页保存目录一致, 需要png图片, 在debug模式运行后, 会对进行切图仅保留被标记部分以减少图片大小
        'default_horizontal_variance': 0.002, #默认x偏移, 查找不传box的时候, 会根据coco坐标, match偏移box内的
        'default_vertical_variance': 0.002, #默认y偏移
        'default_threshold': 0.8, #默认threshold
    },
    'version': version, #版本
    'my_app': ['src.globals', 'Globals'], #可选. 全局单例对象, 可以存放加载的模型, 使用og.my_app调用
    'custom_tabs': [  # 自定义 GUI 页签
        ["src.ui.DashboardTab", "DashboardTab"],  # 看板页签：实时识图画面 + 测谎/角色/怪物共享配置（任务从这里采集参数）
        ["src.ui.MapTab", "MapTab"],  # 地图页签：管理路线挂机的全局小地图底图与彩色指令路线图（资产与配置）
        ["src.ui.LieDetectorTab", "LieDetectorTab"],  # 测谎检验页签, 上传谎言检测器录像验证求解流水线
    ],
    'onetime_tasks': [  # 用户点击触发的任务
        ["src.tasks.MapleIdleTask", "MapleIdleTask"],
        ["src.tasks.MaplePatrolTask", "MaplePatrolTask"],
        ["src.tasks.MapleSingleSpotTask", "MapleSingleSpotTask"],
        ["src.tasks.MapleRouteTask", "MapleRouteTask"],  # 地图路线挂机：消费地图页签的全局小地图底图与彩色指令路线图
    ],
}
