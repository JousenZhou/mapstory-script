# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: 打包 Windows 绿色版 (onedir)。
# exe/进程名用无特征名称, 避免被游戏扫描进程识别。
# 用法: python -m PyInstaller build_exe.spec --clean --noconfirm
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# 项目资源目录 (i18n/ok_templates/icons/assets) 按 exe 同级相对路径读取,
# 由 build_exe.ps1 / CI 构建后复制到 exe 同级, 不放进 _internal。
# 这里只收集三方库自带的数据文件: ok-script 内置资源、OCR 模型、Fluent 主题资源。
datas = []
datas += collect_data_files('ok')
datas += collect_data_files('onnxocr')
datas += collect_data_files('qfluentwidgets')

# 任务/页签/全局单例模块仅以字符串出现在 src/config.py, 由 importlib 动态加载,
# 静态分析扫不到, 必须显式声明; ok 内部采集/交互方式同样按名字实例化, 收集全部子模块。
hiddenimports = [
    'src',
    'src.config',
    'src.globals',
    'src.tasks.MapleIdleTask',
    'src.tasks.MaplePatrolTask',
    'src.ui.DashboardTab',
    'src.dashboard_store',
    'src.gpu_match',
] + collect_submodules('ok') + collect_submodules('openvino')

# cupy/ultralytics 为可选加速依赖 (体积大), 运行时自动降级, 不打入包内。
excludes = [
    'cupy',
    'cupyx',
    'ultralytics',
    'torch',
    'torchvision',
    'pytest',
    'tests',
    'tkinter',
    'IPython',
    'notebook',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    # OpenVINO 插件/前端 DLL 位于 openvino/libs, 需保留子目录结构才能被发现,
    # collect_dynamic_libs 会保持包内相对路径; 否则 available_devices 为空导致 OCR 失败。
    binaries=collect_dynamic_libs('openvino'),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='goodgoodstudydaydayup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 程序, 不弹控制台
    icon='icons/icon.ico',
    uac_admin=True,  # Pynput 等交互需要管理员权限, 与 pyappify.yml admin: true 对齐
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='goodgoodstudydaydayup',
)
