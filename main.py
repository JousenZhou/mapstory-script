import os
import sys

import ok

if getattr(sys, 'frozen', False):  # PyInstaller 打包后, 将工作目录切到 exe 所在目录, 保证 configs/ok_templates 等相对路径一致。
    os.chdir(os.path.dirname(os.path.abspath(sys.executable)))

from src import config as config_module

# 版本号优先级: exe 同级 VERSION.txt (打包时写入) > APP_VERSION 环境变量 > 源码默认值。
_version_file = os.path.join(os.getcwd(), 'VERSION.txt')
_app_version = ""
if os.path.exists(_version_file):
    with open(_version_file, encoding='utf-8') as f:
        _app_version = f.read().strip()
if not _app_version:
    _app_version = os.environ.get("APP_VERSION", "")
if _app_version:
    config_module.version = _app_version
    config_module.config['version'] = _app_version

config = config_module.config

if __name__ == '__main__':
    import src.ui.patch_window_selector  # noqa: F401  应用「选择窗口」可搜索下拉框补丁（需在构建 GUI 前导入）
    import src.ui.patch_hide_tabs  # noqa: F401  屏蔽【脚本】【开发工具】【运行代码】【关于】【任务】菜单（需在构建 GUI 前导入）
    import src.ui.patch_markup_category  # noqa: F401  标注窗口新增「类别」字段（需在构建 GUI 前导入）
    import src.patch_featureset_encoding  # noqa: F401  修复框架 load_json 用 gbk 读 UTF-8 标注失败（需在加载模板标注前导入）
    config = config
    ok = ok.OK(config)
    ok.start()
