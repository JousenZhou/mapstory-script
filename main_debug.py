import ok
from src.config import config

if __name__ == '__main__':
    import src.ui.patch_window_selector  # noqa: F401  应用「选择窗口」可搜索下拉框补丁（需在构建 GUI 前导入）
    import src.ui.patch_hide_tabs  # noqa: F401  屏蔽【脚本】【开发工具】【运行代码】【关于】【任务】菜单（需在构建 GUI 前导入）
    import src.ui.patch_markup_category  # noqa: F401  标注窗口新增「类别」字段（需在构建 GUI 前导入）
    import src.ui.patch_template_import  # noqa: F401  模板页新增「导入图片」按钮（需在构建 GUI 前导入）
    import src.patch_featureset_encoding  # noqa: F401  修复框架 load_json 用 gbk 读 UTF-8 标注失败（需在加载模板标注前导入）
    config = config
    config['debug'] = True
    ok = ok.OK(config)
    ok.start()
