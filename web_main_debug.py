if __name__ == "__main__":
    from src.config import config
    from ok import OK

    config = config
    config["debug"] = True
    config["gui"] = {
        "type": "web",
        "launch_mode": "pywebview",  # default
    }
    import src.patch_featureset_encoding  # noqa: F401  修复框架 load_json 用 gbk 读 UTF-8 标注失败（需在加载模板标注前导入）
    import src.patch_device_selection  # noqa: F401  设备句柄失效时按缓存回选窗口/兜底排除系统窗口，防鼠标绑错窗口（需在设备刷新前导入）
    import src.patch_post_message_keys  # noqa: F401  PostMessage 后台按键补置扩展键 KF_EXTENDED，修方向键不生效（需在交互创建前导入）
    ok = OK(config)
    ok.start()
