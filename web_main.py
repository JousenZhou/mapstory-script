if __name__ == "__main__":
    from src.config import config
    from ok import OK

    config = config
    config["gui"] = {
        "type": "web",
        "launch_mode": "pywebview",  # default
    }
    import src.patch_featureset_encoding  # noqa: F401  修复框架 load_json 用 gbk 读 UTF-8 标注失败（需在加载模板标注前导入）
    ok = OK(config)
    ok.start()
