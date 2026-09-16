# 修复 ok 框架 FeatureSet.load_json 读取 COCO 标注时未指定编码的问题。
#
# 框架 ok/feature/FeatureSet.py 的模块级函数 load_json 用 open(coco_json, 'r') 打开
# ok_templates/coco_annotations.json，未指定 encoding。在 Windows 中文环境下默认按 gbk(cp936)
# 解码，而该标注文件由模板页以 UTF-8 生成、含中文分类名（如「小青蛇」「测谎触发」），gbk 无法
# 解码其字节，抛 UnicodeDecodeError，导致 read_from_json 加载标注失败。独立测谎监控服务
# (src/liedetector/service.py) 每轮值守都依赖 feature_exists -> read_from_json，一旦 feature_set
# 因首次加载或分辨率变化重建缓存就会命中该失败，卡在 annotation missing 而无法值守测谎触发。
#
# 采用运行时猴子补丁（与 src/ui/patch_*.py 模式一致），不直接改 site-packages，重装依赖后仍生效。
# 用 utf-8-sig 读取：兼容带/不带 BOM 的 UTF-8 标注文件（符合项目「中文 JSON 用 utf-8-sig 读取」规范）。
# 只需在加载模板标注（构建 GUI / 启动服务）之前 import 本模块。

import json
import sys

import ok.feature.FeatureSet  # noqa: F401  确保模块已导入并注册到 sys.modules。
from ok.feature.FeatureSet import un_fk_label_studio_path  # 复用框架的路径归一化逻辑。

# 通过 sys.modules 取模块对象，规避 ok.feature 包可能把同名类 FeatureSet 暴露为包属性造成的 module/class 名冲突。
_featureset_module = sys.modules['ok.feature.FeatureSet']

_original_load_json = _featureset_module.load_json  # 保留原函数引用，便于排查或将来回退。


def _patched_load_json(coco_json):  # 以 UTF-8 读取 COCO 标注，其余逻辑与框架原函数保持一致。
    with open(coco_json, 'r', encoding='utf-8-sig') as file:  # 关键修复：显式指定编码，避免 Windows 默认 gbk 解码中文失败。
        data = json.load(file)
        for images in data['images']:
            images['file_name'] = un_fk_label_studio_path(images['file_name'])
        return data


_featureset_module.load_json = _patched_load_json  # 覆盖模块级 load_json，read_from_json 内部调用即命中新版本。
