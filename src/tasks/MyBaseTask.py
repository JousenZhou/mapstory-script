import json  # 导入标准库 json，用于修复标注文件的编码读取。

import ok.feature.FeatureSet as _feature_set_module  # 导入 ok-script 的 FeatureSet 模块，用于修补其 JSON 读取函数。

from ok import BaseTask


def _load_json_utf8(coco_json):  # 重定义标注文件读取函数，强制用 UTF-8 解码。
    with open(coco_json, 'r', encoding='utf-8-sig') as file:  # 用 utf-8-sig 打开文件，兼容带 BOM 的 UTF-8。
        data = json.load(file)  # 解析 COCO 格式标注数据。
        for images in data['images']:  # 遍历图片条目。
            images['file_name'] = _feature_set_module.un_fk_label_studio_path(images['file_name'])  # 保持原库的路径规范化逻辑。
        return data  # 返回解析后的标注数据。


_feature_set_module.load_json = _load_json_utf8  # 替换库内默认按系统 GBK 编码读 JSON 的函数，避免中文分类名导致解码失败；放在基类模块保证所有任务导入时生效。


class MyBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




