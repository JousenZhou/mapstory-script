import json  # 导入标准库 json，用于修复标注文件的编码读取。

import ok.feature.FeatureSet as _feature_set_module  # 导入 ok-script 的 FeatureSet 模块，用于修补其 JSON 读取函数。
import ok.ui.qt.tasks.LabelAndDoubleSpinBox as _double_spinbox_module  # 导入框架小数配置控件模块，用于修补小数位数。

from ok import BaseTask


def _load_json_utf8(coco_json):  # 重定义标注文件读取函数，强制用 UTF-8 解码。
    with open(coco_json, 'r', encoding='utf-8-sig') as file:  # 用 utf-8-sig 打开文件，兼容带 BOM 的 UTF-8。
        data = json.load(file)  # 解析 COCO 格式标注数据。
        for images in data['images']:  # 遍历图片条目。
            images['file_name'] = _feature_set_module.un_fk_label_studio_path(images['file_name'])  # 保持原库的路径规范化逻辑。
        return data  # 返回解析后的标注数据。


_feature_set_module.load_json = _load_json_utf8  # 替换库内默认按系统 GBK 编码读 JSON 的函数，避免中文分类名导致解码失败；放在基类模块保证所有任务导入时生效。

# 需要毫秒级（3 位小数）精度的浮点配置键：框默认只有 2 位小数，无法输入 0.001 级别的位移时长。
_THREE_DECIMAL_KEYS = frozenset({"Move Away Seconds", "Move Back Seconds"})

# 需要放大上限的浮点配置键：框默认上限 99.99，Del 间隔需要输入三位数（如 100 秒）。
_LARGE_MAX_KEYS = frozenset({"Del Key Interval"})

_original_double_spinbox_init = _double_spinbox_module.LabelAndDoubleSpinBox.__init__  # 保存框架浮点控件的原构造函数。


def _patched_double_spinbox_init(self, config_desc, config, key):  # 包装浮点控件构造函数，按键名调整小数位数。
    _original_double_spinbox_init(self, config_desc, config, key)  # 先走原逻辑创建控件。
    from src.ui.spin_wheel_guard import disable_wheel_until_focused  # 延迟导入，避免基类模块与 UI 层的导入顺序耦合。
    spin = getattr(self, "spin_box", None)  # 框架浮点控件内部真正的数字框。
    if spin is not None:  # 存在才打补丁。
        disable_wheel_until_focused(spin)  # 框架浮点框也统一：需点击聚焦后滚轮才生效，避免滚动页面误改数值。
    if key in _THREE_DECIMAL_KEYS:  # 位移时长需要 3 位小数。
        self.spin_box.setDecimals(3)  # 显示与输入精度提升到毫秒级。
        self.spin_box.setSingleStep(0.05)  # 步进缩小，方便微调小数。
        self.spin_box.setMinimum(0.0)  # 时长不允许负数。
        self.spin_box.setValue(self.config.get(self.key))  # 原构造在 2 位小数下赋的值会被截断，提精度后重新赋回完整值。
    if key in _LARGE_MAX_KEYS:  # Del 间隔类配置需要支持三位数秒数。
        self.spin_box.setRange(0.0, 999.0)  # 上限从默认 99.99 提升到 999，下限保持非负。
        self.spin_box.setValue(self.config.get(self.key))  # 超出旧上限的已保存值重新赋回。


_double_spinbox_module.LabelAndDoubleSpinBox.__init__ = _patched_double_spinbox_init  # 替换框默认 2 位小数的行为，导入时全局生效。


class MyBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_shared_config(self):  # 把看板共享配置（角色/怪物/测谎）覆盖进任务配置，全部任务共用同一份参数。
        from src.dashboard_store import load_dashboard_config  # 延迟导入，避免配置层循环依赖。
        try:  # 共享配置读取失败时任务沿用自身配置运行，不能阻断任务。
            shared = load_dashboard_config()  # 读取 configs/Dashboard.json。
        except Exception as e:  # 文件损坏等异常。
            self.log_warning(f"Load dashboard shared config failed: {e}. 看板共享配置读取失败，沿用任务自身配置。")  # 提示后降级。
            return  # 不覆盖任何值。
        self.config.update(shared)  # 看板为单一数据源：任务页已移除这些键，运行时全部以看板为准。




