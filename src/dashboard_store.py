# 看板共享配置存取层：角色/怪物/测谎三栏配置的单一数据源（configs/Dashboard.json）。
#
# 任务运行时通过 MyBaseTask.apply_shared_config() 把这里的值覆盖进任务配置，
# 全部任务共用同一份角色/怪物/测谎参数；看板页签（DashboardTab）负责编辑与保存。
# 首次使用（文件不存在）时自动从既有任务配置迁移用户已调好的数值，无感切换。
import json  # 导入 json，用于读写看板共享配置文件。
import os  # 导入 os，用于配置文件与标注文件路径拼接。

from ok.task.task import VALID_NAMED_KEYS  # 导入框架合法按键名单，看板保存时按键校验与任务一致。
from ok.util.logger import Logger  # 导入框架日志器。

logger = Logger.get_logger(__name__)

DASHBOARD_CONFIG_FILE = os.path.join('configs', 'Dashboard.json')  # 看板共享配置文件路径。
COCO_FILE = os.path.join('ok_templates', 'coco_annotations.json')  # 模板页标注文件，供按类别读取标注。

_ANNOTATION_CACHE = {}  # 标注解析结果缓存，文件未变时直接复用。
_ANNOTATION_CACHE_KEY = None  # 缓存对应的标注文件指纹（修改时间+大小），None 表示尚无可用缓存。

SUPER_LIE_REGION = '测谎'  # 测谎区域标注的类别名（模板页标注的「类别」字段值）。
SUPER_LIE_TRIGGER = '测谎触发'  # 测谎触发标注的类别名（模板页标注的「类别」字段值）。
SUPER_CHARACTER = '角色'  # 角色标注的类别名，供看板角色栏「角色特征/左朝向/右朝向」三个单选下拉取值。
SUPER_MONSTER = '怪物'  # 怪物标注的类别名，供看板怪物栏「怪物特征」多选下拉取值。

DASHBOARD_DEFAULTS = {  # 看板共享配置默认值：键名与任务原配置键一致，任务运行时无缝读取。
    # —— 测谎栏 ——
    'Lie Detector Auto Solve': True,  # 测谎总开关：开启后全部任务运行时值守测谎触发并自动解测谎。
    'Lie Detector Region Feature': '测谎坐标框',  # 测谎区域标注：类别为「测谎」的标注分类名，直接采集坐标作为解测谎输入区域。
    'Lie Detector Trigger Feature': '测谎触发',  # 测谎触发标注：类别为「测谎触发」的标注分类名，画面匹配到即触发解测谎。
    'Lie Detector Threshold': 0.75,  # 测谎触发匹配阈值，越高越严格。
    'Lie Detector Trigger Delay': 5.0,  # 匹配到测谎触发后延迟多少秒才开始解测谎（等弹窗完全展开、图形动画起势），0 表示立即解题。
    'Lie Alarm Sound': 'alarm.mp3',  # 测谎报警音频（支持 wav/mp3，相对路径相对项目根目录），留空不报警。
    # —— 角色栏 ——
    'Character Feature': '角色名',  # 角色标注分类名。
    'Character Threshold': 0.8,  # 角色匹配阈值。
    'Character Facing Left Feature': '',  # 角色左朝向模板分类名，留空禁用朝向校准。
    'Character Facing Right Feature': '',  # 角色右朝向模板分类名，两个朝向模板都配置才启用校准。
    'Attack Key': 'a',  # 常规攻击按键。
    'Melee Attack Key': 'b',  # 近战攻击按键。
    'Melee Distance': 40,  # 近战距离（像素）。
    'Attack Range X Min': -150,  # 攻击区域左边界（符号化像素）。
    'Attack Range X Max': 150,  # 攻击区域右边界（符号化像素）。
    'Attack Range Y Min': -60,  # 攻击区域上边界（符号化像素）。
    'Attack Range Y Max': 60,  # 攻击区域下边界（符号化像素）。
    'Del Key Interval': 0.0,  # 自动按 Del 键间隔（秒），0 禁用。
    # —— 怪物栏 ——
    'Monster Features': '',  # 怪物标注分类名，英文逗号分隔支持多个。
    'Monster Threshold': 0.65,  # 怪物匹配阈值。
    'Monster Mirror Threshold': 0.65,  # 怪物镜像匹配阈值。
}


def _read_json_utf8(path):  # 按 UTF-8 读取 JSON 文件，失败返回 None（与项目标注文件读取惯例一致）。
    try:
        with open(path, encoding='utf-8-sig') as f:  # utf-8-sig 兼容带 BOM 的文件。
            return json.load(f)
    except (OSError, ValueError):
        return None


def _migrate_seed():  # 首次使用时从既有任务配置迁移数值：逐键先到先得，巡逻任务字段最全优先采信。
    seed = {}
    for name in ('MaplePatrolTask', 'MapleIdleTask'):  # 遍历顺序即优先级。
        data = _read_json_utf8(os.path.join('configs', f'{name}.json'))
        if not isinstance(data, dict):
            continue
        for key in DASHBOARD_DEFAULTS:
            if key not in seed and key in data:  # 高优先级任务已有该键时不被后续任务覆盖。
                seed[key] = data[key]
    return seed


def load_dashboard_config():  # 读取看板共享配置；文件不存在时先从既有任务配置迁移再落盘，缺失键补默认值。
    data = _read_json_utf8(DASHBOARD_CONFIG_FILE)
    if not isinstance(data, dict):
        data = dict(DASHBOARD_DEFAULTS)
        data.update(_migrate_seed())  # 迁移用户已调好的数值，避免切换看板后从头再配。
        save_dashboard_config(data)
        logger.info(f'Dashboard config created with migrated values: {DASHBOARD_CONFIG_FILE}')
    for key, value in DASHBOARD_DEFAULTS.items():  # 版本升级新增键时补齐默认值。
        data.setdefault(key, value)
    return data


def save_dashboard_config(data):  # 保存看板共享配置，写失败记日志不抛异常（看板保存按钮另行提示）。
    try:
        os.makedirs(os.path.dirname(DASHBOARD_CONFIG_FILE) or '.', exist_ok=True)
        tmp_path = DASHBOARD_CONFIG_FILE + '.tmp'  # 先写临时文件再原子替换，进程中途被杀不会写坏现有配置。
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, DASHBOARD_CONFIG_FILE)
    except OSError as e:
        logger.error(f'save dashboard config failed: {e}')


def load_annotations_by_supercategory():  # 按类别（supercategory）读取模板页标注，返回 {类别: {分类名: 标注信息}}。
    # 标注信息为 {'x','y','w','h','img_w','img_h'}：坐标为标注图源图尺寸，
    # 使用时需按 当前画面尺寸/源图尺寸 等比缩放到实际游戏画面。
    # 结果按文件指纹缓存：看板每帧刷新都会调本函数，标注文件未变时直接复用上次解析结果，避免 30Hz 反复读盘解析 JSON。
    global _ANNOTATION_CACHE, _ANNOTATION_CACHE_KEY  # 需要写模块级缓存。
    key = _coco_fingerprint()  # 取当前标注文件指纹。
    if key is not None and key == _ANNOTATION_CACHE_KEY:  # 文件自上次解析后未发生变化。
        return _ANNOTATION_CACHE  # 直接命中缓存。
    result = _parse_annotations()  # 重新读盘解析。
    if key is not None:  # 文件存在才写缓存，避免把“文件缺失”的空结果长期钉住。
        _ANNOTATION_CACHE, _ANNOTATION_CACHE_KEY = result, key  # 同时更新结果与指纹。
    return result  # 返回解析结果。


def _coco_fingerprint():  # 取标注文件指纹（修改时间与大小），文件不可读时返回 None。
    try:
        stat = os.stat(COCO_FILE)  # 只读文件元数据，比整文件读取便宜得多。
        return stat.st_mtime_ns, stat.st_size  # 修改时间与大小共同构成指纹，重新标注后必定变化。
    except OSError:  # 文件不存在或不可读。
        return None  # 返回空指纹，调用方不写缓存。


def _parse_annotations():  # 读盘并解析模板页标注文件，返回 {类别: {分类名: 标注信息}}。
    data = _read_json_utf8(COCO_FILE)
    result = {}
    if not isinstance(data, dict):
        return result
    cat_map = {c['id']: c for c in data.get('categories', [])}  # 分类 id -> 分类条目。
    img_map = {i['id']: i for i in data.get('images', [])}  # 图片 id -> 图片条目。
    for ann in data.get('annotations', []):
        cat = cat_map.get(ann.get('category_id'))
        if cat is None:  # 分类缺失的脏数据跳过。
            continue
        super_name = str(cat.get('supercategory') or '').strip()
        if not super_name:  # 未设置类别的标注不参与看板选择。
            continue
        img = img_map.get(ann.get('image_id')) or {}
        bbox = ann.get('bbox') or [0, 0, 0, 0]
        if len(bbox) != 4:  # bbox 异常跳过。
            continue
        result.setdefault(super_name, {})[str(cat.get('name') or '')] = {
            'x': bbox[0], 'y': bbox[1], 'w': bbox[2], 'h': bbox[3],
            'img_w': img.get('width') or 0, 'img_h': img.get('height') or 0,
        }
    return result


def validate_key_name(value):  # 校验按键名是否为键盘上存在的按键，与框架 BaseTask.validate_key 规则一致。
    key = str(value or '')
    k = key.lower()
    if len(k) == 1:
        if not (k.isalnum() or k in ' `~!@#$%^&*()-_=+[{]}\\|;:\'",<.>/?'):
            return False
    elif k not in VALID_NAMED_KEYS:
        return False
    return True
