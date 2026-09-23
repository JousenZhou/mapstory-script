# 全桌面原生尺度模板匹配器。
#
# 背景：框架 FeatureSet.find_feature 会按「当前帧宽 / 标注源图宽」自动缩放模板，
# 用于适配同一设备在不同分辨率下的等比缩放。但掉线重登时截图源从游戏窗口(约1366宽)
# 切到全桌面(如3440宽)，二者并非同一场景的等比放大，框架会把启动器按钮模板放大约1.8~2.5倍，
# 与桌面上原生尺寸的按钮对不上，导致【连接】等模板永远匹配失败、流程超时。
#
# 本匹配器绕开框架缩放：直接从 coco_annotations.json 按 bbox 裁剪「原生尺寸」模板，
# 用 cv2.matchTemplate 在桌面截图上 1:1 匹配，返回物理像素坐标。系统 DPI 缩放为 100% 时，
# 该坐标与 pynput 点击坐标一一对应，可直接用于桌面级点击。
import json  # 解析 COCO 标注文件。
import os  # 路径拼接与标注文件 mtime 检查。
import time  # mtime 检查限频，避免每帧 stat。

import cv2  # 模板匹配与灰度转换。

MATCH_METHOD = cv2.TM_CCOEFF_NORMED  # 归一化相关系数匹配，与框架默认方法一致。
RELOAD_CHECK_INTERVAL = 2.0  # 标注文件 mtime 检查间隔秒数：重新标注后能自动生效，又不必每帧 stat。


class DesktopTemplateMatcher:  # 在原生像素尺度下对全桌面截图做模板匹配（绕开框架按帧宽缩放）。

    def __init__(self, coco_json, logger=None):  # 构造匹配器。
        """
        Args:
            coco_json: coco_annotations.json 路径（与框架 template_matching.coco_feature_json 一致）。
            logger: 日志器，None 时静默。
        """
        self._coco_json = coco_json  # 标注文件路径。
        self._logger = logger  # 日志器。
        self._index = None  # 分类名 -> [(源图路径, bbox), ...] 索引，解析一次后缓存。
        self._templates = {}  # 分类名 -> [原生尺寸 BGR 模板, ...]，按需裁剪缓存。
        self._mtime = None  # 上次加载时标注文件的 mtime，变化则失效缓存重新解析。
        self._last_check = 0.0  # 上次 mtime 检查时间戳，用于限频。

    def _warn(self, message):  # 输出警告日志（有日志器时）。
        if self._logger is not None:
            self._logger.warning(message)

    def _ensure_index(self):  # 建立/刷新「分类名 -> bbox」索引；标注文件变化时重建并清空模板裁剪缓存。
        now = time.time()  # 当前时间。
        if self._index is not None and now - self._last_check < RELOAD_CHECK_INTERVAL:  # 未到检查间隔，直接用缓存。
            return self._index
        self._last_check = now  # 记录本次检查时间。
        try:  # 标注文件可能不存在。
            mtime = os.path.getmtime(self._coco_json)
        except OSError:
            return self._index  # 取不到 mtime，沿用旧索引（可能为 None）。
        if self._index is not None and mtime == self._mtime:  # 文件未变化。
            return self._index
        try:  # 重新解析标注文件。
            with open(self._coco_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:  # 解析失败不能拖垮重登流程。
            self._warn(f"Desktop matcher load annotations failed: {e}. 全桌面匹配器读取标注失败：{e}。")
            return self._index
        folder = os.path.dirname(self._coco_json)  # 源图与标注文件同目录。
        images = {im.get('id'): im for im in data.get('images', [])}  # image_id -> image 记录。
        categories = {c.get('id'): c.get('name') for c in data.get('categories', [])}  # category_id -> 分类名。
        index = {}  # 重建索引。
        for ann in data.get('annotations', []):  # 遍历全部标注。
            name = categories.get(ann.get('category_id'))  # 分类名。
            image = images.get(ann.get('image_id'))  # 所属源图记录。
            if not name or not image:  # 缺分类名或源图。
                continue
            path = os.path.join(folder, image.get('file_name', ''))  # 源图绝对/相对路径。
            index.setdefault(name, []).append((path, ann.get('bbox')))  # 一个分类名可能标注在多张源图上，全部收集。
        self._index = index  # 更新索引缓存。
        self._templates = {}  # 标注变化，裁剪缓存全部失效。
        self._mtime = mtime  # 记录本次加载的 mtime。
        return self._index

    def _get_templates(self, name):  # 取指定分类名的原生尺寸模板列表（首次访问时从源图裁剪并缓存）。
        index = self._ensure_index()  # 确保索引就绪。
        if not index:  # 无标注。
            return []
        if name in self._templates:  # 已裁剪缓存。
            return self._templates[name]
        crops = []  # 裁剪结果。
        image_cache = {}  # 源图读取缓存，避免同名多标注重复读盘。
        for path, bbox in index.get(name, []):  # 遍历该分类名的所有标注。
            if not bbox or len(bbox) < 4:  # bbox 非法。
                continue
            if path not in image_cache:  # 未读过这张源图。
                image_cache[path] = cv2.imread(path)  # 读源图（BGR）。
            whole = image_cache[path]
            if whole is None:  # 源图读取失败。
                continue
            x, y, w, h = (int(round(v)) for v in bbox[:4])  # bbox 取整：[x, y, w, h]。
            x = max(0, x)  # 防越界。
            y = max(0, y)
            crop = whole[y:y + h, x:x + w]  # 按 bbox 裁剪，保持原生尺寸、不缩放。
            if crop.size == 0:  # 裁剪为空。
                continue
            crops.append(crop.copy())  # 拷贝一份，脱离源图内存。
        self._templates[name] = crops  # 缓存裁剪结果（含空列表，避免反复重试）。
        return crops

    def has(self, name):  # 指定分类名是否有可用模板（供流程判断是否跳过未标注的步骤）。
        return len(self._get_templates(name)) > 0

    def find_best(self, frame, name):  # 原生尺度匹配，返回 (best_box_or_None, best_conf)，不判阈值，供诊断/超时日志用。
        """
        Args:
            frame: 全桌面 BGR 截图。
            name: 模板分类名。

        Returns:
            (best_box, best_conf)：best_box 为 (x, y, w, h, confidence) 或 None（无画面/无模板），
            best_conf 为最高分（0~1，未匹配为 0.0），即使低于阈值也返回，便于区分“差一点”还是“根本不在画面”。
        """
        if frame is None:  # 无画面。
            return None, 0.0
        templates = self._get_templates(name)  # 取原生尺寸模板。
        if not templates:  # 该分类名未标注或裁剪失败。
            return None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame  # 灰度匹配，与任务默认一致。
        best = None  # 当前最佳匹配。
        for tpl in templates:  # 逐个模板匹配，取最高分。
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
            th, tw = tpl_gray.shape[:2]  # 模板尺寸。
            if gray.shape[0] < th or gray.shape[1] < tw:  # 模板比帧还大，matchTemplate 会报错，跳过。
                continue
            result = cv2.matchTemplate(gray, tpl_gray, MATCH_METHOD)  # 全帧滑动匹配。
            _, max_val, _, max_loc = cv2.minMaxLoc(result)  # 取最高分位置。
            if best is None or max_val > best[4]:  # 刷新最佳。
                best = (int(max_loc[0]), int(max_loc[1]), tw, th, float(max_val))
        if best is None:  # 没有任何可用模板参与匹配。
            return None, 0.0
        return best, best[4]  # 返回最佳框与最高分（不判阈值）。

    def find(self, frame, name, threshold):  # 在桌面帧上原生尺度匹配模板，达标返回 (x, y, w, h, confidence) 否则 None。
        """
        Args:
            frame: 全桌面 BGR 截图。
            name: 模板分类名。
            threshold: 匹配置信度阈值（0~1），低于阈值视为未匹配。

        Returns:
            (x, y, w, h, confidence) 元组（x/y/w/h 为桌面物理像素），未达标返回 None。
        """
        best, conf = self.find_best(frame, name)  # 取最佳匹配与分数。
        if best is not None and conf >= threshold:  # 达标才返回。
            return best
        return None

    def template_size(self, name):  # 返回该分类名首个原生模板的 (w, h)，供日志确认“未被缩放”；无模板返回 None。
        templates = self._get_templates(name)  # 取模板。
        if not templates:  # 无模板。
            return None
        h, w = templates[0].shape[:2]  # OpenCV 形状为 (高, 宽)。
        return (w, h)
