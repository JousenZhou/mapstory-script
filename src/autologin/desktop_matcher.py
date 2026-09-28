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
#
# 显卡加速：全桌面帧很大（多屏时可达 3440x1440），CPU 全屏 matchTemplate 单次就要数十毫秒，
# 而重登流程每 0.5 秒轮询一次、还要在【卡住监控】里同帧连查当前与下一步两个模板。
# 故匹配可改走 CuPy FFT（src/gpu_feature_match.py 的 GpuFeatureMatcher 显式模板路径），
# 一帧只上传一次显存、只做一次帧变换即可覆盖该分类的全部模板，结果与 CPU 路径等价
# （同为灰度 TM_CCOEFF_NORMED + minMaxLoc 取最高分）。显卡加速隐藏式启用（不设看板开关），
# 无 CuPy/无显卡/运行期异常时自动降级下面的 CPU 路径，行为保持一致。
import json  # 解析 COCO 标注文件。
import os  # 路径拼接与标注文件 mtime 检查。
import time  # mtime 检查限频，避免每帧 stat。

import cv2  # 模板匹配与灰度转换。

MATCH_METHOD = cv2.TM_CCOEFF_NORMED  # 归一化相关系数匹配，与框架默认方法一致。
RELOAD_CHECK_INTERVAL = 2.0  # 标注文件 mtime 检查间隔秒数：重新标注后能自动生效，又不必每帧 stat。


class DesktopTemplateMatcher:  # 在原生像素尺度下对全桌面截图做模板匹配（绕开框架按帧宽缩放）。

    def __init__(self, coco_json, logger=None, gpu_enabled=True):  # 构造匹配器。
        """
        Args:
            coco_json: coco_annotations.json 路径（与框架 template_matching.coco_feature_json 一致）。
            logger: 日志器，None 时静默。
            gpu_enabled: 是否优先走显卡（CuPy FFT），默认 True；显卡加速已改为隐藏式启用，该参数仅供测试构造纯 CPU 参考实现。
        """
        self._coco_json = coco_json  # 标注文件路径。
        self._logger = logger  # 日志器。
        self._index = None  # 分类名 -> [(源图路径, bbox), ...] 索引，解析一次后缓存。
        self._templates = {}  # 分类名 -> [原生尺寸 BGR 模板, ...]，按需裁剪缓存。
        self._mtime = None  # 上次加载时标注文件的 mtime，变化则失效缓存重新解析。
        self._last_check = 0.0  # 上次 mtime 检查时间戳，用于限频。
        self._gpu_enabled = bool(gpu_enabled)  # 显卡加速开关（内部）：默认开启，仅测试可显式关闭构造纯 CPU 参考实现。
        self._gpu = None  # 显卡匹配器（GpuFeatureMatcher），首次需要时创建。
        self._gpu_off = False  # 显卡运行期异常后的永久关闭标志，本次运行内不再重试，避免每帧失败刷日志。

    def _warn(self, message):  # 输出警告日志（有日志器时）。
        if self._logger is not None:
            self._logger.warning(message)

    def _info(self, message):  # 输出信息日志（有日志器时）。
        if self._logger is not None:
            self._logger.info(message)

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

    def fingerprint(self):  # 标注文件指纹：重新标注后 mtime 变化，显卡匹配器据此整体重建模板。
        self._ensure_index()  # 确保已解析过标注（首次调用才真正 stat 与读盘）。
        return self._mtime  # 返回上次加载时的标注文件 mtime，None 表示标注不可读。

    def _gpu_ready(self):  # 显卡加速是否可用：未被运行期异常降级、且本机有可用显卡。
        if not self._gpu_enabled or self._gpu_off:  # 被显式关闭（仅测试），或已被运行期异常永久降级。
            return False  # 走 CPU。
        try:  # 显卡探测与匹配器创建都可能因驱动/显存异常失败。
            from src.gpu_feature_match import GpuFeatureMatcher, gpu_match_available  # 延迟导入：没装 CuPy 时不影响 CPU 路径。
            if not gpu_match_available():  # 无 CuPy 或无可用 NVIDIA 显卡。
                return False  # 走 CPU。
            if self._gpu is None:  # 首次使用，创建显卡匹配器。
                self._gpu = GpuFeatureMatcher(gray=True, logger=self._logger)  # 显式模板路径不绑定框架特征集，模板由本匹配器裁剪后给出。
                self._info("Desktop matcher GPU template match enabled. 全桌面重登匹配已启用显卡模板匹配加速。")  # 记录一次，便于确认确实走了显卡。
            return True  # 可以走显卡。
        except Exception as e:  # 导入或创建失败。
            self._disable_gpu(e)  # 永久关闭加速。
            return False  # 走 CPU。

    def _disable_gpu(self, error):  # 显卡匹配异常后永久关闭加速：本次运行内不再重试，避免每次匹配都失败刷日志。
        if not self._gpu_off:  # 首次失败才记日志。
            self._warn(f"Desktop matcher GPU match failed, fallback to CPU: {error}. 全桌面显卡匹配异常，已回退 CPU 原生匹配。")
        self._gpu_off = True  # 置永久关闭标志。
        self._gpu = None  # 释放显卡匹配器与其显存缓存。

    def _gpu_find(self, frame, name):  # 显卡匹配一次；返回 (best_box_or_None, best_conf)，显卡不可用/异常时返回 None 表示需回退 CPU。
        if not self._gpu_ready():  # 显卡不可用（开关关闭/无显卡/已异常降级）。
            return None  # 交给 CPU 路径。
        try:  # 显卡调用全程包裹异常：任何失败都降级到 CPU，重登流程不能因显卡中断。
            templates = self._get_templates(name)  # 取该分类的原生尺寸模板，与 CPU 路径同一份裁剪结果。
            if not self._gpu.prepare_templates([(name, templates)], frame, self.fingerprint()):  # 注册模板并上传显存（形状未变时直接复用）。
                return None  # 该分类无可显卡化模板，交给 CPU 判定。
            result = self._gpu.best(self._gpu.frame(frame), name)  # 上传本帧并取最高分：一帧一次变换覆盖该分类全部模板。
            if result is None:  # 模板都比画面大等无结果情形。
                return None, 0.0  # 与 CPU 路径一致：无框、0 分。
            return result, result[4]  # 返回五元组与最高分（不判阈值，与 find_best 契约一致）。
        except Exception as e:  # 显卡运行期异常（显存不足/驱动重置等）。
            self._disable_gpu(e)  # 永久关闭加速。
            return None  # 本帧交给 CPU，结果仍然正确。

    def find_best(self, frame, name):  # 原生尺度匹配，返回 (best_box_or_None, best_conf)，不判阈值，供诊断/超时日志用。
        """
        Args:
            frame: 全桌面 BGR 截图。
            name: 模板分类名。

        Returns:
            (best_box, best_conf)：best_box 为 (x, y, w, h, confidence) 或 None（无画面/无模板），
            best_conf 为最高分（0~1，未匹配为 0.0），即使低于阈值也返回，便于区分“差一点”还是“根本不在画面”。

        Note:
            唯一已知的路径差异：标注成纯色块时模板零方差，CPU 的得分图会退化成整张常数并可能误报在 (0,0)，
            显卡路径把分母退化的位置一律记 0 分因而不会去点 (0,0)。这类标注本身就是无效的，显卡的行为更安全。
        """
        if frame is None:  # 无画面。
            return None, 0.0
        templates = self._get_templates(name)  # 取原生尺寸模板。
        if not templates:  # 该分类名未标注或裁剪失败。
            return None, 0.0
        gpu_result = self._gpu_find(frame, name)  # 先试显卡：结果与下面 CPU 路径等价（同为灰度 TM_CCOEFF_NORMED 取最高分）。
        if gpu_result is not None:  # 显卡给出了结果（含「未匹配」），直接返回，不再跑 CPU。
            return gpu_result
        if frame.ndim == 3 and frame.shape[2] == 4:  # 框架 WGC 截图可能带 alpha，先裁掉再转灰度（BGR2GRAY 不接受四通道）。
            frame = frame[:, :, :3]  # 与显卡路径的 _to_bgr 处理保持一致，两条路径结果才可比。
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
