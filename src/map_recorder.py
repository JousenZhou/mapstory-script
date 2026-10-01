# 路线录制工具算法层 + 后台录制线程：把"边走边看实时小地图"的导航过程反向产出为
# MapleRouteTask 可直接消费的地图资产（全局拼接底图 map.png + 彩色指令路线图 routeN.png）。
#
# 核心思想（与消费端 src/map_route.py / MapleRouteTask 完全对称，保证录出来的图一定能被定位）：
#   1. 模板匹配（Minimap Feature）在游戏画面里定位小地图框 → 按 Map Rect 百分比裁出实时小地图内容 live；
#   2. HSV 取色在小地图内容里检测角色黄点 dot（画面坐标）；
#   3. 复用消费端同一个 locate_on_global_map(map.png, live, last_cam) 得到相机左上角 cam —— 由于用的是同一个
#      SQDIFF 函数，录制端与回放端对"cam 的定义"天然一致；
#   4. 把 live 原样贴回全局画布的 cam 处，实现增量拼接（贴的是 1:1 实时小地图像素，故 map.png 与实时小地图同比例）；
#   5. 角色全局坐标 loc = cam + (dot - live左上角)，与消费端 loc_global 公式一模一样；
#      以"当前指令色"把 loc 用线段连笔描到与画布同尺寸的路线图上（通道顺序与 route_canvas 手绘完全一致，
#      经 map_store 的 PNG 无损往返后仍能被 build_color_map/nearest_color 精确反查）。
#   6. 停止时把画布按"已覆盖区域"的最小外接矩形裁剪，map.png 与 routeN.png 同步裁，尺寸一致可直接落盘。
#
# 分层：detect_yellow_dot / compute_map_rect / parse_rect_percent / MapStitcher 均为纯计算，可单测；
#       MapRecorder 是后台守护线程，仅负责按节拍截图 + 调用框架模板匹配 + 驱动 MapStitcher，不碰 Qt。
import threading  # 后台录制守护线程
import time  # 取帧节拍与限频计时

import cv2  # 线段/圆点落笔与 HSV 取色
import numpy as np  # 画布矩阵运算

from ok import Logger  # 线程日志器（不依赖执行器，与测谎服务同源）

from src.map_route import locate_on_global_map  # 复用消费端同款全局定位函数，确保录/放几何一致

logger = Logger.get_logger(__name__)  # 模块日志器

RECORD_FPS = 15  # 录制取帧帧率（比运行时低即可，移动慢；降低截图/匹配压力）
RECORD_INTERVAL = 1.0 / RECORD_FPS  # 每帧节拍秒数
DOT_SAT_MIN = 80  # 黄点 HSV 饱和度下限，滤除灰白噪点（与巡逻任务一致）
DOT_VAL_MIN = 80  # 黄点 HSV 亮度下限，滤除暗色噪点（与巡逻任务一致）
LOCATE_SCORE_MAX = 0.6  # 录制期定位可接受的最大 SQDIFF 得分（比运行期宽松，因底图尚不完整）


def parse_rect_percent(text):  # 解析 Map Rect 文本 "x,y,w,h"（相对小地图框的百分比）为四元浮点组，非法抛 ValueError，空返回 None。
    raw = str(text or '').strip()  # 去空白
    if not raw:  # 空串表示整块小地图框
        return None  # 供调用方按"整个框"处理
    parts = raw.replace('%', '').replace(';', ',').split(',')  # 去百分号统一分隔符后切分
    if len(parts) != 4:  # 必须恰好四个数
        raise ValueError("Map Rect needs 4 values")  # 数量不对
    values = [float(p.strip()) for p in parts]  # 逐个转浮点，非数字自动抛 ValueError
    if any(v < 0 or v > 100 for v in values):  # 百分比越界
        raise ValueError("Map Rect value out of range")  # 非法
    return tuple(values)  # 返回 (x, y, w, h) 百分比


def compute_map_rect(box_x, box_y, box_w, box_h, map_rect_text):  # 由小地图框与 Map Rect 百分比算出实际地图区域（画面坐标 (rx,ry,rw,rh)），留空取整个框。
    parsed = parse_rect_percent(map_rect_text)  # 解析百分比
    if parsed is None:  # 未配置区域：整框即地图
        return box_x, box_y, box_w, box_h  # 直接返回小地图框
    px, py, pw, ph = parsed  # 四个百分比
    rx = box_x + int(box_w * px / 100)  # 区域左上角画面 x
    ry = box_y + int(box_h * py / 100)  # 区域左上角画面 y
    rw = max(1, int(box_w * pw / 100))  # 区域宽（至少 1 像素）
    rh = max(1, int(box_h * ph / 100))  # 区域高（至少 1 像素）
    return rx, ry, rw, rh  # 返回实际地图区域


def detect_yellow_dot(frame, rect, hue_min, hue_max, min_pixels):  # 在地图区域内检测角色黄点，返回画面坐标 (x, y, 面积) 或 None（与巡逻任务同款 HSV+轮廓逻辑）。
    x, y, w, h = rect  # 地图区域
    if w <= 0 or h <= 0:  # 非法区域
        return None  # 无法检测
    roi = frame[y:y + h, x:x + w]  # 裁剪地图区域画面
    if roi.size == 0:  # 区域越界为空
        return None  # 无法检测
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转 HSV 按色相取色
    lower = np.array([hue_min, DOT_SAT_MIN, DOT_VAL_MIN], dtype=np.uint8)  # 取色下限
    upper = np.array([hue_max, 255, 255], dtype=np.uint8)  # 取色上限
    mask = cv2.inRange(hsv, lower, upper)  # 黄点掩码
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算去孤立噪点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取外轮廓
    candidates = []  # 达标候选块
    for contour in contours:  # 逐个轮廓求面积与重心
        area = cv2.contourArea(contour)  # 轮廓面积
        if area < min_pixels:  # 太小不可能是角色黄点
            continue  # 丢弃
        moments = cv2.moments(contour)  # 几何矩求重心
        if moments["m00"] == 0:  # 面积矩为 0 无法求重心
            continue  # 丢弃
        cx = int(moments["m10"] / moments["m00"]) + x  # 重心换算画面坐标
        cy = int(moments["m01"] / moments["m00"]) + y  # 重心换算画面坐标
        candidates.append((cx, cy, area))  # 加入候选
    if not candidates:  # 区域内无黄点
        return None  # 未检测到
    return max(candidates, key=lambda c: c[2])  # 小地图内通常只有角色一个黄点，直接采信面积最大块


class MapStitcher:  # 增量拼接器：维护全局底图画布 + 同尺寸路线图，按定位得到的 cam 贴小地图、描指令轨迹，停止时按覆盖区外接矩形裁成最终资产。
    # 坐标约定与消费端严格对齐：map 画布存 BGR 实时小地图像素；route 画布以 ndarray 通道顺序写指令色三元组，
    # 经 map_store 的 PNG 无损往返后 pixel_key 仍能反查 build_color_map 的键（与 route_canvas 手绘同一条通道约定）。

    def __init__(self, canvas_w, canvas_h, crop_w, crop_h, match_radius=128):  # 画布尺寸（预分配上限）与实时小地图裁剪尺寸、定位局部搜索半径。
        self.map = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # 全局底图画布（BGR，黑=尚未覆盖）
        self.route = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # 指令路线画布，与底图同尺寸
        self.covered = np.zeros((canvas_h, canvas_w), dtype=bool)  # 已覆盖像素掩码，供停止时裁剪到内容外接矩形
        self.crop_w = int(crop_w)  # 单帧小地图内容宽
        self.crop_h = int(crop_h)  # 单帧小地图内容高
        self.match_radius = max(8, int(match_radius))  # 定位局部搜索半径
        self.cam = None  # 上一帧相机左上角（画布坐标），None 表示尚未种下第一块
        self.prev_point = None  # 上一落点（画布坐标），用于把离散落点连成线段
        self.pasted = 0  # 已贴帧计数（诊断用）
        self.located = 0  # 成功定位并落点计数（诊断用）
        self.lost = 0  # 定位失败/丢帧计数（诊断用）

    def locate_cam(self, live):  # 用实时小地图在已拼接底图上定位相机左上角；首帧直接种在画布中心，返回 (cam(x,y) 或 None, score)。
        if live is None or live.size == 0:  # 无裁剪内容
            return None, 1.0  # 定位失败
        if live.shape[0] != self.crop_h or live.shape[1] != self.crop_w:  # 小地图框尺寸与首帧不符（窗口被缩放），拒绝避免错位拼接
            return None, 1.0  # 视为本帧不可用
        if self.cam is None:  # 首帧：把 live 种在画布中心，作为坐标系原点，之后向四周生长
            cx = max(0, (self.map.shape[1] - self.crop_w) // 2)  # 中心 x
            cy = max(0, (self.map.shape[0] - self.crop_h) // 2)  # 中心 y
            return (cx, cy), 0.0  # 首帧得分视为完美（自匹配）
        cam, score = locate_on_global_map(self.map, live, self.cam, radius=self.match_radius, score_max=LOCATE_SCORE_MAX)  # 复用消费端同款定位
        if cam is None or score > LOCATE_SCORE_MAX:  # 底图不完整或匹配失败：本帧丢弃，保留旧 cam 供下帧重试
            return None, score  # 未定位
        return cam, score  # 定位成功

    def place_and_paste(self, cam, live):  # 把 live 原样贴回底图 cam 处（1:1 同比例），并标记覆盖区；cam 会被夹到画布边界内。
        if cam is None or live is None:  # 无定位结果
            return None  # 不贴
        x, y = cam  # 相机左上角
        x = max(0, min(self.map.shape[1] - self.crop_w, int(x)))  # 横向夹到界内
        y = max(0, min(self.map.shape[0] - self.crop_h, int(y)))  # 纵向夹到界内
        self.map[y:y + self.crop_h, x:x + self.crop_w] = live  # 贴小地图内容（通道顺序原样保留）
        self.covered[y:y + self.crop_h, x:x + self.crop_w] = True  # 标记已覆盖
        self.cam = (x, y)  # 记录相机供下帧局部搜索
        self.pasted += 1  # 计数
        return (x, y)  # 返回实际贴合坐标

    def trace(self, loc, color, thickness=2):  # 以指令色 color=(r,g,b)（ndarray 通道顺序）在路线图把全局落点 loc 连笔描下，与上一落点用线段相连。
        if loc is None:  # 无落点
            return  # 不描
        x, y = int(loc[0]), int(loc[1])  # 全局落点
        if not (0 <= x < self.route.shape[1] and 0 <= y < self.route.shape[0]):  # 越界（画出预分配画布）
            self.lost += 1  # 计入丢失
            return  # 跳过该点，避免越界写入异常
        col = (int(color[0]), int(color[1]), int(color[2]))  # 指令色三元组（直接写通道，与 route_canvas 同约定）
        radius = max(0, thickness // 2)  # 落笔半径
        if self.prev_point is not None:  # 已有上一点：连线段保证快速移动不断点
            cv2.line(self.route, self.prev_point, (x, y), col, max(1, thickness), cv2.LINE_8)  # 粗线段
        else:  # 首点画圆点
            cv2.circle(self.route, (x, y), radius, col, -1, cv2.LINE_8)  # 实心点
        self.prev_point = (x, y)  # 更新上一点
        self.located += 1  # 成功落点计数

    def finalize(self):  # 按已覆盖区最小外接矩形裁出最终 (map, route)，两图同步裁剪保证尺寸一致；无覆盖内容时返回 (None, None)。
        if not self.covered.any():  # 从没贴过任何帧
            return None, None  # 无资产
        ys, xs = np.where(self.covered)  # 覆盖区所有行列坐标
        y0, y1 = int(ys.min()), int(ys.max()) + 1  # 纵向外接（含右端点）
        x0, x1 = int(xs.min()), int(xs.max()) + 1  # 横向外接
        cropped_map = self.map[y0:y1, x0:x1].copy()  # 裁底图
        cropped_route = self.route[y0:y1, x0:x1].copy()  # 裁路线图（同矩形，尺寸必然一致）
        return cropped_map, cropped_route  # 返回最终资产


class MapRecorder:  # 后台录制线程：按节拍截图 + 框架模板匹配定位小地图 + 驱动 MapStitcher 拼接描线，停止后落盘地图资产。
    # 线程模型完全对齐 LieDetectorService：截图 device_manager.capture_method.get_frame（内部加锁、线程安全），
    # 模板匹配 executor.feature_set.find_feature（自带锁），只读当前选择色通过回调原子获取，不触碰任何 Qt 对象。

    def __init__(self, map_name, route_file, meta, color_getter, on_finished=None):  # 目标地图名、目标路线图文件名、地图 meta（定位参数）、当前指令色取用回调、完成回调。
        self.map_name = map_name  # 录制写入的地图目录名
        self.route_file = route_file  # 录制写入的路线图文件名
        self.meta = dict(meta or {})  # 定位参数快照（Minimap Feature/Threshold/Map Rect/色相等）
        self._color_getter = color_getter  # 无参回调，返回当前指令色三元组 (r,g,b)
        self._on_finished = on_finished  # 可选：录制结束时以 (map, route) 回调（在工作线程调用，勿直接碰 Qt）
        self._thread = None  # 工作线程句柄
        self._stop_event = threading.Event()  # 停止请求标志
        self.finished = False  # 线程是否已结束（供 GUI 轮询）
        self.saved = False  # 是否已成功落盘
        self.error = None  # 结束原因/错误信息（供 GUI 展示）
        self.status = 'idle'  # 当前状态文本（idle/locating/lost/saving）
        self.stitcher = None  # 拼接器实例（首帧拿到小地图尺寸后创建）
        self.stats = {'pasted': 0, 'located': 0, 'lost': 0}  # 诊断统计快照

    def start(self):  # 启动录制线程；已启动则忽略。
        if self._thread is not None:  # 已启动
            return  # 幂等
        self._thread = threading.Thread(target=self._run, name='MapRecorder', daemon=True)  # 守护线程随进程退出
        self._thread.start()  # 启动
        logger.info(f'Map recorder started for map={self.map_name} route={self.route_file}. 路线录制已启动：地图 {self.map_name}，路线 {self.route_file}。')  # 记录

    def stop(self):  # 请求停止录制（非阻塞，线程在下一拍检查到停止后收尾落盘）。
        self._stop_event.set()  # 置停止标志

    def wait(self, timeout=5.0):  # 等待录制线程结束（供 GUI 在关闭页签/切换地图前确保已落盘），返回是否在超时内结束。
        if self._thread is None:  # 未启动
            return True  # 视为已结束
        return self._thread.join(timeout) is None  # join 返回 None 表示线程已结束

    # ------------------------------------------------------------------ 工作线程

    def _capture(self):  # 采集一帧游戏画面，失败返回 None（与测谎服务同款取帧路径）。
        from ok import og  # 延迟导入全局对象，避免模块级依赖执行器
        device_manager = getattr(og, 'device_manager', None)  # 取设备管理器
        method = getattr(device_manager, 'capture_method', None) if device_manager is not None else None  # 取截图方法
        if method is None:  # 未选择窗口等
            return None  # 无画面
        try:  # 截图异常不能让线程崩溃
            return method.get_frame()  # 取一帧（内部加锁，线程安全）
        except Exception:  # 截图失败
            return None  # 按无画面处理

    def _match_minimap(self, frame, feature_name, threshold):  # 全屏模板匹配小地图框，返回置信度最高框或 None。
        from ok import og  # 延迟导入取执行器特征集
        executor = getattr(og, 'executor', None)  # 取执行器
        feature_set = getattr(executor, 'feature_set', None) if executor is not None else None  # 取特征集
        if feature_set is None:  # 特征集不可用
            return None  # 无法匹配
        try:  # 标注缺失会抛 ValueError
            boxes = feature_set.find_feature(frame, feature_name, 1, 1, threshold, True, limit=1)  # variance=1 全屏搜索，灰度匹配与任务一致
        except ValueError:  # 未标注该分类
            return None  # 未匹配
        except Exception as e:  # 匹配异常
            logger.warning(f'Map recorder minimap match failed: {e}. 录制小地图匹配失败：{e}。')  # 记录
            return None  # 未匹配
        if not boxes:  # 无达标框
            return None  # 未匹配
        return max(boxes, key=lambda b: getattr(b, 'confidence', 0))  # 取最高分框

    def _run(self):  # 线程主循环：截图→定位小地图→裁 live→检测黄点→定位 cam→贴图画布→算全局坐标→描路线，直到停止后落盘。
        feature_name = str(self.meta.get('Minimap Feature') or '').strip()  # 小地图模板名
        minimap_threshold = float(self.meta.get('Minimap Threshold') or 0.8)  # 小地图匹配阈值
        map_rect_text = str(self.meta.get('Map Rect') or '').strip()  # 实际地图区域百分比
        hue_min = int(self.meta.get('Dot Hue Min') or 18)  # 黄点色相下限
        hue_max = int(self.meta.get('Dot Hue Max') or 38)  # 黄点色相上限
        dot_min = max(1, int(self.meta.get('Dot Min Pixels') or 4))  # 黄点最小面积
        canvas_w, canvas_h = self._parse_canvas_size()  # 预分配画布尺寸
        thickness = max(1, int(self.meta.get('Record Trace Thickness') or 2))  # 描线粗细
        try:  # 包裹主循环，任何异常都要收尾落盘并置 finished
            while not self._stop_event.is_set():  # 录制循环直到停止
                frame = self._capture()  # 采集一帧
                if frame is None:  # 无画面（窗口未就绪）
                    self.status = 'no frame'  # 记录状态
                    self._sleep(RECORD_INTERVAL)  # 短等重试
                    continue  # 下一拍
                box = self._match_minimap(frame, feature_name, minimap_threshold)  # 定位小地图框
                if box is None:  # 没找到小地图
                    self.status = 'minimap lost'  # 记录
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                rx, ry, rw, rh = compute_map_rect(box.x, box.y, box.width, box.height, map_rect_text)  # 实际地图区域
                rx = max(0, rx); ry = max(0, ry)  # 夹到画面内
                rw = min(rw, frame.shape[1] - rx); rh = min(rh, frame.shape[0] - ry)  # 防越界
                if rw <= 0 or rh <= 0:  # 区域退化
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                live = frame[ry:ry + rh, rx:rx + rw]  # 实时小地图内容裁剪
                if self.stitcher is None:  # 首帧确定裁剪尺寸后建画布
                    self.stitcher = MapStitcher(canvas_w, canvas_h, rw, rh)  # 创建拼接器
                dot = detect_yellow_dot(frame, (rx, ry, rw, rh), hue_min, hue_max, dot_min)  # 检测角色黄点（画面坐标）
                cam, score = self.stitcher.locate_cam(live)  # 定位相机左上角
                if cam is None:  # 定位失败
                    self.lost_tick()  # 累计丢失
                    self.status = f'locate lost s={score:.2f}'  # 记录得分供排查
                    self._sleep(RECORD_INTERVAL)  # 短等
                    continue  # 下一拍
                self.stitcher.place_and_paste(cam, live)  # 贴小地图到画布（增量拼接）
                if dot is not None:  # 有黄点才能算全局落点
                    loc = (cam[0] + (dot[0] - rx), cam[1] + (dot[1] - ry))  # 与消费端一模一样的 loc_global 公式
                    color = self._current_color()  # 取当前指令色
                    if color is not None:  # 有有效色才描线（橡皮/无选择色时跳过）
                        self.stitcher.trace(loc, color, thickness)  # 描指令轨迹
                    self.status = f'@({loc[0]},{loc[1]}) s={score:.2f}'  # 记录当前全局坐标
                else:  # 贴了图但没检测到黄点
                    self.status = f'dot lost s={score:.2f}'  # 记录
                self._refresh_stats()  # 更新统计快照
                self._sleep(RECORD_INTERVAL)  # 按节拍下帧
            self._finish_and_save()  # 停止请求：裁剪落盘
        except Exception as e:  # 循环异常兜底
            self.error = str(e)  # 记录错误
            logger.warning(f'Map recorder loop error: {e}. 路线录制循环异常：{e}。')  # 记录
            try:  # 异常也尽量落盘已录内容
                self._finish_and_save()  # 收尾
            except Exception:  # 落盘再失败则忽略
                pass
        finally:  # 无论成败都置 finished 供 GUI 轮询
            self.finished = True  # 标记线程结束

    def _finish_and_save(self):  # 停止后：裁剪最终资产并写入 map_store（覆盖底图与目标路线图），仅在有覆盖内容时落盘。
        if self.stitcher is None:  # 一帧都没录
            self.error = self.error or 'no frames captured'  # 记原因
            return  # 不落盘
        cropped_map, cropped_route = self.stitcher.finalize()  # 裁到内容外接矩形
        if cropped_map is None:  # 无覆盖内容
            self.error = self.error or 'nothing stitched'  # 记原因
            return  # 不落盘
        from src import map_store  # 延迟导入存取层，落盘资产
        map_store.save_map_image(self.map_name, cropped_map)  # 覆盖写 map.png
        map_store.save_route_image(self.map_name, self.route_file, cropped_route)  # 写目标路线图
        self.saved = True  # 标记已落盘
        self._refresh_stats()  # 末次刷新统计
        self.status = f'saved {cropped_map.shape[1]}x{cropped_map.shape[0]}'  # 记录最终尺寸
        logger.info(f'Map recorder saved map={self.map_name} size={cropped_map.shape[1]}x{cropped_map.shape[0]}. 路线录制已保存：地图 {self.map_name}，尺寸 {cropped_map.shape[1]}x{cropped_map.shape[0]}。')  # 记录
        if self._on_finished is not None:  # 回调（在工作线程，调用方勿直接操作 Qt）
            try:  # 回调异常不影响落盘结果
                self._on_finished(cropped_map, cropped_route)  # 通知
            except Exception as e:  # 回调异常
                logger.warning(f'Map recorder on_finished callback error: {e}. 录制完成回调异常：{e}。')  # 记录

    def _current_color(self):  # 通过回调取当前指令色三元组，非法/None 返回 None。
        try:  # 回调可能异常
            color = self._color_getter()  # 取色
        except Exception:  # 回调失败
            return None  # 无色
        if color is None:  # 未选色
            return None  # 跳过
        try:  # 规整为三元整数
            return (int(color[0]), int(color[1]), int(color[2]))  # 三元组
        except (TypeError, ValueError, IndexError):  # 非法色
            return None  # 跳过

    def _parse_canvas_size(self):  # 解析 meta 的预分配画布尺寸 "w,h"，非法回退 1600x1200。
        text = str(self.meta.get('Record Canvas Size') or '').strip()  # 尺寸文本
        try:  # 解析两数
            w, h = [int(p.strip()) for p in text.replace('×', ',').replace('x', ',').split(',')]  # 兼容 x/×分隔
        except (TypeError, ValueError):  # 非法
            w, h = 1600, 1200  # 默认值
        return max(200, w), max(200, h)  # 下限保护，至少能容一块小地图

    def lost_tick(self):  # 定位失败计数（写进 stitcher 若已建，否则忽略）。
        if self.stitcher is not None:  # 已有拼接器
            self.stitcher.lost += 1  # 累加

    def _refresh_stats(self):  # 把拼接器计数同步到对外 stats 快照，供 GUI 轮询展示。
        if self.stitcher is not None:  # 已创建
            self.stats = {'pasted': self.stitcher.pasted, 'located': self.stitcher.located, 'lost': self.stitcher.lost}  # 拷贝计数

    def _sleep(self, seconds):  # 分片短等，随时响应停止事件，避免停止后还卡在整拍。
        end = time.time() + seconds  # 结束时间
        while not self._stop_event.is_set():  # 未停止时循环
            remaining = end - time.time()  # 剩余
            if remaining <= 0:  # 已到点
                return  # 返回
            time.sleep(min(remaining, 0.05))  # 每片最多 0.05 秒，保证及时停止
