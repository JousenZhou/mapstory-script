# 地图路线挂机任务：消费"地图"页签（MapTab）产出的全局小地图底图 map.png 与彩色指令路线图 routeN.png，
# 在游戏画面上定位小地图框 → 用实时小地图在全局图上做 SQDIFF 匹配得到角色全局坐标 → 解码路线色点得到
# "左右/上下/动作"三元指令 → 驱动按住方向键/爬梯/跳跃/瞬移沿路线行走；途中攻击范围内有怪则停下打怪，
# 走到 goal 黄点自动切换下一条路线图；全局位置长期停滞触发看门狗脱困。定位与解码算法见 src/map_route.py。
#
# 复用关系：继承 MaplePatrolTask（→ MapleIdleTask）以复用 find_minimap/map_rect/detect_dot、GPU/CPU 怪物匹配、
# 攻击换向、松键兜底与看板共享配置采集；本任务只重写 run() 与新增路线跟随相关配置与辅助方法。
import time  # 定位/瞬移/跳跃/看门狗计时。

import cv2  # 画面标注绘制。
import numpy as np  # 实时小地图裁剪矩阵。

from ok import og  # 全局对象，推送带标注画面供 UI 展示。
from qfluentwidgets import FluentIcon  # 任务图标。

from src.tasks.MaplePatrolTask import MaplePatrolTask, CAPTURE_MIN_INTERVAL  # 复用定位/黄点/攻击/松键能力。
from src import map_store  # 地图资产存取层（meta、map.png、路线图）。
from src.map_route import (  # 路线跟随纯算法。
    build_color_map, nearest_color, combine_cmd, locate_on_global_map, is_near_edge)

MOVE_LEFT_KEY = "left"  # 左方向键。
MOVE_RIGHT_KEY = "right"  # 右方向键。
MOVE_UP_KEY = "up"  # 上方向键：爬梯。
MOVE_DOWN_KEY = "down"  # 下方向键：下梯。
JUMP_MIN_INTERVAL = 0.35  # 连续跳跃指令的最小触发间隔秒数，避免每帧重复触发导致乱跳。
ATTACK_TO_MOVE_WAIT = 0.5  # 攻击结束后恢复移动前的等待秒数，攻击后摇会吞方向键（沿用巡逻时序）。


class MapleRouteTask(MaplePatrolTask):  # 地图路线挂机任务，继承巡逻任务复用其全部底层能力。

    def __init__(self, *args, **kwargs):  # 构造：先复用父类配置，再裁剪巡逻专属键、补充路线跟随专属配置。
        super().__init__(*args, **kwargs)  # 父类已填好角色/怪物/攻击/小地图/黄点等全部配置与帮助文本。
        self.name = "地图路线挂机"  # 任务显示名称。
        self.description = "Follow pre-recorded colored routes on the global minimap: locate the minimap box by template, match the live minimap against the stitched global map (SQDIFF) to get the character's global position, decode the nearest route color pixel into a left/right, up/down, action command, then hold movement / climb / jump / teleport to walk the route; stop and attack monsters that enter attack range, switch to the next route on a goal marker, and a watchdog breaks out when stuck. Map assets and route color tables come from the Map tab; character/monster parameters come from the dashboard. The lie detector is an independent service.  # 任务描述：全局小地图路线跟随。地图与路线资产来自地图页签，角色/怪物参数来自看板；测谎由独立服务值守。"
        self.icon = FluentIcon.GLOBE  # 任务图标。
        self._held_ud_key = None  # 当前持续按住的上下方向键（爬梯/下梯）；加入 pop_held_keys 供测谎服务暂停时释放。
        self._t_last_jump = 0.0  # 上次跳跃时间戳，预置避免未运行 run() 时 apply_cmd 读不到属性。
        self._t_last_tp = 0.0  # 上次瞬移时间戳，预置供 apply_cmd 与主循环共用。
        for key in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent",  # 巡逻往返与卡住脱困由路线跟随+看门狗取代，裁剪之。
                    "Del Key Interval Variance", "Stuck Seconds", "Resume Wait Seconds"):
            self.default_config.pop(key, None)  # 移除默认值，框架加载旧配置时自动清理旧键。
            self.config_description.pop(key, None)  # 同步移除帮助文本。
        for key in ("Map Rect", "Dot Hue Min", "Dot Hue Max", "Dot Min Pixels"):  # 小地图几何与黄点检测与地图资产绑定，改由地图页签的 meta 单一数据源提供，任务页不再重复暴露。
            self.default_config.pop(key, None)  # 移除继承自巡逻父类的重复配置项。
            self.config_description.pop(key, None)  # 同步移除帮助文本。
        self.default_config.update({  # 路线跟随专属配置项（地图名/指令色表等来自地图页签的 meta，不在此重复）。
            "Map Name": "",  # 目标地图目录名；留空表示使用"地图"页签设定的默认地图。
            "Jump Key": "space",  # 跳跃键：路线 jump 指令按下它（需配合当前按住的方向键产生方向跳）。
            "Teleport Key": "",  # 瞬移技能键：留空则把路线/边缘的瞬移指令降级为跳跃。
            "Locate Score Max": 0.5,  # 全局定位 SQDIFF 归一化得分上限（越小越严格），超过视为定位失败保持静止。
            "Route Attack Enabled": True,  # 跟随路线途中遇怪是否停下攻击；关闭则纯跟随不攻击。
            "Teleport To Edge": True,  # 接近平台边缘标记色时用瞬移（无瞬移键则跳跃）回拉，防止坠落。
            "Watchdog Range": 5.0,  # 看门狗位移阈值（全局图像素）：角色移动小于该值视为未移动。
            "Watchdog Seconds": 8.0,  # 按住移动但全局位置长期不动达该秒数则触发脱困，0 禁用。
        })
        self.config_description.update({  # 路线跟随专属配置项帮助文本。
            "Map Name": "Map folder name under maps/ to follow; empty means the default map set in the Map tab. 目标地图目录名（maps/ 下）；留空使用地图页签设定的默认地图。",
            "Jump Key": "Jump key pressed on a route jump command (hold a direction key for a directional jump). 跳跃键：路线 jump 指令按下它，配合按住的方向键形成方向跳。",
            "Teleport Key": "Teleport skill key; leave empty to downgrade teleport commands to a jump. 瞬移技能键：留空则把瞬移指令降级为跳跃。",
            "Locate Score Max": "Max normalized SQDIFF score to trust global localization; higher is looser. 全局定位 SQDIFF 归一化得分上限，越低越严格；超过则视为定位失败保持静止。",
            "Route Attack Enabled": "Whether to stop and attack monsters within range while following the route; off = pure follow. 跟随路线途中是否在攻击范围内停下打怪；关闭则只跟随。",
            "Teleport To Edge": "Teleport (or jump if no teleport key) back when close to a platform edge marker color to avoid falling off. 接近平台边缘标记色时用瞬移（无瞬移键则跳跃）回拉防坠落。",
            "Watchdog Range": "Global-map pixels the character must move to reset the stuck watchdog. 看门狗位移阈值（全局图像素），移动超过该值视为在动。",
            "Watchdog Seconds": "Seconds of no global movement while holding a direction key before a stuck escape; 0 disables. 按住移动但全局位置长期不动达该秒数则脱困，0 禁用。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
        error = super().validate_config(key, value)  # 复用父类（巡逻/挂机）校验。
        if error is not None:  # 父类不通过直接返回。
            return error  # 透传错误。
        if key == "Locate Score Max":  # 定位得分上限须落在归一化 0-1。
            try:  # 解析浮点。
                score = float(value)  # 用户输入。
            except (TypeError, ValueError):  # 非数字。
                return "Locate score must be a number. 定位得分必须是数字。"  # 阻止保存并提示。
            if not 0.0 <= score <= 1.0:  # 越界。
                return "Locate score must be within 0-1. 定位得分必须在 0-1 之间。"  # 阻止保存并提示。
        return None  # 其余不校验。

    def run(self):  # 任务入口：全局小地图路线跟随 + 途中打怪的主循环。
        self.apply_shared_config()  # 从看板读取角色/怪物/攻击共享参数覆盖任务配置（单一数据源）。
        map_name = str(self.config.get("Map Name") or '').strip() or map_store.get_default_map()  # 目标地图名，留空取默认。
        if not map_name:  # 没有任何地图资产。
            self.log_warning("No map selected and no default map. Create one in the Map tab first. 未选择地图且无默认地图，请先在地图页签创建地图。")  # 提示。
            return  # 结束任务。
        meta = map_store.load_meta(map_name)  # 读取该地图配置（指令色表/搜索半径/定位/地图几何/怪物覆盖）。
        map_rect_meta = str(meta.get('Map Rect') or '').strip()  # 地图录制时确定的实际地图区域百分比，非空则覆盖任务配置，保证与录制端定位几何一致。
        if map_rect_meta:  # meta 配了 Map Rect 才写入 self.config，供继承的 map_rect() 读取；留空时尊重已有任务配置。
            self.config['Map Rect'] = map_rect_meta  # 覆盖任务配置的地图区域。
        map_img = map_store.load_map_image(map_name)  # 全局拼接底图 BGR。
        route_files = map_store.list_routes(map_name)  # 全部路线图文件名。
        if map_img is None or not route_files:  # 缺底图或缺路线无法跟随。
            self.log_warning(f"Map '{map_name}' has no map.png or routes. Draw routes in the Map tab first. 地图 {map_name} 缺少底图或路线图，请先在地图页签绘制路线。")  # 提示。
            return  # 结束任务。
        routes = [map_store.load_route_image(map_name, rf) for rf in route_files]  # 预加载全部路线图 BGR。
        routes = [r for r in routes if r is not None]  # 过滤读取失败的。
        if not routes:  # 全部路线图都读不到。
            self.log_warning("All route images failed to load. 全部路线图加载失败，任务退出。")  # 提示。
            return  # 结束任务。
        main_map, ud_map = build_color_map(meta)  # 主/上下指令色查表字典。
        edge_key = str(meta.get('Edge Color') or '').strip()  # 平台边缘标记色（RGB 串），留空禁用边缘保护。
        search_range = int(meta.get('Search Range') or 10)  # 最近色点搜索半径（全局图像素）。
        use_tp_walk = bool(meta.get('Use Teleport To Walk'))  # 行走时也持续瞬移加速（法师向）。
        tp_cd = float(meta.get('Teleport Cooldown') or 1.0)  # 瞬移技能冷却秒数。
        minimap_name = str(meta.get('Minimap Feature') or self.config.get('Minimap Feature') or '').strip()  # 小地图模板名，meta 优先。
        minimap_threshold = float(meta.get('Minimap Threshold') or self.config.get('Minimap Threshold') or 0.8)  # 小地图匹配阈值，meta 优先。
        monster_text = str(meta.get('Monster Features') or '').strip() or str(self.config.get('Monster Features') or '')  # 怪物分类：meta 覆盖，否则用看板共享。
        monster_names = self.parse_monster_names(monster_text)  # 解析逗号分隔怪物分类名。
        frame = self.wait_frame()  # 先取一帧让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示。
            return  # 结束。
        attack_enabled = bool(self.config.get("Route Attack Enabled")) and bool(monster_names)  # 途中打怪需开关开启且配了怪物。
        if not self.feature_ready(minimap_name):  # 小地图模板未标注无法定位。
            self.log_warning(f"Minimap template not ready, annotate it in the Template tab: {minimap_name}. 小地图模板未就绪，请先在模板页标注：{minimap_name}。")  # 提示。
            return  # 结束。
        if attack_enabled:  # 打怪需要角色与怪物标注就绪。
            missing = [name for name in [self.config.get("Character Feature")] + monster_names if not self.feature_ready(name)]  # 检查标注。
            if missing:  # 缺失无法打怪。
                self.log_warning(f"Template not ready for attack: {', '.join(missing)}. 打怪所需模板未就绪：{'、'.join(missing)}。")  # 提示（不阻断纯跟随，见下）。
                attack_enabled = False  # 降级为只跟随不攻击。
                self.log_info("Downgraded to follow-only (no attack). 已降级为只跟随不打怪。")  # 提示。
        jump_key = str(self.config.get("Jump Key") or 'space')  # 跳跃键。
        tp_key = str(self.config.get("Teleport Key") or '')  # 瞬移键，空则瞬移降级跳跃。
        tp_to_edge = bool(self.config.get("Teleport To Edge")) and bool(edge_key)  # 边缘保护是否生效。
        score_max = float(self.config.get("Locate Score Max"))  # 定位可信得分上限。
        wd_range = float(self.config.get("Watchdog Range") or 0)  # 看门狗位移阈值。
        wd_seconds = float(self.config.get("Watchdog Seconds") or 0)  # 看门狗停滞秒数，0 禁用。
        char_threshold = self.config.get("Character Threshold")  # 角色匹配阈值。
        attack_key = self.config.get("Attack Key")  # 常规攻击键。
        melee_key = self.config.get("Melee Attack Key")  # 近战攻击键。
        melee_distance = float(self.config.get("Melee Distance") or 0)  # 近战距离像素。
        attack_x_min, attack_x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 攻击区域左右边界，填反自动交换。
        attack_y_min, attack_y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 上下边界。
        hue_min, hue_max = sorted((int(meta.get('Dot Hue Min') or 18), int(meta.get('Dot Hue Max') or 38)))  # 黄点色相范围，与录制端同源取 meta。
        dot_min_pixels = max(1, int(meta.get('Dot Min Pixels') or 4))  # 黄点最小面积，取 meta。
        frame_interval = float(self.config.get("Frame Interval") or 0)  # 配置帧间隔。
        loop_interval = max(frame_interval, CAPTURE_MIN_INTERVAL)  # 截图节拍固定 30FPS。
        gpu = self.build_gpu_matcher(self.config.get("Character Feature"), monster_names) if attack_enabled else None  # 打怪才建 GPU 匹配器。
        idx_route = 0  # 当前跟随的路线图索引，走到 goal 黄点后循环切换。
        last_cam = None  # 上一帧全局相机左上角，供定位局部搜索加速。
        facing = None  # 攻击换向用：1=右、-1=左、None=未知。
        self._held_move_key = None  # 持续按住的左右移动键（继承自巡逻属性）。
        self._held_attack_key = None  # 持续按住的攻击键。
        self._held_ud_key = None  # 持续按住的上下梯键。
        self._t_last_tp = 0.0  # 上次瞬移时间，配合冷却（实例属性，apply_cmd 与主循环共用一份计时）。
        self._t_last_jump = 0.0  # 上次跳跃时间，防每帧乱跳（实例属性）。
        wd_anchor = None  # 看门狗全局位置锚点 (x, y)。
        wd_anchor_time = time.time()  # 锚点更新时间。
        patrol_resume_at = 0.0  # 攻击结束后允许恢复移动的时点。
        last_diag_time = 0.0  # 诊断日志限频时戳。
        loop_start = time.time()  # 本轮循环起点。
        self.log_info(f"Route task start: map={map_name}, routes={len(routes)}, attack={'on' if attack_enabled else 'off'}. 路线挂机启动：地图 {map_name}，路线 {len(routes)} 条，打怪{'开' if attack_enabled else '关'}。")  # 记录启动信息。
        try:  # 包裹主循环，退出兜底松开全部按住键。
            while True:  # 实时识图循环，直到用户停止。
                wait = loop_interval - (time.time() - loop_start)  # 计算本轮剩余等待钉住 30FPS。
                if wait > 0:  # 未到下一帧时点先等待。
                    self.sleep(wait)  # 补齐帧间隔。
                loop_start = time.time()  # 记录本轮起点。
                frame = self.next_frame()  # 取最新一帧并清除旧帧。
                if frame is None:  # 取不到帧。
                    self.sleep(loop_interval)  # 按节拍拍等待。
                    loop_start = time.time()  # 重置起点。
                    continue  # 下一帧。
                # —— 怪物匹配（打怪开启时）：与巡逻一致，提交朝向/小地图与角色怪物匹配 ——
                batch = None  # CPU 并发批次。
                character, monsters = None, []  # 角色框与怪物框。
                if attack_enabled:  # 打怪开启才做角色/怪物匹配。
                    batch = self._match_frame(frame, minimap_name, minimap_threshold, char_threshold, monster_names, gpu)  # 提交并回收本帧匹配。
                    character = batch[1]  # 角色框。
                    monsters = batch[2]  # 怪物框列表。
                    minimap = batch[0]  # 小地图框。
                else:  # 只跟随：仅定位小地图。
                    minimap = self.find_minimap(minimap_name, frame, minimap_threshold)  # 模板匹配小地图框。
                # —— 全局定位：小地图框 → 实时小地图裁剪 → 与 map.png SQDIFF 匹配 → 角色全局坐标 ——
                rect = self.map_rect(minimap) if minimap is not None else None  # 小地图内的实际地图区域（画面坐标）。
                dot = self.detect_dot(frame, rect, hue_min, hue_max, dot_min_pixels) if rect is not None else None  # 区域内黄点（画面坐标）。
                loc_global, score = None, 1.0  # 角色全局坐标与定位得分。
                if rect is not None and dot is not None:  # 小地图与黄点都在才尝试全局定位。
                    rx, ry, rw, rh = rect  # 地图区域。
                    live = frame[ry:ry + rh, rx:rx + rw]  # 实时小地图内容裁剪。
                    cam, score = locate_on_global_map(map_img, live, last_cam, radius=search_range * 2, score_max=score_max)  # 局部优先命中阈值才直用，否则全局回退取最优。
                    if cam is not None and score <= score_max:  # 得分达标才信任该位置。
                        last_cam = cam  # 记录相机左上角供下帧局部搜索。
                        loc_global = (cam[0] + (dot[0] - rx), cam[1] + (dot[1] - ry))  # 全局坐标 = 相机左上 + 黄点相对地图区域左上。
                    else:  # 定位失败，保留 last_cam 供下帧局部重试但本帧不移动。
                        loc_global = None  # 视为未定位。
                if time.time() - last_diag_time >= 1.0:  # 每秒一次诊断日志。
                    last_diag_time = time.time()  # 记录时戳。
                    self.log_info(f"Locate score={score:.3f} loc_global={loc_global} route_idx={idx_route}. 定位得分={score:.3f}，全局坐标={loc_global}，当前路线={idx_route}。")  # 输出定位质量。
                # —— 途中打怪：攻击范围内有怪则停下攻击（优先于跟随）——
                target = None  # 当前攻击目标。
                if character is not None and monsters:  # 角色与怪物都在。
                    kept = []  # 过滤后怪物。
                    for monster in monsters:  # 逐只剔除压在角色身上的误匹配。
                        if not self.overlapping(character, monster):  # 不与角色重叠才保留。
                            kept.append(monster)  # 保留。
                    monsters = kept  # 替换列表。
                    if monsters:  # 仍有怪物。
                        nearest = min(monsters, key=lambda m: self.center_distance(character, m))  # 最近一只。
                        ndx, ndy = self.center_offset(character, nearest)  # 最近怪相对角色偏移。
                        self.info_set("Distance", f"dx={ndx} dy={ndy}")  # GUI 显示距离。
                        in_range = []  # 攻击区域内怪物。
                        for monster in monsters:  # 逐只判定是否在范围。
                            dx, dy = self.center_offset(character, monster)  # 偏移。
                            if attack_x_min <= dx <= attack_x_max and attack_y_min <= dy <= attack_y_max:  # 落在攻击框内。
                                in_range.append(monster)  # 加入候选。
                        if in_range:  # 有可攻击目标。
                            target = min(in_range, key=lambda m: self.center_distance(character, m))  # 取最近。
                og.my_app.update_vision(self.draw_overlay(frame, minimap, rect, dot, loc_global, target, idx_route, score))  # 推送带标注画面。
                if target is not None:  # 攻击范围内有怪：停下打怪。
                    self._release_move_and_ud()  # 松开移动/上下键，站着打。
                    dx, dy = self.center_offset(character, target)  # 目标方向。
                    want_direction = 1 if dx > 0 else -1  # 1=怪在右，-1=在左。
                    want_key = melee_key if abs(dx) <= melee_distance else attack_key  # 近战/常规切换。
                    if self._held_attack_key is not None and self._held_attack_key != want_key:  # 换攻击键先松旧键。
                        self.send_key_up(self._held_attack_key)  # 松开旧攻击键。
                        self._held_attack_key = None  # 清空。
                    if facing != want_direction:  # 朝向与目标方向不一致则转身。
                        if self._held_attack_key is not None:  # 转身前松攻击键避免被后摇吞。
                            self.send_key_up(self._held_attack_key)  # 松开攻击键。
                            self._held_attack_key = None  # 清空。
                            self.sleep(0.5)  # 等攻击后摇。
                        self.send_key(MOVE_RIGHT_KEY if want_direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单按方向键转身。
                        facing = want_direction  # 记录朝向。
                        self.sleep(0.08)  # 等转身生效。
                    if self._held_attack_key is None:  # 未按住攻击键才按下。
                        self.send_key_down(want_key)  # 持续按住攻击。
                        self._held_attack_key = want_key  # 记录。
                    self.info_set("Status", "攻击中")  # GUI 状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重识别。
                    continue  # 目标消失后自动恢复跟随。
                if self._held_attack_key is not None:  # 目标消失松开攻击键并短暂等待再移动。
                    self.send_key_up(self._held_attack_key)  # 松开攻击键。
                    self._held_attack_key = None  # 清空。
                    patrol_resume_at = time.time() + ATTACK_TO_MOVE_WAIT  # 攻击后摇期暂不移动。
                if time.time() < patrol_resume_at:  # 后摇等待期只识图不动。
                    self.info_set("Status", "恢复跟随中")  # GUI 状态。
                    continue  # 下一帧。
                # —— 路线跟随：解码最近色点并执行三元指令 ——
                if loc_global is None:  # 未定位成功：保持静止等待恢复。
                    self._release_move_and_ud()  # 松开移动/上下键，防止盲走。
                    self.info_set("Status", "未定位(静止)")  # GUI 状态。
                    continue  # 下一帧重试定位。
                route_img = routes[idx_route]  # 当前路线图。
                nearest, nearest_ud = nearest_color(route_img, loc_global, search_range, main_map, ud_map)  # 最近主色点与上下色点。
                move_x, move_y, action = combine_cmd(nearest, nearest_ud)  # 合成三元指令。
                now = time.time()  # 当前时间戳。
                # 边缘保护：接近平台边缘色则瞬移/跳跃回拉（移植参考库 edge_teleport）。
                if tp_to_edge and action not in ("teleport", "goal") and is_near_edge(route_img, loc_global, edge_key) \
                        and now - self._t_last_tp > tp_cd:  # 命中边缘且过冷却。
                    action = "teleport"  # 触发瞬移回拉。
                    self._t_last_tp = now  # 更新瞬移时间。
                # 行走也瞬移加速（法师向，meta 开关）。
                if use_tp_walk and action == "none" and now - self._t_last_tp > tp_cd:  # 纯行走且过冷却。
                    action = "teleport"  # 持续瞬移加速。
                    self._t_last_tp = now  # 更新时间。
                # 无瞬移键时瞬移降级为跳跃。
                if not tp_key and action == "teleport":  # 未配置瞬移键。
                    action = "jump"  # 降级跳跃。
                if action == "goal":  # 走到路线终点。
                    idx_route = (idx_route + 1) % len(routes)  # 切换到下一条路线图（循环）。
                    routes[idx_route] = map_store.load_route_image(map_name, route_files[idx_route]) or routes[idx_route]  # 重载新路线以便绘制修改即时生效。
                    self.log_info(f"Reached goal, switch to route {idx_route} ({route_files[idx_route]}). 到达终点，切换到路线 {idx_route}（{route_files[idx_route]}）。")  # 记录切换。
                    move_x, move_y, action = 'none', 'none', 'none'  # 切换帧不执行动作。
                self.apply_cmd(move_x, move_y, action, jump_key, tp_key, tp_cd)  # 执行指令：按住左右/上下，离散触发跳跃/瞬移（内部更新自持计时）。
                # 看门狗：全局位置长期停滞则脱困。
                wd_anchor, wd_anchor_time = self.update_watchdog(loc_global, wd_anchor, wd_anchor_time, wd_range, wd_seconds)  # 更新卡住判定。
                self.info_set("Status", f"跟随路线{idx_route}")  # GUI 状态。
                self.info_set("Position", f"({loc_global[0]},{loc_global[1]}) s={score:.2f}")  # 显示全局坐标与得分。
                self.sleep(loop_interval)  # 按节拍下帧。
        finally:  # 停止/异常兜底松键，防止按键卡住。
            if self._held_attack_key is not None:  # 残留攻击键。
                self.send_key_up(self._held_attack_key)  # 松开。
            self._release_move_and_ud()  # 松开移动与上下键。

    def _match_frame(self, frame, minimap_name, minimap_threshold, char_threshold, monster_names, gpu):  # 本帧提交并回收小地图/角色/怪物匹配，返回 (小地图框, 角色框, 怪物框列表)，与巡逻一致只走 GPU 或 CPU 一条路径。
        from src.tasks.MapleIdleTask import MatchBatch  # 局部导入并发批次类。
        batch = MatchBatch()  # 新建本帧批次。
        batch.submit("minimap", lambda: self.find_minimap(minimap_name, frame, minimap_threshold))  # 小地图模板匹配定位小地图框。
        cpu_submitted = gpu is None  # 角色/怪物匹配是否已随本批次提交（GPU 可用时改走显卡）。
        if cpu_submitted:  # CPU 路径：角色与全部怪物匹配一并投递并发。
            self.submit_char_monster_matches(batch, frame, self.config.get("Character Feature"), monster_names)  # 提交角色与全部怪物匹配。
        minimap = batch.get("minimap")  # 取回小地图框。
        gm = None  # GPU 帧句柄。
        if gpu is not None:  # GPU 可用。
            try:  # 上传/计算可能因显存等原因异常。
                gm = gpu.match_frame(frame)  # 上传本帧做批量匹配。
            except Exception as e:  # 本帧 GPU 异常。
                self.log_warning(f"GPU match failed at runtime, fall back to CPU: {e}. GPU 匹配运行异常，本帧回退 CPU。")  # 记录。
        if gm is not None:  # GPU 路径：一帧内批量算完角色与全部怪物。
            character = self.gpu_lookup_one(gm, self.config.get("Character Feature"), char_threshold)  # 角色取最高分框。
            monsters = []  # 收集本帧全部怪物匹配框。
            for name in monster_names:  # 逐个怪物分类取全部达标框。
                monsters.extend(self.gpu_lookup_all(gm, name, self.config.get("Monster Threshold"), self.config.get("Monster Mirror Threshold")))  # 原始与镜像合并。
        else:  # CPU 路径。
            if not cpu_submitted:  # 本帧 GPU 刚异常回退，角色/怪物匹配还没提交。
                self.submit_char_monster_matches(batch, frame, self.config.get("Character Feature"), monster_names)  # 补提交。
            character, monsters = self.collect_char_monster_matches(batch, monster_names)  # 取回角色框与全部怪物框。
        return minimap, character, monsters  # 返回三元组。

    def apply_cmd(self, move_x, move_y, action, jump_key, tp_key, tp_cd):  # 执行三元指令：按住左右/上下，离散触发跳跃/瞬移（带冷却）。
        now = time.time()  # 当前时间。
        # —— 左右：单键按住，切换先松旧键 ——
        move_key = MOVE_LEFT_KEY if move_x == 'left' else MOVE_RIGHT_KEY if move_x == 'right' else None  # 需要的移动键。
        if move_key is not None:  # 有左右指令。
            if self._held_move_key != move_key:  # 与当前按住的不同才换键。
                if self._held_move_key is not None:  # 有旧键。
                    self.send_key_up(self._held_move_key)  # 松开旧键。
                self.send_key_down(move_key)  # 按住新键。
                self._held_move_key = move_key  # 记录。
        elif self._held_move_key is not None:  # 左右为 none/stop 且还按着键。
            self.send_key_up(self._held_move_key)  # 松开停止移动。
            self._held_move_key = None  # 清空。
        # —— 上下：爬梯/下梯单键按住 ——
        ud_key = MOVE_UP_KEY if move_y == 'up' else MOVE_DOWN_KEY if move_y == 'down' else None  # 需要的上下键。
        if ud_key is not None:  # 有上下指令。
            if self._held_ud_key != ud_key:  # 换键先松旧。
                if self._held_ud_key is not None:  # 有旧键。
                    self.send_key_up(self._held_ud_key)  # 松开。
                self.send_key_down(ud_key)  # 按住新键。
                self._held_ud_key = ud_key  # 记录。
        elif self._held_ud_key is not None:  # 上下为 none 且还按着。
            self.send_key_up(self._held_ud_key)  # 松开停止爬/下。
            self._held_ud_key = None  # 清空。
        # —— 动作：跳跃/瞬移为离散触发，带最小间隔与冷却，不重复刷屏 ——
        if action == 'jump' and now - self._t_last_jump >= JUMP_MIN_INTERVAL:  # 跳跃且过最小间隔。
            self.send_key(jump_key, down_time=0.05)  # 短按跳跃键（配合按住的方向键形成方向跳）。
            self._t_last_jump = now  # 记录跳跃时间。
        elif action == 'teleport' and tp_key and now - self._t_last_tp >= tp_cd:  # 瞬移且有键、过冷却。
            self.send_key(tp_key, down_time=0.05)  # 短按瞬移键（朝当前按住方向瞬移）。
            self._t_last_tp = now  # 记录瞬移时间。

    def _release_move_and_ud(self):  # 松开当前按住的移动键与上下键（打怪/未定位/退出时保持静止）。
        if self._held_move_key is not None:  # 有按住移动键。
            self.send_key_up(self._held_move_key)  # 松开。
            self._held_move_key = None  # 清空。
        if self._held_ud_key is not None:  # 有按住上下键。
            self.send_key_up(self._held_ud_key)  # 松开。
            self._held_ud_key = None  # 清空。

    def update_watchdog(self, loc_global, anchor, anchor_time, wd_range, wd_seconds):  # 全局位置看门狗：移动超过阈值重置，按住移动但长期停滞则反向脱困。返回 (锚点, 锚点时间)。
        if wd_seconds <= 0 or self._held_move_key is None:  # 禁用或当前没在按住移动键则不判定停滞。
            return loc_global, time.time()  # 以当前位置重置锚点。
        if anchor is None:  # 无历史锚点。
            return loc_global, time.time()  # 建立锚点。
        moved = abs(loc_global[0] - anchor[0]) + abs(loc_global[1] - anchor[1])  # 曼哈顿位移。
        if moved > wd_range:  # 明显移动了。
            return loc_global, time.time()  # 重置锚点与时间。
        if time.time() - anchor_time >= wd_seconds:  # 按住移动却长期停滞=卡住。
            self.info_set("Status", "卡住脱困中")  # GUI 状态。
            self.log_warning("Watchdog: stuck on global map, reverse escape. 看门狗：全局位置停滞，反向脱困。")  # 记录。
            self.send_key_up(self._held_move_key)  # 松开当前移动键。
            back_key = MOVE_LEFT_KEY if self._held_move_key == MOVE_RIGHT_KEY else MOVE_RIGHT_KEY  # 反方向键。
            self._held_move_key = None  # 清空按住状态。
            self.send_key(back_key, down_time=0.5)  # 反向短移离开卡住点。
            self.send_key(str(self.config.get("Jump Key") or 'space'), down_time=0.05)  # 补一跳帮助脱困。
            return None, time.time()  # 清空锚点重新计时。
        return anchor, anchor_time  # 未达停滞时长，保留锚点继续累计。

    def pop_held_keys(self):  # 重写：巡逻版本只报移动+攻击键，这里追加上下梯键，供测谎服务暂停任务时全部释放。
        keys = super().pop_held_keys()  # 取回移动键+攻击键（父类会清空并返回列表）。
        if self._held_ud_key is not None:  # 还按着上下键。
            keys.append(self._held_ud_key)  # 加入待释放。
            self._held_ud_key = None  # 清空。
        return keys  # 返回全部持有键。

    def draw_overlay(self, frame, minimap, rect, dot, loc_global, target, idx_route, score):  # 在画面上绘制定位与跟随标注（重写父类，去掉巡逻边界线，改为全局坐标与指令状态）。
        canvas = frame.copy()  # 复制帧避免污染原图。
        if target is not None:  # 有攻击目标时高亮。
            self.draw_target(canvas, target, (255, 255, 255), "TARGET", cross=False)  # 白色框。
        if minimap is not None:  # 小地图框。
            cv2.rectangle(canvas, (minimap.x, minimap.y), (minimap.x + minimap.width, minimap.y + minimap.height), (0, 255, 0), 2)  # 绿框。
            self.draw_text(canvas, "MINIMAP", (minimap.x, max(minimap.y - 6, 14)), (0, 255, 0))  # 上方标签。
        if rect is not None:  # 实际地图区域。
            rx, ry, rw, rh = rect  # 区域。
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)  # 青色框。
        if dot is not None:  # 黄点。
            cv2.circle(canvas, (dot[0], dot[1]), 6, (0, 255, 255), 2)  # 黄色圆圈。
        status = f"route{idx_route} score={score:.2f}"  # 状态文本。
        if loc_global is not None:  # 已定位。
            status += f" @({loc_global[0]},{loc_global[1]})"  # 追加全局坐标。
        else:  # 未定位。
            status += " LOST"  # 标记定位丢失。
        self.draw_text(canvas, status, (10, 20), (0, 0, 255) if loc_global is None else (0, 255, 0))  # 左上角状态，未定位红色。
        return canvas  # 返回标注画面。
