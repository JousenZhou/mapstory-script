import time  # 导入标准库 time，用于超时与节奏控制。

import cv2  # 导入 OpenCV，用于镜像模板匹配和实时画面标注绘制。
import numpy as np  # 导入 NumPy，用于镜像匹配的阈值筛选。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from ok.feature.Box import Box  # 导入 Box 类，用于镜像匹配结果的包装。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

from src.tasks.MyBaseTask import MyBaseTask  # 导入项目任务基类，导入时会同时生效标注文件 UTF-8 读取补丁。

MOVE_LEFT_KEY = "left"  # 左方向键：单击用于换方向，转身攻击时短敲用于转身。
MOVE_RIGHT_KEY = "right"  # 右方向键：单击用于换方向，转身攻击时短敲用于转身。


class MapleIdleTask(MyBaseTask):  # 定义冒险岛挂机任务，继承项目基类。

    def __init__(self, *args, **kwargs):  # 构造函数，先初始化父类再设置任务元数据。
        super().__init__(*args, **kwargs)  # 必须先调用父类构造。
        self.name = "Maple Idle"  # 任务显示名称。
        self.description = "Single-spot camping: stay in place, single-tap direction key only to turn, keep attacking the nearest monster within attack range until it dies; also shows live vision."  # 任务描述：单点挂机，不持续移动，仅单击换方向，攻击范围内最近怪直到消失。
        self.icon = FluentIcon.FLAG  # 任务图标。
        self.default_config.update({  # 用户可在 GUI 编辑的配置项。
            "Character Feature": "角色名",  # 角色名：模板页标注的分类名，按分类匹配角色。
            "Monster Features": "小青蛇,绿水灵",  # 怪物：模板页标注的分类名，英文逗号分隔支持多个怪物分类。
            "Attack Range X Min": -150,  # 攻击区域左边界：怪物横向偏移大于等于该值才可攻击，负值=角色左侧。
            "Attack Range X Max": 150,  # 攻击区域右边界：怪物横向偏移小于等于该值才可攻击，正值=角色右侧。
            "Attack Range Y Min": -60,  # 攻击区域上边界：怪物纵向偏移大于等于该值才可攻击，负值=角色上方。
            "Attack Range Y Max": 60,  # 攻击区域下边界：怪物纵向偏移小于等于该值才可攻击，正值=角色下方。
            "Attack Key Left": "a",  # 左侧攻击按键：目标在角色左侧时持续按住该键攻击。
            "Attack Key Right": "b",  # 右侧攻击按键：目标在角色右侧时持续按住该键攻击。
            "Del Key Interval": 0.0,  # 每隔该秒数自动按一下 Del 键，设为 0 表示禁用。
            "Turn Interval": 30.0,  # 每隔该秒数停止全部状态做一次转身攻击：停攻 1 秒→朝朝向反向敲击一下方向键转身→停 1 秒→按住攻击键攻击 1 秒→停 1 秒→再敲击一下方向键转回原朝向，设为 0 表示禁用。
            "Character Threshold": 0.8,  # 角色匹配阈值：越高匹配越严格。
            "Monster Threshold": 0.65,  # 怪物匹配阈值：怪物漏检时可适当调低。
            "Monster Mirror Threshold": 0.65,  # 怪物镜像匹配阈值：怪物转向后精灵图镜像，镜像命中得分通常略低，可单独调低。
            "Use Gray Scale": True,  # 是否转灰度匹配，对颜色差异更稳定。
            "Frame Interval": 0.05,  # 每帧处理之间的最小间隔秒数，控制检测节奏。
        })
        self.config_description.update({  # 各配置项的帮助文本。
            "Character Feature": "Category name annotated for the character in the Template tab. 角色名：在模板页标注的分类名。",
            "Monster Features": "Category names for monsters, separated by English commas. 怪物：模板页标注的分类名，英文逗号隔开支持多个。",
            "Attack Range X Min": "Left boundary of attack zone in signed pixels relative to character, negative=left side. 攻击区域左边界（像素，负值=左侧）。",
            "Attack Range X Max": "Right boundary of attack zone in signed pixels relative to character, positive=right side. 攻击区域右边界（像素，正值=右侧）。",
            "Attack Range Y Min": "Top boundary of attack zone in signed pixels relative to character, negative=above. 攻击区域上边界（像素，负值=上方）。",
            "Attack Range Y Max": "Bottom boundary of attack zone in signed pixels relative to character, positive=below. 攻击区域下边界（像素，正值=下方）。",
            "Attack Key Left": "Key held to attack targets on the left, must exist on the keyboard. 左侧攻击按键：打左边怪用，仅支持键盘存在的按键。",
            "Attack Key Right": "Key held to attack targets on the right, must exist on the keyboard. 右侧攻击按键：打右边怪用，仅支持键盘存在的按键。",
            "Del Key Interval": "Seconds between automatic Del key presses; 0 disables it. 每隔该秒数自动按一下 Del 键，0 禁用。",
            "Turn Interval": "Seconds between turn-attack actions: stop attacking and wait 1s, tap direction key once to turn opposite to facing, wait 1s, hold attack key for 1s, wait 1s, then tap direction key once to turn back; 0 disables it. 每隔该秒数停攻等 1 秒后转身攻击一次（反向敲击转身、攻击 1 秒、再敲击转回），0 禁用。",
            "Character Threshold": "Template match threshold for the character, higher means stricter. 角色匹配阈值，越高越严格。",
            "Monster Threshold": "Template match threshold for monsters; lower it if monsters are missed. 怪物匹配阈值，漏检可调低。",
            "Monster Mirror Threshold": "Threshold for matching horizontally flipped monster templates; flipped sprites usually score a bit lower. 怪物镜像匹配阈值，镜像得分通常略低可单独调。",
            "Use Gray Scale": "Match in grayscale, more robust to color differences. 是否转灰度匹配，对颜色差异更稳定。",
            "Frame Interval": "Minimum seconds between processed frames. 每帧处理之间的最小间隔秒数。",
        })

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
        if key in ("Attack Key Left", "Attack Key Right"):  # 两侧攻击按键都必须是键盘上存在的按键。
            try:  # 用框架自带的按键校验逻辑。
                self.validate_key(value)  # 非法按键会抛出异常。
            except Exception:  # 按键非法时阻止保存并提示。
                return "Attack key must exist on the keyboard. 攻击按键必须是键盘上存在的按键。"
        return None  # 其他配置项不做额外校验。

    def run(self):  # 任务运行入口。
        char_name = self.config.get("Character Feature")  # 读取角色标注分类名。
        monster_names = self.parse_monster_names(self.config.get("Monster Features"))  # 解析逗号分隔的怪物分类名列表。
        frame = self.wait_frame()  # 先取到一帧画面，让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面时无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示取不到画面。
            return  # 直接结束任务。
        if not monster_names:  # 未配置任何怪物分类时无法运行。
            self.log_warning("No monster feature configured. 未配置怪物分类名，任务退出。")  # 提示配置缺失。
            return  # 直接结束任务。
        missing = [name for name in [char_name] + monster_names if not self.feature_ready(name)]  # 检查角色与全部怪物标注是否存在。
        if missing:  # 必要标注缺失时无法运行。
            self.log_warning(f"Template not ready, please annotate in the Template tab: {', '.join(missing)}. 模板未就绪，请先在模板页标注：{'、'.join(missing)}。")  # 提示用户去模板页标注。
            return  # 标注不可用时直接结束任务。
        attack_key_left = self.config.get("Attack Key Left")  # 读取左侧攻击按键，目标在左时按住它。
        attack_key_right = self.config.get("Attack Key Right")  # 读取右侧攻击按键，目标在右时按住它。
        attack_x_min, attack_x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 读取攻击区域左右边界（符号化像素），填反时自动交换。
        attack_y_min, attack_y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 读取攻击区域上下边界（符号化像素），填反时自动交换。
        del_interval = float(self.config.get("Del Key Interval") or 0)  # 读取自动按 Del 键的间隔秒数，0 表示禁用。
        last_del_time = time.time()  # 上次按 Del 键的时间，从任务启动开始计时。
        turn_interval = float(self.config.get("Turn Interval") or 0)  # 读取转身攻击的间隔秒数，0 表示禁用。
        last_turn_time = time.time()  # 上次做转身攻击的时间，从任务启动开始计时。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，只在需要换向时单击方向键。
        held_key = None  # 当前持续按住的攻击键，换侧/换向/目标消失/退出时必须松开它。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        try:  # 包裹主循环，退出时兜底松开持续按住的攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                if del_interval > 0 and time.time() - last_del_time >= del_interval:  # 到达定时间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                if turn_interval > 0 and time.time() - last_turn_time >= turn_interval:  # 到达转身间隔时停止全部状态做一次转身攻击。
                    last_turn_time = time.time()  # 重置转身计时。
                    if held_key is not None:  # 先松开持续按住的攻击键，转身期间不攻击。
                        self.send_key_up(held_key)  # 松开当前攻击键。
                        held_key = None  # 清空按住状态。
                    self.info_set("Status", "Turning")  # 在 GUI 显示转身状态。
                    self.sleep(1.0)  # 停止攻击后等待 1 秒再转身，等攻击后摇结束避免输入被吞。
                    facing_assumed = False  # 本次转身的朝向是否来自假定，假定值不可信需在结束后清空。
                    if facing is None:  # 从未转身过导致朝向未知，先用角色模板匹配探测，失败才假定朝右。
                        facing = self.detect_facing(char_name)  # 原始模板命中=朝右、镜像命中=朝左。
                        if facing is None:  # 探测不到角色时退回假定。
                            facing = 1  # 假定角色当前朝右。
                            facing_assumed = True  # 标记朝向为假定值。
                            self.log_info("Facing unknown before turn attack, assume facing right. 转身攻击前朝向未知，假定角色朝右。")  # 记录朝向假定供排查。
                    turn_key = MOVE_LEFT_KEY if facing == 1 else MOVE_RIGHT_KEY  # 转身按键：当前朝向的反方向。
                    self.log_info(f"Turn attack start: facing={facing} turn_key={turn_key}. 转身攻击开始：当前朝向与转身按键。")  # 记录转身前朝向供排查。
                    self.send_key(turn_key, down_time=0.05)  # 短敲一下方向键只触发转身动画，不产生位移。
                    facing = -facing  # 转身后朝向翻转。
                    self.sleep(1.0)  # 转身后停止 1 秒再攻击，等转身动作生效。
                    turn_attack_key = attack_key_right if facing == 1 else attack_key_left  # 按转身后朝向选对应侧攻击键。
                    self.send_key(turn_attack_key, down_time=1.0)  # 按住攻击键攻击 1 秒，结束自动松开。
                    self.sleep(1.0)  # 攻击结束后停止 1 秒再转回。
                    self.send_key(turn_key, down_time=0.05)  # 短敲一下方向键转回原朝向，该键即原朝向方向。
                    facing = -facing  # 转回后朝向恢复为转身前的原朝向。
                    self.log_info(f"Turn attack done: facing={facing}. 转身攻击结束：朝向已转回。")  # 记录转回后朝向供排查。
                    if facing_assumed:  # 朝向为假定值时序列得到的朝向同样不可信。
                        facing = None  # 清空让下次触发重新探测，避免错误假定被长期沿用。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时短暂等待后重试。
                    self.sleep(self.config.get("Frame Interval"))  # 等待一个帧间隔。
                    continue  # 进入下一帧处理。
                character = self.find_one_feature(char_name, frame, self.config.get("Character Threshold"))  # 用角色标注模板在本帧做匹配，角色用独立阈值。
                monsters = []  # 收集本帧全部怪物匹配框。
                for name in monster_names:  # 逐个怪物分类匹配，支持多个怪物。
                    monsters.extend(self.find_all_features(name, frame, self.config.get("Monster Threshold"), self.config.get("Monster Mirror Threshold")))  # 追加该分类的全部匹配框，怪物与怪物镜像各用独立阈值。
                if character is not None and monsters:  # 角色存在时先剔除压在角色身上的怪物框。
                    kept = []  # 过滤后的怪物框列表。
                    for monster in monsters:  # 逐只检查是否与角色框重叠。
                        if self.overlapping(character, monster):  # 低阈值时怪物模板容易误匹配到角色自身，中心重合就是典型特征。
                            self.log_warning(f"Drop monster box overlapping character: {monster.name} conf={monster.confidence:.2f} x={monster.x} y={monster.y}. 剔除与角色重叠的怪物框，疑似误匹配，建议调高 Template/Mirror Threshold。")  # 记录被剔除的框供排查。
                            continue  # 丢弃该框不参与距离与攻击判定。
                        kept.append(monster)  # 保留正常怪物框。
                    monsters = kept  # 用过滤后的列表替换原列表。
                nearest = None  # 距离角色最近的怪物框。
                if character is not None and monsters:  # 角色与怪物都找到时才计算距离。
                    nearest = min(monsters, key=lambda m: self.center_distance(character, m))  # 选中心点距离最近的怪物。
                    dx, dy = self.center_offset(character, nearest)  # 计算角色到怪物的中心点 xy 距离。
                    self.info_set("Distance", f"dx={dx} dy={dy}")  # 在 GUI 状态区显示 xy 距离。
                elif character is not None:  # 角色在但没匹配到任何怪物。
                    self.info_set("Distance", "-")  # 明确显示无怪，避免残留旧值造成误判。
                if time.time() - last_diag_time >= 1.0:  # 每秒限频输出一次诊断日志。
                    last_diag_time = time.time()  # 记录本次诊断时间。
                    char_desc = f"x={character.x} y={character.y} conf={character.confidence:.2f}" if character is not None else "None"  # 角色框位置与置信度。
                    mob_desc = "; ".join(f"{m.name}@({m.x},{m.y}) conf={m.confidence:.2f}" for m in monsters) or "None"  # 全部怪物框位置与置信度。
                    self.log_info(f"Match diag: CHAR[{char_desc}] MOBS[{mob_desc}] 匹配诊断：角色与怪物框坐标及置信度。")  # 输出诊断日志供排查误匹配。
                target = None  # 当前要攻击的目标怪物。
                if character is not None and monsters:  # 角色与怪物都存在时筛选攻击目标。
                    in_range = []  # 收集攻击区域内的全部怪物。
                    for monster in monsters:  # 逐只怪物检查是否在攻击区域内。
                        dx, dy = self.center_offset(character, monster)  # 计算角色到该怪物的 xy 偏移（符号化）。
                        if attack_x_min <= dx <= attack_x_max and attack_y_min <= dy <= attack_y_max:  # 偏移落在上下左右四条边界围成的区域内。
                            in_range.append(monster)  # 加入可攻击列表。
                    if in_range:  # 区域内有怪物时选定目标。
                        same_side = [m for m in in_range if facing is not None and (1 if self.center_offset(character, m)[0] > 0 else -1) == facing]  # 筛出当前朝向同侧的怪物。
                        pool = same_side if same_side else in_range  # 同侧还有怪就锁定该侧，清完才允许换侧，避免两侧反复转身。
                        target = min(pool, key=lambda m: self.center_distance(character, m))  # 取候选池中最近的一只，一直攻击直到它消失。
                og.my_app.update_vision(self.draw_overlay(frame, character, monsters, nearest, target))  # 把带标注画面推送给 UI 实时展示。
                if target is not None:  # 攻击范围内有怪物时原地攻击。
                    dx, dy = self.center_offset(character, target)  # 计算目标怪物相对角色的方向。
                    direction = 1 if dx > 0 else -1  # 1=怪在右侧，-1=怪在左侧。
                    want_key = attack_key_right if direction == 1 else attack_key_left  # 目标在右用右键，在左用左键。
                    if held_key is not None and held_key != want_key:  # 换目标侧时先松开旧键，避免两键同时按住。
                        self.send_key_up(held_key)  # 松开当前按住的攻击键。
                        held_key = None  # 清空按住状态。
                    if facing != direction:  # 朝向与怪物方向不一致时才单击方向键换向，绝不持续按住造成移动。
                        self.send_key(MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if held_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住对应侧攻击键不放。
                        held_key = want_key  # 记录当前按住的键。
                    self.info_set("Status", "Attacking")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if held_key is not None:  # 目标消失时松开持续按住的攻击键。
                    self.send_key_up(held_key)  # 松开攻击键。
                    held_key = None  # 清空按住状态。
                self.info_set("Status", "Camping" if character is not None else "Character not found")  # 无目标时原地待命，显示当前状态。
                self.sleep(self.config.get("Frame Interval"))  # 等待一个帧间隔后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if held_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(held_key)  # 松开它。
                held_key = None  # 清空按住状态。

    def parse_monster_names(self, text):  # 把逗号分隔的怪物分类名文本解析成列表。
        return [name.strip() for name in str(text or '').split(',') if name.strip()]  # 去掉空白项与首尾空格。

    def draw_overlay(self, frame, character, monsters, nearest, target):  # 在一帧画面上绘制全部标注，返回新画面。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if character is not None:  # 匹配到角色时绘制角色标注。
            self.draw_target(canvas, character, (0, 255, 0), "CHAR" + ("-flip" if getattr(character, "flipped", False) else ""))  # 绿色框+十字延长线，镜像命中时标注 -flip。
            cx, cy = self.box_center(character)  # 攻击范围基准点：角色框中心，与 run() 中攻击判定公式一致。
            x_min, x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 左右边界，填反自动交换，与判定公式一致。
            y_min, y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 上下边界，填反自动交换，与判定公式一致。
            cv2.rectangle(canvas, (int(cx + x_min), int(cy + y_min)), (int(cx + x_max), int(cy + y_max)), (255, 0, 255), 2)  # 紫色矩形框标出攻击范围，怪物中心点落入框内才会被攻击。
            self.draw_text(canvas, "ATK RANGE", (int(cx + x_min), max(int(cy + y_min) - 6, 14)), (255, 0, 255))  # 范围框左上角标注文本。
        for monster in monsters:  # 绘制每只怪物的标注。
            self.draw_target(canvas, monster, (0, 0, 255), "MOB" + ("-flip" if getattr(monster, "flipped", False) else ""))  # 红色框+十字延长线，镜像命中时标注 -flip，方便确认镜像匹配生效。
        if target is not None:  # 存在当前攻击目标时额外高亮。
            self.draw_target(canvas, target, (255, 255, 255), "TARGET", cross=False)  # 白色框标出正在攻击的怪物，消失后不再绘制。
        if character is not None and nearest is not None:  # 角色与最近怪物都存在时绘制距离信息。
            cx, cy = self.box_center(character)  # 角色中心点。
            mx, my = self.box_center(nearest)  # 怪物中心点。
            cv2.line(canvas, (int(cx), int(cy)), (int(mx), int(my)), (255, 255, 0), 1)  # 角色与怪物中心连线。
            dx, dy = self.center_offset(character, nearest)  # 计算 xy 距离。
            self.draw_text(canvas, f"dx={dx} dy={dy}", (int((cx + mx) / 2), int((cy + my) / 2)), (255, 255, 0))  # 在连线中点显示 xy 距离。
        return canvas  # 返回绘制完成的画面。

    def draw_target(self, canvas, box, color, label, cross=True):  # 绘制一个匹配框，可选十字延长线到画面边缘。
        x1, y1 = box.x, box.y  # 匹配框左上角。
        x2, y2 = box.x + box.width, box.y + box.height  # 匹配框右下角。
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)  # 绘制匹配框。
        cx, cy = self.box_center(box)  # 计算匹配框中心点。
        if cross:  # 移动目标需要十字线辅助观察位置。
            cv2.line(canvas, (0, int(cy)), (canvas.shape[1], int(cy)), color, 1)  # 以中心为原点画横向延长线到画面左右边缘。
            cv2.line(canvas, (int(cx), 0), (int(cx), canvas.shape[0]), color, 1)  # 以中心为原点画纵向延长线到画面上下边缘。
        if label:  # 有标签文本时画在框上方。
            self.draw_text(canvas, label, (x1, max(y1 - 6, 14)), color)  # 标签显示在匹配框左上角。

    def draw_text(self, canvas, text, position, color):  # 在画面上绘制带黑色描边的文字，保证可读性。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # 先画黑色描边。
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  # 再画彩色文字。

    def wait_frame(self):  # 等待取到一帧画面，最多等 10 秒。
        for _ in range(100):  # 最多尝试 100 次，每次间隔 0.1 秒。
            frame = self.next_frame()  # 取最新一帧画面。
            if frame is not None:  # 取到画面就返回。
                return frame  # 返回画面矩阵。
            self.sleep(0.1)  # 未取到画面时短暂等待。
        return None  # 超时仍取不到画面返回 None。

    def feature_ready(self, feature_name):  # 检查模板页中是否存在指定分类名的标注。
        try:  # 标注文件损坏或编码异常时避免任务崩溃。
            return self.executor.feature_set.feature_exists(feature_name)  # 查询 FeatureSet 中是否加载到该标注。
        except Exception as e:  # 加载标注文件失败。
            self.log_warning(f"Failed to load template annotations: {e}")  # 记录加载失败日志。
            return False  # 标注不可用。

    def detect_facing(self, char_name):  # 用角色模板匹配探测朝向：原始模板命中视为朝右、镜像命中视为朝左，找不到角色返回 None，假定标注模板为朝右立绘。
        frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
        if frame is None:  # 取不到画面无法探测。
            return None  # 返回未知。
        box = self.find_one_feature(char_name, frame, self.config.get("Character Threshold"))  # 先原始朝向匹配，未命中再用镜像模板匹配。
        if box is None:  # 两种朝向都找不到角色。
            return None  # 返回未知。
        return -1 if getattr(box, "flipped", False) else 1  # 镜像命中=朝左，原始命中=朝右。

    def find_one_feature(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配一个标注模板，返回置信度最高的框或 None，镜像未单独给阈值时沿用主阈值。
        try:  # 标注不存在时框架会抛 ValueError，不能中断主流程。
            box = self.find_one(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1)  # variance=1 表示全屏搜索，目标会移动不能只在标注位置附近找。
        except ValueError:  # 该分类名未在模板页标注。
            return None  # 按未匹配处理。
        if box is not None:  # 原始朝向匹配成功。
            return box  # 直接返回结果。
        return self.find_flipped(feature_name, frame, threshold=mirror_threshold or threshold)  # 目标转向时精灵图会水平镜像，用翻转模板再匹配一次。

    def find_all_features(self, feature_name, frame, threshold, mirror_threshold=None):  # 在一帧画面中匹配标注模板的全部出现位置，镜像未单独给阈值时沿用主阈值。
        boxes = list(self.find_feature(feature_name, frame=frame, threshold=threshold, use_gray_scale=self.config.get("Use Gray Scale"), horizontal_variance=1, vertical_variance=1))  # variance=1 表示全屏搜索，返回全部匹配框。
        boxes.extend(self.find_flipped(feature_name, frame, find_all=True, threshold=mirror_threshold or threshold))  # 补充镜像朝向的匹配结果，目标转向后也能识别。
        return self.merge_boxes(boxes)  # 两种朝向的框合并后统一去重，避免同一目标出重复框。

    def merge_boxes(self, boxes):  # 对重叠的匹配框做非极大值抑制，按置信度从高到低保留。
        merged = []  # 去重后的结果列表。
        for box in sorted(boxes, key=lambda b: -b.confidence):  # 按置信度从高到低遍历。
            if any(self.overlapping(box, kept) for kept in merged):  # 与已保留框重叠视为同一目标。
                continue  # 跳过低分的重复框。
            merged.append(box)  # 保留该框。
        return merged  # 返回去重后的全部匹配框。

    def overlapping(self, a, b):  # 判断两个框中心是否足够近，近则视为同一目标。
        return abs(a.x - b.x) < min(a.width, b.width) / 2 and abs(a.y - b.y) < min(a.height, b.height) / 2  # 左上角差距小于较小框一半尺寸即重叠。

    def find_flipped(self, feature_name, frame, find_all=False, threshold=0.8):  # 用水平镜像的模板在整帧匹配，返回 Box 或 Box 列表，阈值由调用方按目标类型传入。
        try:  # 获取标注模板失败时不能影响主流程。
            feature_set = self.executor.feature_set  # 取执行器的特征集。
            feature_set.ensure_feature(feature_name)  # 确保该标注已加载进缓存。
            feature = feature_set.feature_dict.get(feature_name)  # 从缓存取出特征对象。
        except Exception as e:  # 标注加载异常。
            self.log_warning(f"Failed to load feature for flip match: {e}")  # 记录异常日志。
            return [] if find_all else None  # 异常时返回空结果。
        if feature is None:  # 标注不存在。
            return [] if find_all else None  # 返回空结果。
        template = cv2.flip(feature.mat, 1)  # 水平镜像模板，对应目标反朝向的精灵图。
        search = frame  # 待匹配画面。
        if self.config.get("Use Gray Scale"):  # 配置为灰度匹配时统一转灰度。
            search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)  # 画面转灰度。
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # 模板转灰度。
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)  # 整帧模板匹配。
        if not find_all:  # 只需最佳结果时。
            _, score, _, loc = cv2.minMaxLoc(result)  # 取最高分与位置。
            if score < threshold:  # 最高分低于阈值。
                return None  # 镜像方向也没找到。
            box = Box(loc[0], loc[1], template.shape[1], template.shape[0], confidence=score, name=feature_name)  # 包装成 Box。
            box.flipped = True  # 标记为镜像命中，供画面标注区分朝向。
            return box  # 返回镜像匹配框。
        boxes = []  # 收集全部镜像匹配框。
        positions = np.argwhere(result >= threshold)  # 所有达到阈值的位置（y, x）。
        for y, x in sorted(positions, key=lambda p: -result[p[0], p[1]]):  # 按得分从高到低遍历。
            if any(abs(x - b.x) < template.shape[1] / 2 and abs(y - b.y) < template.shape[0] / 2 for b in boxes):  # 与已选框距离过近视为同一目标。
                continue  # 跳过重复框实现简单非极大值抑制。
            box = Box(int(x), int(y), template.shape[1], template.shape[0], confidence=float(result[y, x]), name=feature_name)  # 构造镜像匹配框。
            box.flipped = True  # 标记为镜像命中，供画面标注区分朝向。
            boxes.append(box)  # 加入结果列表。
        return boxes  # 返回全部镜像匹配框。

    def box_center(self, box):  # 计算匹配框中心点坐标。
        return box.x + box.width / 2, box.y + box.height / 2  # 返回中心点 (x, y)。

    def center_distance(self, a, b):  # 计算两个匹配框中心点的直线距离，用于选最近目标。
        ax, ay = self.box_center(a)  # 第一个框的中心。
        bx, by = self.box_center(b)  # 第二个框的中心。
        return (ax - bx) ** 2 + (by - ay) ** 2  # 返回距离平方，比较大小无需开方。

    def center_offset(self, a, b):  # 计算从框 a 中心到框 b 中心的 xy 偏移（像素）。
        ax, ay = self.box_center(a)  # 第一个框的中心。
        bx, by = self.box_center(b)  # 第二个框的中心。
        return int(round(bx - ax)), int(round(by - ay))  # 返回 (dx, dy)。
