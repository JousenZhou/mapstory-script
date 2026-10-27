import time  # 导入标准库 time，用于折返暂停、归位滞回、卡住计时与攻击节奏。

import cv2  # 导入 OpenCV，用于画面标注绘制（地台框、定点色框、巡逻区间边界线）。

from ok import og  # 导入全局对象，用于把带标注画面推送给 UI 实时展示。
from ok.feature.Box import Box  # 导入 Box，用于构造地台局域匹配的搜索区域。
from qfluentwidgets import FluentIcon  # 导入 Fluent 图标，用于任务在 GUI 中显示图标。

# 导入巡逻任务复用角色/怪物匹配、朝向校准、卡住脱困与攻击辅助能力；本任务不用小地图定位，
# 角色位置改为「地台模板匹配 + 角色模板匹配」同帧换算：镜头平移对两者同时生效，相对位置不变，天然抗镜头跟随。
# 测谎监控与自动解题由独立服务 src/liedetector/service.py 值守，任务只需通过 pop_held_keys 上报持有键供服务暂停时释放。
from src.tasks.MaplePatrolTask import (MaplePatrolTask, MatchBatch, CAPTURE_MIN_INTERVAL,
                                       MOVE_LEFT_KEY, MOVE_RIGHT_KEY)

PLATFORM_SEARCH_MARGIN = 0.3  # 地台局域搜索外扩比例（相对上一帧地台框长边），兼顾镜头小幅平移与抑制重复贴图远处错配。
PLATFORM_FREEZE_MISSING = 10  # 地台连续匹配失败达该帧数后松开移动键原地冻结，防止定位盲走坠落；短时丢失沿用上一帧位置。
PLATFORM_EXPECT_MARGIN = 0.2  # 首帧在标注期望区域（按当前画面尺寸缩放）四周外扩的比例，超差再回退全屏搜索。
BOUND_TURN_PAUSE = 0.2  # 巡逻到达区间边界折返时松键停顿的秒数，避免撞边界继续走。
TARGET_LOSS_BUFFER = 0.3  # 目标丢失缓冲默认秒数（已提升为看板角色配置共享键 Target Loss Buffer Seconds，任务经 MyBaseTask.target_loss_buffer() 读取，此常量仅作历史参照）。
SPOT_BOX_FALLBACK_WIDTH = 40  # 未匹配到角色时定点色框的兜底宽度（像素）。
SPOT_BOX_FALLBACK_HEIGHT = 60  # 未匹配到角色时定点色框的兜底高度（像素）。
JUMP_MIN_INTERVAL = 0.35  # 攻击中跳跃最小间隔（秒）：配置低于它按它执行防每帧连跳（与路线任务 JUMP_MIN_INTERVAL 一致）。


class MaplePlatformTask(MaplePatrolTask):  # 定义冒险岛地台定点挂机任务，继承巡逻任务复用匹配与攻击能力。

    def __init__(self, *args, **kwargs):  # 构造函数，先复用父类全部配置再按地台挂机场景裁剪。
        super().__init__(*args, **kwargs)  # 父类构造已填好小地图、黄点、巡逻等全部配置项与帮助文本。
        self.name = "地台定点巡逻挂机"  # 任务显示名称。
        self.description = "Platform camping with spot patrol: pick a template annotated under category '地台'; each frame locate the platform on screen by template matching and the character by its own template, then compare the character's screen x with the patrol band from spot x minus Patrol Left Pixels to spot x plus Patrol Right Pixels (screen pixels); patrol left/right inside the band and attack monsters entering the attack range; when the character is pushed beyond the band edge by more than Band Exit Margin Pixels (e.g. by monster collision) walking back to the spot takes top priority before patrolling again, while merely reaching an edge only turns the patrol around; the live vision marks the platform box, the spot with a green box, the band with a colored frame and boundary lines, and the exit margin with blue wireframes. Character/monster parameters come from the dashboard config; the lie detector is handled by an independent service, decoupled from this task.  # 任务描述：地台定点挂机，选择类别为「地台」的标注；每帧模板匹配地台与角色，用角色画面 x 与巡逻区间比较：区间 = 定点画面x − 左偏移px ~ 定点画面x + 右偏移px；区间内左右巡逻打怪；边界折返；被怪撞开越过边界再多 Band Exit Margin Pixels 才算真脱区，优先走回定点再恢复巡逻；叠加层标出地台框、定点色框、区间色框与蓝色脱区余量线框。角色/怪物参数统一从看板采集，测谎由独立服务值守。"
        self.icon = FluentIcon.PIN  # 任务图标。
        for key in ("Patrol Enabled", "Patrol Left Percent", "Patrol Right Percent",  # 小地图巡逻体系整体由地台区间取代；本任务不用小地图定位，裁剪对应配置。
                    "Minimap Feature", "Minimap Threshold", "Map Rect",
                    "Dot Hue Min", "Dot Hue Max", "Dot Min Pixels"):
            self.default_config.pop(key, None)  # 移除默认值，框架加载旧配置时会自动清理已保存的旧键。
            self.config_description.pop(key, None)  # 同步移除帮助文本。
        self.default_config.update({  # 地台挂机专属配置项，父类没有需自行补充。
            "Platform Feature": "",  # 地台标注：模板页类别为「地台」的标注分类名，看板任务卡以下拉框选择。
            "Spot Percent": 50.0,  # 定点：地台左→右的横向百分比，50 表示地台正中间。
            "Patrol Left Pixels": 100,  # 巡逻左偏移（像素）：区间左端 = 定点画面 x − 该值，按当前分辨率画面计。
            "Patrol Right Pixels": 100,  # 巡逻右偏移（像素）：区间右端 = 定点画面 x + 该值，与左偏移可不对称。
            "Platform Threshold": 0.7,  # 地台模板匹配阈值：越高越严格。
            "Band Exit Margin Pixels": 100,  # 脱区触发余量（像素）：越过区间边界再多该值才算真脱区触发归位；折返惯性冲过头在余量内只继续折返。
            "Home Tolerance Pixels": 10,  # 归位容差（像素）：偏移回落到定点该像素以内才算归位完成（滞回，防边界抖动）。
        })
        self.config_description.update({  # 地台挂机专属配置项的帮助文本。
            "Platform Feature": "Category name annotated under supercategory '地台' in the Template tab; the character camps on this platform. 地台标注：模板页类别为「地台」的标注分类名，角色在选中的地台上定点挂机。",
            "Spot Percent": "Home spot as percent along the platform width from left to right, 50 means the platform center; drawn as a green box in the live vision. 定点（百分比）：地台左→右的横向百分比，50 为地台中间，叠加层用绿色色框标出。",
            "Patrol Left Pixels": "Screen pixels from the spot to the band's left edge; allowed band = spot x − this ~ spot x + Patrol Right Pixels; leave the band triggers a top-priority return to the spot. 巡逻左偏移（像素）：区间左端 = 定点画面 x − 该值；脱离区间优先走回定点，叠加层用色框标出区间。",
            "Patrol Right Pixels": "Screen pixels from the spot to the band's right edge; can differ from the left offset. 巡逻右偏移（像素）：区间右端 = 定点画面 x + 该值，与左偏移可不对称。",
            "Platform Threshold": "Template match threshold for the platform, higher means stricter. 地台模板匹配阈值，越高越严格。",
            "Band Exit Margin Pixels": "Extra screen pixels beyond the band edge before the return-to-spot kicks in; reaching an edge only turns the patrol around, and turn-around inertia within this margin is ignored, so only a real push by a monster triggers homing; drawn as blue wireframes outside the band. 脱区触发余量（像素）：越过区间边界再多该值才触发归位；走到边界只折返不归位，折返惯性冲过头在余量内忽略，叠加层用蓝色线框标出余量带。",
            "Home Tolerance Pixels": "Screen pixels offset from the spot that counts as home, hysteresis to avoid walk-stop jitter. 归位容差（像素）：偏移回落到定点该像素以内才算归位完成，避免阈值边界反复走停。",
        })
        self.config_type = {  # GUI 控件元数据：地台标注由看板任务卡渲染成类别过滤的下拉框（见 src/ui/DashboardTaskPanel.py）。
            "Platform Feature": {"category": "地台"},
        }
        self._expect_first = True  # 首轮地台局域搜索用标注期望框的外扩比例，命中后改用上一帧外扩比例；run() 启动时会重置，构造期初始化供单测/局部调用不报缺键。

    def validate_config(self, key, value):  # 配置保存前校验，返回错误提示或 None。
        error = super().validate_config(key, value)  # 复用父类校验（父类已裁剪掉小地图相关键的校验分支，此处主要兜底按键校验）。
        if error is not None:  # 父类校验不通过时直接返回。
            return error  # 透传错误提示。
        if key == "Spot Percent":  # 百分比键必须是 0-100 的数字。
            try:  # 尝试按浮点数解析。
                percent = float(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非数字输入。
                return f"{key} must be a number. {key} 必须是数字。"  # 阻止保存并提示。
            if not 0 <= percent <= 100:  # 超出百分比范围。
                return f"{key} must be within 0-100. {key} 必须在 0-100 之间。"  # 阻止保存并提示。
        if key in ("Patrol Left Pixels", "Patrol Right Pixels", "Band Exit Margin Pixels", "Home Tolerance Pixels"):  # 像素偏移键必须是非负数字。
            try:  # 尝试按浮点数解析。
                pixels = float(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非数字输入。
                return f"{key} must be a number. {key} 必须是数字。"  # 阻止保存并提示。
            if pixels < 0:  # 负偏移没有意义。
                return f"{key} must be >= 0. {key} 必须大于等于 0。"  # 阻止保存并提示。
        if key == "Platform Threshold":  # 匹配阈值必须是 (0, 1] 的数字。
            try:  # 尝试按浮点数解析。
                threshold = float(value)  # 解析用户输入。
            except (TypeError, ValueError):  # 非数字输入。
                return "Platform threshold must be a number. 地台阈值必须是数字。"  # 阻止保存并提示。
            if not 0.01 <= threshold <= 1.0:  # 超出合法阈值范围。
                return "Platform threshold must be within 0.01-1. 地台阈值必须在 0.01-1 之间。"  # 阻止保存并提示。
        return None  # 其他配置项不做额外校验。

    @staticmethod
    def band_pixels(spot_x, left_px, right_px):  # 计算允许走动区间左右边界的画面像素坐标：定点画面 x 减左偏移 ~ 加右偏移。
        return spot_x - max(0.0, float(left_px)), spot_x + max(0.0, float(right_px))  # (左边界x, 右边界x)，不夹进地台范围：用户显式指定像素值，超出部分同样直观可见。

    @staticmethod
    def percent_to_x(plat_x, plat_w, pct):  # 把地台横向百分比换算成画面 x 坐标，百分比夹到 0-100。
        return plat_x + plat_w * max(0.0, min(100.0, pct)) / 100  # 返回画面 x 坐标（像素）。

    @staticmethod
    def out_of_band(x, lo_x, hi_x, exit_margin=0.0):  # 角色是否真正脱离允许走动区间（被怪撞开等），脱区触发优先归位；全部按画面像素判定。
        return x < lo_x - exit_margin or x > hi_x + exit_margin  # 边界再外扩 exit_margin 才算脱区：贴边界与折返惯性冲过头只算巡逻折返，不触发归位。

    @staticmethod
    def home_reached(x, spot_x, home_tol_px):  # 归位完成滞回判定：偏移回落到定点容差像素以内才算回到位，避免阈值边界反复走停。
        return abs(x - spot_x) <= home_tol_px  # 在容差内返回 True。

    @staticmethod
    def patrol_turn(direction, x, lo_x, hi_x):  # 边界折返判定：沿当前方向走到对侧边界时返回折返方向，否则返回 None；贴边界归折返管，折返惯性冲过头由脱区外扩余量兜住，不误触归位。
        if direction == 1 and x >= hi_x:  # 向右走到右边界。
            return -1  # 折返改为向左。
        if direction == -1 and x <= lo_x:  # 向左走到左边界。
            return 1  # 折返改为向右。
        return None  # 未达边界不折返。

    def find_platform(self, platform_name, frame, threshold, last_rect):  # 匹配地台当前画面位置：优先在上一帧位置（首帧为标注期望区域）外扩的局域内搜，未命中回退全屏，标注缺失返回 None。
        if last_rect is not None:  # 有参考区域（上一帧地台或首帧标注缩放区域）。
            x, y, w, h = last_rect  # 参考框。
            margin = int(max(w, h) * (PLATFORM_EXPECT_MARGIN if self._expect_first else PLATFORM_SEARCH_MARGIN))  # 首帧用标注外扩比例，之后用上一帧外扩比例。
            box = Box(max(0, int(x) - margin), max(0, int(y) - margin), int(w) + margin * 2, int(h) + margin * 2)  # 外扩后的局域搜索框。
            try:  # 局域搜索可能因模板大于搜索框等原因异常，异常时按未命中走全屏回退。
                hit = self.find_one(platform_name, frame=frame, threshold=threshold, box=box)  # 传 box 时框架只在框内搜索，不受 variance 影响。
                if hit is not None:  # 局域命中。
                    return hit  # 直接返回，抑制重复贴图在远处错配。
            except ValueError:  # 该分类名未在模板页标注。
                return None  # 按未匹配处理，调用方已前置校验，这里只兜底。
            except Exception:  # 局域搜索异常。
                pass  # 回退全屏。
        return self.find_one_raw(platform_name, frame, threshold)  # 全屏搜索：镜头大幅平移或首帧标注区域不准时的回退。

    def release_all_keys(self):  # 松开当前按住的全部按键（移动键+攻击键），进入归位/冻结/状态切换前调用。
        if self._held_attack_key is not None:  # 有按住的攻击键。
            self.send_key_up(self._held_attack_key)  # 松开攻击键。
            self._held_attack_key = None  # 清空按住状态。
            self._attack_released_at = time.time()  # 记录松键时间，攻击后摇会吞方向键，走动前需短暂等待。
        if self._held_move_key is not None:  # 有按住的移动键。
            self.send_key_up(self._held_move_key)  # 松开移动键。
            self._held_move_key = None  # 清空按住状态。

    def run(self):  # 任务运行入口：地台定位换算百分比，区间内巡逻打怪，脱离区间优先归位。
        self.apply_shared_config()  # 从看板读取共享的角色/怪物/测谎参数覆盖任务配置（单一数据源）。
        char_name = self.config.get("Character Feature")  # 读取角色标注分类名。
        monster_names = self.parse_monster_names(self.config.get("Monster Features"))  # 解析逗号分隔的怪物分类名列表。
        platform_name = str(self.config.get("Platform Feature") or '').strip()  # 读取地台标注分类名。
        facing_left_name = str(self.config.get("Character Facing Left Feature") or '').strip()  # 读取角色左朝向模板标注分类名，留空禁用朝向校准。
        facing_right_name = str(self.config.get("Character Facing Right Feature") or '').strip()  # 读取角色右朝向模板标注分类名。
        frame = self.wait_frame()  # 先取到一帧画面，让 FeatureSet 确定画面尺寸。
        if frame is None:  # 取不到画面时无法运行。
            self.log_warning("No frame captured, cannot run. 取不到画面，任务退出。")  # 提示取不到画面。
            return  # 直接结束任务。
        if not platform_name:  # 未选择地台标注时无法定位区间。
            self.log_warning("No platform annotation selected, pick one in the task card first. 未选择地台标注，请先在任务卡下拉框选择类别为「地台」的标注，任务退出。")  # 提示配置缺失。
            return  # 直接结束任务。
        if not monster_names:  # 未配置任何怪物分类时无法运行。
            self.log_warning("No monster feature configured. 未配置怪物分类名，任务退出。")  # 提示配置缺失。
            return  # 直接结束任务。
        missing = [name for name in [platform_name, char_name] + monster_names if not self.feature_ready(name)]  # 检查地台、角色与全部怪物标注是否存在。
        if missing:  # 必要标注缺失时无法运行。
            self.log_warning(f"Template not ready, please annotate in the Template tab: {', '.join(missing)}. 模板未就绪，请先在模板页标注：{'、'.join(missing)}。")  # 提示用户去模板页标注。
            return  # 标注不可用时直接结束任务。
        if facing_left_name and not self.feature_ready(facing_left_name):  # 左朝向模板已配置但未在模板页标注。
            self.log_warning(f"Left facing template not annotated, facing calibration disabled: {facing_left_name}. 左朝向模板未标注，朝向校准已禁用：{facing_left_name}。")  # 提示并禁用校准，不阻断任务。
            facing_left_name = ''  # 清空后不再参与校准。
        if facing_right_name and not self.feature_ready(facing_right_name):  # 右朝向模板已配置但未在模板页标注。
            self.log_warning(f"Right facing template not annotated, facing calibration disabled: {facing_right_name}. 右朝向模板未标注，朝向校准已禁用：{facing_right_name}。")  # 提示并禁用校准，不阻断任务。
            facing_right_name = ''  # 清空后不再参与校准。
        facing_check = bool(facing_left_name and facing_right_name)  # 左右两个朝向模板都就绪才启用图像朝向校准。
        attack_key = self.config.get("Attack Key")  # 读取常规攻击按键，横向距离超过近战距离时按住它。
        melee_key = self.config.get("Melee Attack Key")  # 读取近战攻击按键，横向距离在近战距离内时按住它。
        melee_distance = float(self.config.get("Melee Distance") or 0)  # 读取近战距离（像素），横向距离绝对值不超过该值用近战键。
        attack_x_min, attack_x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 读取攻击区域左右边界（符号化像素），填反时自动交换。
        attack_y_min, attack_y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 读取攻击区域上下边界（符号化像素），填反时自动交换。
        attack_mode = str(self.config.get("Attack Mode") or 'normal').strip()  # 读取攻击模式（看板角色配置共享键）：normal=站立攻击（本次改动前最原始的攻击策略，停下站着打，初始默认）；move_attack=边攻击边位移；move_jump=位移跳跃攻击。
        move_attack = attack_mode in ('move_attack', 'move_jump')  # 位移攻击开关（两种位移模式共用）：攻击时同时按住朝目标方向键走打，缓解被怪碰撞顶走。
        jump_key = str(self.config.get("Attack Jump Key") or '').strip()  # 读取攻击跳跃键（看板角色配置共享键，区别于路线任务私有的 Jump Key 配置）：位移跳跃模式按它周期跳跃，留空则只位移不跳。
        try:  # 攻击中跳跃间隔容错解析：配置残留非数字时不致任务崩溃。
            jump_interval_cfg = max(0.0, float(self.config.get("Jump Interval Seconds") or 0))  # 配置间隔（秒），0 表示不跳。
        except (TypeError, ValueError):  # 未配置或非数字。
            jump_interval_cfg = 0.6  # 回退看板默认间隔（与 DASHBOARD_DEFAULTS 的 Jump Interval Seconds 一致）。
        jump_interval = max(JUMP_MIN_INTERVAL, jump_interval_cfg) if jump_interval_cfg > 0 else 0.0  # 生效间隔：配置低于最小间隔按最小间隔执行，防每帧连跳。
        jump_active = move_attack and attack_mode == 'move_jump' and bool(jump_key) and jump_interval > 0  # 仅位移跳跃攻击模式周期跳跃：边攻击边位移模式不跳，跳跃键留空或间隔为 0 也不跳。
        del_interval = float(self.config.get("Del Key Interval") or 0)  # 读取自动按 Del 键的间隔秒数，0 表示禁用。
        last_del_time = time.time()  # 上次按 Del 键的时间，从任务启动开始计时。
        platform_threshold = float(self.config.get("Platform Threshold"))  # 读取地台匹配阈值。
        char_threshold = self.config.get("Character Threshold")  # 读取角色匹配阈值，朝向校准沿用同一阈值。
        spot_pct = float(self.config.get("Spot Percent") or 0)  # 读取定点百分比（地台左→右）。
        left_px = max(0.0, float(self.config.get("Patrol Left Pixels") or 0))  # 读取巡逻左偏移像素：区间左端 = 定点画面 x − 该值。
        right_px = max(0.0, float(self.config.get("Patrol Right Pixels") or 0))  # 读取巡逻右偏移像素：区间右端 = 定点画面 x + 该值。
        home_tol_px_cfg = max(1.0, float(self.config.get("Home Tolerance Pixels") or 0))  # 归位容差（像素）：偏移回落到定点该值以内算归位完成；填 0 或非法按 1px 兜底防止永不退出归位。
        band_exit_px = max(0.0, float(self.config.get("Band Exit Margin Pixels") or 0))  # 脱区触发余量（像素）：越过边界再多该值才触发归位；0 表示越界即归位（旧行为）。
        attack_to_move_wait = self.attack_to_move_wait()  # 攻击后摇等待秒数：已提升为看板角色配置全局共享键，全部任务统一生效。
        target_loss_buffer = self.target_loss_buffer()  # 目标丢失缓冲秒数：看板角色配置共享键，攻击中怪框短暂没匹配到不松攻击键，消抖帧“打一下停一下”。
        if left_px <= 0 and right_px <= 0:  # 两侧偏移都为 0 没有可巡逻空间。
            self.log_warning("Patrol band is empty (both patrol pixel offsets are 0). 巡逻左右偏移均为 0，无可巡逻区间，任务退出。")  # 提示配置问题。
            return  # 直接结束任务。
        stuck_seconds = float(self.config.get("Stuck Seconds") or 0)  # 读取卡住判定时长（沿用巡逻默认值），0 表示禁用。
        resume_wait = float(self.config.get("Resume Wait Seconds") or 0)  # 读取脱困后的等待秒数。
        frame_interval = float(self.config.get("Frame Interval") or 0)  # 读取配置帧间隔，0 表示不额外限速。
        loop_interval = max(frame_interval, CAPTURE_MIN_INTERVAL)  # 截图采集固定 30FPS，配置更慢则尊重配置。
        expected_rect = self.platform_expected_rect(platform_name, frame)  # 首帧搜索参考：标注区域按当前画面尺寸缩放，取不到时为 None（直接全屏搜）。
        gpu = self.build_gpu_matcher(char_name, monster_names)  # 尝试构建 GPU 匹配器并注册角色与全部怪物模板，失败返回 None 走 CPU；地台匹配始终走 CPU 局域搜索。
        self._expect_first = True  # 首轮局域搜索用标注外扩比例，命中后改用上一帧外扩比例。
        plat_rect = None  # 地台当前画面框 (x, y, w, h)，匹配失败时沿用上一帧，连续失败超限冻结。
        plat_missing = 0  # 地台连续匹配失败帧数。
        facing = None  # 角色当前朝向：1=右、-1=左、None=未知，只在需要换向时单击方向键。
        self._held_move_key = None  # 当前持续按住的移动方向键，归位/换向/退出时必须松开它。
        self._held_attack_key = None  # 当前持续按住的攻击键（近战或常规），切换/目标消失/退出时必须松开它。
        self._attack_released_at = 0.0  # 上次松开攻击键的时间戳，攻击后摇会吞方向键，走动前需等待。
        returning = False  # 是否处于归位状态，带滞回：脱离区间进入，像素偏移回落到容差内退出。
        last_target_seen = 0.0  # 上次在攻击区域内命中目标的时间戳，配合目标丢失缓冲：抖帧丢框不松攻击键，超时才真松。
        direction = 1  # 巡逻方向：1=向右、-1=向左，默认先向右走。
        turn_until = 0.0  # 边界折返松键停顿的截止时间，停顿期内只识图不移动。
        anchor_x = None  # 归位/巡逻卡住判定的位置锚点（画面像素），位置明显变化时重置。
        anchor_time = time.time()  # 位置锚点上次更新时间。
        last_diag_time = 0.0  # 上次诊断日志的时间戳，限频避免刷日志。
        last_facing_check = 0.0  # 上次朝向校准的时间戳，0 表示启动后立即校准一次。
        last_jump_time = 0.0  # 上次攻击中/归位中跳跃的时间戳，控制位移跳跃攻击的跳跃节奏。
        loop_start = time.time()  # 本轮循环起点，用于把截图采集节拍钉在固定 30FPS。
        try:  # 包裹主循环，退出时兜底松开移动键与攻击键。
            while True:  # 实时识图循环，直到用户手动停止任务。
                wait = loop_interval - (time.time() - loop_start)  # 计算本轮剩余等待，把取帧节拍钉在固定 30FPS。
                if wait > 0:  # 未到下一帧时点时先等待，sleep 同时承担用户停止检查。
                    self.sleep(wait)  # 补齐帧间隔。
                loop_start = time.time()  # 记录本轮起点供下轮计算。
                if del_interval > 0 and time.time() - last_del_time >= del_interval:  # 到达固定间隔时自动按一下 Del 键。
                    last_del_time = time.time()  # 重置计时。
                    self.send_key("delete", down_time=0.05)  # 短按一下 Del 键。
                frame = self.next_frame()  # 取最新一帧画面并清除旧帧。
                if frame is None:  # 取不到画面时等待下一帧时点再重试。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    loop_start = time.time()  # 重置本轮起点，避免下轮再补等待。
                    continue  # 进入下一帧处理。
                facing_due = facing_check and time.time() - last_facing_check >= 0.2  # 本轮是否需要用图像校准朝向，提前算好以便把朝向匹配一并提交并发。
                batch = MatchBatch()  # 本帧并发匹配批次：地台、朝向走 CPU 线程池，角色/怪物匹配仅在 GPU 不可用时一并提交。
                batch.submit("platform", lambda: self.find_platform(platform_name, frame, platform_threshold, expected_rect if plat_rect is None else plat_rect))  # 地台局域优先匹配，首帧以标注缩放区域为参考。
                if facing_due:  # 朝向校准到期才提交，未到期不白烧两个核心。
                    batch.submit("face_l", lambda: self.find_one_raw(facing_left_name, frame, char_threshold))  # 左朝向模板匹配，朝向模板绝不能用镜像否则左右会互串。
                    batch.submit("face_r", lambda: self.find_one_raw(facing_right_name, frame, char_threshold))  # 右朝向模板匹配。
                cpu_matches_submitted = gpu is None  # 角色/怪物匹配是否已随本批次一并提交，GPU 可用时改由显卡批量匹配。
                if cpu_matches_submitted:  # CPU 路径：角色与全部怪物匹配也一并投递，与地台/朝向匹配并行跑。
                    self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 提交角色与全部怪物的原始/镜像匹配。
                if facing_due:  # 配置了朝向模板时定期用图像校准实际朝向。
                    last_facing_check = time.time()  # 记录本次校准时间。
                    actual_facing = self.resolve_template_facing(batch.get("face_l"), batch.get("face_r"))  # 左右朝向模板取置信度高者。
                    if actual_facing is not None and facing is None:  # 朝向未知（如任务刚启动）时用图像结果初始化。
                        facing = actual_facing  # 直接采信图像朝向。
                    elif actual_facing is not None and actual_facing != facing:  # 图像识别的朝向与系统记录不一致。
                        self.log_info(f"Facing corrected by template: {facing} -> {actual_facing}. 朝向已由图像校准修正：{facing} -> {actual_facing}。")  # 记录修正供排查。
                        facing = actual_facing  # 及时按图像修正，后续转身判定会自动补发转身键。
                plat_box = batch.get("platform")  # 取回地台匹配结果。
                if plat_box is not None:  # 地台匹配命中，更新当前画面框并清空失败计数。
                    self._expect_first = False  # 已有实测位置，后续局域搜索改用上一帧外扩比例。
                    plat_rect = (plat_box.x, plat_box.y, plat_box.width, plat_box.height)  # 地台框（画面坐标）。
                    plat_missing = 0  # 恢复计数清零。
                elif plat_rect is not None:  # 本帧没匹配到（遮挡/变暗），短时间内沿用上一帧位置。
                    plat_missing += 1  # 累计连续失败帧数。
                    if plat_missing > PLATFORM_FREEZE_MISSING:  # 连续失败超限，位置不可信，冻结防盲走坠落。
                        plat_rect = None  # 丢弃位置，进入定位丢失状态。
                gm = None  # 本帧 GPU 匹配句柄，默认不可用。
                if gpu is not None:  # GPU 匹配器可用时才尝试批量匹配。
                    try:  # 上传/计算可能因显存等原因异常。
                        gm = gpu.match_frame(frame)  # 上传本帧做批量匹配。
                    except Exception as e:  # 本帧 GPU 处理异常。
                        gpu = None  # 永久回退 CPU，避免每帧重复报错。
                        self.log_warning(f"GPU match failed at runtime, fall back to CPU: {e}. GPU 匹配运行异常，已回退 CPU。")  # 记录回退原因。
                if gm is not None:  # GPU 路径：一帧内批量算完角色与全部怪物（含镜像）。
                    character = self.gpu_lookup_one(gm, char_name, self.config.get("Character Threshold"))  # 角色取最高分框，镜像命中带 flipped 标记。
                    monsters = []  # 收集本帧全部怪物匹配框。
                    for name in monster_names:  # 逐个怪物分类取全部达标框。
                        monsters.extend(self.gpu_lookup_all(gm, name, self.config.get("Monster Threshold"), self.config.get("Monster Mirror Threshold")))  # 原始与镜像各自阈值，合并后去重。
                else:  # CPU 路径：按与串行完全一致的顺序取回线程池里的匹配结果，行为与 GPU 路径一致。
                    if not cpu_matches_submitted:  # 本帧刚走 GPU 分支异常回退 CPU，角色/怪物匹配还没提交。
                        self.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 补提交本帧全部匹配。
                    character, monsters = self.collect_char_monster_matches(batch, monster_names)  # 取回角色框与全部怪物框。
                if character is not None and monsters:  # 角色存在时先剔除压在角色身上的怪物框。
                    kept = []  # 过滤后的怪物框列表。
                    for monster in monsters:  # 逐只检查是否与角色框重叠。
                        if self.overlapping(character, monster):  # 低阈值时怪物模板容易误匹配到角色自身，中心重合就是典型特征。
                            self.log_warning(f"Drop monster box overlapping character: {monster.name} conf={monster.confidence:.2f} x={monster.x} y={monster.y}. 剔除与角色重叠的怪物框，疑似误匹配，建议调高 Template/Mirror Threshold。")  # 记录被剔除的框供排查。
                            continue  # 丢弃该框不参与距离与攻击判定。
                        kept.append(monster)  # 保留正常怪物框。
                    monsters = kept  # 用过滤后的列表替换原列表。
                pct = None  # 角色在地台上的横向百分比（左→右 0-100），仅供展示与诊断，判定全部改用画面像素；定位不全为 None。
                cx = None  # 角色框中心 x（画面像素），定位不全为 None；脱区/归位/折返判定都用它。
                spot_x = lo_x = hi_x = None  # 定点与巡逻区间左右边界（画面像素），地台定位不全为 None。
                if plat_rect is not None and plat_rect[2] > 0:  # 地台就绪即可换算定点与区间，不依赖角色定位。
                    spot_x = self.percent_to_x(plat_rect[0], plat_rect[2], spot_pct)  # 定点画面 x 坐标，随当前地台框变化，镜头平移不影响区间相对位置。
                    lo_x, hi_x = self.band_pixels(spot_x, left_px, right_px)  # 区间边界 = 定点 − 左偏移 ~ 定点 + 右偏移（像素）。
                    if character is not None:  # 角色与地台都就绪才换算位置。
                        cx, _ = self.box_center(character)  # 角色框中心 x。
                        pct = (cx - plat_rect[0]) / plat_rect[2] * 100  # 相对地台宽度的横向百分比（展示用）。
                        self.info_set("Position", f"x={cx:.0f} 定点x={spot_x:.0f} 区间 {lo_x:.0f}~{hi_x:.0f}px")  # 在 GUI 状态区显示角色与区间的像素坐标。
                nearest = None  # 距离角色最近的怪物框。
                if character is not None and monsters:  # 角色与怪物都找到时才计算距离。
                    nearest = min(monsters, key=lambda m: self.center_distance(character, m))  # 选中心点距离最近的怪物。
                    dx, dy = self.center_offset(character, nearest)  # 计算角色到怪物的中心点 xy 距离。
                    self.info_set("Distance", f"dx={dx} dy={dy}")  # 在 GUI 状态区显示 xy 距离。
                elif character is not None:  # 角色在但没匹配到任何怪物。
                    self.info_set("Distance", "-")  # 明确显示无怪，避免残留旧值造成误判。
                if time.time() - last_diag_time >= 1.0:  # 每秒限频输出一次诊断日志。
                    last_diag_time = time.time()  # 记录本次诊断时间。
                    plat_desc = f"x={plat_rect[0]} y={plat_rect[1]} w={plat_rect[2]} miss={plat_missing}" if plat_rect is not None else "None"  # 地台框位置与连续失败数。
                    char_desc = f"x={character.x} y={character.y} conf={character.confidence:.2f}" if character is not None else "None"  # 角色框位置与置信度。
                    mob_desc = "; ".join(f"{m.name}@({m.x},{m.y}) conf={m.confidence:.2f}" for m in monsters) or "None"  # 全部怪物框位置与置信度。
                    pct_desc = f"{pct:.1f}%" if pct is not None else "None"  # 地台百分比描述。
                    self.log_info(f"Platform diag: PLAT[{plat_desc}] PCT[{pct_desc}] CHAR[{char_desc}] MOBS[{mob_desc}] 地台挂机诊断：地台框、角色百分比与怪物框坐标及置信度。")  # 输出诊断日志供排查误匹配。
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
                        target = min(pool, key=lambda m: self.center_distance(character, m))  # 取候选池中最近的一只，同侧换更近的怪不松键不断攻击。
                if target is not None:  # 目标框计时刷新：命中目标就记录时间，供下方目标丢失缓冲判定。
                    last_target_seen = time.time()  # 重置丢框计时，连续命中则缓冲期永远不触发松键。
                og.my_app.update_vision(self.draw_overlay(frame, plat_rect, pct, spot_pct, left_px, right_px, band_exit_px, returning, character, monsters, nearest, target))  # 把带标注画面推送给 UI 实时展示。
                if plat_rect is None or character is None or cx is None:  # 地台或角色定位不可信时无法判定位置：松开移动键原地冻结，位置恢复自动继续。
                    if self._held_move_key is not None:  # 有按住的移动键。
                        self.send_key_up(self._held_move_key)  # 松开方向键，位置未知时走动有越界坠落风险。
                        self._held_move_key = None  # 清空按住状态。
                    if self._held_attack_key is not None and time.time() - last_target_seen > target_loss_buffer:  # 定位丢失且超目标丢失缓冲仍按着攻击键：被怪顶着遮挡定位时会带着攻击键原地盲打且归位永不触发，必须松键（短暂抖帧在缓冲期内不松，保持攻击节奏）。
                        self.send_key_up(self._held_attack_key)  # 松开攻击键停止盲打。
                        self._held_attack_key = None  # 清空按住状态。
                        self._attack_released_at = time.time()  # 记录松键时间，恢复走动前等攻击后摇。
                        self.log_warning("Tracking lost while attack key held, release after target loss buffer. 定位丢失超过目标丢失缓冲，已松开攻击键，防止被怪顶着时原地盲打不归位。")  # 记录事件供确认根因。
                    anchor_x = None  # 定位丢失时卡住锚点失效。
                    self.info_set("Status", "地台定位丢失" if plat_rect is None else "未找到角色")  # 在 GUI 显示定位丢失状态。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    continue  # 定位恢复后自动继续区间判定。
                outside = self.out_of_band(cx, lo_x, hi_x, band_exit_px)  # 角色是否真正脱离区间：边界处归巡逻折返管，越过外扩余量才算被怪撞飞触发归位。
                if outside and not returning:  # 刚脱离区间，进入归位状态（最高优先级，归位途中不攻击）。
                    returning = True  # 切换为归位状态。
                    self.log_info(f"Out of band [{lo_x:.0f}px, {hi_x:.0f}px] by more than {band_exit_px:.0f}px at x={cx:.0f}, return to spot x={spot_x:.0f} first. 角色 x={cx:.0f} 越过区间 [{lo_x:.0f}, {hi_x:.0f}] 外扩 {band_exit_px:.0f}px，优先走回定点 x={spot_x:.0f}。")  # 记录归位事件供排查。
                if returning:  # 归位状态：朝定点走回，像素偏移回落到容差内（滞回）才退出恢复巡逻。
                    if self.home_reached(cx, spot_x, home_tol_px_cfg):  # 已回到定点容差内。
                        returning = False  # 退出归位状态，恢复巡逻（保留原巡逻方向）。
                        self.log_info(f"Back to spot at x={cx:.0f}, resume patrol. 已归位到 x={cx:.0f}，恢复区间巡逻。")  # 记录归位完成供排查。
                    else:  # 尚未归位，本帧继续朝定点走动，不进入攻击与巡逻分支。
                        if not move_attack and self._held_attack_key is not None:  # 站立攻击模式归位优先：松开按住的攻击键，归位途中不攻击（最原始版本行为）；两种位移模式保持输出边走边打。
                            self.send_key_up(self._held_attack_key)  # 松开攻击键。
                            self._held_attack_key = None  # 清空按住状态。
                            self._attack_released_at = time.time()  # 记录松键时间，走动前需等攻击后摇结束。
                        if move_attack and self._held_attack_key is not None and target is None and time.time() - last_target_seen > target_loss_buffer:  # 位移模式归位途中目标消失超缓冲：松攻击键，无怪可打不空挥。
                            self.send_key_up(self._held_attack_key)  # 松开攻击键。
                            self._held_attack_key = None  # 清空按住状态。
                            self._attack_released_at = time.time()  # 记录松键时间。
                        if move_attack and attack_key and self._held_attack_key is None and target is not None:  # 位移模式归位途中有目标：按住常规攻击键朝移动方向继续输出。
                            self.send_key_down(attack_key)  # 持续按住攻击键。
                            self._held_attack_key = attack_key  # 记录当前按住的键。
                        walk_direction = 1 if cx < spot_x else -1  # 归位走动方向：定点在右侧向右走，在左侧向左走。
                        anchor_x, anchor_time = self.update_stuck_anchor(int(cx), anchor_x, anchor_time, plat_rect[2])  # 用角色画面 x 更新卡住锚点。
                        if stuck_seconds > 0 and self._held_move_key is not None and time.time() - anchor_time >= stuck_seconds:  # 按住方向键但位置长期不动视为卡住。
                            self.recover_stuck(self._held_move_key, walk_direction, resume_wait)  # 反向短移脱困，脱困后本帧松键，下一帧按最新偏移继续归位。
                            self._held_move_key = None  # 恢复过程已松键。
                            anchor_x, anchor_time = None, time.time()  # 脱困后重置卡住锚点。
                        if not move_attack and time.time() < self._attack_released_at + attack_to_move_wait:  # 站立攻击模式攻击刚结束的等待期内只识图不移动，方向键会被攻击后摇吞掉（时长看板可调）；位移模式攻击键未松开无吞键问题。
                            self.info_set("Status", "归位等待")  # 在 GUI 显示归位等待状态。
                            self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                            continue  # 等待期结束后自动恢复走动。
                        want_key = MOVE_RIGHT_KEY if walk_direction == 1 else MOVE_LEFT_KEY  # 归位方向需要的方向键。
                        if self._held_move_key != want_key:  # 换向时先松开旧键再按新键，避免两键同时按住；同向保持不重复松按。
                            if self._held_move_key is not None:  # 有旧键按住。
                                self.send_key_up(self._held_move_key)  # 松开旧方向键。
                            self.send_key_down(want_key)  # 持续按住新方向键走回定点。
                            self._held_move_key = want_key  # 记录当前按住的键。
                            facing = walk_direction  # 走动时朝向与移动方向一致，供攻击换向判定直接使用。
                        if jump_active and time.time() - last_jump_time >= jump_interval:  # 位移跳跃模式归位途中同样按间隔跳跃：空中不受地面碰撞推移，防止走回途中再被顶开；边攻击边位移模式不跳。
                            self.send_key(jump_key, down_time=0.05)  # 短按跳跃键，配合按住的方向键形成方向跳。
                            last_jump_time = time.time()  # 记录跳跃时间。
                        self.info_set("Status", f"{'向右' if walk_direction == 1 else '向左'}归位 (偏移 {abs(cx - spot_x):.0f}px)" + (" 边打" if move_attack else ""))  # 在 GUI 显示归位方向与当前像素偏移。
                        self.sleep(loop_interval)  # 按固定 30FPS 节拍等待后处理下一帧。
                        continue  # 归位中不进入攻击与巡逻分支。
                if target is not None:  # 区间内且攻击范围内有怪：普通模式停下巡逻原地转向攻击；位移跳跃模式按住朝怪方向键边走边跳边打。
                    dx, dy = self.center_offset(character, target)  # 计算目标怪物相对角色的方向。
                    want_direction = 1 if dx > 0 else -1  # 1=怪在右侧，-1=怪在左侧。
                    want_key = melee_key if abs(dx) <= melee_distance else attack_key  # 横向距离在近战距离内用近战键，否则用常规攻击键，怪物走近走远时自动切换。
                    if move_attack:  # 位移攻击模式（边攻击边位移/位移跳跃攻击共用）：按住朝目标方向键边走边打，朝向与移动方向一致攻击不落空，空中也不受地面碰撞推移。
                        want_move_key = MOVE_RIGHT_KEY if want_direction == 1 else MOVE_LEFT_KEY  # 朝目标方向的移动键。
                        if self._held_move_key != want_move_key:  # 换向先松旧键再按新键（首次按下也走此分支）。
                            if self._held_move_key is not None:  # 有旧键按住。
                                self.send_key_up(self._held_move_key)  # 松开旧方向键。
                            self.send_key_down(want_move_key)  # 持续按住新方向键朝目标移动。
                            self._held_move_key = want_move_key  # 记录当前按住的键。
                            facing = want_direction  # 移动时朝向与移动方向一致，转身序列不再触发。
                    elif self._held_move_key is not None:  # 普通模式进入攻击前先松开移动键，站着打不边走边打。
                        self.send_key_up(self._held_move_key)  # 松开方向键。
                        self._held_move_key = None  # 清空按住状态。
                        anchor_x = None  # 停止走动后卡住锚点失效。
                    if self._held_attack_key is not None and self._held_attack_key != want_key:  # 换攻击键（近战↔常规）：同一帧内松旧按新不中断输出，不计入后摇也不影响朝向；只有真正要转身才走下方松键→等待→点方向键序列。
                        self.send_key_up(self._held_attack_key)  # 松开当前按住的攻击键。
                        self._held_attack_key = None  # 清空按住状态，紧下方会立即按新键，帧内无缝衔接。
                    if facing != want_direction:  # 朝向与怪物方向不一致时执行转身序列（位移跳跃模式移动键已同步朝向，通常不触发）。
                        if self._held_attack_key is not None:  # 先松开攻击键：攻击后摇期间方向键输入会被游戏吞掉，导致转身失败后长时间空打。
                            self.send_key_up(self._held_attack_key)  # 松开攻击键。
                            self._held_attack_key = None  # 清空按住状态，转身完成后重新按住。
                            self.sleep(attack_to_move_wait)  # 等攻击后摇结束再发方向键，时长与攻击后恢复走动同一配置值。
                        self.send_key(MOVE_RIGHT_KEY if want_direction == 1 else MOVE_LEFT_KEY, down_time=0.05)  # 单次短按方向键只触发转身动画，按住时间越短越不容易产生位移。
                        facing = want_direction  # 记录当前朝向，同一方向不再重复按键，避免持续位移。
                        self.sleep(0.08)  # 等待转身动作生效后再攻击。
                    if self._held_attack_key is None:  # 当前没有按住攻击键时才按下，已按住则保持不重复发送。
                        self.send_key_down(want_key)  # 持续按住近战或常规攻击键不放。
                        self._held_attack_key = want_key  # 记录当前按住的攻击键。
                    if jump_active and time.time() - last_jump_time >= jump_interval:  # 位移跳跃模式到达跳跃间隔：短按跳跃键，配合按住的方向键形成方向跳；边攻击边位移模式不跳。
                        self.send_key(jump_key, down_time=0.05)  # 短按跳跃键。
                        last_jump_time = time.time()  # 记录跳跃时间。
                    self.info_set("Status", "位移攻击中" if move_attack else "攻击中")  # 在 GUI 显示攻击状态。
                    self.sleep(0.1)  # 按住期间每 0.1 秒重新识别一次校准目标。
                    continue  # 目标消失时自动停止攻击重新扫描。
                if self._held_attack_key is not None and time.time() - last_target_seen <= target_loss_buffer:  # 目标丢失缓冲（看板角色配置可调）：怪物框连续一两帧没匹配到时不松攻击键，消掉抖帧导致的打一下停一下；超时确实没怪才落入下方松键流程。
                    self.info_set("Status", "攻击中")  # 缓冲期沿用攻击状态，按键保持不断。
                    self.sleep(0.1)  # 与攻击节奏一致的识别节拍，怪框恢复后立即重新校准目标。
                    continue  # 跳过本帧松键与巡逻恢复。
                if self._held_attack_key is not None:  # 目标消失超过缓冲时长：松开持续按住的攻击键并准备恢复巡逻。
                    self.send_key_up(self._held_attack_key)  # 松开攻击键。
                    self._held_attack_key = None  # 清空按住状态。
                    self._attack_released_at = time.time()  # 记录松键时间，供走动前等待攻击后摇结束。
                if time.time() < self._attack_released_at + attack_to_move_wait:  # 攻击刚结束的等待期内只识图不移动（时长看板可调）。
                    self.info_set("Status", "恢复巡逻中")  # 在 GUI 显示恢复巡逻等待状态。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    continue  # 等待期结束后自动恢复巡逻。
                if time.time() < turn_until:  # 边界折返停顿期内保持松键，避免继续顶向边界。
                    self.info_set("Status", "巡逻折返中")  # 在 GUI 显示折返状态。
                    self.sleep(loop_interval)  # 按固定 30FPS 节拍等待。
                    continue  # 停顿结束后按新方向继续巡逻。
                turn = self.patrol_turn(direction, cx, lo_x, hi_x)  # 沿当前巡逻方向走到对侧区间边界时返回折返方向，未达边界为 None。
                if turn is not None:  # 边界折返：换向并松键短暂停顿，避免继续顶向边界。
                    direction = turn  # 采用折返后的巡逻方向。
                    turn_until = time.time() + BOUND_TURN_PAUSE  # 松键短暂停顿再换向。
                    if self._held_move_key is not None:  # 松开旧方向键。
                        self.send_key_up(self._held_move_key)  # 松键。
                        self._held_move_key = None  # 清空按住状态。
                    self.log_info(f"Reached {'right' if turn == -1 else 'left'} band x={hi_x if turn == -1 else lo_x:.0f}, turn {'left' if turn == -1 else 'right'}. 到达{'右' if turn == -1 else '左'}边界 x={hi_x if turn == -1 else lo_x:.0f}，改为向{'左' if turn == -1 else '右'}巡逻。")  # 记录折返供排查。
                anchor_x, anchor_time = self.update_stuck_anchor(int(cx), anchor_x, anchor_time, plat_rect[2])  # 更新巡逻卡住判定锚点。
                if stuck_seconds > 0 and self._held_move_key is not None and time.time() - anchor_time >= stuck_seconds:  # 按住方向键但位置长期不动视为卡住。
                    direction = self.recover_stuck(self._held_move_key, direction, resume_wait)  # 反向短移脱困并翻转巡逻方向。
                    self._held_move_key = None  # 恢复过程已松键。
                    anchor_x, anchor_time = None, time.time()  # 脱困后重置卡住锚点。
                want_key = MOVE_RIGHT_KEY if direction == 1 else MOVE_LEFT_KEY  # 巡逻方向需要的方向键。
                if self._held_move_key != want_key:  # 换向时先松开旧键再按新键，避免两键同时按住。
                    if self._held_move_key is not None:  # 有旧键按住。
                        self.send_key_up(self._held_move_key)  # 松开旧方向键。
                    self.send_key_down(want_key)  # 持续按住新方向键保持移动。
                    self._held_move_key = want_key  # 记录当前按住的键。
                if self._held_move_key is not None:  # 确实在移动时朝向与移动方向一致。
                    facing = direction  # 同步朝向，供攻击换向判定直接使用，减少转身探测。
                self.info_set("Status", "向右巡逻" if direction == 1 else "向左巡逻")  # 在 GUI 显示当前巡逻方向。
                self.sleep(loop_interval)  # 按固定 30FPS 节拍等待后处理下一帧。
        finally:  # 用户停止任务或异常退出时兜底松键，防止按键卡住。
            if self._held_attack_key is not None:  # 有按住未松的攻击键。
                self.send_key_up(self._held_attack_key)  # 松开它。
            if self._held_move_key is not None:  # 有按住未松的方向键。
                self.send_key_up(self._held_move_key)  # 松开它。

    def platform_expected_rect(self, platform_name, frame):  # 取标注的地台区域按当前画面尺寸缩放后的期望框 (x, y, w, h)，取不到返回 None（首帧直接全屏搜）。
        try:  # 标注框读取可能因文件缺失/分类不存在异常。
            box = self.get_box_by_name(platform_name)  # 框架返回已按当前画面尺寸缩放的标注框。
            if box is None:  # 无标注框信息。
                return None  # 首帧回退全屏搜索。
            return box.x, box.y, box.width, box.height  # 返回期望框。
        except Exception:  # 任意异常都不阻断任务。
            return None  # 首帧回退全屏搜索。

    def draw_overlay(self, frame, plat_rect, pct, spot_pct, left_px, right_px, band_exit_px, returning, character, monsters, nearest, target):  # 在一帧画面上绘制地台、定点、巡逻区间色框、脱区余量线框与攻击的全部标注，返回新画面。
        canvas = frame.copy()  # 复制画面避免污染原始帧。
        if character is not None:  # 匹配到角色时绘制角色标注。
            self.draw_target(canvas, character, (0, 255, 0), "CHAR" + ("-flip" if getattr(character, "flipped", False) else ""))  # 绿色框+十字延长线，镜像命中时标注 -flip。
            cx, cy = self.box_center(character)  # 攻击范围基准点：角色框中心，与 run() 中攻击判定公式一致。
            x_min, x_max = sorted((int(self.config.get("Attack Range X Min")), int(self.config.get("Attack Range X Max"))))  # 左右边界，填反自动交换，与判定公式一致。
            y_min, y_max = sorted((int(self.config.get("Attack Range Y Min")), int(self.config.get("Attack Range Y Max"))))  # 上下边界，填反自动交换，与判定公式一致。
            cv2.rectangle(canvas, (int(cx + x_min), int(cy + y_min)), (int(cx + x_max), int(cy + y_max)), (255, 0, 255), 2)  # 紫色矩形框标出攻击范围，怪物中心点落入框内才会被攻击。
            self.draw_text(canvas, "ATK RANGE", (int(cx + x_min), max(int(cy + y_min) - 6, 14)), (255, 0, 255))  # 范围框左上角标注文本。
            melee_distance = int(float(self.config.get("Melee Distance") or 0))  # 近战距离，与 run() 中近战/常规攻击键切换公式一致。
            if melee_distance > 0:  # 配置了近战距离时用蓝线框出近战范围。
                cv2.rectangle(canvas, (int(cx - melee_distance), int(cy + y_min)), (int(cx + melee_distance), int(cy + y_max)), (255, 0, 0), 2)  # 蓝色矩形框标出近战范围，怪物中心点落入框内用近战键。
                self.draw_text(canvas, f"MELEE {melee_distance}", (int(cx - melee_distance), min(int(cy + y_max) + 16, canvas.shape[0] - 6)), (255, 0, 0))  # 近战框下方标注近战距离，避免与上方的 ATK RANGE 标注重叠。
        for monster in monsters:  # 绘制每只怪物的标注。
            self.draw_target(canvas, monster, (0, 0, 255), "MOB" + ("-flip" if getattr(monster, "flipped", False) else ""), cross=False)  # 红色框不带十字延长线，避免多只怪物时红线交叉刷屏，镜像命中时标注 -flip。
        if target is not None:  # 存在当前攻击目标时额外高亮。
            self.draw_target(canvas, target, (255, 255, 255), "TARGET", cross=False)  # 白色框标出正在攻击的怪物，消失后不再绘制。
        if character is not None and nearest is not None:  # 角色与最近怪物都存在时绘制距离信息。
            cx, cy = self.box_center(character)  # 角色中心点。
            mx, my = self.box_center(nearest)  # 怪物中心点。
            cv2.line(canvas, (int(cx), int(cy)), (int(mx), int(my)), (255, 255, 0), 1)  # 角色与怪物中心连线。
            dx, dy = self.center_offset(character, nearest)  # 计算 xy 距离。
            self.draw_text(canvas, f"dx={dx} dy={dy}", (int((cx + mx) / 2), int((cy + my) / 2)), (255, 255, 0))  # 在连线中点显示 xy 距离。
        if plat_rect is not None:  # 地台就绪时绘制地台框、巡逻区间色框与边界线、定点色框。
            px, py, pw, ph = plat_rect  # 地台框（画面坐标）。
            cv2.rectangle(canvas, (int(px), int(py)), (int(px + pw), int(py + ph)), (255, 255, 0), 2)  # 青色框标出地台当前画面位置。
            self.draw_text(canvas, "PLATFORM", (int(px), max(int(py) - 6, 14)), (255, 255, 0))  # 地台框上方标签。
            spot_x = int(self.percent_to_x(px, pw, spot_pct))  # 定点画面 x 坐标。
            band_lo, band_hi = self.band_pixels(spot_x, left_px, right_px)  # 巡逻区间左右边界（像素）。
            band_color = (0, 0, 255) if returning else (0, 255, 255)  # 归位中区间转红提示，平时黄色。
            height = canvas.shape[0]  # 画面高度，边界线贯穿全高方便观察。
            for edge_x in (int(band_lo), int(band_hi)):  # 区间左右端纵向边界线。
                cv2.line(canvas, (edge_x, 0), (edge_x, height), band_color, 1)  # 边界纵向线。
            box_w = int(character.width) if character is not None else SPOT_BOX_FALLBACK_WIDTH  # 定点色框宽度：与角色框同宽，未匹配到角色时兜底处理。
            box_h = int(character.height) if character is not None else SPOT_BOX_FALLBACK_HEIGHT  # 定点色框高度：与角色框同高。
            top = max(0, int(py) - box_h)  # 色框立于地台顶边上方（角色站立面），防止越出画面。
            cv2.rectangle(canvas, (int(band_lo), top), (int(band_hi), int(py)), band_color, 2)  # 区间带状色框横跨左右边界，直观展示允许走动范围。
            self.draw_text(canvas, f"PATROL -{spot_x - band_lo:.0f}/+{band_hi - spot_x:.0f}px", (int(band_lo) + 4, top + 14), band_color)  # 区间色框内标注左右像素偏移。
            if band_exit_px > 0:  # 脱区触发余量不为 0 时，用蓝色细线框单独标出边界外“越界也不归位”的缓冲带，直观看出被撞开多远才触发归位。
                exit_color = (255, 0, 0)  # 蓝色（BGR）区别于区间的黄/红色框。
                for seg_lo, seg_hi in ((int(band_lo - band_exit_px), int(band_lo)), (int(band_hi), int(band_hi + band_exit_px))):  # 左右两段余量带。
                    cv2.rectangle(canvas, (seg_lo, top), (seg_hi, int(py)), exit_color, 1)  # 细线框标出余量带，越过它区间框才会转红归位。
                self.draw_text(canvas, f"EXIT ±{band_exit_px:.0f}px", (int(band_lo - band_exit_px) + 4, top + 30), exit_color)  # 余量带标签。
            cv2.rectangle(canvas, (spot_x - box_w // 2, top), (spot_x + box_w // 2, int(py)), (0, 255, 0), 2)  # 绿色色框标出定点站位。
            self.draw_text(canvas, f"SPOT {spot_pct:g}% x={spot_x}", (spot_x - box_w // 2, max(top - 6, 14)), (0, 255, 0))  # 定点标签。
            if pct is not None:  # 角色百分比就绪时在定点标签下方补充当前位置。
                self.draw_text(canvas, f"NOW {pct:.1f}%", (spot_x - box_w // 2, max(top + 28, 40)), (0, 255, 255))  # 当前位置百分比。
        return canvas  # 返回绘制完成的画面。
