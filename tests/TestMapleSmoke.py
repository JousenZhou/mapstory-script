# MapleIdleTask 回归测试：验证配置校验、解析、模板匹配与画面标注逻辑。
import unittest

import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MapleIdleTask import MapleIdleTask, MatchBatch


class TestMapleIdleSmoke(TaskTestCase):
    task_class = MapleIdleTask

    config = config

    def test_parse_monster_names(self):
        names = self.task.parse_monster_names("小青蛇, 绿水灵 ,")
        self.assertEqual(["小青蛇", "绿水灵"], names)

    def test_shared_config_trimmed(self):
        # 角色/怪物/攻击等共享配置已搬到看板，任务页只保留动作节奏类配置。
        from src.dashboard_store import DASHBOARD_DEFAULTS  # 看板共享配置默认值。
        for shared_key in DASHBOARD_DEFAULTS:  # 看板共享键不再出现在任务页。
            self.assertNotIn(shared_key, self.task.default_config)
        for kept in ("Move Interval", "Move Away Seconds", "Move Back Seconds", "Turn Interval", "Use Gray Scale", "GPU Match", "Frame Interval"):
            self.assertIn(kept, self.task.default_config)  # 动作节奏类配置保留。

    def test_validate_key_name(self):
        # 按键合法性校验随配置搬到看板，看板存取层提供与框架规则一致的校验函数。
        from src.dashboard_store import validate_key_name  # 看板按键校验。
        self.assertTrue(validate_key_name("a"))  # 单字母合法。
        self.assertTrue(validate_key_name("delete"))  # 框架命名键合法。
        self.assertFalse(validate_key_name("not_a_key"))  # 非法按键拦截。

    def test_find_monster_flipped(self):
        # 整帧水平镜像后怪物朝向反转，镜像匹配应能命中并标记 flipped。
        self.task.config['Monster Threshold'] = 0.8  # 测试图用严格阈值，避免用户为实际游戏调低阈值后在测试图上误检。
        self.task.config['Monster Mirror Threshold'] = 0.8  # 怪物镜像匹配同样用严格阈值。
        self.set_image('ok_templates/0.png')
        frame = cv2.flip(self.task.frame.copy(), 1)  # 复制后再翻转，避免 cv2.flip 就地操作污染共享内存。
        monsters = self.task.find_all_features("绿水灵", frame, self.task.config['Monster Threshold'], self.task.config['Monster Mirror Threshold'])
        self.assertEqual(1, len(monsters))
        self.assertTrue(getattr(monsters[0], "flipped", False))

    def test_find_character_and_draw_overlay(self):
        # 共享键已搬到看板，测试直接注入运行期等效值（与任务启动时 apply_shared_config 后的读取方式一致）。
        self.task.config.update({
            'Character Feature': '帽子',  # 测试图上的角色标注分类名，跟随模板页实际标注名。
            'Attack Range X Min': -150, 'Attack Range X Max': 150,  # 绘制攻击范围框所需。
            'Attack Range Y Min': -60, 'Attack Range Y Max': 60,
            'Melee Distance': 40,  # 绘制近战范围框所需。
        })
        self.task.config['Character Threshold'] = 0.75  # 角色模板在测试图上的得分为 0.79，用略低于该值的阈值避免临界抖动。
        self.task.config['Monster Threshold'] = 0.8  # 怪物匹配同样用严格阈值。
        self.task.config['Monster Mirror Threshold'] = 0.8  # 怪物镜像匹配同样用严格阈值。
        self.set_image('ok_templates/0.png')
        frame = self.task.frame
        character = self.task.find_one_feature(self.task.config['Character Feature'], frame, self.task.config['Character Threshold'])  # 角色分类名从配置读取，跟随模板页实际标注名。
        self.assertIsNotNone(character)
        monsters = self.task.find_all_features("绿水灵", frame, self.task.config['Monster Threshold'], self.task.config['Monster Mirror Threshold'])
        self.assertEqual(1, len(monsters))
        canvas = self.task.draw_overlay(frame, character, monsters, monsters[0], monsters[0])
        self.assertEqual(frame.shape, canvas.shape)
        self.assertFalse((canvas == frame).all())
        dx, dy = self.task.center_offset(character, monsters[0])
        self.assertIsInstance(dx, int)
        self.assertIsInstance(dy, int)

    def test_concurrent_match_consistent_with_serial(self):
        # 并发匹配批次与串行逐个匹配对同一帧的结果必须完全一致（框数量、坐标、置信度、镜像标记与顺序）。
        self.task.config.update({
            'Character Feature': '帽子',  # 测试图上的角色标注分类名，跟随模板页实际标注名。
            'Character Threshold': 0.75,  # 与 test_find_character_and_draw_overlay 保持一致的角色阈值。
            'Monster Threshold': 0.8,  # 测试图用严格阈值，避免误检干扰一致性对比。
            'Monster Mirror Threshold': 0.8,  # 怪物镜像匹配同样用严格阈值。
        })
        self.set_image('ok_templates/0.png')
        frame = self.task.frame
        char_name = self.task.config['Character Feature']  # 角色分类名。
        monster_names = ["绿水灵"]  # 测试图上有标注的怪物分类。
        serial_char = self.task.find_one_feature(char_name, frame, self.task.config['Character Threshold'])  # 串行角色匹配。
        serial_monsters = []  # 串行怪物匹配结果。
        for name in monster_names:  # 逐个分类串行匹配，与改造前 run() 的 CPU 路径一致。
            serial_monsters.extend(self.task.find_all_features(name, frame, self.task.config['Monster Threshold'], self.task.config['Monster Mirror Threshold']))  # 原始与镜像合并去重。
        batch = MatchBatch()  # 新建本帧并发匹配批次。
        self.task.submit_char_monster_matches(batch, frame, char_name, monster_names)  # 一次性提交角色与全部怪物的原始/镜像匹配。
        conc_char, conc_monsters = self.task.collect_char_monster_matches(batch, monster_names)  # 按串行原顺序取回结果。
        self.assertIsNotNone(conc_char)  # 测试图上角色必须能匹配到，否则本用例没有真正对比到东西。
        self.assertEqual(self.describe_box(serial_char), self.describe_box(conc_char))  # 角色框完全一致。
        self.assertEqual([self.describe_box(b) for b in serial_monsters],  # 全部怪物框的内容与顺序都一致。
                         [self.describe_box(b) for b in conc_monsters])

    @staticmethod
    def describe_box(box):  # 把匹配框压成可比较的元组，None 保持 None。
        if box is None:  # 未匹配到目标。
            return None  # 用 None 参与对比。
        return box.name, box.x, box.y, box.width, box.height, round(box.confidence, 6), bool(getattr(box, "flipped", False))  # 分类名、位置、尺寸、置信度与镜像标记。

    def test_gpu_match_consistent_with_cpu(self):
        # GPU 路径与 CPU 路径对同一帧的最佳匹配位置与分数应一致，位置容差 3 像素、分数容差 0.02。
        try:  # CuPy 未安装时跳过，不阻断无显卡环境的测试。
            from src.gpu_match import GpuTemplateMatcher, gpu_available
        except Exception as e:
            self.skipTest(f"cupy not installed: {e}")
        if not gpu_available():  # 无可用 NVIDIA 显卡。
            self.skipTest("no available NVIDIA GPU")
        self.set_image('ok_templates/0.png')
        frame = self.task.frame
        feature_set = self.task.executor.feature_set
        self.task.config['Character Feature'] = '帽子'  # 角色分类名已搬到看板，测试直接注入模板页实际标注名。
        names = [self.task.config['Character Feature'], "绿水灵"]  # 角色与怪物各验一个。
        matcher = GpuTemplateMatcher(gray=bool(self.task.config.get("Use Gray Scale")))  # 按任务灰度配置建匹配器。
        for name in names:  # 注册原始与镜像模板，与任务运行时的注册方式一致。
            feature_set.ensure_feature(name)
            feature = feature_set.feature_dict.get(name)
            matcher.add_template(name, feature.mat)
            matcher.add_template(name + "__flip", cv2.flip(feature.mat, 1))
        gm = matcher.match_frame(frame)
        for name in names:  # 逐个分类对比 GPU 与 CPU 的最佳匹配。
            cpu_box = self.task.find_one_feature(name, frame, 0.6)
            gx, gy, gscore = gm.best(name)
            if cpu_box is None:  # CPU 找不到时 GPU 分数也必须不达标。
                self.assertLess(gscore, 0.6)
                continue
            self.assertLessEqual(abs(gx - cpu_box.x), 3)
            self.assertLessEqual(abs(gy - cpu_box.y), 3)
            self.assertLess(abs(gscore - cpu_box.confidence), 0.02)


if __name__ == '__main__':
    unittest.main()
