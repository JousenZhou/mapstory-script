# MapleIdleTask 回归测试：验证配置校验、解析、模板匹配与画面标注逻辑。
import unittest

import cv2

from src.config import config
from ok.test.TaskTestCase import TaskTestCase

from src.tasks.MapleIdleTask import MapleIdleTask


class TestMapleIdleSmoke(TaskTestCase):
    task_class = MapleIdleTask

    config = config

    def test_parse_monster_names(self):
        names = self.task.parse_monster_names("小青蛇, 绿水灵 ,")
        self.assertEqual(["小青蛇", "绿水灵"], names)

    def test_validate_attack_key(self):
        for key in ("Attack Key Left", "Attack Key Right"):  # 两侧攻击按键都要走按键合法性校验。
            self.assertIsNone(self.task.validate_config(key, "a"))
            self.assertIsNotNone(self.task.validate_config(key, "not_a_key"))

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
