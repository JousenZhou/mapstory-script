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


if __name__ == '__main__':
    unittest.main()
