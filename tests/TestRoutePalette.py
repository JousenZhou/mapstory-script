# 指令色表工具（src/route_palette.py）单元测试。
# 覆盖：色表反查（主表优先）、指令归属哪张表（与消费端 combine_cmd 的双色互补语义一致）、
# 自动扩色不与内置色/边缘色/黑冲突且互不重复、色池耗尽返回 None、路线图上未知色的检出。
# 全部为纯函数断言，不依赖 Qt/图像文件；只有 unknown_route_colors 用 numpy 造一张小路线图。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 离屏平台设置，避免任何 Qt 初始化触碰屏幕。

import unittest

import numpy as np  # 构造测试路线图。

import src.map_store as map_store  # 内置默认色表与边缘色常量。
import src.route_palette as rp  # 被测色表模块。


def default_tables():  # 返回 (主表副本, 上下表副本)，每个用例独立修改不污染常量。
    return (dict(map_store.DEFAULT_COLOR_CODE), dict(map_store.DEFAULT_COLOR_CODE_UP_DOWN))


class TestKeyAndCommandFormat(unittest.TestCase):  # 色串与指令串的解析/生成。

    def test_rgb_from_key(self):  # 合法/中文逗号/非法/越界。
        self.assertEqual((255, 0, 0), rp.rgb_from_key('255,0,0'))  # 标准写法。
        self.assertEqual((255, 0, 0), rp.rgb_from_key(' 255，0，0 '))  # 兼容中文逗号与空格。
        self.assertIsNone(rp.rgb_from_key('255,0'))  # 段数不足。
        self.assertIsNone(rp.rgb_from_key('255,0,x'))  # 非数字。
        self.assertIsNone(rp.rgb_from_key('256,0,0'))  # 越界。
        self.assertIsNone(rp.rgb_from_key(None))  # 空值。

    def test_key_from_rgb_roundtrip(self):  # 三元组与串互转保持等价。
        self.assertEqual('12,230,7', rp.key_from_rgb((12, 230, 7)))  # 生成串。
        self.assertEqual((12, 230, 7), rp.rgb_from_key(rp.key_from_rgb((12, 230, 7))))  # 往返一致。

    def test_command_text_and_split(self):  # 指令串补齐与规整。
        self.assertEqual('left none jump', rp.command_text('left', None, 'jump'))  # 缺省补 none。
        self.assertEqual('left none none', rp.command_text(' LEFT ', '', ''))  # 大写与多余空格被规整。
        self.assertEqual(('up', 'none', 'none'), rp.split_command('up'))  # 段数不足时在末尾补 none。
        self.assertEqual(('left', 'none', 'none'), rp.split_command('left none none extra'))  # 多余段丢弃。
        self.assertEqual(('none', 'none', 'none'), rp.split_command(''))  # 空串全 none。


class TestBuildReverse(unittest.TestCase):  # {色串: 指令} -> {指令: 色}。

    def test_hit_both_tables(self):  # 主表与上下表都能反查到。
        main, ud = default_tables()  # 默认两表。
        reverse = rp.build_reverse(main, ud)  # 反查。
        self.assertEqual((255, 0, 0), reverse['left none none'])  # 主表：红=向左。
        self.assertEqual((127, 127, 127), reverse['none up none'])  # 上下表：灰=爬梯。

    def test_main_table_wins_on_duplicate_command(self):  # 同一指令在两表都出现时保留主表色（与人工编辑习惯一致）。
        main = {'1,1,1': 'left none none'}  # 主表色。
        ud = {'2,2,2': 'left none none'}  # 上下表同指令。
        self.assertEqual((1, 1, 1), rp.build_reverse(main, ud)['left none none'])  # 主表优先。

    def test_illegal_color_key_skipped(self):  # 非法色串不进反查表，不影响其他项。
        reverse = rp.build_reverse({'bad,key': 'left none none', '0,0,255': 'right none none'}, {})  # 混入非法键。
        self.assertEqual({'right none none': (0, 0, 255)}, reverse)  # 只保留合法项。

    def test_case_and_space_insensitive_lookup(self):  # 色表里写成大写/多空格也能反查到规范指令。
        reverse = rp.build_reverse({'3,4,5': '  LEFT  none  none '}, {})  # 脏数据。
        self.assertEqual((3, 4, 5), reverse.get(rp.command_text('left', 'none', 'none')))  # 规范串命中。


class TestTargetTable(unittest.TestCase):  # 新色该写进哪张表。

    def test_with_move_x_or_action_goes_main(self):  # 含左右或动作 → 主表。
        self.assertEqual(rp.MAIN_TABLE, rp.target_table('left none none'))  # 有左右。
        self.assertEqual(rp.MAIN_TABLE, rp.target_table('none up jump'))  # 有动作。
        self.assertEqual(rp.MAIN_TABLE, rp.target_table('left up teleport'))  # 都有。

    def test_pure_vertical_goes_up_down(self):  # 纯上下 → 上下表（保证 combine_cmd 互补逻辑能找到）。
        self.assertEqual(rp.UD_TABLE, rp.target_table('none up none'))  # 只上。
        self.assertEqual(rp.UD_TABLE, rp.target_table('none down none'))  # 只下。
        self.assertEqual(rp.UD_TABLE, rp.target_table('none none none'))  # 全 none 也归上下表（不会用到）。


class TestResolveAndAllocate(unittest.TestCase):  # 命中与自动扩色。

    def test_known_command_returns_no_addition(self):  # 色表已有定义时不扩色。
        main, ud = default_tables()  # 默认两表。
        rgb, addition = rp.resolve_color('left none none', main, ud)  # 反查。
        self.assertEqual((255, 0, 0), rgb)  # 命中内置红。
        self.assertIsNone(addition)  # 无新增。

    def test_unknown_command_allocates_color_and_table(self):  # 没定义的组合自动分配新色并给出归属表。
        main, ud = default_tables()  # 默认两表。
        rgb, addition = rp.resolve_color('left up none', main, ud)  # 左+上（表里没有）。
        self.assertIsNotNone(rgb)  # 分到颜色。
        self.assertEqual((rp.MAIN_TABLE, rp.key_from_rgb(rgb), 'left up none'), addition)  # 新增项三元组。
        self.assertTrue(rp.apply_addition(main, ud, addition))  # 写进内存表。
        self.assertEqual(rgb, rp.build_reverse(main, ud)['left up none'])  # 同指令复用同色。

    def test_pure_vertical_addition_goes_ud_table(self):  # 纯上下的新增项必须落上下表，否则回放时 combine_cmd 补不到上下分量。
        rgb, addition = rp.resolve_color('none down none', dict(map_store.DEFAULT_COLOR_CODE), {})  # 上下表清空后重取。
        self.assertIsNotNone(rgb)  # 分到颜色。
        self.assertEqual(rp.UD_TABLE, addition[0])  # 归属上下表。
        main = dict(map_store.DEFAULT_COLOR_CODE)  # 主表。
        ud = {}  # 空上下表。
        self.assertTrue(rp.apply_addition(main, ud, addition))  # 写回上下表。
        self.assertEqual(rgb, rp.build_reverse(main, ud)['none down none'])  # 反查命中同色。
        self.assertNotIn(rp.key_from_rgb(rgb), main)  # 不应污染主表。

    def test_allocates_avoid_builtin_edge_and_black(self):  # 分配色不得与内置指令色/边缘色/黑重复，且色池内部自身不重色。
        main, ud = default_tables()  # 默认两表。
        used = rp.used_colors(main, ud)  # 占用集。
        self.assertIn((0, 0, 0), used)  # 黑保留。
        edge = rp.rgb_from_key(map_store.DEFAULT_EDGE_COLOR)  # 默认边缘色。
        self.assertIn(edge, used)  # 边缘色也在占用集内。
        self.assertEqual(len(set(rp.ALLOC_POOL)), len(rp.ALLOC_POOL))  # 色池自身无重复项。
        for rgb in rp.ALLOC_POOL:  # 逐个检查色池。
            self.assertNotIn(rgb, used, f'色池里的 {rgb} 与内置色冲突')  # 池内色不得已被占用。

    def test_successive_allocations_are_distinct(self):  # 连续扩色互不重复。
        main, ud = default_tables()  # 默认两表。
        used = rp.used_colors(main, ud)  # 起始占用集。
        picked = []  # 已分配色。
        for index in range(10):  # 连续分配 10 个。
            rgb = rp.allocate_color(used)  # 取一个。
            self.assertIsNotNone(rgb)  # 池足够大。
            self.assertNotIn(rgb, picked)  # 不与本局已分配重复。
            picked.append(rgb)  # 记录。
            used.add(rgb)  # 归档占用。

    def test_pool_exhausted_returns_none(self):  # 色池耗尽返回 None，调用方按"本拍不描线"处理。
        used = set(rp.ALLOC_POOL)  # 占满整池。
        self.assertIsNone(rp.allocate_color(used))  # 无可用色。

    def test_resolve_returns_none_when_pool_exhausted(self):  # 扩不出色时指令无落色。
        main, ud = default_tables()  # 默认两表。
        used = set(rp.ALLOC_POOL)  # 人为占满。
        rgb, addition = rp.resolve_color('left up none', main, ud, used)  # 需要扩色。
        self.assertIsNone(rgb)  # 无颜色。
        self.assertIsNone(addition)  # 无新增项。

    def test_apply_addition_noop_for_empty(self):  # 无新增项时不改动色表。
        main, ud = default_tables()  # 默认两表。
        before = dict(main)  # 快照。
        self.assertFalse(rp.apply_addition(main, ud, None))  # 空新增。
        self.assertEqual(before, main)  # 未改动。


class TestUnknownRouteColors(unittest.TestCase):  # 路线图未知色提示。

    def test_detects_colors_missing_from_table(self):  # 色表被清掉后，路线上的色应被列为未知。
        main, ud = default_tables()  # 默认两表。
        img = np.zeros((4, 4, 3), dtype=np.uint8)  # 黑底（无指令）。
        img[0, 0] = (255, 0, 0)  # 内置红：已定义。
        img[1, 1] = (9, 9, 9)  # 色表里没有的色。
        unknown = rp.unknown_route_colors(img, main, ud)  # 检出。
        self.assertEqual(['9,9,9'], unknown)  # 只报未定义色。

    def test_edge_color_can_be_whitelisted(self):  # 自定义边缘色要能通过 edge_rgb 加进白名单，不传则报未知。
        custom = (10, 20, 30)  # 非内置边缘色。
        img = np.zeros((2, 2, 3), dtype=np.uint8)  # 黑底。
        img[0, 0] = custom  # 画一个自定义边缘标记。
        self.assertEqual(['10,20,30'], rp.unknown_route_colors(img, {}, {}))  # 未白名单时报未知。
        self.assertEqual([], rp.unknown_route_colors(img, {}, {}, edge_rgb=custom))  # 白名单生效。

    def test_builtin_edge_color_is_always_reserved(self):  # 内置边缘色始终保留（它不是指令色，不该被报为未知）。
        edge = rp.rgb_from_key(map_store.DEFAULT_EDGE_COLOR)  # 默认边缘色。
        img = np.zeros((2, 2, 3), dtype=np.uint8)  # 黑底。
        img[0, 0] = edge  # 画默认边缘标记。
        self.assertEqual([], rp.unknown_route_colors(img, {}, {}))  # 无需额外传 edge_rgb 也不报。

    def test_empty_image_returns_empty(self):  # 无图/空图返回空列表。
        self.assertEqual([], rp.unknown_route_colors(None, {}, {}))  # None。
        self.assertEqual([], rp.unknown_route_colors(np.zeros((0, 0, 3), dtype=np.uint8), {}, {}))  # 空矩阵。


if __name__ == '__main__':
    unittest.main()
