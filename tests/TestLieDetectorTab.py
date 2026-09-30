# 测谎检验页签（LieDetectorTab）精度下拉与触发延迟的构造/加载/持久化回归测试。
#
# 设计要点：
#   - 无头离屏构造 Qt 控件（QT_QPA_PLATFORM=offscreen 须在导入 PySide6 前设置），仅做控件状态检查，不真起工作线程。
#   - 全部用例 patch save_dashboard_config，绝不污染仓库里的 configs/Dashboard.json。
#   - 重点覆盖：GPU 五档齐全并回填配置精度/延迟、CPU 移除高/极高/最强并 clamp 到中等且加载期不回写（_loading 守卫）、
#     用户改档合并写回并通知测谎服务热更新、start 把选中精度档传给 worker、worker 保存 tier。
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无显示环境下用离屏平台构造 Qt 控件，须在导入 PySide6 前设置。

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication


class TestLieDetectorTabWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # 全部用例共享一个离屏应用实例。

    def test_precision_combo_gpu_populated_and_loaded(self):
        # GPU 下精度下拉五档齐全，从 Dashboard.json 回填 ultra 与触发延迟 7.5，后端显示 GPU。
        from src.ui.LieDetectorTab import LieDetectorTab
        with patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=True), \
                patch('src.ui.LieDetectorTab.load_dashboard_config',
                      side_effect=lambda: {'Lie Detector Precision': 'ultra', 'Lie Detector Trigger Delay': 7.5}), \
                patch('src.ui.LieDetectorTab.save_dashboard_config'):
            tab = LieDetectorTab()
            self.addCleanup(tab.timer.stop)  # 停掉画面刷新定时器，避免测试结束后回调已销毁控件。
            keys = [tab.precision_combo.itemData(i) for i in range(tab.precision_combo.count())]
            self.assertEqual(['low', 'medium', 'high', 'ultra', 'extreme'], keys)  # GPU 五档齐全，顺序由低到高。
            self.assertEqual('最强', tab.precision_combo.itemText(4))  # 顶级档中文标签为「最强」。
            self.assertIn('GPU', tab.backend_label.text())  # 运算后端显示 GPU。
            self.assertEqual('ultra', tab.precision_combo.currentData())  # 回填选中配置的极高。
            self.assertEqual((0.0, 60.0), (tab.delay_spin.minimum(), tab.delay_spin.maximum()))  # 延迟范围 0~60 秒。
            self.assertEqual(1, tab.delay_spin.decimals())  # 一位小数，与看板一致。
            self.assertEqual(7.5, tab.delay_spin.value())  # 回填触发延迟。

    def test_precision_combo_cpu_clamps_and_no_writeback_on_load(self):
        # CPU 下精度下拉只剩低/中等，配置 high 被 clamp 到 medium；加载期由 _loading 守卫不回写配置。
        from src.ui.LieDetectorTab import LieDetectorTab
        with patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=False), \
                patch('src.ui.LieDetectorTab.load_dashboard_config',
                      side_effect=lambda: {'Lie Detector Precision': 'high', 'Lie Detector Trigger Delay': 5.0}), \
                patch('src.ui.LieDetectorTab.save_dashboard_config') as save:
            tab = LieDetectorTab()
            self.addCleanup(tab.timer.stop)
            keys = [tab.precision_combo.itemData(i) for i in range(tab.precision_combo.count())]
            self.assertEqual(['low', 'medium'], keys)  # CPU 移除高/极高/最强。
            self.assertIn('CPU', tab.backend_label.text())  # 运算后端显示 CPU。
            self.assertEqual('medium', tab.precision_combo.currentData())  # 配置 high 被 clamp 到 medium。
            save.assert_not_called()  # 加载期回填控件值触发的信号被 _loading 守卫拦下，不回写。
            tab._set_precision_value('extreme')  # 默认档（最强）在 CPU 下也不可选。
            self.assertEqual('medium', tab.precision_combo.currentData())  # 同样 clamp 到 medium，不会选中不存在的项。

    def test_persist_merges_config_and_notifies_service(self):
        # 用户改精度/延迟：合并写回 Dashboard.json（保留看板其它字段）并通知测谎服务热更新。
        from src.ui.LieDetectorTab import LieDetectorTab
        with patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=True), \
                patch('src.ui.LieDetectorTab.load_dashboard_config',
                      side_effect=lambda: {'Character Feature': 'role1',
                                           'Lie Detector Precision': 'high',
                                           'Lie Detector Trigger Delay': 5.0}), \
                patch('src.ui.LieDetectorTab.save_dashboard_config') as save, \
                patch('src.ui.LieDetectorTab.og') as og:
            tab = LieDetectorTab()
            self.addCleanup(tab.timer.stop)
            tab._set_precision_value('ultra')  # 用户改精度档，触发持久化。
            tab.delay_spin.setValue(9.0)  # 用户改触发延迟，再次触发持久化。
            saved = save.call_args[0][0]  # 取最后一次落盘的配置字典。
            self.assertEqual('ultra', saved['Lie Detector Precision'])  # 精度写回。
            self.assertEqual(9.0, saved['Lie Detector Trigger Delay'])  # 延迟写回。
            self.assertEqual('role1', saved['Character Feature'])  # 合并写保留看板其它字段，未被覆盖。
            self.assertTrue(og.my_app.lie_service.reload_config.called)  # 通知独立测谎服务热更新配置。

    def test_start_passes_selected_tier_to_worker(self):
        # start() 把精度下拉当前档传给 LieDetectorWorker，实现与线上服务同档复算。
        from src.ui.LieDetectorTab import LieDetectorTab
        with patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=True), \
                patch('src.ui.LieDetectorTab.load_dashboard_config',
                      side_effect=lambda: {'Lie Detector Precision': 'ultra', 'Lie Detector Trigger Delay': 5.0}), \
                patch('src.ui.LieDetectorTab.save_dashboard_config'), \
                patch('src.ui.LieDetectorTab.LieDetectorWorker') as worker_cls:
            tab = LieDetectorTab()
            self.addCleanup(tab.timer.stop)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
                tmp_path = tf.name  # 造一个真实存在的临时 mp4，绕过 start() 的存在性检查。
            self.addCleanup(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
            tab.video_path = tmp_path
            self.assertEqual('ultra', tab.precision_combo.currentData())  # 下拉已选中极高。
            tab.start()
            args = worker_cls.call_args[0]  # 取构造 worker 的实参。
            self.assertEqual(tmp_path, args[0])  # 视频路径。
            self.assertEqual('ultra', args[1])  # 选中精度档传给 worker。
            self.assertIs(tab, args[2])  # parent 为页签自身。

    def test_worker_stores_tier(self):
        # LieDetectorWorker 保存传入的精度档，缺省为 high（构造不起线程，无需真实视频）。
        from src.ui.LieDetectorTab import LieDetectorWorker
        worker = LieDetectorWorker("dummy.mp4", "ultra")
        self.assertEqual("ultra", worker.tier)  # 显式传档。
        default_worker = LieDetectorWorker("dummy.mp4")
        self.assertEqual("high", default_worker.tier)  # 缺省高档。

    def test_worker_stores_trigger_delay(self):
        # LieDetectorWorker 保存触发延迟秒数，缺省 0（不延迟），负值被夹到 0。
        from src.ui.LieDetectorTab import LieDetectorWorker
        self.assertEqual(7.5, LieDetectorWorker("dummy.mp4", "high", None, delay=7.5).delay)  # 显式传延迟。
        self.assertEqual(0.0, LieDetectorWorker("dummy.mp4").delay)  # 缺省不延迟。
        self.assertEqual(0.0, LieDetectorWorker("dummy.mp4", "high", None, delay=-3.0).delay)  # 负值夹到 0。

    def test_start_passes_trigger_delay_to_worker(self):
        # start() 把触发延迟数字框的值以关键字 delay 传给 worker，位置参数仍是 (path, tier, parent)。
        from src.ui.LieDetectorTab import LieDetectorTab
        with patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=True), \
                patch('src.ui.LieDetectorTab.load_dashboard_config',
                      side_effect=lambda: {'Lie Detector Precision': 'ultra', 'Lie Detector Trigger Delay': 6.0}), \
                patch('src.ui.LieDetectorTab.save_dashboard_config'), \
                patch('src.ui.LieDetectorTab.LieDetectorWorker') as worker_cls:
            tab = LieDetectorTab()
            self.addCleanup(tab.timer.stop)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
                tmp_path = tf.name  # 造一个真实存在的临时 mp4，绕过 start() 的存在性检查。
            self.addCleanup(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
            tab.video_path = tmp_path
            self.assertEqual(6.0, tab.delay_spin.value())  # 延迟数字框已回填 6.0。
            tab.start()
            call = worker_cls.call_args
            self.assertEqual((tmp_path, 'ultra', tab), call[0])  # 位置参数不变：路径、精度档、父对象。
            self.assertEqual(6.0, call.kwargs['delay'])  # 触发延迟以关键字传入。

    # ------------------------------------------------------------------ 历史记录：结果置末尾与删除

    def _build_tab(self, records):
        # 构造一个离屏页签：屏蔽显卡探测/配置读写，并把历史列举定向到传入的假记录（不碰仓库 lie_records）。
        from src.ui.LieDetectorTab import LieDetectorTab
        stack = [patch('src.ui.LieDetectorTab.gpu_backend_available', return_value=True),
                 patch('src.ui.LieDetectorTab.load_dashboard_config',
                       side_effect=lambda: {'Lie Detector Precision': 'high', 'Lie Detector Trigger Delay': 5.0}),
                 patch('src.ui.LieDetectorTab.save_dashboard_config'),
                 patch('src.ui.LieDetectorTab.list_records', return_value=records)]
        for p in stack:
            p.start()
            self.addCleanup(p.stop)
        tab = LieDetectorTab()
        self.addCleanup(tab.timer.stop)
        return tab

    def test_format_history_label_result_readable_at_end(self):
        # 历史摘要把结果映射为中文（success->成功/failure->失败）并固定放在每条记录的最后。
        tab = self._build_tab([])
        rec = {'timestamp': '2026-09-30T12:34:00', 'score': 0.92, 'tier': 'high', 'outcome': 'success'}
        label = tab._format_history_label(rec)
        self.assertEqual('09-30 12:34 分0.92 高 成功', label)  # 时间 分 档 结果，结果在末尾且为中文。
        rec['outcome'] = 'failure'
        self.assertTrue(tab._format_history_label(rec).endswith('失败'))  # 失败结果同样置末尾。
        rec['outcome'] = 'weird'
        self.assertTrue(tab._format_history_label(rec).endswith('weird'))  # 未知结果原样显，不报错。

    def test_delete_selected_history_removes_and_refreshes(self):
        # 选中一条历史录像点删除：弹确认框确认后按 mp4 路径删除，当前视频被删则清空选择并复位标签。
        rec = {'timestamp': '2026-09-30T12:34:00', 'score': 0.9, 'tier': 'high', 'outcome': 'success',
               'video': 'rec1.mp4', 'path': os.path.join('lie_records', 'rec1.mp4')}
        tab = self._build_tab([rec])
        with patch('src.ui.LieDetectorTab.delete_record', return_value=True) as delete, \
                patch('qfluentwidgets.Dialog') as dialog_cls:
            dialog_cls.return_value.exec.return_value = 1  # 用户确认删除。
            tab.history_combo.setCurrentIndex(1)  # 选中第一条真实记录（0 为占位）。
            tab.video_path = rec['path']  # 模拟当前正验证该录像。
            tab._delete_selected_history()
        delete.assert_called_once_with(rec['path'])  # 按选中项 mp4 路径删除。
        self.assertEqual('', tab.video_path)  # 删的是当前视频 -> 清空选择。
        self.assertEqual('未选择视频', tab.path_label.text())  # 路径标签复位。

    def test_delete_selected_history_cancel_does_not_delete(self):
        # 确认框取消时不调删除，也不改当前选择。
        rec = {'timestamp': '2026-09-30T12:34:00', 'score': 0.9, 'tier': 'high', 'outcome': 'success',
               'video': 'rec1.mp4', 'path': os.path.join('lie_records', 'rec1.mp4')}
        tab = self._build_tab([rec])
        with patch('src.ui.LieDetectorTab.delete_record', return_value=True) as delete, \
                patch('qfluentwidgets.Dialog') as dialog_cls:
            dialog_cls.return_value.exec.return_value = 0  # 用户取消。
            tab.history_combo.setCurrentIndex(1)
            tab._delete_selected_history()
        delete.assert_not_called()  # 取消不删。

    def test_delete_selected_history_no_selection_noop(self):
        # 选中占位首项（无路径）时：不弹确认框、不删除，仅记日志。
        tab = self._build_tab([])
        with patch('src.ui.LieDetectorTab.delete_record') as delete, \
                patch('qfluentwidgets.Dialog') as dialog_cls:
            tab.history_combo.setCurrentIndex(0)  # 占位首项，data=None。
            tab._delete_selected_history()
        delete.assert_not_called()  # 无可删。
        dialog_cls.assert_not_called()  # 无可删时不弹确认框。


if __name__ == "__main__":
    unittest.main()
