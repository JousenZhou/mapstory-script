# 测谎触发录像记录（LieRecorder）回归测试。
#
# 设计要点：
#   - 全部用例把 recorder.LIE_RECORD_DIR 重定向到临时目录，绝不往仓库里的 lie_records/ 写测试产物。
#   - 依赖 mp4v 编码器的用例（真录一段 mp4）在编码器不可用时自动 skipTest，其余纯文件逻辑仍全量验证。
#   - 重点覆盖：start/write/stop 产出 mp4+json、边车字段正确、滚动保留按 mtime 删旧、
#     list_records 倒序且跳过缺失 mp4 的脏记录、以及各条「最佳努力」守卫（异常/非法输入一律吞掉不抛）。
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.liedetector import recorder as recorder_module  # 被测模块：patch 其 LIE_RECORD_DIR / LIE_RECORD_KEEP。
from src.liedetector.recorder import LieRecorder, delete_record, list_records, LIE_RECORD_FPS  # 录像器、历史列举/删除与帧率常量。


def _frame(value, height=120, width=160):
    """合成一张 (height,width,3) uint8 的纯色 BGR 帧，模拟一次触发采集到的原始画面。"""

    return np.full((height, width, 3), value % 256, dtype=np.uint8)


def _touch_pair(folder, stem, mtime):
    """在 folder 下造一对同名 mp4+json 占位文件并把 mtime 设为指定值，用于滚动保留/列举排序测试。"""

    paths = []
    for ext in (".mp4", ".json"):
        path = os.path.join(folder, stem + ext)
        with open(path, "w", encoding="utf-8"):
            pass  # 占位空文件即可，测试只关心存在性与 mtime。
        os.utime(path, (mtime, mtime))  # 显式设置修改时间，制造确定的新旧顺序。
        paths.append(path)
    return paths


class TestLieRecorder(unittest.TestCase):
    """LieRecorder：起停录像、边车记录、滚动保留与历史列举。"""

    def setUp(self):
        # 每个用例独享一个临时目录，并把 recorder 的录像目录常量指过去，测试产物随目录一起清理。
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(recorder_module, "LIE_RECORD_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ------------------------------------------------------------------ 录像产出与边车

    def test_start_write_stop_produces_mp4_and_json(self):
        # start -> 写若干帧 -> stop：临时目录下产出一对同名 mp4+json，边车字段与录像内容一致。
        rec = LieRecorder()
        rec.start((120, 160), {"score": 0.79, "tier": "high"})  # 起流：分辨率 (高, 宽) + 元数据。
        if rec._writer is None:  # 本机 mp4v 编码器不可用（start 已吞掉并置空写入流）。
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        for i in range(8):  # 写 8 帧原始画面。
            rec.write(_frame(i * 10))
        rec.stop("solved")  # 收尾：排空写线程 -> 关流 -> 写边车。

        mp4s = [n for n in os.listdir(self.tmp) if n.endswith(".mp4")]
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        self.assertEqual(1, len(mp4s), "应产出一个 mp4")
        self.assertEqual(1, len(jsons), "应产出一个 json 边车")
        self.assertEqual(os.path.splitext(mp4s[0])[0], os.path.splitext(jsons[0])[0], "mp4 与 json 应同名")
        self.assertGreater(os.path.getsize(os.path.join(self.tmp, mp4s[0])), 0, "mp4 不应为空文件")

        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertEqual(mp4s[0], data["video"], "边车记录的 video 应为 mp4 文件名")
        self.assertEqual(0.79, data["score"], "触发分应写入边车")
        self.assertEqual("high", data["tier"], "精度档应写入边车")
        self.assertEqual("solved", data["outcome"], "结束原因应写入边车")
        self.assertEqual(8, data["frames"], "写入帧数应等于实际录帧数")
        self.assertEqual(LIE_RECORD_FPS, data["fps"], "帧率应为常量 LIE_RECORD_FPS")
        self.assertEqual(160, data["width"], "边车宽应为起流宽度")
        self.assertEqual(120, data["height"], "边车高应为起流高度")
        self.assertEqual(round(8 / float(LIE_RECORD_FPS), 3), data["duration"], "时长 = 帧数 / 帧率")
        self.assertTrue(data["timestamp"], "应记录起录时间戳")
        self.assertIn("score0.79", mp4s[0], "文件名应含触发分")
        self.assertIn("high", mp4s[0], "文件名应含精度档")

    def test_slow_capture_writes_real_frames_without_padding(self):
        # 采集慢于标称帧率时，写线程不再用上一帧补齐：帧数==实际采集次数（杜绝重复帧），真实慢帧率只记进边车 measured_fps。
        rec = LieRecorder()
        rec.start((60, 80), {"score": 0.5, "tier": "ultra"})
        if rec._writer is None:  # 本机 mp4v 编码器不可用。
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        for i in range(6):  # 以 ~10fps 采集 6 帧，横跨约 0.5s（远慢于标称 30fps）。
            rec.write(_frame(i * 10, height=60, width=80))
            time.sleep(0.1)
        rec.stop("solved")
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            data = json.load(f)
        # 不补帧：写多少帧就存多少帧，帧数恒等于采集次数（旧版会补齐到远多于 6）。
        self.assertEqual(6, data["frames"], "慢采集不再补帧，帧数应等于实际采集次数")
        self.assertEqual(round(6 / float(LIE_RECORD_FPS), 3), data["duration"], "容器时长 = 帧数 / 标称帧率（不补帧后回放会变快）")
        # measured_fps 应反映真实慢采集（明显低于标称 30fps），而非被补帧到 30。
        self.assertIn("measured_fps", data, "边车应记录真实采集帧率 measured_fps")
        self.assertGreater(data["measured_fps"], 0.0, "measured_fps 应为正")
        self.assertLess(data["measured_fps"], float(LIE_RECORD_FPS), "慢采集时 measured_fps 应明显低于标称帧率")

    def test_self_capture_records_frames_without_external_write(self):
        # 自采集模式：start 传 capture 回调后，录像器起独立线程按 LIE_RECORD_FPS 拓帧写入，无需外部 write；
        # 帧数随采集时间增长、measured_fps 接近标称（与慢采集补帧场景的 ~10fps 区分）。
        rec = LieRecorder()
        counter = {"n": 0}

        def fake_capture():  # 每次返回一张互异纯色帧，模拟独立 30fps 采集。
            counter["n"] += 1
            return _frame(counter["n"] * 7, height=60, width=80)

        rec.start((60, 80), {"score": 0.5, "tier": "extreme"}, None, capture=fake_capture)
        if rec._writer is None:  # 本机 mp4v 编码器不可用。
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        time.sleep(0.5)  # 让采集线程独立跑约 0.5s（~15 帧@30fps）。
        rec.stop("solved")
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertGreaterEqual(counter["n"], 5, "自采集线程应在 0.5s 内独立抓多帧")
        self.assertGreaterEqual(data["frames"], 5, "写入帧数应随采集增长（无需外部 write）")
        self.assertLessEqual(abs(data["frames"] - counter["n"]), 2, "写入帧数应≈采集次数（仅收尾竞态可能差 1~2 帧）")
        self.assertGreater(data["measured_fps"], 12.0, "自采集按 30fps 节拍，真实帧率应接近标称（远高于慢采集的 ~10fps）")

    def test_write_is_noop_in_self_capture_mode(self):
        # 自采集模式下手动 write 应被忽略（录像只由采集线程驱动），避免与采集线程重复喂帧。
        rec = LieRecorder()
        rec.start((60, 80), {"score": 0.5, "tier": "high"}, None, capture=lambda: None)  # 采集恒返回 None：采集线程不产帧。
        if rec._writer is None:  # 本机 mp4v 编码器不可用。
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        for i in range(5):
            rec.write(_frame(i * 10, height=60, width=80))  # 自采集模式下这些 write 应全部空转。
        rec.stop("solved")
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertEqual(0, data["frames"], "自采集模式下手动 write 应被忽略，帧数为 0")

    def test_outcome_is_recorded_in_sidecar(self):
        # 不同结束原因（gone/timeout/abandoned）都应如实写进边车 outcome 字段。
        for outcome in ("gone", "timeout", "abandoned"):
            with self.subTest(outcome=outcome):
                folder = os.path.join(self.tmp, outcome)  # 每种原因单独子目录，避免文件名/清理相互干扰。
                os.makedirs(folder, exist_ok=True)
                with patch.object(recorder_module, "LIE_RECORD_DIR", folder):
                    rec = LieRecorder()
                    rec.start((60, 80), {"score": 0.5, "tier": "medium"})
                    if rec._writer is None:
                        rec.stop(outcome)
                        self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
                    rec.write(_frame(30, height=60, width=80))
                    rec.stop(outcome)
                    jsons = [n for n in os.listdir(folder) if n.endswith(".json")]
                    self.assertEqual(1, len(jsons))
                    with open(os.path.join(folder, jsons[0]), encoding="utf-8-sig") as f:
                        self.assertEqual(outcome, json.load(f)["outcome"])

    def test_start_with_crop_records_region_only(self):
        # 指定裁剪区域起录：只录【测谎区域标注】区域，输出分辨率收敛为区域尺寸，边车记录 region 字段。
        rec = LieRecorder()
        rec.start((120, 160), {"score": 0.66, "tier": "ultra"}, (10, 20, 50, 60))  # 整帧 120x160，裁剪区域 (x=10,y=20,w=50,h=60)。
        if rec._writer is None:  # 本机 mp4v 编码器不可用。
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        self.assertEqual((10, 20, 50, 60), rec._crop, "有效裁剪区域应被采纳")
        self.assertEqual(50, rec._width, "输出宽应收敛为区域宽")
        self.assertEqual(60, rec._height, "输出高应收敛为区域高")
        for i in range(6):  # 写 6 帧整帧，录像器内部裁剪为 50x60 区域后入队。
            rec.write(_frame(i * 10))
        rec.stop("solved")
        mp4s = [n for n in os.listdir(self.tmp) if n.endswith(".mp4")]
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        self.assertEqual(1, len(mp4s), "应产出一个 mp4")
        self.assertGreater(os.path.getsize(os.path.join(self.tmp, mp4s[0])), 0, "mp4 不应为空文件")
        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertEqual([10, 20, 50, 60], data["region"], "边车应记录裁剪区域 [x,y,w,h]")
        self.assertEqual(50, data["width"], "边车宽应为区域宽")
        self.assertEqual(60, data["height"], "边车高应为区域高")
        self.assertEqual(6, data["frames"], "写入帧数应等于实际录帧数")

    def test_crop_out_of_bounds_falls_back_to_full_frame(self):
        # 裁剪区域完全越界：夹取后退化，回退录整帧，边车 region 为 None。
        rec = LieRecorder()
        rec.start((120, 160), {"score": 0.5, "tier": "high"}, (999, 999, 50, 50))  # 区域左上角已在帧外。
        if rec._writer is None:
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        self.assertIsNone(rec._crop, "完全越界的裁剪区域应退回录整帧（None）")
        self.assertEqual(160, rec._width, "回退整帧宽")
        self.assertEqual(120, rec._height, "回退整帧高")
        rec.write(_frame(10))
        rec.stop("solved")
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        with open(os.path.join(self.tmp, jsons[0]), encoding="utf-8-sig") as f:
            self.assertIsNone(json.load(f)["region"], "回退整帧时边车 region 应为 None")

    def test_resolve_crop_clamps_to_frame(self):
        # _resolve_crop 纯逻辑：帧内区域原样、跨边界夹取、完全越界/非法结构退回 None。
        rec = LieRecorder()
        self.assertEqual((10, 20, 30, 40), rec._resolve_crop((10, 20, 30, 40), 160, 120), "帧内区域原样返回")
        self.assertEqual((150, 110, 10, 10), rec._resolve_crop((150, 110, 50, 50), 160, 120), "跨右下边界夹取到帧内")
        self.assertIsNone(rec._resolve_crop((999, 999, 50, 50), 160, 120), "完全越界退回 None")
        self.assertIsNone(rec._resolve_crop(None, 160, 120), "未指定区域返回 None（录整帧）")
        self.assertIsNone(rec._resolve_crop("bad", 160, 120), "非法结构返回 None")

    # ------------------------------------------------------------------ 最佳努力守卫

    def test_start_invalid_shape_is_skipped(self):
        # 非法分辨率（截图异常等）不起流、不产出边车，也不抛异常。
        rec = LieRecorder()
        rec.start((0, 0), {"score": 0.5, "tier": "high"})
        self.assertIsNone(rec._writer, "非法分辨率不应创建写入流")
        rec.write(_frame(10))  # 未起流时写帧应为空转。
        rec.stop("solved")
        self.assertEqual([], [n for n in os.listdir(self.tmp) if n.endswith(".json")], "非法分辨率不应产出边车")

    def test_writer_open_failure_is_swallowed(self):
        # VideoWriter 打不开（编码器缺失/路径不可写）时 start 吞掉异常、标记未起流，后续 write/stop 全空转。
        fake_writer = type("W", (), {"isOpened": lambda self: False, "release": lambda self: None,
                                     "write": lambda self, frame: None})()
        with patch.object(recorder_module.cv2, "VideoWriter", return_value=fake_writer):
            rec = LieRecorder()
            rec.start((120, 160), {"score": 0.5, "tier": "high"})  # 不应抛异常。
            self.assertIsNone(rec._writer, "打不开写入流时应置空")
            rec.write(_frame(10))  # 空转。
            rec.stop("solved")  # 空转，不产出边车。
        self.assertEqual([], [n for n in os.listdir(self.tmp) if n.endswith(".json")], "起流失败不应产出边车")

    def test_write_before_start_and_stop_without_start_are_noop(self):
        # 未起流就 write、以及从未起流就 stop，都必须安全空转不抛异常。
        rec = LieRecorder()
        rec.write(_frame(10))  # start 之前写帧：空转。
        rec.stop("solved")  # 从未起流就收尾：空转。
        self.assertEqual([], os.listdir(self.tmp), "空转不应产生任何文件")

    def test_write_after_stop_is_noop(self):
        # stop 之后再 write 应被忽略（_running 已置 False），不得复活写线程或抛异常。
        rec = LieRecorder()
        rec.start((60, 80), {"score": 0.5, "tier": "low"})
        if rec._writer is None:
            rec.stop("solved")
            self.skipTest("mp4v 编码器不可用，跳过真实录像产出用例")
        rec.write(_frame(10, height=60, width=80))
        rec.stop("solved")
        rec.write(_frame(20, height=60, width=80))  # 收尾后写帧：应被忽略。
        jsons = [n for n in os.listdir(self.tmp) if n.endswith(".json")]
        self.assertEqual(1, len(jsons), "stop 后再 write 不应产生第二条记录")

    # ------------------------------------------------------------------ 滚动保留

    def test_prune_keeps_most_recent_groups(self):
        # 超过 LIE_RECORD_KEEP 组时按 mtime 删最旧，mp4 与 json 成对删除，只留最近 N 组。
        for i in range(5):  # 造 5 组占位记录，mtime 递增（i 越大越新）。
            _touch_pair(self.tmp, f"rec{i}", time.time() - (100 - i * 10))
        with patch.object(recorder_module, "LIE_RECORD_KEEP", 3):  # 只保留最近 3 组。
            LieRecorder()._prune()
        remaining = sorted(os.listdir(self.tmp))
        self.assertEqual(["rec2.json", "rec2.mp4", "rec3.json", "rec3.mp4", "rec4.json", "rec4.mp4"], remaining,
                         "应删掉最旧的 rec0/rec1，保留最近 3 组")

    def test_prune_tolerates_missing_dir(self):
        # 录像目录不存在时 _prune 直接返回，不抛异常（服务从未触发过录像的场景）。
        with patch.object(recorder_module, "LIE_RECORD_DIR", os.path.join(self.tmp, "not_exist")):
            LieRecorder()._prune()  # 不应抛异常。

    # ------------------------------------------------------------------ 历史列举

    def test_list_records_orders_desc_and_skips_missing_mp4(self):
        # list_records 按边车 mtime 倒序返回，path 拼到传入目录；mp4 缺失的脏记录被跳过。
        for i, stem in enumerate(["old", "mid", "new"]):  # 三条正常记录，mtime 递增。
            mp4 = os.path.join(self.tmp, stem + ".mp4")
            with open(mp4, "w", encoding="utf-8"):
                pass
            json_path = os.path.join(self.tmp, stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"video": stem + ".mp4", "score": 0.1 * i, "tier": "high", "outcome": "solved"}, f)
            mtime = time.time() - (100 - i * 10)
            os.utime(json_path, (mtime, mtime))  # 排序只看边车 mtime。
        # 脏记录：有 json 但 mp4 不存在，应被跳过。
        with open(os.path.join(self.tmp, "ghost.json"), "w", encoding="utf-8") as f:
            json.dump({"video": "ghost.mp4", "tier": "high"}, f)

        records = list_records(folder=self.tmp)
        self.assertEqual(["new", "mid", "old"], [os.path.splitext(r["video"])[0] for r in records],
                         "应按时间倒序（新->旧）返回，且跳过缺失 mp4 的记录")
        self.assertEqual(os.path.join(self.tmp, "new.mp4"), records[0]["path"], "path 应拼到传入目录下")

    def test_list_records_missing_dir_returns_empty(self):
        # 目录不存在时返回空列表，供 UI 下拉安全兜底。
        self.assertEqual([], list_records(folder=os.path.join(self.tmp, "not_exist")))

    # ------------------------------------------------------------------ 历史删除

    def test_delete_record_removes_mp4_and_json_pair(self):
        # delete_record 按 mp4 路径成对删除同名 mp4+json，返回 True，其余记录不受影响。
        _touch_pair(self.tmp, "keep", time.time())
        mp4, json_path = _touch_pair(self.tmp, "drop", time.time())
        removed = delete_record(mp4, folder=self.tmp)  # 传入 mp4 路径与临时目录。
        self.assertTrue(removed, "删掉 mp4 应返回 True")
        self.assertFalse(os.path.exists(mp4), "mp4 应被删除")
        self.assertFalse(os.path.exists(json_path), "同名 json 边车应一并删除")
        self.assertEqual(["keep.json", "keep.mp4"], sorted(os.listdir(self.tmp)), "只删选中那组，其余保留")

    def test_delete_record_by_json_path_also_works(self):
        # 传入 json 路径也能按同名 stem 成对删除（UI 传 mp4 路径，此处验证口径健壮性）。
        mp4, json_path = _touch_pair(self.tmp, "rec", time.time())
        self.assertTrue(delete_record(json_path, folder=self.tmp), "按 json 路径也应删掉 mp4 并返回 True")
        self.assertFalse(os.path.exists(mp4) or os.path.exists(json_path), "mp4+json 都应被删除")

    def test_delete_record_missing_returns_false_without_raise(self):
        # 删除不存在的记录（或空基名）返回 False，不抛异常。
        self.assertFalse(delete_record(os.path.join(self.tmp, "ghost.mp4"), folder=self.tmp), "mp4 不存在应返回 False")
        self.assertFalse(delete_record("", folder=self.tmp), "空路径应返回 False")
        self.assertFalse(delete_record(os.path.join(self.tmp, "x.mp4"), folder=os.path.join(self.tmp, "not_exist")), "目录不存在应返回 False")


if __name__ == "__main__":
    unittest.main()
