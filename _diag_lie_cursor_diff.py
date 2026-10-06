# -*- coding: utf-8 -*-
"""准星足迹掩膜的零扰动对照：逐素材比较 curon/curoff 两次回放的逐帧中心偏差，并量出掩膜面积占比。

判据：掩膜只该遮「小面积高对比绿色十字」，所以
  - 足迹面积占比（遮掉的像素 / 全图）应远小于 5%；
  - 无绿色准星的素材应逐比特等于关掩膜（0 扰动）；
  - 有准星的素材允许中心改变，但必须逐帧自己核对是否变得更贴目标（本脚本只报偏差，判读看轨迹表）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_cursor_diff.py [tag_on tag_off]
"""

import glob  # 找回放产物。
import json  # 读 JSON。
import os  # 路径。
import sys  # 参数。

import cv2  # 解码素材量绿色像素
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "_diag_out"  # 回放产物目录
RECORD_DIR = "lie_records"  # 生产录像目录
CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"  # 外部参考视频
TIER = "extreme"  # 只比最强档
GREEN_MIN = 90  # 与生产同口径的绿色判据
GREEN_DOMINANCE = 12
DILATE = 7  # 与生产 cursor_mask_dilate 同口径，量「实际被遮面积」而非绿像素面积


def load(tag, name):  # 读某 tag 某素材的回放产物。
    hits = [h for h in glob.glob(os.path.join(OUT_DIR, f"{tag}_{TIER}_{name}.json")) if f"_{TIER}_" in h]
    if not hits:
        return None
    with open(sorted(hits)[-1], "r", encoding="utf-8") as handle:
        return json.load(handle)


def green_share(path, scale, frames, step=17):  # 抽样量：最大绿色连通块膨胀后的面积占全图比例（= 实际被遮比例）。
    cap = cv2.VideoCapture(path)  # 打开素材
    shares, tick = [], 0  # 抽样结果
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1  # 总帧数（用于定步长）
    picks = set(range(0, total, max(1, total // frames)))  # 均匀抽 frames 帧
    while True:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick in picks:  # 命中抽样
            proc = cv2.resize(frame, None, fx=scale, fy=scale)  # 处理尺度
            b = proc[:, :, 0].astype(np.int16)  # 拆通道
            g = proc[:, :, 1].astype(np.int16)
            r = proc[:, :, 2].astype(np.int16)
            green = (((g > GREEN_MIN) & (g - np.maximum(r, b) > GREEN_DOMINANCE)).astype(np.uint8)) * 255  # 绿主导
            count, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)  # 连通块
            if count > 1:  # 有绿色块
                blob = (labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8) * 255  # 最大块
                blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE)))  # 同生产膨胀
                shares.append(float(cv2.countNonZero(blob)) / float(proc.shape[0] * proc.shape[1]))  # 被遮占比
        tick += 1
    cap.release()  # 释放
    return shares


def stats(payload):  # 逐素材质量指标：置信、信噪、丢观测帧数（掩膜不该让这些变差）。
    rows = payload["frames"]
    conf = [r[4] for r in rows if r[4] is not None]
    snr = [r[5] for r in rows if len(r) > 5 and r[5] is not None]
    return {
        "conf": float(np.mean(conf)) if conf else float("nan"),
        "snr": float(np.mean(snr)) if snr else float("nan"),
        "pred": sum(1 for r in rows if r[1] == "prediction"),
        "none": sum(1 for r in rows if r[2] is None),
    }


def main():
    tags = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("curon", "curoff")  # 开/关掩膜两个 tag
    names = sorted({os.path.basename(h).split(f"_{TIER}_")[1][:-5] for h in
                    glob.glob(os.path.join(OUT_DIR, f"{tags[0]}_{TIER}_*.json"))})  # 两侧都有的素材
    print(f"对照 {tags[0]}(掩膜开) vs {tags[1]}(掩膜关)  tier={TIER}")
    print(f"{'素材':38} {'可比帧':>6} {'中位偏差':>8} {'p95偏差':>8} {'最大':>7} {'被遮中位':>9} {'被遮最大':>9} {'source差异':>26}")
    for name in names:
        on, off = load(tags[0], name), load(tags[1], name)  # 两侧产物
        if on is None or off is None:
            print(f"{name:38} 缺产物")
            continue
        rows_on = {r[0]: r for r in on["frames"]}  # tick → 帧
        rows_off = {r[0]: r for r in off["frames"]}  # 同上
        common = sorted(set(rows_on) & set(rows_off))  # 共同帧
        dev, diff_src = [], set()  # 偏差与 source 差异
        for t in common:
            a, b = rows_on[t], rows_off[t]  # 两侧行
            if a[2] is not None and b[2] is not None:  # 都有中心
                dev.append(float(np.hypot(a[2] - b[2], a[3] - b[3])))
            if a[1] != b[1]:  # source 不同
                diff_src.add(f"{t}:{b[1]}→{a[1]}")
        path = os.path.join(RECORD_DIR, name)  # 素材路径
        if not os.path.exists(path):  # 外部参考视频
            path = os.path.join(CASE_DIR, name)
        shares = green_share(path, float(on.get("effective_scale", 0.5)), 40) if os.path.exists(path) else []  # 抽样掩膜强度
        d = np.array(dev) if dev else np.array([np.nan])  # 偏差数组
        med = float(np.nanmedian(d))  # 中位
        p95 = float(np.nanpercentile(d, 95))  # p95
        mx = float(np.nanmax(d))  # 最大
        flag = "零扰动" if mx < 0.5 else ("轻微" if mx < 20 else "显著")  # 判定标签
        print(f"{name:38} {len(common):6d} {med:8.2f} {p95:8.2f} {mx:7.1f} "
              f"{(np.median(shares) * 100 if shares else 0.0):8.2f}% "
              f"{(np.max(shares) * 100 if shares else 0.0):8.2f}% "
              f"{flag + ' | ' + ', '.join(sorted(diff_src)[:4]) if diff_src else flag:>26}")

    # 第二表：偏差大不等于变差 —— 用置信/信噪/丢观测帧数判方向（掩膜开应不劣于关）。
    print("\n== 质量方向对照（遮 vs 不遮；pred=纯预测帧，none=无中心帧）==")
    print(f"{'素材':38} {'conf遮':>7} {'conf不':>7} {'snr遮':>7} {'snr不':>7} {'pred遮':>7} {'pred不':>7} {'判定':>10}")
    for name in names:
        on, off = load(tags[0], name), load(tags[1], name)
        if on is None or off is None:
            continue
        a, b = stats(on), stats(off)
        good = a["conf"] >= b["conf"] - 0.02 and a["pred"] <= b["pred"] + 5 and a["none"] <= b["none"] + 5
        print(f"{name:38} {a['conf']:7.3f} {b['conf']:7.3f} {a['snr']:7.2f} {b['snr']:7.2f} "
              f"{a['pred']:7d} {b['pred']:7d} {'不劣化' if good else '需复核':>10}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
