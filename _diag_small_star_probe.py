# 小面积星星失败局离线探针：直接对录像逐帧跑 detect_white_shapes，看候选存活情况。
# 一次性诊断脚本，不接入主流程。
import sys  # 读取命令行参数。
import cv2  # 读视频帧。
import numpy as np  # 数值。

sys.path.insert(0, ".")  # 保证从仓库根目录导入 src。
from src.liedetector.shape_tracking import detect_white_shapes, resize_for_processing  # 被测函数。

path = sys.argv[1] if len(sys.argv) > 1 else r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 录像路径。
cap = cv2.VideoCapture(path)  # 打开录像。
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧尺寸。
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数。
side = max(w, h)  # 区域长边。
scale = min(0.55, 384.0 / side)  # extreme 档实际处理尺度（与 session._resolve 一致）。
play_area = (w * scale) * (h * scale)  # 处理尺度有效面积。
min_area = max(160.0, play_area * 0.0014)  # 候选面积下限。
print(f"frame={w}x{h} n={n} scale={scale:.4f} proc={w*scale:.0f}x{h*scale:.0f} min_area={min_area:.0f} (全尺度约 {min_area/scale**2:.0f})")  # 门限概览。
idx = 0  # 帧号。
shown = 0  # 已打印行数。
while shown < 40:  # 只看前 40 个有信息量的帧。
    ok, frame = cap.read()  # 逐帧读。
    if not ok:  # 读完为止。
        break
    if idx % 3 == 0:  # 每 3 帧采一次就够看趋势。
        proc = resize_for_processing(frame, scale)  # 缩到处理尺度。
        dets = detect_white_shapes(proc, 1.0)  # 白色候选检测。
        if dets or idx < 150:  # 有候选或在前 150 帧（学习窗口）时打印。
            desc = " | ".join(
                f"area={d.area:.0f} conf={d.confidence:.2f} bbox={tuple(int(v) for v in cv2.boundingRect(d.contour))}"
                for d in dets[:4]
            )
            print(f"f{idx:4d} cands={len(dets):2d} {desc}")  # 候选明细。
            shown += 1
    idx += 1  # 帧号递增。
cap.release()  # 释放。
