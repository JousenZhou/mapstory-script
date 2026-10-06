# -*- coding: utf-8 -*-
"""9s 窗肉眼取证：每帧叠上四方候选标记，直接看图判断答案究竟跟谁重合。

标记约定（便于同一张图里区分）：
    绿十字 = 回放跟踪中心（被测者）
    品字   = 游戏内绿色准星（玩家鼠标，白相已证明它会瞄答案）
    黄圈   = 「t452(395,55) → t702(610.6,333.7) 匀速右下」真值线上的预测点
    橙圈   = 旋转见证器候选
    青圈   = 该帧离黄圈最近的 YOLO 透明形状检测

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_frames.py [t_from] [t_to] [step]
"""

import glob  # 找回放产物。
import json  # 读。
import os  # 路径。
import sys  # 参数。

import cv2  # 画图与解码。
import numpy as np  # 数值。

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
FRAMES_TAG = os.environ.get("FRAMES_TAG", "anchor2").strip().lower() or "anchor2"  # 看哪一轮回放（验收看 curcap=掩膜开、nomask=掩膜关）。
REPLAY = sorted(glob.glob(f"_diag_out/{FRAMES_TAG}_extreme_20261001_190829_*.json"))[-1]  # 回放。
CURSOR = "_diag_out/cursor_traj_190829.json"  # 准星轨迹。
DETS = "_diag_out/yolo_dets_190829.json"  # YOLO 检测中心。
OUT_DIR = f"_diag_out/nine_s_marked_{FRAMES_TAG}"  # 输出目录（按 tag 分开，不覆盖旧的）。
A_TICK, A_POS = 452, np.array([395.0, 55.0])  # 真值线起点。
B_TICK, B_POS = 702, np.array([610.6, 333.7])  # 真值线终点。


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 452  # 起始帧。
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 640  # 结束帧。
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 20  # 存图步长。
    rows = {r[0]: r for r in json.load(open(REPLAY, encoding="utf-8"))["frames"]}  # 回放。
    cur = {int(k): v for k, v in json.load(open(CURSOR, encoding="utf-8")).items()}  # 准星。
    dets = {int(k): v for k, v in json.load(open(DETS, encoding="utf-8")).items()}  # 检测。
    os.makedirs(OUT_DIR, exist_ok=True)
    vel = (B_POS - A_POS) / (B_TICK - A_TICK)  # 匀速。
    wanted = list(range(t_from, t_to + 1, step))
    cap = cv2.VideoCapture(VIDEO)
    tick = 0
    saved = []
    while tick <= t_to:
        ok, frame = cap.read()
        if not ok:
            break
        if tick in wanted:
            canvas = frame.copy()
            pred = A_POS + vel * (tick - A_TICK)  # 真值线点。
            cv2.circle(canvas, tuple(pred.astype(int)), 14, (0, 255, 255), 2)  # 黄圈(真值线) 用 BGR=(0,255,255)。
            cv2.line(canvas, (int(pred[0]) - 22, int(pred[1])), (int(pred[0]) + 22, int(pred[1])), (0, 255, 255), 1)
            cv2.line(canvas, (int(pred[0]), int(pred[1]) - 22), (int(pred[0]), int(pred[1]) + 22), (0, 255, 255), 1)
            dd = np.array(dets.get(tick, []), dtype=np.float64)  # 本帧检测。
            if len(dd):
                d = np.linalg.norm(dd - pred, axis=1)
                k = int(np.argmin(d))
                if d[k] < 120:
                    cv2.circle(canvas, tuple(dd[k].astype(int)), 45, (255, 255, 0), 2)  # 青圈=最近检测。
            c = cur.get(tick)
            if c:
                cv2.drawMarker(canvas, (int(c[0]), int(c[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            row = rows.get(tick)
            if row and row[2] is not None:
                cv2.drawMarker(canvas, (int(row[2]), int(row[3])), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            if row and row[6]:
                cv2.circle(canvas, (int(row[6][0]), int(row[6][1])), 12, (0, 128, 255), 2)
            cv2.putText(canvas, f"t{tick} yel=truthline grn=tracker mag=cursor org=witness cyn=yolo",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            path = os.path.join(OUT_DIR, f"m{tick}.png")
            cv2.imwrite(path, canvas)
            saved.append(path)
        tick += 1
    cap.release()
    print("已存：" + " ".join(saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
