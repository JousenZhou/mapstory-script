# -*- coding: utf-8 -*-
"""把指定帧号的原图按全尺度存成 PNG，供肉眼核对（结尾揭示阶段尤其关键）。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_dump.py <t0> <t1> <step> [up]
产物：_diag_out\\dump_t<tick>.png（upscale 倍放大，默认 1）
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码与写图
import numpy as np  # 数值

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
OUT = r"_diag_out"  # 产物目录


def main():
    t0 = int(sys.argv[1]) if len(sys.argv) > 1 else 830  # 起
    t1 = int(sys.argv[2]) if len(sys.argv) > 2 else 880  # 止
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # 步长
    up = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # 放大倍率
    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    tick = 0  # 帧号
    while tick <= t1:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick >= t0 and tick % step == 0:  # 目标帧
            out = cv2.resize(frame, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST) if up != 1.0 else frame  # 放大
            path = os.path.join(OUT, f"dump_t{tick}.png")  # 产物名
            cv2.imwrite(path, out)  # 写盘
            print(f"t{tick}: {path} 尺寸 {out.shape[1]}x{out.shape[0]}")
        tick += 1
    cap.release()  # 释放
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
