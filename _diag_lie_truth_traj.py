# -*- coding: utf-8 -*-
"""给「答案图形」建立全程真值轨迹，作为所有修复方案的评分地基。

为什么不能直接用探针端点：探针 winner #311 只在旋转期（帧 702–776）有残差证据，端点 (610.6,333.7)→(594.4,412.6)
之外的帧没有证据；而本素材**相机在持续平移**（地形整体在画面里移动），所以答案图形在旋转前后都会漂到别处。
把真值写死成 (594,412) 会在大部分帧上误判（实测帧 806 之后星形已漂走，红叉却停在右下）。

做法：以帧 702 为参考、探针 start 为锚点，**相邻帧相位相关（cv2.phaseCorrelate）累计相机平移**，
真值(t) = 锚点 + (累计平移(t) − 累计平移(702))。自检：外推到帧 776 应接近探针 end=(594.4,412.6)。
（试过 findTransformECC(MOTION_EUCLIDEAN) 直接对参考帧全帧对齐：位移超出搜索域，除参考帧本身外 cc 仅 0.1~0.2
 且解回退成单位变换，故改为相邻帧小位移递推。）

用法：
    .venv\\Scripts\\python.exe _diag_lie_truth_traj.py                 # 默认 190829
    .venv\\Scripts\\python.exe _diag_lie_truth_traj.py 165429          # 换素材

输出：
    _diag_out/truth/traj_<tag>.json   tick -> [x, y]（含自检误差、右下窗口帧表）
    _diag_out/truth/traj_<tag>.png    每 40 帧一格，绿圈应压在答案星形上（肉眼可核）
"""

import glob  # 素材过滤。
import json  # 落盘。
import os  # 路径。
import sys  # 参数。

import cv2  # 图像与对齐。
import numpy as np  # 矩阵。

RECORD_DIR = "lie_records"  # 录像目录。
OUT_DIR = os.path.join("_diag_out", "truth")  # 输出目录。
REF_TICK = 702  # 参考帧（探针 winner 出生帧）。
ANCHOR = (610.6, 333.7)  # 参考帧上的答案图形坐标（探针 winner #311 start）。
CHECK_TICK, CHECK_END = 776, (594.4, 412.6)  # 自检帧与期望值（探针 winner end）。
RIGHT_BOTTOM = (500.0, 330.0)  # 「右下角」判定的最小 x/y（与探针 winner 判据同口径）。
GRID_STRIDE = 40  # 叠图取样步长。


def find_video(name_filter):  # 取匹配的录像。
    hits = [p for p in sorted(glob.glob(os.path.join(RECORD_DIR, "*.mp4"))) if name_filter in os.path.basename(p)]
    return hits[-1] if hits else None


def frame_shift(prev, cur):  # 相邻帧平移 (dx, dy, 可信度)；黑屏/无纹理/位移过大返回 None。
    if float(cur.std()) < 6.0 or float(prev.std()) < 6.0:  # 黑屏（场景结束）无纹理可配。
        return None
    win = np.hanning(prev.shape[0])[:, None] * np.hanning(prev.shape[1])[None, :]  # 汉宁窗抑制边缘绕回。
    (dx, dy), response = cv2.phaseCorrelate(prev.astype(np.float32) * win, cur.astype(np.float32) * win)  # 相位相关。
    if response < 0.02 or abs(dx) > 40 or abs(dy) > 40:  # 单帧位移上限，超界当配准失败（宁可断档也不要跳飞）。
        return None
    return float(dx), float(dy), float(response)


def scan(path, keep_ticks):  # 一次流式遍历：返回逐帧累计平移、可信度、留存的核对帧。
    cap = cv2.VideoCapture(path)  # 打开。
    cumulative = [(0.0, 0.0)]  # cumulative[t] = 自帧 0 起累计平移（配不准的帧沿用上一值）。
    confidence = [None]  # 每帧相位相关可信度。
    frames = {}  # tick -> 灰度图（仅 keep）。
    prev = None  # 上一帧。
    tick = 0
    while True:
        ok, frame = cap.read()  # 读一帧。
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰。
        if prev is not None:  # 递推。
            shift = frame_shift(prev, gray)  # 相邻位移。
            if shift is not None:
                cumulative.append((cumulative[-1][0] + shift[0], cumulative[-1][1] + shift[1]))  # 累加。
                confidence.append(shift[2])  # 记可信度。
            else:
                cumulative.append(cumulative[-1])  # 沿用（视为无位移）。
                confidence.append(None)  # 标断档。
        if tick in keep_ticks:  # 留存核对帧。
            frames[tick] = gray.copy()  # 存图。
        prev = gray  # 滚动。
        tick += 1
    cap.release()  # 释放。
    return cumulative, confidence, frames, tick


def main():  # 入口。
    name_filter = sys.argv[1] if len(sys.argv) > 1 else "190829"  # 素材。
    path = find_video(name_filter)  # 找录像。
    if path is None:
        print(f"no video matched {name_filter}")
        return 1
    cap = cv2.VideoCapture(path)  # 探元数据。
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 总帧数。
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0  # 容器帧率。
    cap.release()  # 释放。
    total = min(total, 940)  # 上限保护。
    keep_ticks = {t for t in range(0, total, GRID_STRIDE)}  # 核对帧集合。
    cumulative, confidence, frames, read_count = scan(path, keep_ticks)  # 流式扫描。
    if REF_TICK >= len(cumulative):
        print(f"reference frame {REF_TICK} not reached (read {read_count})")
        return 1
    base = cumulative[REF_TICK]  # 差分基准。
    traj = {}  # tick -> [x, y, cc]
    for tick in range(0, len(cumulative), GRID_STRIDE // 8 or 1):  # 输出步长 5 帧。
        dx = cumulative[tick][0] - base[0]  # 相对参考帧的相机平移。
        dy = cumulative[tick][1] - base[1]  # 同上。
        traj[tick] = [round(ANCHOR[0] + dx, 1), round(ANCHOR[1] + dy, 1), confidence[tick]]  # 真值 + 可信度。
    check = min(traj, key=lambda t: abs(t - CHECK_TICK))  # 自检帧（取最近采样）。
    self_err = round(float(np.hypot(traj[check][0] - CHECK_END[0], traj[check][1] - CHECK_END[1])), 1)  # 自检误差。
    rb = [t for t, v in traj.items() if v[0] >= RIGHT_BOTTOM[0] and v[1] >= RIGHT_BOTTOM[1]]  # 真值落在右下的帧。
    tag = os.path.splitext(os.path.basename(path))[0].split("_")[1]  # 短 tag。
    os.makedirs(OUT_DIR, exist_ok=True)  # 建目录。
    with open(os.path.join(OUT_DIR, f"traj_{tag}.json"), "w", encoding="utf-8") as handle:  # 落盘。
        json.dump({"video": os.path.basename(path), "ref_tick": REF_TICK, "anchor": list(ANCHOR), "src_fps": fps,
                   "self_check_tick": check, "self_check_err": self_err, "right_bottom_min": list(RIGHT_BOTTOM),
                   "right_bottom_ticks": rb, "traj": traj}, handle)
    tiles = []  # 叠图块。
    for tick in sorted(keep_ticks):  # 按序出图。
        gray = frames.get(tick)  # 核对帧。
        if gray is None or tick not in traj:
            continue
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 转彩以便画色标。
        x, y = int(traj[tick][0]), int(traj[tick][1])  # 真值。
        if 0 <= x < color.shape[1] and 0 <= y < color.shape[0]:  # 在画面内才画。
            cv2.circle(color, (x, y), 45, (0, 255, 0), 1)  # 命中圈。
            cv2.drawMarker(color, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)  # 十字。
        cv2.putText(color, f"t{tick} truth=({x},{y}) cc={traj[tick][2]}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)  # 标签。
        tiles.append(color[:320, 340:728])  # 裁右半区域放大观察。
    if tiles:  # 拼 3 列网格。
        width_min = min(t.shape[1] for t in tiles)  # 统一宽。
        height_min = min(t.shape[0] for t in tiles)  # 统一高。
        tiles = [t[:height_min, :width_min] for t in tiles]  # 裁齐。
        rows = []  # 行图。
        for start in range(0, len(tiles), 3):  # 分行。
            row_tiles = tiles[start:start + 3]  # 本行。
            while len(row_tiles) < 3:  # 末行补齐，避免 vconcat 断言。
                row_tiles.append(np.zeros_like(tiles[0]))  # 黑块。
            rows.append(cv2.hconcat(row_tiles))  # 拼行。
        grid = cv2.vconcat(rows)  # 堆叠。
        out = os.path.join(OUT_DIR, f"traj_{tag}.png")
        cv2.imwrite(out, grid)  # 落盘。
        print(f"grid -> {out} ({grid.shape[1]}x{grid.shape[0]})")
    print(f"video={os.path.basename(path)} frames={len(cumulative)} traj_points={len(traj)} "
          f"selfcheck t{check} err={self_err}px (expect<=45)")
    print(f"truth-in-right-bottom ticks: {rb[:5]} ... {rb[-5:] if rb else []} (count {len(rb)})")
    for tick in (200, 300, 400, 500, 510, 560, 600, 617, 660, 700, 740, 775, 805, 845, 865):  # 关键帧表。
        near = min(traj, key=lambda t: abs(t - tick))  # 最近采样点。
        print(f"  t{tick:>4} ({tick / fps:5.1f}s@{fps:.0f}fps | {tick / 30.0:5.1f}s@30fps) "
              f"truth=({traj[near][0]:6.1f},{traj[near][1]:6.1f}) cc={traj[near][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
