# -*- coding: utf-8 -*-
"""验证 9s 真值模型：从旋转窗实测点 t702=(610.6,333.7) 用 YOLO 检测反向接龙到 9s，看是否落在白相外推的位置上。

两条独立证据必须对上，才算真值成立：
    证据 A（白相外推）：白色轮廓阶段 t19..336 是游戏亲自描出的答案，位置精确；t336 在 (401.7,134.4)，
        之后先上行、约 t420..450 在顶部掉头，再往右下走到 t853=(606.4,401.6)。
    证据 B（旋转实测）：探针 #311 在旋转窗 t702..776 实测答案 (610.6,333.7)→(594.4,412.6)。
把 B 的起点用逐帧 YOLO 检测反向接龙（透明相位答案可检），若 t≈452 落在 A 的掉头区，模型成立。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_verify.py
"""

import json  # 读检测缓存。
import sys  # 参数（起点帧 / 方向 / 自检点）。

import numpy as np  # 数值。

CACHE = "_diag_out/yolo_dets_190829.json"  # 逐帧检测中心缓存（由 _diag_lie_field_drift.py 生成）。
START_TICK = 702  # 反向接龙起点（探针 #311 起点帧）。
START_POS = np.array([610.6, 333.7])  # 起点坐标。
END_TICK = 400  # 反到 9s 之前。
GATE = 22.0  # 逐帧接龙门限（答案自身速度约 1.4px/帧，留 15px 余量）。


def main():
    # 命令行可改写起点与方向：argv = [start_tick, x, y, end_tick, gate]
    start_tick = int(sys.argv[1]) if len(sys.argv) > 1 else START_TICK  # 起点帧。
    pos0 = np.array([float(sys.argv[2]), float(sys.argv[3])]) if len(sys.argv) > 3 else START_POS  # 起点。
    end_tick = int(sys.argv[4]) if len(sys.argv) > 4 else END_TICK  # 终点帧。
    gate = float(sys.argv[5]) if len(sys.argv) > 5 else GATE  # 门限。
    step = -1 if end_tick < start_tick else 1  # 方向自动跟随起止顺序。
    frames = {int(k): v for k, v in json.load(open(CACHE, encoding="utf-8")).items()}  # tick -> [[x,y],...]
    pos = pos0.copy()  # 当前位置
    vel = np.zeros(2)  # 常速
    rows = []  # 输出
    for t in range(start_tick, end_tick + step, step):
        dets = np.array(frames.get(t, []), dtype=np.float64) if frames.get(t) else np.zeros((0, 2))  # 本帧检测
        if len(dets):
            dist = np.linalg.norm(dets - pos, axis=1)  # 距预测
            k = int(np.argmin(dist))
            if dist[k] <= gate:  # 接上
                vel = 0.7 * vel + 0.3 * (dets[k] - pos)
                pos = dets[k]
                rows.append((t, pos.copy(), float(dist[k]), False))
                continue
        pos = pos + vel  # 接不上：常速外推
        rows.append((t, pos.copy(), -1.0, True))
    filled = sum(1 for r in rows if r[3])  # 补帧数
    print(f"接龙 t{start_tick}..t{end_tick} 步长 {step} 门限 {gate} 补帧 {filled}/{len(rows)}")
    for t, p, d, gap in rows:
        if t % 10 == 0:
            print(f"t{t:4d} 答案=({p[0]:6.1f},{p[1]:6.1f}) 接单距={d:5.1f}{' 补' if gap else ''}")
    for t, want in ((702, (610.6, 333.7)), (776, (594.4, 412.6)), (760, (593.7, 412.8)), (452, None)):  # 自检
        row = [r for r in rows if r[0] == t]
        if not row:
            continue
        got = row[0][1]
        if want is None:
            print(f"关键帧 t{t}: ({got[0]:.1f},{got[1]:.1f}) 补帧={row[0][3]}")
            continue
        err = float(np.linalg.norm(got - np.array(want)))
        print(f"自检 t{t}: 链=({got[0]:6.1f},{got[1]:6.1f}) 旋转实测=({want[0]:6.1f},{want[1]:6.1f}) 误差={err:6.1f}px"
              f"{'  <== 链可信' if err < 30 else '  <== 链已认错了形状'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
