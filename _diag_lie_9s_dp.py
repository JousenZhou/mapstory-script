# -*- coding: utf-8 -*-
"""9s 窗逐帧真值：把证据图的全图打分存成「得分体」，再用带速度态的 Viterbi 串出最可能的移动轨迹。

为什么前面的逐帧贪心不行（本脚本要解决的取证瓶颈）：
    全图扫描（_diag_lie_9s_scan.py）显示证据里答案轮廓的最强匹配从 t440 (413,65) 一路右下走到
    t760 (641,377)，但单帧最优在周期星形地形里逐帧抖动 ±40~60px：贪心串链 58% 帧接不上，
    而且「当前锁」的单帧得分常常更高（margin 普遍为负）。单帧得分定不了真值，必须换成
    「一整段路径的累计得分 + 匀速性」这个统计量——正是 Viterbi 在做的事。

采样步长为什么取 10 帧：
    真目标每帧约走 1.2 全尺度像素（t440→t720 位移 (192,276)/280 帧），比位置网格还细；
    每 2 帧采样时步内位移不足一格，速度态退化成「原地不动」，匀速约束失效。取 10 帧后
    每步位移约 7 全尺度 ≈ 2 格，速度离散才有意义。

两遍走（solve 可反复重跑调参）：
    scan  ：生产路径逐帧推进，旁路抓证据图，对位置×角度网格打分，落盘 (T,nx,ny) 得分体；
            同帧另算「跟踪中心邻域最佳得分」= 锁得分，作为对照轨迹。
    solve ：读得分体做带速度态 Viterbi（转移代价 = 速度突变惩罚 + 速度绝对值惩罚 + 硬门限），
            回溯最优路径，逐帧与跟踪中心对比，并给出「路径累计得分 vs 锁累计得分」差值。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_dp.py scan  [t_from] [t_to] [step]
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_dp.py solve [lambda_smooth] [mu_slow] [max_disp]
"""

import os  # 路径。
import sys  # 参数。
import time  # 计时。

import cv2  # 解码录像。
import numpy as np  # 数值。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector import shape_tracking as st  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材。
NPZ = "_diag_out/nine_s_dp.npz"  # 得分体落盘。
PATH_NPZ = "_diag_out/nine_s_dp_path.npz"  # 解出的轨迹落盘。
TIER = "extreme"  # 生产默认档。
GRID = 4.0  # 位置网格步长（处理尺度像素）。
ANGLE_STEP = 18.0  # 角度步长（度）：对称周期 72° 铺 4 个。
LOCK_RADIUS = 10.0  # 锁得分的局部搜索半径（处理尺度像素）。
MAX_DISP = 12.0  # 每步最大位移（处理尺度像素）：10 帧真位移约 3.5px，留 3 倍余量。
ANCHORS = {440: (413.0, 65.0), 700: (605.0, 293.0), 720: (605.5, 339.3)}  # 自检锚点（全尺度）。


def _install_capture():  # 装旁路抓证据图，原样转发不改变生产行为；返回抓取字典。
    captured = {}  # 本帧抓取结果。
    origin = st.ParticleShapeTracker.observe_border  # 原实现。

    def patched(self, temporal_map):  # 拦截入口。
        captured["ev"] = st._to_numpy_array(temporal_map)  # 证据图 (H,W)。
        captured["tr"] = self  # 跟踪器引用。
        return origin(self, temporal_map)

    st.ParticleShapeTracker.observe_border = patched  # 打补丁。
    return captured


def _make_grid(ev, template, play_height):  # 生成位置与角度网格（处理尺度）。
    rx = float(template.nominal_size) * 0.5  # 贴边留出的半径。
    xs = np.arange(rx, ev.shape[1] - rx, GRID, dtype=np.float32)  # x 网格。
    ys = np.arange(rx, min(play_height, ev.shape[0]) - rx, GRID, dtype=np.float32)  # y 网格。
    period = float(template.symmetry_period or 360.0)  # 对称周期。
    angles = np.arange(0.0, period, ANGLE_STEP, dtype=np.float32)  # 角度网格。
    return xs, ys, angles


def _score_at(ev, template, play_height, states, scale):  # 批量打分，返回 (N,) 主存得分。
    out = st.score_shape_contours(ev, states.astype(np.float32), template, scale, play_height)  # 打分。
    return np.asarray(out.scores, dtype=np.float32)  # 得分。


def _surface(ev, template, play_height, xs, ys, angles, scale):  # 全图打分 → (nx,ny) 角度最大值。
    X, Y, A = np.meshgrid(xs, ys, angles, indexing="ij")  # 三轴。
    states = np.zeros((X.size, 6), dtype=np.float32)  # 只用 x,y,angle 三维。
    states[:, 0] = X.ravel()  # 列。
    states[:, 1] = Y.ravel()  # 行。
    states[:, 4] = A.ravel()  # 角。
    scores = _score_at(ev, template, play_height, states, scale)  # 打分。
    return scores.reshape(len(xs), len(ys), len(angles)).max(axis=2)  # 角度取最大：路径不约束自转。


def _lock_score(ev, template, play_height, tracker, center_full, p_scale):  # 跟踪中心邻域最佳得分。
    period = float(template.symmetry_period or 360.0)  # 对称周期。
    angle = float(tracker.estimate()[2])  # 当前角度。
    offs = np.arange(-LOCK_RADIUS, LOCK_RADIUS + GRID * 0.5, GRID, dtype=np.float32)  # 局部位置偏移。
    ang_off = np.array([-ANGLE_STEP, 0.0, ANGLE_STEP], dtype=np.float32)  # 局部角度偏移。
    DX, DY, DA = np.meshgrid(offs, offs, ang_off, indexing="ij")  # 三轴。
    states = np.zeros((DX.size, 6), dtype=np.float32)  # 假设矩阵。
    states[:, 0] = center_full[0] * p_scale + DX.ravel()  # 处理尺度 = 全尺度 × 处理尺度。
    states[:, 1] = center_full[1] * p_scale + DY.ravel()  # 处理尺度 y。
    states[:, 4] = (angle + DA.ravel()) % period  # 围绕当前角度。
    return float(np.max(_score_at(ev, template, play_height, states, float(tracker.scale))))  # 邻域最佳。


def do_scan(t_from, t_to, step):  # 第一遍：逐帧打分落盘。
    captured = _install_capture()  # 装旁路。
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier=TIER), logger=None)  # 生产同参。
    cap = cv2.VideoCapture(VIDEO)  # 打开素材。
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率。
    surfaces, meta = [], []  # 得分体与逐帧元数据。
    xs = ys = angles = None  # 网格轴（首帧确定后复用）。
    play_height = 0  # 有效高度（首帧确定）。
    p_scale = 0.5  # 处理尺度（首帧确定）。
    tick = 0  # 帧号。
    started = time.time()  # 计时。
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧。
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局。
        captured.clear()  # 重置旁路。
        result = session.update(frame, float(fps))  # 生产路径推进。
        if tick >= t_from and (tick - t_from) % step == 0 and captured.get("ev") is not None and session.template is not None:
            ev, tracker = captured["ev"], captured["tr"]  # 证据图与跟踪器。
            p_scale = float(session.effective_scale)  # 处理尺度。
            play_height = int(frame.shape[0] * session.params.play_height_ratio * p_scale)  # 有效高度。
            if xs is None:  # 首帧定网格。
                xs, ys, angles = _make_grid(ev, session.template, play_height)  # 网格。
                print(f"证据 {ev.shape} 网格 {len(xs)}x{len(ys)}x{len(angles)} = "
                      f"{len(xs)*len(ys)*len(angles)} 假设/帧  nominal={session.template.nominal_size:.1f}")
            surfaces.append(_surface(ev, session.template, play_height, xs, ys, angles, float(tracker.scale)).astype(np.float16))  # 存得分体。
            center = np.array(result.center, dtype=np.float32) if result.center is not None else np.full(2, np.nan)  # 跟踪中心。
            lock = _lock_score(ev, session.template, play_height, tracker, center, p_scale) if np.isfinite(center).all() else float("nan")  # 锁得分。
            meta.append([tick, center[0], center[1], lock, result.confidence, result.border_snr, float(tracker.scale)])  # 元数据。
            if len(meta) % 8 == 0:
                print(f"  t{tick:4d} 已扫 {len(meta)} 帧 用时 {time.time() - started:.1f}s", flush=True)
        tick += 1
    cap.release()  # 释放。
    os.makedirs("_diag_out", exist_ok=True)  # 输出目录。
    np.savez_compressed(NPZ, surfaces=np.stack(surfaces), xs=xs, ys=ys,  # 得分体与轴。
                        meta=np.array(meta, dtype=np.float64), grid=GRID, angle_step=ANGLE_STEP,  # 元数据与参数。
                        scale=p_scale, play_height=play_height)  # 尺度与高度。
    print(f"落盘 {NPZ}：{len(surfaces)} 帧，用时 {time.time() - started:.1f}s")
    return 0


def _shift(arr, dx, dy):  # 把 (nx,ny) 表按格位移 (dx,dy) 平移：out[i,j] = arr[i-dx, j-dy]，越界填 +inf。
    nx, ny = arr.shape  # 形状。
    out = np.full_like(arr, np.inf)  # 默认不可达。
    ix = slice(max(dx, 0), nx - max(-dx, 0))  # 目标 x 段。
    jx = slice(max(-dx, 0), nx - max(dx, 0))  # 源 x 段。
    iy = slice(max(dy, 0), ny - max(-dy, 0))  # 目标 y 段。
    jy = slice(max(-dy, 0), ny - max(dy, 0))  # 源 y 段。
    out[ix, iy] = arr[jx, jy]  # 平移赋值（两组切片长度相同）。
    return out  # 平移结果。


def do_solve(lam, mu, max_disp):  # 第二遍：带速度态 Viterbi。
    data = np.load(NPZ)  # 读得分体。
    surf = data["surfaces"].astype(np.float32)  # (T,nx,ny)。
    xs, ys, meta = data["xs"], data["ys"], data["meta"]  # 轴与元数据。
    p_scale = float(data["scale"])  # 处理尺度。
    ticks = meta[:, 0].astype(int)  # 帧号。
    emit = -surf  # 发射代价（得分越高代价越低）。
    nx, ny = surf.shape[1], surf.shape[2]  # 网格尺寸。
    reach = int(np.floor(max_disp / GRID))  # 每步最大格位移。
    vel = [(a, b) for a in range(-reach, reach + 1) for b in range(-reach, reach + 1)  # 速度集合。
           if a * a + b * b <= reach * reach]
    va = np.array([GRID * v[0] for v in vel], dtype=np.float32)  # 速度 x（处理尺度）。
    vb = np.array([GRID * v[1] for v in vel], dtype=np.float32)  # 速度 y。
    nv = len(vel)  # 速度个数。
    print(f"Viterbi：T={len(surf)} 网格 {nx}x{ny} 速度态 {nv}（每步位移 ≤ {max_disp}px） λ={lam} μ={mu}")
    cost = np.broadcast_to(emit[0][:, :, None], (nx, ny, nv)).astype(np.float32).copy()  # 首帧：各速度代价相同。
    back = np.zeros((len(surf), nx, ny, nv), dtype=np.uint8)  # 回溯表：上一帧的速度索引。
    for t in range(1, len(surf)):  # 逐帧递推。
        best = np.full((nx, ny, nv), np.inf, dtype=np.float32)  # 本帧最优累计代价。
        from_v = np.zeros((nx, ny, nv), dtype=np.uint8)  # 最优入速度索引。
        for u in range(nv):  # 上一帧速度。
            shifted = _shift(cost[:, :, u], vel[u][0], vel[u][1])  # (nx,ny)：从 (i-u, j-u) 迁来（旧速度承载位移）。
            if not np.isfinite(shifted).any():
                continue  # 整片不可达。
            pen = lam * ((va - va[u]) ** 2 + (vb - vb[u]) ** 2) + mu * (va ** 2 + vb ** 2)  # 速度突变 + 速度绝对值惩罚（对新速度 v）。
            cand = shifted[:, :, None] + pen[None, None, :] + emit[t][:, :, None]  # 候选代价。
            upd = cand < best  # 刷新的格子。
            best = np.where(upd, cand, best)  # 取小。
            from_v = np.where(upd, np.uint8(u), from_v)  # 记录入速度。
        cost = best  # 推进。
        back[t] = from_v  # 存档。
    flat = np.argmin(cost)  # 终态最优索引。
    i, j, v = np.unravel_index(int(flat), cost.shape)  # 终点格子与速度。
    path = []  # 回溯结果。
    for t in range(len(surf) - 1, -1, -1):  # 逐帧回退。
        path.append((t, i, j, v))  # 记录。
        v = int(back[t][i, j, v]) if t > 0 else v  # 上一帧速度。
        i -= vel[v][0]  # 回退 x。
        j -= vel[v][1]  # 回退 y。
        i = int(np.clip(i, 0, nx - 1))  # 越界钳制（首帧边界）。
        j = int(np.clip(j, 0, ny - 1))  # 越界钳制。
    path = path[::-1]  # 正向。
    idx = np.array([(p[1], p[2]) for p in path])  # 格子序列。
    vel_used = np.array([p[3] for p in path])  # 速度序列。
    pts = np.column_stack([xs[idx[:, 0]] / p_scale, ys[idx[:, 1]] / p_scale]).astype(np.float32)  # 全尺度轨迹。
    score = np.array([surf[t, i2, j2] for t, (i2, j2) in zip(range(len(path)), idx)], dtype=np.float32)  # 路径得分。
    lock = meta[:, 3]  # 锁得分。
    center = meta[:, 1:3]  # 跟踪中心。
    d = np.linalg.norm(pts - center, axis=1)  # 与跟踪中心逐帧距离。
    print(f"路径累计均分 {np.nanmean(score):.2f}  锁累计均分 {np.nanmean(lock):.2f}  Δ均分 {np.nanmean(score)-np.nanmean(lock):+.2f}")
    print(f"路径端点 t{ticks[0]}=({pts[0,0]:.1f},{pts[0,1]:.1f}) → t{ticks[-1]}=({pts[-1,0]:.1f},{pts[-1,1]:.1f})  "
          f"Δ与跟踪中心：中位 {np.nanmedian(d):.1f} P90 {np.nanpercentile(d,90):.1f} 最大 {np.nanmax(d):.1f}")
    print("锚点自检（DP 路径 vs 独立锚）：" + "  ".join(
        f"t{t} Δ={np.linalg.norm(pts[int(np.where(ticks == t)[0][0])] - np.array(p)):.1f}px"
        for t, p in ANCHORS.items() if t in ticks))
    for k in range(len(ticks)):  # 全表打印（每步 10 帧，一共几十行）。
        print(f"t{ticks[k]:4d} DP=({pts[k,0]:6.1f},{pts[k,1]:6.1f}) 得分={score[k]:5.2f} 锁得分={lock[k]:5.2f} "
              f"margin={score[k]-lock[k]:+5.2f} 跟踪=({center[k,0]:6.1f},{center[k,1]:6.1f}) Δ={d[k]:6.1f} "
              f"conf={meta[k,4]:.2f} snr={meta[k,5]:.2f} 速度=({va[vel_used[k]]:.1f},{vb[vel_used[k]]:.1f})")
    np.savez_compressed(PATH_NPZ, pts=pts, score=score, ticks=ticks, meta=meta,  # 落盘轨迹。
                        lam=lam, mu=mu, max_disp=max_disp)  # 参数。
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"  # 模式。
    if mode == "scan":  # 第一遍。
        raise SystemExit(do_scan(int(sys.argv[2]) if len(sys.argv) > 2 else 440,
                                 int(sys.argv[3]) if len(sys.argv) > 3 else 760,
                                 int(sys.argv[4]) if len(sys.argv) > 4 else 10))
    raise SystemExit(do_solve(float(sys.argv[2]) if len(sys.argv) > 2 else 0.30,
                              float(sys.argv[3]) if len(sys.argv) > 3 else 0.02,
                              float(sys.argv[4]) if len(sys.argv) > 4 else MAX_DISP))
