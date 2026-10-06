# -*- coding: utf-8 -*-
"""生产打分口径下的三方对比：同一帧证据上，「跟踪器当前锁」「真值折线位置」「全场最优」各得几分。

为什么要做（前面所有争论都卡在这里）：
    9s 之后目标完全透明，跟踪器停在画面顶部而真值在右下方。到底是
      (i)  真值处根本没有证据 —— 那就是信息缺失，任何打分/接管都救不了；还是
      (ii) 真值处证据更强，只是跟踪器出不去自己所在的局部 —— 那是搜索/接管机制的问题，可修。
    本段已经用两种做法（inpaint、prevfill）摘过准星，跟踪器不但没离开反而贴得更紧；
    再把准星整体平移 216px、原位补洞，跟踪器一次都没跟过去（贴假准星 0%）——
    说明「准星是吸引子」这个上一段的定案是错的，必须换个可证伪的量直接问证据本身。

做法（不改任何画面，纯读生产口径）：
    钩住 tracker.observe_border 拿到本帧 temporal.normalized 证据图；用生产同一个
    score_shape_contours 给三批假设打分：
      锁   = 跟踪器自己的 (center, angle, scale)；
      真值 = 真值折线在该帧的位置（角度扫遍对称周期取最优，尺度同锁）；
      全场 = 处理尺度 12px 粗网格、锁角度下的最优（另附其坐标）。
    逐帧打印三者得分与排名，看真值到底赢不赢、赢多少。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_scores.py [t_from] [t_to] [step]
"""

import os  # 路径。
import sys  # 参数。

import cv2  # 解码
import numpy as np  # 数值

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.liedetector import shape_session as ss  # noqa: E402
from src.liedetector.shape_session import ShapeTrackParams, ShapeTrackSession  # noqa: E402
from src.liedetector.shape_tracking import score_shape_contours, _to_numpy_array  # noqa: E402

VIDEO = r"lie_records\20261001_190829_score1.00_extreme.mp4"  # 素材
TURN_TICK, TURN_POS = 430.0, np.array([410.0, 48.0])  # 准星仍在目标上的最后可信点（全尺度）
END_TICK, END_POS = 853.0, np.array([606.4, 401.6])  # 结尾揭示的答案星
V_TRUTH = (END_POS - TURN_POS) / (END_TICK - TURN_TICK)  # 真值速度（右下）
GRID = 12  # 全场粗扫步长（处理尺度像素）


def truth(tick):  # 真值折线（全尺度）
    return TURN_POS + V_TRUTH * (tick - TURN_TICK)


def main():
    t_from = int(sys.argv[1]) if len(sys.argv) > 1 else 440  # 起始帧
    t_to = int(sys.argv[2]) if len(sys.argv) > 2 else 720  # 结束帧
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 6  # 抽样步长
    session = ShapeTrackSession(params=ShapeTrackParams(precision_tier="extreme"), logger=None)  # 生产同参
    captured = {}  # observe_border 收到的证据图
    BOX = {"origin": None}  # 原 observe_border（用容器避免闭包只读）

    def hooked(self, temporal_map):  # 只旁路取证据，不改行为
        captured["map"] = _to_numpy_array(temporal_map)  # 统一下载为 NumPy（显存张量也能读）
        return BOX["origin"](self, temporal_map)  # 原样执行

    tracker_ref = {}  # 跟踪器实例（首帧后取）

    def hook_after_update(session):  # 每帧装一次钩子：跟踪器是延迟创建的
        tr = session.tracker  # 当前跟踪器
        if tr is not None and tracker_ref.get("obj") is not tr:  # 新建或换了实例
            if BOX["origin"] is None:
                BOX["origin"] = type(tr).observe_border  # 保存原实现（类级）
                setattr(type(tr), "observe_border", hooked)  # 类级替换
            tracker_ref["obj"] = tr  # 记录实例

    cap = cv2.VideoCapture(VIDEO)  # 打开素材
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.207  # 帧率
    tick = 0  # 帧号
    wins = 0  # 真值胜过锁的帧数
    total = 0  # 参与统计的帧数
    print(f"{'t':>5} {'相位':>9} {'锁(x,y)':>15} {'锁分':>6} {'真值(x,y)':>14} {'真值分':>7} {'全场分':>7} {'最优(x,y)':>13} {'真值-锁':>8} {'全场-锁':>8}")
    while tick <= t_to:
        ok, frame = cap.read()  # 读一帧
        if not ok:
            break
        if tick == 0:
            session.reset(frame.shape[1], frame.shape[0])  # 建局
        hook_after_update(session)  # 保证跟踪器实例的类已装钩
        result = session.update(frame, float(fps))  # 推进
        tr = session.tracker  # 跟踪器
        if (tick >= t_from and tick % step == 0 and tr is not None and tr.template is not None
                and result.center is not None and captured.get("map") is not None):
            ev = np.asarray(captured["map"], dtype=np.float32)  # 证据图（处理尺度）
            sc = session.effective_scale  # 尺度
            period = tr.template.symmetry_period  # 对称周期
            lock_c, _, lock_a = tr.estimate()  # 锁的中心/角度（处理尺度）
            lock_states = np.zeros((1, 6), np.float32)  # 单假设
            lock_states[0, 0], lock_states[0, 1], lock_states[0, 4] = float(lock_c[0]), float(lock_c[1]), lock_a  # 位姿
            lock_s = float(score_shape_contours(ev, lock_states, tr.template, np.array([tr.scale], np.float32),
                                                tr.play_height, tr.backend).scores[0])  # 锁得分
            tp = truth(tick) * sc  # 真值（处理尺度）
            n_ang = max(2, int(round(period / 15.0)))  # 角度扫描数
            t_states = np.zeros((n_ang, 6), np.float32)  # 真值假设批
            t_states[:, 0], t_states[:, 1] = tp[0], tp[1]  # 位置
            t_states[:, 4] = np.linspace(0.0, period, n_ang, endpoint=False)  # 铺满对称周期
            t_scores = score_shape_contours(ev, t_states, tr.template, np.full(n_ang, tr.scale, np.float32),
                                            tr.play_height, tr.backend).scores  # 打分
            truth_s = float(np.max(t_scores))  # 真值最优
            h, w = ev.shape[:2]  # 证据尺寸
            gx = np.arange(GRID, w - GRID, GRID)  # 粗网格
            gy = np.arange(GRID, h - GRID, GRID)  # 网格 y
            XX, YY = np.meshgrid(gx, gy)  # 展开
            g_states = np.zeros((XX.size, 6), np.float32)  # 网格假设
            g_states[:, 0], g_states[:, 1] = XX.ravel(), YY.ravel()  # 位置
            g_states[:, 4] = lock_a  # 角度同锁（只看位置维的最优）
            g_scores = np.asarray(score_shape_contours(ev, g_states, tr.template, np.full(XX.size, tr.scale, np.float32),
                                                       tr.play_height, tr.backend).scores)  # 打分
            k = int(np.argmax(g_scores))  # 全场最优
            best_s = float(g_scores[k])  # 最优分
            best_xy = (g_states[k, 0] / sc, g_states[k, 1] / sc)  # 全尺度坐标
            total += 1  # 计数
            wins += int(truth_s > lock_s)  # 真值是否胜过锁
            print(f"t{tick:4d} {str(result.source):>9} ({lock_c[0] / sc:6.1f},{lock_c[1] / sc:6.1f}) {lock_s:6.2f} "
                  f"({tp[0] / sc:6.1f},{tp[1] / sc:6.1f}) {truth_s:7.2f} {best_s:7.2f} "
                  f"({best_xy[0]:6.1f},{best_xy[1]:6.1f}) {truth_s - lock_s:+8.2f} {best_s - lock_s:+8.2f}")
        tick += 1
    cap.release()  # 释放
    if BOX["origin"] is not None and session.tracker is not None:  # 卸钩
        setattr(type(session.tracker), "observe_border", BOX["origin"])  # 还原
    if total:
        print(f"\nt{t_from}..{t_to} 共 {total} 帧：真值处得分高于锁的帧数 = {wins}/{total} = {wins / total * 100:.1f}%")
        print("解读：>60% ⇒ 证据里有目标、是跟踪器出不去自己所在局部（可修）；"
              "<30% ⇒ 透明相证据里根本没有目标（信息缺失，得换信息源）。")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径稳定
    raise SystemExit(main())
