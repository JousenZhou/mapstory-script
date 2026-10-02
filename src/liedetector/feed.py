"""测谎「逐帧喂入层」：实时解题与离线回放共用的单一口径工具集（纯计算，无 I/O、不依赖 Qt）。

存在意义（消除"回放准、实时不准"的口径差）：
- 帧间隔 → fps 的换算只在这里发生一次，实时不再写死 30fps、回放不再写死 src_fps，
  两边都按「上一次真实喂入帧到本帧的真实间隔 dt」换算，时间阈值口径一致。
- 区域 reset 判定（首帧、位移/尺寸超阈值）只在这里实现一次，实时与回放共用，杜绝两份手写漂移。
- 实时把每 tick 的喂帧日程（dt、相对裁剪区域、是否 reset、命中的 source/conf）录成 feed_trace
  写进录像边车；回放「仿实时模式」据这份日程精确重喂同一帧子集、同一 dt，复现实时的掉帧与阈值行为。

设计边界（明确不假装解决）：离线无法让游戏响应鼠标，故不复现「跟踪结果→移鼠标→改变下一帧画面」的
物理闭环本身；但 mp4 录的正是实时移动鼠标之后的画面序列，仿实时按同一天花板重放这份画面，即间接把
闭环的影响体现在回放里。区域以「相对录像裁剪原点的偏移」记录，保证在已裁剪的 mp4 上正确复切。
"""

from __future__ import annotations

from dataclasses import dataclass  # 单步喂帧记录的数据类。
from typing import Iterable, Optional, Sequence  # 类型标注。

# 有效帧率的合理夹取区间：下限防开局卡顿把 dt 拉爆导致阈值过长，上限防瞬时连写把阈值压没。
EFFECTIVE_FPS_MIN = 5.0  # 有效帧率下限。
EFFECTIVE_FPS_MAX = 60.0  # 有效帧率上限（与录像采集节拍上限一致）。
EFFECTIVE_FPS_DEFAULT = 30.0  # dt 非法时的兜底帧率（算法即按 30fps 设计）。
REGION_RESET_THRESHOLD = 4  # 区域任一边位移/尺寸变化超过该像素数视为需重置会话（与旧两份实现同量级）。


def compute_effective_fps(dt: float, default_fps: float = EFFECTIVE_FPS_DEFAULT) -> float:
    """把「本次喂入相对上次喂入的真实间隔 dt（秒）」换算成 session.update 用的帧率口径。

    dt>1e-6 时返回 clamp(1/dt, [EFFECTIVE_FPS_MIN, EFFECTIVE_FPS_MAX])；否则回退 default_fps。
    这是实时与回放共用的唯一 fps 来源：session.update 只用它做「秒↔帧数」阈值换算，
    喂入节奏真实多快，这里就如实反映多快，阈值不再因写死 30 而漂。
    """

    try:
        value = float(dt)
    except (TypeError, ValueError):
        return float(default_fps)
    if value <= 1e-6:
        return float(default_fps)
    fps = 1.0 / value
    return max(EFFECTIVE_FPS_MIN, min(fps, EFFECTIVE_FPS_MAX))


def should_reset(
    last_region: Optional[Sequence[float]],
    new_region: Optional[Sequence[float]],
    threshold: int = REGION_RESET_THRESHOLD,
) -> bool:
    """判断本帧区域相对上次是否需要重置光流会话（清空模板/粒子/光流历史）。

    规则与两份旧实现合并：new_region 为空不重置（沿用旧区域继续）；last_region 为空（首帧）需重置；
    否则任一边（x/y/w/h）变化 > threshold 需重置。实时与回放都走这里，行为一致。
    """

    if new_region is None:  # 本帧没采到区域：不因此重置（沿用上一有效区域）。
        return False
    if last_region is None:  # 首次出现区域：必须重置以新尺寸建会话。
        return True
    for i in range(4):  # 逐边比较 x/y/w/h。
        if abs(float(new_region[i]) - float(last_region[i])) > threshold:
            return True
    return False


@dataclass
class FeedStep:
    """实时解题循环一次「真正喂入 session」的日程记录，供边车持久化、回放仿实时复现。

    dt_ms：本帧距上次喂入的真实毫秒数（首帧为自起点耗时）。
    rect：本帧实际使用的裁剪区域，坐标已相对「首个有效区域原点」归一，None 表示本帧未采到区域沿用旧区域。
    reset：本帧喂入前是否触发了 session.reset（区域重置）。
    source / conf：session.update 返回的该帧 source 与 confidence，用于回放结果与实时逐一对照。
    """

    dt_ms: int
    rect: Optional[tuple]  # (x, y, w, h) 相对裁剪原点；None 表示沿用旧区域。
    reset: bool
    source: str
    conf: float


def encode_trace(trace: Iterable[FeedStep]) -> list:
    """把 FeedStep 列表紧凑序列化为可 JSON 化的 list[list]：[dt_ms, x, y, w, h, reset, source, conf]。

    rect 为 None 时四个坐标写 null。紧凑数组而非对象，减小边车体积（一局可达近千条）。
    """

    rows = []
    for step in trace:
        rect = step.rect
        if rect is None:
            x = y = w = h = None
        else:
            x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        rows.append([int(step.dt_ms), x, y, w, h, bool(step.reset), str(step.source), round(float(step.conf), 4)])
    return rows


def decode_trace(rows: Optional[Sequence]) -> list:
    """把边车里的 encode_trace 结果还原成 FeedStep 列表；脏数据/旧记录（无此字段）安全返回空列表。"""

    steps = []
    if not rows:  # None 或空：旧记录未带该字段，视为无日程。
        return steps
    for row in rows:
        try:
            dt_ms = int(row[0])
            x, y, w, h = row[1], row[2], row[3], row[4]
            reset = bool(row[5])
            source = str(row[6])
            conf = float(row[7]) if len(row) > 7 else 0.0
        except (TypeError, ValueError, IndexError):
            continue  # 单行损坏跳过，不影响其余。
        rect = None if x is None else (int(x), int(y), int(w), int(h))
        steps.append(FeedStep(dt_ms=dt_ms, rect=rect, reset=reset, source=source, conf=conf))
    return steps


@dataclass
class ReplayStep:
    """回放「仿实时模式」对单条 FeedStep 的解析结果：该喂 mp4 的第几帧、传什么 fps、用什么子区域、是否 reset。"""

    frame_index: int  # 对应 mp4 中的帧序号（0 基）。
    fps: float  # 传给 session.update 的有效帧率（由该步 dt 换算，复现实时口径）。
    rect: Optional[tuple]  # 相对裁剪原点的子区域 (x, y, w, h)，None 表示用整帧（兼容 crop_mode）。
    reset: bool  # 该步是否需要先 session.reset。


def iter_replay_plan(trace: Sequence[FeedStep], src_fps: float, start_frame: int = 0) -> Iterable[ReplayStep]:
    """按实时喂帧日程生成回放执行计划：用每步 dt 驱动 mp4 读取游标，复现实时的掉帧与阈值行为。

    start_frame：首次喂帧在 mp4 中的帧序号对齐偏移。录像在「触发确认瞬间」就起录，而解题要再等
    报警+触发延迟才开题，因此 mp4 前段已录了 K 帧解题才喂第一帧；实时侧首喂时快照录像器已入帧数
    写进边车 feed_start_frame，回放据此从第 K 帧起跳，否则全程错位、跟踪器面对的不是实时看过的帧序列。
    旧记录无该字段时缺省 0（含提前段，尽量向后兼容）。

    游标从 start_frame 开始，每步前进 `max(1, round(dt_s * src_fps))` 帧——dt 越大跳得越多，正是实时因
    处理慢而 latest-wins 丢弃中间帧的效果（两路独立采集，靠 dt 对齐到游戏真实帧序列即可，无需时间戳配对）。
    fps 由该步 dt 经 compute_effective_fps 换算；rect/reset 原样透传。src_fps 非法回退 30。
    """

    try:
        fps_base = float(src_fps)
    except (TypeError, ValueError):
        fps_base = 0.0
    if not (1.0 <= fps_base <= 240.0):
        fps_base = EFFECTIVE_FPS_DEFAULT
    try:
        index = int(start_frame)  # 起点对齐偏移：解题首喂帧对应的 mp4 帧号。
    except (TypeError, ValueError):
        index = 0
    if index < 0:
        index = 0
    for step in trace:
        dt_s = step.dt_ms / 1000.0 if step.dt_ms > 0 else 0.0
        yield ReplayStep(
            frame_index=index,
            fps=compute_effective_fps(dt_s),
            rect=step.rect,
            reset=step.reset,
        )
        # 本帧已消费，按「这一步真实过了多少秒 × 游戏帧率」把游标推进到实时下一喂入帧对应的位置。
        step_frames = max(1, int(round(dt_s * fps_base))) if dt_s > 0 else 1
        index += step_frames
