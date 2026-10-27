import os

# vision 包目录（ocsort.yaml / models 随包走）与宿主项目根目录（权重、TRT 缓存在宿主）。
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT)))

# === YOLO ===
# The model ships inside the project; YOLO_MODEL_PATH overrides it for an external checkpoint.
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH") or os.path.join(APP_ROOT, "weights", "best.pt")
YOLO_CONF_THRESHOLD = 0.3
YOLO_NMS_THRESHOLD = 0.45
# Measured on the 768x508 test video, CPU (torch 2.14 cpu, 16 threads), full 560-frame run:
#   640 -> p50 161ms, 43 TARGET_LOST frames   416 -> p50 94ms, 39   320 -> p50 70ms, 72
# 416 is the only setting that meets doc 4.4's <100ms budget here, and it also tracks longest
# without a loss because the smaller boxes jitter less, so OC-SORT spawns fewer duplicates.
YOLO_INPUT_SIZE = (416, 416)
YOLO_DEVICE = None  # None -> auto (GPU/TensorRT INT8 if the full stack is present, else CPU)

# === GPU (TensorRT INT8) ===
# The engine is built once per (checkpoint, imgsz) and cached next to the checkpoint; a build
# takes minutes, a cached load is milliseconds. Calibration pulls frames from the video being
# analysed, so INT8 quantization sees the game's actual pixel distribution.
TRT_CALIB_MAX_IMAGES = 256  # evenly spaced across the video; TensorRT's own guidance is >300
TRT_CALIB_MIN_IMAGES = 32  # below this the cache is considered stale and re-extracted
TRT_WORKSPACE_GB = 4
TRT_CALIB_DIR = os.path.join(APP_ROOT, "cache", "trt_calib")

# === Class mapping ===
# The doc's id table does NOT match best.pt: its actual classes are
#   {0: circle, 1: square, 2: triangle, 3: pentagram, 4: twographic, 5: threegraphic, 6: fourgraphic}
# so logical shape names are resolved to real ids from model.names at startup.
LOGICAL_SHAPES = ["square", "circle", "star", "triangle", "duo_shape", "trio_shape", "quad_shape"]
BASIC_SHAPES = ["square", "circle", "star", "triangle"]
COMBO_SHAPES = ["duo_shape", "trio_shape", "quad_shape"]

# Real class label -> logical shape name.
SHAPE_ALIASES = {
    "square": "square",
    "circle": "circle",
    "triangle": "triangle",
    "star": "star",
    "pentagram": "star",
    "duo_shape": "duo_shape",
    "twographic": "duo_shape",
    "trio_shape": "trio_shape",
    "threegraphic": "trio_shape",
    "quad_shape": "quad_shape",
    "fourgraphic": "quad_shape",
}

# NOTE: rotation observability is measured per track from the contour signature contrast,
# not assumed from the class label - a circle simply produces a flat signature.

# === OC-SORT (ultralytics built-in) ===
OC_SORT_MAX_AGE = 30
OC_SORT_MIN_HITS = 3
OC_SORT_IOU_THRESHOLD = 0.3
TRACKER_CONFIG_PATH = os.path.join(PACKAGE_ROOT, "ocsort.yaml")

# === Target determination ===
# OC-SORT ids are volatile: overlapping stars swap ids and a star whose box jitters can be
# re-created as a new id. The target is therefore anchored on geometry, and a new id that
# takes over the anchor box is adopted instead of declaring the target lost.
TARGET_HANDOFF_IOU = 0.30
# A track whose rotation fit has not converged carries no identity evidence, so adopting it
# is only defensible when the box is essentially the anchor itself (OC-SORT re-labelled it).
TARGET_HANDOFF_BLIND_IOU = 0.55
TARGET_OMEGA_HISTORY = 7  # median of the last N fits: one contaminated frame must not flip the sign
TARGET_HANDOFF_MAX_AREA_DRIFT = 0.6  # relative bbox area change allowed across a handoff
# Measured on the real video: the countdown digits are detected as pentagrams too, but their
# white fill stays at 0.12-0.16 while the target sits at 0.26-0.43.
TARGET_HANDOFF_MAX_RATIO_GAP = 0.05
TARGET_LOST_TOLERANCE_FRAMES = 3  # doc 15.2: a few missed detections are bridged by the EKF
TRANSITION_TIMEOUT_FRAMES = 120  # never hold TRANSITION open if the target cannot be measured
OMEGA_THRESHOLD = 0.02  # rad/frame, calibrate against real video
OMEGA_WINDOW = 15
SIGNATURE_BINS = 96  # angular resolution of the rotation signature (3.75 deg per bin)
SIGNATURE_HARMONICS = 8  # keep only low harmonics: shape structure, not pixelation ripple
MAX_SHIFT_DEG = 25.0  # per-frame rotation search window; rejects symmetry aliases
MIN_ROTATION_MODULATION = 0.05  # below this the shape carries no angular signal (circle)
MIN_CONTOUR_AREA_RATIO = 0.15  # ROI area below this ratio -> angle invalid this frame
WHITE_BINARY_THRESHOLD = 240  # gray threshold for white filled target
WHITE_HSV_RANGE = ((0, 0, 200), (180, 30, 255))
WHITE_FILL_RATIO = 0.20  # measured: filled star 0.26-0.43 inside its bbox, outlines <0.15
FADE_DONE_RATIO = 0.12  # measured: target drops below this ~46 frames after countdown ends
FADE_CONFIRM_FRAMES = 5  # sustained, so a glinting distractor does not end TRANSITION

# === Trail ===
MAX_TRAIL_LENGTH = 300
EDGE_MARGIN = 5

# === Kinematics / overlap identity (measured on the validation set) ===
# Distractors share one motion model and move as a flock: cos(vel, group median) ~ +0.97.
# The target wanders independently: cos ~ +0.1 or negative. This is the widest identity
# separation available after the fade, so it gates handoff/recovery.
KIN_SMOOTH_WINDOW = 15  # measured intervals in the robust velocity; ~0.5s
KIN_MIN_SAMPLES = 6  # fewer samples than this and the velocity is not trustworthy
KIN_MIN_SPEED = 0.5  # px/frame; below it a direction is box jitter, not motion
KIN_GROUP_MIN_TRACKS = 4  # flock median needs at least this many live confirmed tracks
KIN_FLOCK_COS_HI = 0.8  # cos >= this with the flock = distractor (measured 0.97)
KIN_FLOCK_COS_LO = 0.2  # cos <= this = clearly independent (measured target ~0.11)
KIN_NEAR_FRAME_TOLERANCE = 5  # frames of slack when looking up where a track was
TARGET_GRACE_FRAMES = 3  # locked track missing <= this many frames: wait, do not hand off
TARGET_OMEGA_CERT_FLOOR = 0.02  # |cert omega| below this carries no rotation evidence
# Two stars' boxes touch long before their contours merge: at IoU 0.15 the largest contour
# in the ROI is still the track's own star, and holding the angle there starved omega for
# the whole countdown in dense scenes (measured: 0/30 measured samples f120-240). Only a
# deep overlap, where the merged contour is genuinely ambiguous, is worth holding.
OVERLAP_OMEGA_IOU = 0.35

# === Recovery ===
RECOVERY_TIMEOUT_FRAMES = 90
# Measured on the real video: a star moves 2-4 px/frame, so a candidate further away than
# this over the elapsed time is a different object, not the target coming back.
RECOVERY_MAX_SPEED = 6.0
RECOVERY_OMEGA_TOLERANCE = 0.09  # measured: the true target's own omega fits swing +/-0.08 across a run
# The score is normalized by the evidence mass actually available this run (no flock
# estimate early, no certificate for a circle), so one number works for every scene shape.
RECOVERY_MIN_RATIO = 0.55
TARGET_NEAR_RADIUS_FACTOR = 1.5  # body lengths for the "was near the anchor when last measured" check
# Appearance (Hu-signature) evidence: measured NET-NEGATIVE on valset_7x2x20 (duo-edge
# 15->14, trio-free 16->15, star-free even) - the offline pairwise-dissimilarity signal
# (Hu-NCC ~0 between same-class compounds) does not survive the full pipeline: rotating
# outline discretization + fade residue make the certificate too noisy. Code kept behind
# this switch; revisit only with a noise-robust descriptor (e.g. learned embedding).
APPEARANCE_EVIDENCE_ENABLED = False
RECOVERY_SCORE_WEIGHTS = {
    "group": 1.0,  # independence from the flock (also a hard gate, see KIN_FLOCK_COS_HI)
    "path": 0.8,   # closeness to the constant-velocity prediction
    "omega": 0.6,  # agreement with the certificate (no evidence when cert floor not met)
    "appearance": 0.5,  # Hu-signature match vs the fade-complete certificate (same-class
                        # distractors are mostly dissimilar shapes, measured Hu-NCC ~0)
    "area": 0.3,
    "iou": 0.3,
}
SWAP_DETECTOR_ENABLED = True
# 10 sustained flock-aligned frames is the calibrated trigger; the measured race (gates
# passed before the counter filled, then the near gate closed) is fixed by referencing
# the last healthy snapshot in the near gate, not by shortening the wait
SWAP_SUSPECT_FRAMES = 6
# a swap candidate must match the locked track's object scale: a 15px micro-detection is
# not the 127px star, however "independently" it jitters (measured on score1.00_high.mp4)
SWAP_MIN_AREA_RATIO = 0.3

# presentation-only smoothing of the follow box: a 3-frame median absorbs 1-2 frame
# detection drags when a distractor grazes the target (a 5-frame window costs 2 frames
# per correction instead of 1 and thins the final-frame margin, measured)
FOLLOW_SMOOTH_FRAMES = 3

# Countdown confirmation filters. The digit is white AND often whiter than the target
# (fill 0.52 vs a compound's 0.42, measured) but sits top-centre: across all 280 scenes
# the GT target never enters that zone during the countdown, so zone membership alone is
# a clean digit discriminator there; a lock sitting on a digit-zone track for
# COUNTDOWN_DIGIT_ESCAPE_FRAMES consecutive frames while a real white candidate exists is
# re-locked. Compound targets double-detect (compound box + brighter primitive inside);
# the inner detection is dropped when >=80% covered by a >=2x-area white candidate.
COUNTDOWN_DIGIT_ZONE_X = (0.35, 0.65)
COUNTDOWN_DIGIT_ZONE_Y = 0.40
COUNTDOWN_DIGIT_ESCAPE_FRAMES = 3
COUNTDOWN_FRAGMENT_MIN_AREA_RATIO = 2.0

# A direction-less (too slow / too young) candidate that only shows up late in a recovery
# has no independence evidence at all: it must clear a higher bar than an early one, or a
# slow distractor drifting through the loss area gets adopted (measured: scene_2078906197
# adopted a slow star at IoU 0.0 with the target on frame 433 of a 72-frame recovery).
RECOVERY_LATE_FRAMES = 30
RECOVERY_LATE_MIN_RATIO = 0.70

# === Flow-primary session (host addition, not in D:\lie) ===
# 身份由倒计时白目标锁定后交给 LK 光流像素携带；YOLO 降级为校验（杀漂移/查滑移）与
# 丢失重获（try_recover 四路证据 + 光流末位锚）。FLOW_PRIMARY=False 时 solver 走原
# Pipeline 路径，行为与 d:\lie 完全一致。
#
# ⚠ 当前默认关闭：d/lie 验证集 35 场景真值 A/B（failed_videos + trajectory.json）
# 平均在靶率 d/lie 60.9% vs 光流 9.8%（3/35 不劣化）。失败链：组合体（duo/trio/quad）
# 检测框内多个子形状+大片地形，环带采样点落在地形上→双峰位移跟错→框漂离→NCC 死亡
# →重获上下文弱（near_positions 空）→误采纳错误候选→光流长期锁定错误对象。
# 待解决：形状上点选择（区别地形/子形状间隙）、身份感知校验、强恢复上下文。
FLOW_PRIMARY = False
FLOW_MAX_POINTS = 40        # 锁定框内 goodFeaturesToTrack 特征点上限
FLOW_MIN_POINTS = 8         # 一帧追踪后存活点低于此数判死亡
FLOW_FB_THRESHOLD = 1.5     # 前向-后向往返误差(px)超过即剔除该点
FLOW_MOTION_GATE_SIGMA = 2.0  # 点位移偏离中位数超过 sigma*MAD 剔除（抗穿越拉扯）
FLOW_MIN_NCC = 0.5          # 帧间 NCC（上一帧 patch vs 当前 patch）低于此判死亡
FLOW_SEED_NCC_FLOOR = 0.25  # 种子 NCC 粗筛下限：半透明轮廓换地形天然衰减，只拦 gross 变化
FLOW_VERIFY_INTERVAL = 5    # 每 N 帧 YOLO 全帧校验一次（吸附杀漂移）
FLOW_VERIFY_IOU = 0.5       # 校验检测框与光流框 IoU>=此值即吸附
FLOW_VERIFY_MISS = 2        # 连续 N 次校验无匹配判滑移，进入重获
FLOW_SMOOTH_FRAMES = 3      # follow bbox 中值平滑窗（与 FOLLOW_SMOOTH_FRAMES 同口径）
FLOW_HANDOFF_STABLE = 3     # phase-1 交接：TRACKING 且 follow_bbox 连续 N 帧
FLOW_CARRIER_MAX_FRAMES = 90  # 光流载波寿命上限（校验吸附会重置），防漂移发散
FLOW_COAST_ON_LOST = 10     # 重获期沿光流惯性外推 follow 的最大帧数

# === Visualization ===
TARGET_COLOR = (0, 255, 0)
DISTRACTOR_COLOR = (100, 100, 100)
VELOCITY_COLOR = (0, 255, 255)
ANGLE_COLOR = (255, 255, 255)
CENTROID_RADIUS = 4
BOX_THICKNESS = 2
TRAIL_THICKNESS = 1

# === Game timing (seconds, from prototype readme: 15s total incl. 3s countdown) ===
COUNTDOWN_DURATION_SEC = 5.0  # this game counts down from 5
GAME_TOTAL_DURATION_SEC = 15.0

STATE_LIGHT_COLORS = {
    "IDLE": (128, 128, 128),
    "COUNTDOWN": (255, 165, 0),
    "TRANSITION": (0, 255, 255),
    "TRACKING": (0, 200, 0),
    "TARGET_LOST": (0, 0, 255),
    "GAME_OVER": (200, 0, 200),
}

# OC-SORT id-theft guard: when the locked track re-measures after coasting at least this
# many frames (a real gap, past the 3-frame grace) and its rotation contradicts the
# certificate, the resumption is rejected as an impostor and recovery searches instead.
# Measured on the edge categories: 15/16 final-frame failures had the locked id survive
# through the failure onset (re-associated to a different object while coasting).
RESUME_THEFT_MIN_GAP = 4

# Corrector candidate independence, temporally smoothed: in edge scenes the target's
# flock-cosine oscillates +-1 with a ~10-frame period, so a single-frame gate excludes
# the true carrier about half the time. A candidate qualifies if its cosine dipped to
# SWAP_INDEP_HIST_COS at least SWAP_INDEP_HIST_MIN times in the last
# SWAP_INDEP_HIST_WINDOW frames (measured: runs/analysis6).
SWAP_INDEP_HIST_WINDOW = 15
SWAP_INDEP_HIST_COS = -0.3
SWAP_INDEP_HIST_MIN = 2

# Theft watch: after a gap-bridged return to TRACKING the resumed carrier is watched for
# THEFT_WATCH_FRAMES. A stolen id reveals itself by jumping into the flock's direction
# (cos jump >= THEFT_WATCH_JUMP vs the pre-loss relation, or a velocity reversal across
# the gap) sustained for THEFT_WATCH_ALIGNED_FRAMES. The response is a re-capture attempt
# anchored at the pre-loss snapshot through the normal recovery gates - finding nothing
# costs nothing (measured: runs/analysis6, TP 9-10/13, FP price is a delayed re-lock).
THEFT_WATCH_ENABLED = True
THEFT_WATCH_FRAMES = 30
THEFT_WATCH_JUMP = 0.2
THEFT_WATCH_ALIGNED_FRAMES = 5

# Recovery evidence threshold after a poisoned (suspected-mis-lock) loss: the path,
# anchor-IoU and omega score terms are dropped there by design, shrinking the evidence
# mass to group+area - and in edge scenes the true carrier flies with the flock (cos
# 0.64-1.0, measured), capping the achievable ratio near 0.4. All hard gates stay.
RECOVERY_MIN_RATIO_POISONED = 0.40

# === Neural motion-evidence channel (V7 trajectory regression, vendor port) ===
# Second-opinion channel consumed ONLY under distress (swap-suspect / theft-watch /
# TARGET_LOST); never overwrites tracker state directly. Its failure modes (regression
# confusion on multi-object panels, drift toward the panel prior under fast tail motion)
# are disjoint from ours; see runs/sts_tail_check + NEURAL_FUSION_PLAN.md for the measured
# complementarity (matched-protocol oracle union 271/280). Motion evidence, not appearance:
# orthogonal to the falsified Hu channel (APPEARANCE_EVIDENCE_ENABLED).
NEURAL_EVIDENCE_ENABLED = os.environ.get("NEURAL_EVIDENCE_ENABLED", "0") == "1"
NEURAL_ONNX_PATH = os.environ.get("NEURAL_ONNX_PATH") or os.path.join(
    PACKAGE_ROOT, "models", "shape_trajectory_v7.onnx")
NEURAL_EVIDENCE_STRIDE = 2     # feed cadence; 2 is the V7 runtime's native source_stride
NEURAL_DIVERGENCE_FRAMES = 10  # K: distress frames of sustained |target - neural| > tau before M1 acts
NEURAL_DIVERGENCE_TAU = 1.0    # tau = factor x body_length of the locked track
NEURAL_NEAR_FACTOR = 0.9       # M1/M2 candidate must sit within factor x body of the neural center
NEURAL_CONFIRM_FRAMES = 2      # same candidate must win this many consecutive frames before adoption
NEURAL_M2_MIN_LOST_FRAMES = 30  # M2 only enters after this long in TARGET_LOST (v17 recovery first;
                                # measured: adopting inside a young loss hijacked recoverable locks)
NEURAL_PRIOR_MIN_DIST = 80.0   # px from the network's (320,212) prior: closer = invalid evidence
NEURAL_MAX_SPEED = 15.0        # px/frame neural continuity ceiling: faster = invalid (re-lock jump)
