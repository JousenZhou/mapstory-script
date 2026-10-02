# 诊断测谎录像是否被加速：对比「容器播放时长」与「真实录制墙钟时长」，并统计互异帧率。
import glob, json, os, re
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(ROOT, "lie_records")
LOGS = sorted(glob.glob(os.path.join(ROOT, "logs", "ok-script*.log")))

from datetime import datetime
ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
started_re = re.compile(r"Lie record started:\s*(\S+\.mp4)")
stopped_re = re.compile(r"Lie record stopped:\s*(\S+\.mp4)")

def parse_ts(line):
    m = ts_re.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None

# 从日志抓每条录像的真实录制墙钟时长（started -> stopped）
wall = {}
pend = {}
for fp in LOGS:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = parse_ts(line)
            if t is None:
                continue
            sm = started_re.search(line)
            if sm:
                pend[os.path.basename(sm.group(1))] = t
                continue
            tm = stopped_re.search(line)
            if tm:
                name = os.path.basename(tm.group(1))
                if name in pend:
                    wall[name] = (tm.group(1) and (t - pend[name]).total_seconds())

print(f"{'video':42} {'side_fps':>8} {'meas_fps':>8} {'frames':>6} {'cont_dur':>8} {'real_dur':>8} {'cfid':>7} {'wall_dur':>8} {'speedup':>8} {'distinct':>8}")
for mp4 in sorted(glob.glob(os.path.join(REC, "*.mp4"))):
    name = os.path.basename(mp4)
    js = os.path.splitext(mp4)[0] + ".json"
    side = {}
    if os.path.exists(js):
        with open(js, "r", encoding="utf-8") as f:
            side = json.load(f)
    cap = cv2.VideoCapture(mp4)
    cfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cont_dur = nframes / cfps if cfps > 0 else 0.0
    # 统计互异帧：与上一帧完全相同则算重复
    distinct = 0
    prev = None
    got = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        got += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is None or not np.array_equal(g, prev):
            distinct += 1
        prev = g
    cap.release()
    wd = wall.get(name)
    side_fps = side.get("fps", "-")
    meas = side.get("measured_fps", "-")
    sframes = side.get("frames", nframes)
    real_dur = side.get("real_duration")  # 真实采集跨度（首帧->末帧），即视频内容对应的真实时长
    speedup = (wd / cont_dur) if (wd and cont_dur > 0) else float("nan")
    # 内容真速比 = 真实采集跨度 / 容器播放时长：视频内容实际被播放的倍速（≈1.0 即忠实），不含场景结束后的静止结算尾巴
    cfid = (real_dur / cont_dur) if (real_dur and cont_dur > 0) else float("nan")
    distinct_rate = (distinct / wd) if (wd and wd > 0) else float("nan")
    wd_s = f"{wd:.2f}" if wd else "-"
    rd_s = f"{real_dur:.2f}" if real_dur else "-"
    sp_s = f"{speedup:.2f}x" if speedup == speedup else "-"
    cf_s = f"{cfid:.3f}x" if cfid == cfid else "-"
    dr_s = f"{distinct_rate:.1f}" if distinct_rate == distinct_rate else "-"
    print(f"{name:42} {str(side_fps):>8} {str(meas):>8} {str(sframes):>6} {cont_dur:8.2f} {rd_s:>8} {cf_s:>7} {wd_s:>8} {sp_s:>8} {dr_s:>8}")
