# 从日志统计每个精度档的“求解循环帧率”（tick/时间），区别于录像采集帧率。
import glob, os, re
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
files = sorted(glob.glob(os.path.join(LOG_DIR, "ok-script*.log")))

ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
start_re = re.compile(r"Lie record started:.*?tier=(\w+)")
stop_re = re.compile(r"Lie record stopped:")
diag_re = re.compile(r"Lie solve diag:.*?tick=(\d+)")

def parse_ts(line):
    m = ts_re.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")

sessions = []          # 每个 solve 会话: dict(tier, ticks=[(ts,tick)])
cur = None
for fp in files:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sm = start_re.search(line)
            if sm:
                cur = {"tier": sm.group(1), "file": os.path.basename(fp), "ticks": []}
                sessions.append(cur)
                continue
            if stop_re.search(line):
                cur = None
                continue
            dm = diag_re.search(line)
            if dm and cur is not None:
                ts = parse_ts(line)
                if ts is not None:
                    cur["ticks"].append((ts, int(dm.group(1))))

# 逐会话算 fps：diag 每秒限频一条，用首末 tick 与首末时间差。
rows = []
for s in sessions:
    tk = s["ticks"]
    if len(tk) < 2:
        continue
    t0, k0 = tk[0]
    t1, k1 = tk[-1]
    dt = (t1 - t0).total_seconds()
    dk = k1 - k0
    if dt <= 0 or dk <= 0:
        continue
    rows.append((s["tier"], s["file"], t0.strftime("%m-%d %H:%M:%S"), dk, dt, dk / dt, k1))

# 按档位聚合
by_tier = {}
for r in rows:
    by_tier.setdefault(r[0], []).append(r)

print("== 逐会话求解循环帧率 ==")
print(f"{'tier':8} {'start':16} {'Δtick':>6} {'Δt(s)':>7} {'fps':>7} {'endTick':>7}  file")
for tier in sorted(by_tier):
    for r in sorted(by_tier[tier], key=lambda x: x[2]):
        print(f"{r[0]:8} {r[2]:16} {r[3]:6d} {r[4]:7.2f} {r[5]:7.2f} {r[6]:7d}  {r[1]}")

print("\n== 按档位汇总（求解循环帧率）==")
print(f"{'tier':10} {'sessions':>8} {'mean_fps':>9} {'min_fps':>8} {'max_fps':>8}")
for tier in sorted(by_tier):
    fps_list = [r[5] for r in by_tier[tier]]
    print(f"{tier:10} {len(fps_list):8d} {sum(fps_list)/len(fps_list):9.2f} {min(fps_list):8.2f} {max(fps_list):8.2f}")
