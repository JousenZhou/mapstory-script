# -*- coding: utf-8 -*-
"""对 test_case 最慢视频逐阶段计时，定位 session.update 内部耗时大头。"""
import glob
import sys
import time

sys.path.insert(0, '.')
from src.liedetector import shape_tracking as st
from src.liedetector import shape_session as ss

CASE_DIR = r"D:\workspace\新建文件夹\maoxiandao\test_case"

acc = {}


def wrap(mod, name):
    orig = getattr(mod, name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        item = acc.setdefault(name, [0, 0.0])
        item[0] += 1
        item[1] += time.perf_counter() - t0
        return r

    setattr(mod, name, timed)


wrap(st, 'detect_white_shapes')
wrap(st, 'choose_shape_candidate')
wrap(st, 'estimate_detection_angle')
wrap(st, 'score_shape_contours')
wrap(st, 'score_rotated_borders')
wrap(st, 'transform_template_points')

for name in ('detect_white_shapes', 'choose_shape_candidate'):
    orig = getattr(ss, name)

    def make_f(o, n):
        def timed(*a, **kw):
            t0 = time.perf_counter()
            r = o(*a, **kw)
            item = acc.setdefault(n, [0, 0.0])
            item[0] += 1
            item[1] += time.perf_counter() - t0
            return r

        return timed

    setattr(ss, name, make_f(orig, name))

orig_update = ss.DenseTemporalAligner.update


def timed_align(self, frame):
    t0 = time.perf_counter()
    r = orig_update(self, frame)
    item = acc.setdefault('aligner.update(光流)', [0, 0.0])
    item[0] += 1
    item[1] += time.perf_counter() - t0
    return r


ss.DenseTemporalAligner.update = timed_align

for m in ('observe_color', 'observe_border', 'propagate', '_resample_if_needed', '_proposal_states'):
    orig = getattr(st.ParticleShapeTracker, m)

    def make_t(o, n):
        def timed(self, *a, **kw):
            t0 = time.perf_counter()
            r = o(self, *a, **kw)
            item = acc.setdefault(n, [0, 0.0])
            item[0] += 1
            item[1] += time.perf_counter() - t0
            return r

        return timed

    setattr(st.ParticleShapeTracker, m, make_t(orig, m))

videos = sorted(glob.glob(CASE_DIR + r"\*.mp4"))
for path in videos:
    import cv2
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    acc.clear()
    sess = ss.ShapeTrackSession(params=ss.ShapeTrackParams())
    sess.reset(w, h)
    t_all = 0.0
    n_frames = 0
    for i in range(90):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        sess.update(frame, 30.0)
        if i >= 5:
            t_all += time.perf_counter() - t0
            n_frames += 1
    cap.release()
    if n_frames == 0:
        continue
    print(f"\n===== {path.split(chr(92))[-1]} {w}x{h} scale={sess.effective_scale:.3f} "
          f"track={t_all / n_frames * 1000:.1f}ms avg over {n_frames} frames")
    print('%-28s %6s %10s %9s' % ('stage', 'calls', 'total_ms', 'ms/call'))
    for name, (n, tot) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
        print('%-28s %6d %10.1f %9.2f' % (name, n, tot * 1000, tot / n * 1000))
    print('source counts:', sess.source_counts)
