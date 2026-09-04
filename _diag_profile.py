# -*- coding: utf-8 -*-
"""逐阶段计时：定位 session.update 内部各函数的真实耗时。"""
import sys
import time

sys.path.insert(0, '.')
import numpy as np
import cv2
from src.liedetector import shape_tracking as st
from src.liedetector import shape_session as ss


def make_frame(t, with_circle=True):
    f = np.full((253, 378, 3), 40, np.uint8)
    rng = np.random.default_rng(t)
    f = f + rng.integers(0, 6, f.shape, dtype=np.uint8)
    if with_circle:
        cx, cy = 120 + t % 150, 130
        y, x = np.ogrid[:253, :378]
        m = (x - cx) ** 2 + (y - cy) ** 2 < 30 ** 2
        f[m] = 250
    return f


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
wrap(st, 'resample_closed_contour')
wrap(st, 'build_shape_template')
wrap(st, 'transform_template_points')

# shape_session 用 from import 绑定，需在调用方模块再包一层。
for name in ('detect_white_shapes', 'choose_shape_candidate', 'build_shape_template'):
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
    item = acc.setdefault('aligner.update', [0, 0.0])
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

sess = ss.ShapeTrackSession(params=ss.ShapeTrackParams())
sess.reset(378, 253)
t_all = 0.0
for t in range(60):
    f = make_frame(t, t < 15 or t % 7 == 0)
    t0 = time.perf_counter()
    sess.update(f, 30.0)
    t_all += time.perf_counter() - t0
print(f'total update: {t_all / 60 * 1000:.1f} ms/frame avg over 60 frames')
print('%-26s %6s %10s %9s' % ('stage', 'calls', 'total_ms', 'ms/call'))
for name, (n, tot) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
    print('%-26s %6d %10.1f %9.2f' % (name, n, tot * 1000, tot / n * 1000))
print('source counts:', sess.source_counts)
