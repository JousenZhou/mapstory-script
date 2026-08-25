# 临时脚本：验证 GPU 模板匹配器在桩模块下的端到端结果与 CPU 一致性。
import sys
import time

sys.path.insert(0, '.')
import cv2
import numpy as np

from src.gpu_match import gpu_available, GpuTemplateMatcher

assert gpu_available(), 'GPU not available'

rng = np.random.default_rng(42)
frame = rng.integers(0, 200, (768, 1366, 3), dtype=np.uint8)
tpl = frame[300:340, 500:560].copy()  # 从画面中截取一块当模板，保证满分命中。

matcher = GpuTemplateMatcher(gray=True)
matcher.add_template('t', tpl)
matcher.add_template('t_flip', cv2.flip(tpl, 1))

start = time.time()
gm = matcher.match_frame(frame)
x, y, score = gm.best('t')
elapsed = time.time() - start
print(f'gpu match: x={x} y={y} score={score:.4f} elapsed={elapsed:.2f}s (first run includes JIT)')
assert (x, y) == (500, 300), f'wrong position: {(x, y)}'
assert score > 0.99, f'score too low: {score}'

cpu_score = cv2.matchTemplate(
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
    cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).astype(np.float32),
    cv2.TM_CCOEFF_NORMED)
cpu_min, cpu_max, cpu_min_loc, cpu_max_loc = cv2.minMaxLoc(cpu_score)
print(f'cpu match: loc={cpu_max_loc} score={cpu_max:.4f}')
assert abs(cpu_max - score) < 0.01, 'gpu/cpu score mismatch'
print('ALL OK')
