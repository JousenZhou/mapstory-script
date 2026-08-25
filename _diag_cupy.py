# 临时脚本：验证桩模块生效后 CuPy 的可用性与报错细节。
import sys
import traceback

sys.path.insert(0, '.')
import src.gpu_match as g  # 先走项目的桩逻辑再导入。

print('cp module:', g.cp)
import cupy

print('is_available:', cupy.cuda.is_available())
try:
    n = cupy.cuda.runtime.getDeviceCount()
    print('device count:', n)
except Exception:
    traceback.print_exc()
try:
    a = cupy.arange(10)
    print('cupy compute ok:', float(a.sum()))
except Exception:
    traceback.print_exc()
