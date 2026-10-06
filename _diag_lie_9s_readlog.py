# -*- coding: utf-8 -*-
"""读取 _diag_lie_9s_suppress.py 的逐帧明细日志：按步长打印轨迹，并给出分段统计。

用法：
    .venv\\Scripts\\python.exe -X utf8 _diag_lie_9s_readlog.py <log> [t_lo] [t_hi] [stride]
"""

import io  # 读文件
import re  # 解析
import sys  # 参数

import numpy as np  # 数值

# 明细行：PowerShell 管道会把中文转码成乱码，因此只按 ASCII 骨架匹配（帧号/相位/坐标/conf/两个距离）。
LINE = re.compile(r"\s*t\s*(\d+)\s+([A-Za-z_]+)\s*\(\s*([\d.\-]+),\s*([\d.\-]+)\)\s*conf=([\d.]+)\D+([\d.]+)\D+([\d.\-]+)\s*$")  # 明细行


def parse(path):  # 解析日志 → [(tick, source, x, y, conf, 距准星, 距真值)]
    rows = []  # 结果
    for ln in io.open(path, encoding="utf-8", errors="replace"):  # 逐行
        m = LINE.match(ln)  # 匹配
        if m:
            rows.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)),
                         float(m.group(5)), float(m.group(6)), float(m.group(7))))  # 记录
    return rows  # 返回


def main():
    path = sys.argv[1]  # 日志
    t_lo = int(sys.argv[2]) if len(sys.argv) > 2 else 430  # 明细起始
    t_hi = int(sys.argv[3]) if len(sys.argv) > 3 else 640  # 明细结束
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 6  # 步长
    rows = parse(path)  # 解析
    if not rows:
        print("未解析到明细行（该次运行没开 t_print？）")
        return 1
    print(f"共 {len(rows)} 帧明细（{rows[0][0]}..{rows[-1][0]}）")
    for r in rows:  # 明细
        if t_lo <= r[0] <= t_hi and r[0] % stride == 0:
            print(f"t{r[0]:4d} {r[1]:>10} ({r[2]:6.1f},{r[3]:6.1f}) conf={r[4]:.2f} 距准星={r[5]:6.1f} 距真值={r[6]:6.1f}")
    print("---- 分段统计 ----")
    for lo, hi in ((430, 452), (452, 612), (612, 787), (787, 860), (452, 787)):  # 分段
        s = [r for r in rows if lo <= r[0] < hi]  # 子集
        if not s:
            continue
        d = np.array([x[6] for x in s], dtype=np.float64)  # 距真值
        c = np.array([x[5] for x in s], dtype=np.float64)  # 距准星
        print(f"t{lo:4d}..{hi:4d} n={len(s):4d} 距真值中位={np.median(d):6.1f} p90={np.percentile(d, 90):6.1f} "
              f"最大={d.max():6.1f} 末={d[-1]:6.1f}｜距准星中位={np.nanmedian(c):6.1f} 重合<35px={np.nanmean(c < 35) * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
