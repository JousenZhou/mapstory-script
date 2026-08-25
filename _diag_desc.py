# 临时脚本：打印 pathfinder 动态库描述表中关键库的 DLL 名与搜索目录。
import re

src = open(r'.venv\Lib\site-packages\cuda\pathfinder\_dynamic_libs\descriptor_catalog.py', encoding='utf-8').read()
lines = src.splitlines()
for name in ('nvrtc', 'cudart'):
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('name="' + name + '"'):
            start = i - 1
            break
    if start is None:
        print(name, 'NOT FOUND')
        continue
    depth = 0
    for j in range(start, len(lines)):
        depth += lines[j].count('(') - lines[j].count(')')
        print(lines[j])
        if depth <= 0 and j > start:
            break
    print('---')
