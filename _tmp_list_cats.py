import json, io
d = json.load(io.open('ok_templates/coco_annotations.json', encoding='utf-8'))
lines = [f"{c.get('supercategory') or '(无类别)'} | {c['name']}" for c in d['categories']]
io.open('_cats.txt', 'w', encoding='utf-8').write('\n'.join(lines))
