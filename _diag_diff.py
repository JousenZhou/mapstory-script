# -*- coding: utf-8 -*-
"""对比外部算法脚本与项目内移植模块的函数体差异。"""
import ast
import sys

EXTERNAL = {
    r"D:\workspace\新建文件夹\maoxiandao\track_transparent_shape.py": "shape",
    r"D:\workspace\新建文件夹\maoxiandao\track_transparent_square_v2.py": "v2",
}
INTERNAL = r"d:\workspace\mapstory-script\src\liedetector\shape_tracking.py"


def strip_docstring(node):
    """去掉函数体开头的 docstring 常量，避免注释差异干扰对比。"""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def normalize(node):
    """把函数定义归一化成可比较的 AST dump（去 docstring）。"""
    clone = ast.parse(ast.unparse(node)).body[0]
    clone.body = strip_docstring(clone)
    return ast.dump(clone)


def extract_functions(path):
    """提取文件中所有顶层函数与类内方法，返回 {名称: 归一化源码}。"""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    functions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        name = f"{node.name}.{item.name}"
                        functions[name] = normalize(item)
            else:
                functions[node.name] = normalize(node)
    return functions


internal = extract_functions(INTERNAL)
for path, tag in EXTERNAL.items():
    external = extract_functions(path)
    print(f"\n===== {tag} ({path}) =====")
    for name, body in sorted(external.items()):
        if "." in name:  # 类方法单独标注
            cls, method = name.split(".", 1)
            internal_key = name if name in internal else f"ParticleShapeTracker.{method}"
        else:
            internal_key = name
        if internal_key not in internal:
            print(f"  [NOT PORTED] {name}")
            continue
        if internal[internal_key] == body:
            print(f"  [IDENTICAL ] {name}")
        else:
            print(f"  [DIFFERENT ] {name} (internal key: {internal_key})")
