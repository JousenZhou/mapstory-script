# -*- coding: utf-8 -*-
"""门面语义探针：核对 Phase 3 粒子滤波要用到的算子在两套门面下语义一致（临时脚本，收尾删）。"""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.liedetector.torch_array import numpy_api, torch_api

FAILS = []


def check(label, left, right, exact=True, tol=0.0):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        FAILS.append(f"{label}: shape {a.shape} != {b.shape}")
        print(f"FAIL {label}: shape {a.shape} != {b.shape}")
        return
    diff = float(np.abs(a - b).max()) if a.size else 0.0
    ok = (diff == 0.0) if exact else (diff <= tol)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: max|Δ|={diff:.3e} dtype_np={np.asarray(left).dtype} dtype_tp={np.asarray(right).dtype}")
    if not ok:
        FAILS.append(f"{label}: max|Δ|={diff:.3e}")


def main():
    np_api = numpy_api()
    tp_api = torch_api()
    period = 90.0

    # [1] 取模：负角度折回周期域，numpy 的 % 与 torch 的 % 必须同为 floor-mod。
    angles = np.array([-200.0, -90.5, -1e-6, 0.0, 45.0, 89.9, 90.0, 181.0, -720.0], dtype=np.float32)
    check("[1] mod 负角度", angles % period, tp_api.to_numpy(tp_api.asarray(angles) % period))

    # [2] periodic_angle_difference 的整条表达式。
    ref = 37.5
    expr_np = abs((angles - ref + period * 0.5) % period - period * 0.5)
    t = tp_api.asarray(angles)
    expr_tp = tp_api.abs((t - ref + period * 0.5) % period - period * 0.5)
    check("[2] periodic_angle_difference", expr_np, tp_api.to_numpy(expr_tp))

    # [3] 系统重采样：cumsum + searchsorted(right) + clip + take。
    rng = np.random.default_rng(20260902)
    weights = rng.random(420).astype(np.float64)
    weights /= weights.sum()
    positions = (rng.random() + np.arange(420)) / 420
    states = rng.normal(0.0, 5.0, size=(420, 6)).astype(np.float32)
    cum_np = np.cumsum(weights)
    idx_np = np.clip(np.searchsorted(cum_np, positions, side="right"), 0, 419)
    picked_np = np.take(states, idx_np, axis=0)
    tp_states = tp_api.asarray(states)
    cum_tp = tp_api.cumsum(tp_api.asarray(weights))
    idx_tp = tp_api.clip(tp_api.searchsorted(cum_tp, tp_api.asarray(positions), side="right"), 0, 419)
    picked_tp = tp_api.take(tp_states, idx_tp, axis=0)
    check("[3a] searchsorted 索引", idx_np, tp_api.to_numpy(idx_tp))
    check("[3b] take 结果", picked_np, tp_api.to_numpy(picked_tp))

    # [4] where 接 Python 标量 / 0 维布尔条件广播。
    column = rng.normal(0.0, 3.0, size=(420, 1)).astype(np.float32)
    mask = column < 0.0
    check("[4a] where(标量, 数组)", np.where(mask, 1.5, column), tp_api.to_numpy(tp_api.where(tp_api.asarray(mask), 1.5, tp_api.asarray(column))))
    total = np.float64(0.0)
    degenerate = (~np.isfinite(total)) | (total <= 1e-18)
    tp_total = tp_api.asarray(np.float64(0.0))
    tp_degenerate = (~tp_api.isfinite(tp_total)) | (tp_total <= 1e-18)
    check("[4b] 0 维布尔广播", np.where(degenerate, np.full(5, 0.2), np.arange(5, dtype=np.float64)),
          tp_api.to_numpy(tp_api.where(tp_degenerate, tp_api.full((5,), 0.2, dtype=tp_api.float64), tp_api.arange(5, dtype=tp_api.float64))))

    # [5] put_rows：按索引覆写行。
    target_np = states.copy()
    replace = np.argsort(weights)[:30]
    top = rng.integers(0, 420, 30).astype(np.int64)
    target_np[replace] = states[top]
    tp_target = tp_api.asarray(states).clone()
    tp_api.put_rows(tp_target, tp_api.asarray(replace), tp_api.take(tp_states, tp_api.asarray(top)))
    check("[5] put_rows", target_np, tp_api.to_numpy(tp_target))

    # [6] argmax / argsort(descending) 与 numpy 的等价性。
    scores = rng.normal(0.0, 1.0, 720).astype(np.float32)
    check("[6a] argmax", np.argmax(scores), tp_api.to_numpy(tp_api.argmax(tp_api.asarray(scores))))
    check("[6b] argsort 降序", np.argsort(scores)[::-1], tp_api.to_numpy(tp_api.argsort(tp_api.asarray(scores), descending=True)))

    # [7] column / ravel。
    check("[7a] column(4)", states[:, 4], tp_api.to_numpy(tp_api.column(tp_states, 4)))
    check("[7b] stack 成列", np.stack([weights], axis=1).ravel(), tp_api.to_numpy(tp_api.stack([tp_api.asarray(weights)], axis=1)).ravel())

    # [8] linalg_norm + median + sum 的 dtype 提升行为（states float32 × weights float64）。
    mixed_np = np.sum(states[:, :2] * weights[:, None], axis=0)
    mixed_tp = tp_api.sum(tp_api.narrow(tp_states, 1, 0, 2) * tp_api.stack([tp_api.asarray(weights)], axis=1), axis=0)
    check("[8a] 混合精度求和", mixed_np, tp_api.to_numpy(mixed_tp), exact=False, tol=1e-9)
    check("[8b] median float32", np.median(scores), tp_api.to_numpy(tp_api.median(tp_api.asarray(scores))), exact=False, tol=1e-6)
    check("[8c] linalg_norm", np.linalg.norm(states[:, 2:4], axis=1),
          tp_api.to_numpy(tp_api.linalg_norm(tp_api.narrow(tp_states, 1, 2, 2), axis=1)), exact=False, tol=1e-5)

    print("PASS" if not FAILS else f"FAIL {len(FAILS)}: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
