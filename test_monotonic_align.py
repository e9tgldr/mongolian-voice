"""Verify numpy_monotonic_align against a literal brute-force transcription
of the original Cython algorithm before we let it steer training."""
import numpy as np
import torch

from numpy_monotonic_align import _maximum_path_each, maximum_path

MAX_NEG_VAL = -1e9


def brute_force(value_in, t_y, t_x):
    """Literal triple-loop version of core.pyx (reference semantics)."""
    value = value_in.copy()
    path = np.zeros((t_y, t_x), dtype=np.int32)
    index = t_x - 1
    for y in range(t_y):
        x0 = max(0, t_x + y - t_y)
        x1 = min(t_x, y + 1)
        for x in range(x0, x1):
            if x == y:
                v_cur = MAX_NEG_VAL
            else:
                v_cur = value[y - 1, x]  # only reached when y > 0
            if x == 0:
                v_prev = 0.0 if y == 0 else MAX_NEG_VAL
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and y != 0 and (index == y or
                                      value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
    return path, value


rng = np.random.default_rng(0)
cases = [(1, 1), (1, 5), (5, 1), (3, 3), (7, 4), (4, 9), (12, 12), (2, 11), (11, 2)]
for _ in range(30):
    cases.append((int(rng.integers(1, 40)), int(rng.integers(1, 40))))

fails = 0
for t_y, t_x in cases:
    v_in = rng.standard_normal((t_y, t_x)).astype(np.float32)
    ref_path, ref_value = brute_force(v_in, t_y, t_x)
    v_fast = v_in.copy()
    fast_path = _maximum_path_each(v_fast, t_y, t_x)
    if not np.array_equal(ref_path, fast_path):
        fails += 1
        print(f"PATH MISMATCH at t_y={t_y} t_x={t_x}")
    if not np.allclose(ref_value, v_fast):
        fails += 1
        print(f"VALUE MISMATCH at t_y={t_y} t_x={t_x}")

# batch API sanity incl. masking
neg_cent = torch.randn(3, 9, 7)
mask = torch.ones(3, 9, 7)
mask[0, 6:] = 0
mask[1, :, 5:] = 0
out = maximum_path(neg_cent, mask)
assert out.shape == (3, 9, 7) and out.dtype == neg_cent.dtype  # original cython returns input dtype
assert set(np.unique(out.numpy())) <= {0.0, 1.0}
print("batch API OK, dtype:", out.dtype)

if fails == 0:
    print(f"ALL {len(cases)} PATH CASES MATCH BRUTE-FORCE REFERENCE — patch is correct")
else:
    raise SystemExit(f"{fails} FAILURES — DO NOT PATCH TRAINING WITH THIS")
