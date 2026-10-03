"""Pure-numpy replacement for piper's compiled monotonic_align extension.

This machine has no MSVC compiler, so the Cython core cannot be built.
This module reproduces the exact algorithm from VITS `monotonic_align/core.pyx`
(jaywalnut310/vits) in numpy and is installed as a drop-in patch.
Verified against a brute-force reference in test_monotonic_align.py.
"""
import numpy as np
import torch

MAX_NEG_VAL = -1e9


def _maximum_path_each(value, t_y, t_x):
    """value: [t_y, t_x] float32 (modified in place); returns path [t_y, t_x]."""
    path = np.zeros((t_y, t_x), dtype=np.int32)
    index = t_x - 1
    for y in range(t_y):
        x0 = max(0, t_x + y - t_y)
        x1 = min(t_x, y + 1)
        xs = np.arange(x0, x1)
        if y == 0:
            v_cur = np.full(xs.shape, MAX_NEG_VAL, dtype=np.float32)
            v_prev = np.where(xs == 0, 0.0, MAX_NEG_VAL).astype(np.float32)
        else:
            v_cur = np.where(xs == y, MAX_NEG_VAL, value[y - 1, xs])
            v_prev = np.where(xs == 0, MAX_NEG_VAL,
                              value[y - 1, np.maximum(xs - 1, 0)])
        value[y, xs] += np.maximum(v_prev, v_cur)
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and y != 0 and (index == y or
                                      value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
    return path


def maximum_path(neg_cent, mask):
    """neg_cent: [b, t_y, t_x]; mask: [b, t_y, t_x] -> int path tensor."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    values = neg_cent.data.cpu().numpy().astype(np.float32)
    mask_np = mask.data.cpu().numpy().astype(np.float32)
    b, t_y, t_x = values.shape
    t_ys = mask_np.sum(1)[:, 0].astype(np.int32)
    t_xs = mask_np.sum(2)[:, 0].astype(np.int32)
    path = np.zeros((b, t_y, t_x), dtype=np.int32)
    for i in range(b):
        v = values[i].copy()
        path[i, : t_ys[i], : t_xs[i]] = _maximum_path_each(v, t_ys[i], t_xs[i])
    return torch.from_numpy(path).to(device=device, dtype=dtype)
