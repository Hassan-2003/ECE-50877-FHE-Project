"""SURF kernel spec for Scripts/run_kernel_test.py (plaintext reference: surf.py)."""

import numpy as np

import surf
from mlir_builder import Kernel

SIZES = [256]
TOL = 1e-4
CASES = {
    "det_9": dict(filter_size=9, response="det"),
    "trace_9": dict(filter_size=9, response="trace"),
    "det_15": dict(filter_size=15, response="det"),
    "trace_15": dict(filter_size=15, response="trace"),
    "det_21": dict(filter_size=21, response="det"),
    "trace_21": dict(filter_size=21, response="trace"),
}


def make_input(n):
    return surf.make_input(n)


def plaintext(x, filter_size=9, response="det"):
    return surf.hessian_response(x, filter_size, response)


def _axis_filter(k, x, weights, axis, scale=1.0):
    rows, cols = np.indices(surf.IMAGE_SHAPE)
    center = len(weights) // 2
    acc = None
    for index, weight in enumerate(weights):
        if weight == 0:
            continue
        offset = index - center
        dy, dx = (offset, 0) if axis == 0 else (0, offset)
        # Mask both coordinates so a packed rotation implements zero-padded filtering.
        valid = ((rows + dy >= 0) & (rows + dy < 16) &
                 (cols + dx >= 0) & (cols + dx < 16))
        mask = valid.ravel() * weight * scale
        term = k.mul_const(k.rotate(x, 16 * dy + dx), mask)
        acc = term if acc is None else k.add(acc, term)
    return acc


def mlir(n, filter_size=9, response="det"):
    if n != surf.IMAGE_SIZE:
        raise ValueError("SURF expects n=256.")
    if response not in ("det", "trace"):
        raise ValueError("response must be 'det' or 'trace'.")
    band, second, mixed = surf.filter_vectors(filter_size)
    k = Kernel(n, "surf")
    scale = 1 / filter_size**2

    # Bay et al., Sec. 3: separable box filtering produces the same Dxx and Dyy maps.
    vertical = _axis_filter(k, k.input, band, axis=0)
    horizontal = _axis_filter(k, k.input, band, axis=1)
    dxx = _axis_filter(k, vertical, second, axis=1, scale=scale)
    dyy = _axis_filter(k, horizontal, second, axis=0, scale=scale)
    if response == "trace":
        return k.build(k.add(dxx, dyy))

    # Fold Bay's 0.9 correction into public weights before the determinant products.
    mixed_rows = _axis_filter(k, k.input, mixed, axis=0)
    dxy = _axis_filter(k, mixed_rows, mixed, axis=1, scale=surf.DXY_WEIGHT * scale)
    determinant = k.sub(k.mul(dxx, dyy), k.mul(dxy, dxy))
    # Shortell, Sec. 5: decrypt this map before running keypoint comparisons.
    return k.build(determinant)
