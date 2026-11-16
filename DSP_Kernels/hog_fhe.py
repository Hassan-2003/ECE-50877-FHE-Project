"""HOG kernel spec for Scripts/run_kernel_test.py (plaintext reference: hog.py)."""

import numpy as np

import hog
from mlir_builder import Kernel

SIZES = [256]
TOL = 1e-3
CASES = {
    "grad_x": dict(stage="grad_x"),
    "grad_y": dict(stage="grad_y"),
    "magnitude_sq": dict(stage="magnitude_sq"),
    "histogram_poly": dict(stage="histogram_poly"),
    "descriptor_poly": dict(stage="descriptor_poly"),
}


def make_input(n):
    return hog.make_input(n)


def plaintext(x, stage="grad_x"):
    return hog.features(x, stage)


def _gradients(k):
    rows, cols = np.indices(hog.IMAGE_SHAPE)
    # Dalal-Triggs, Sec. 6.2: move the two neighbors into each packed pixel's slot.
    gx = k.sub(k.rotate(k.input, 1), k.rotate(k.input, -1))
    gy = k.sub(k.rotate(k.input, 16), k.rotate(k.input, -16))
    gx = k.mul_const(gx, ((cols > 0) & (cols < 15)).ravel())
    gy = k.mul_const(gy, ((rows > 0) & (rows < 15)).ravel())
    return gx, gy


def _cell_sums(k, votes):
    # Sum 8x8 windows in six steps; only the four cell anchors are retained below.
    for shift in (1, 2, 4, 16, 32, 64):
        votes = k.add(votes, k.rotate(votes, shift))
    return votes


def _pack_bin(k, cell_sums, bin_index):
    acc = None
    for cell, source in enumerate((0, 8, 128, 136)):
        target = cell * hog.BINS + bin_index
        mask = np.zeros(hog.IMAGE_SIZE)
        mask[target] = hog.VOTE_SCALE / hog.CELL_SIZE**2
        # Gather each cell anchor into the same cell-major layout as hog.features().
        term = k.mul_const(k.rotate(cell_sums, source - target), mask)
        acc = term if acc is None else k.add(acc, term)
    return acc


def mlir(n, stage="grad_x"):
    if n != hog.IMAGE_SIZE:
        raise ValueError("HOG expects n=256.")
    if stage not in hog.STAGES:
        raise ValueError(f"stage must be one of {hog.STAGES}.")
    k = Kernel(n, "hog")
    gx, gy = _gradients(k)
    if stage == "grad_x":
        return k.build(gx)
    if stage == "grad_y":
        return k.build(gy)
    if stage == "magnitude_sq":
        return k.build(k.add(k.mul(gx, gx), k.mul(gy, gy)))

    histogram = None
    for index, angle in enumerate(hog.ANGLES):
        # Our polynomial substitutes for Shortell's angle/magnitude histogram stage.
        projection = k.add(k.mul_const(gx, np.cos(angle) / 2),
                           k.mul_const(gy, np.sin(angle) / 2))
        square = k.mul(projection, projection)
        votes = k.mul(square, square)
        term = _pack_bin(k, _cell_sums(k, votes), index)
        histogram = term if histogram is None else k.add(histogram, term)
    if stage == "histogram_poly":
        return k.build(histogram)

    # Dalal's normalization stage starts with the squared norm of the 36 packed bins.
    sumsq = k.mul(histogram, histogram)
    for shift in (1, 2, 4, 8, 16, 32, 64, 128):
        sumsq = k.add(sumsq, k.rotate(sumsq, shift))
    centered = k.add_const(sumsq, -hog.NORM_SUMSQ_MAX / 2)
    t = k.mul_const(centered, 1 / hog.NORM_CENTER)
    t2 = k.mul(t, t)
    scale = 1 / np.sqrt(hog.NORM_CENTER)
    # Our bounded quadratic matches hog.inverse_sqrt_polynomial(); constants are public.
    linear = k.mul_const(t, -0.5 * scale)
    quadratic = k.mul_const(t2, 0.375 * scale)
    inverse_norm = k.add_const(k.add(linear, quadratic), scale)
    return k.build(k.mul(histogram, inverse_norm))
