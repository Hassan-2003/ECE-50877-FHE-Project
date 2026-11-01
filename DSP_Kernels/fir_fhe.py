"""FIR filter kernel spec for Scripts/run_kernel_test.py (template for new kernels)."""
import numpy as np
from scipy.signal import firwin

import fir_filters as fir
from mlir_builder import Kernel

FS = 1000.0
NUM_TAPS = 101
SIZES = [128, 256, 512]
TOL = 1e-3          # max abs error allowed under CKKS (f32 + approximation)

CASES = {
    "lowpass": dict(kind="lowpass"),
    "highpass": dict(kind="highpass"),
    "bandpass": dict(kind="bandpass"),
}


def make_input(n):
    # Seeded so every run and teammate gets the same "random" signal.
    return np.random.default_rng(0).standard_normal(n)


def plaintext(x, kind):
    """Reference: the library implementation from fir_filters.py."""
    if kind == "lowpass":
        return fir.fir_lowpass(x, 100, FS, NUM_TAPS)
    if kind == "highpass":
        return fir.fir_highpass(x, 200, FS, NUM_TAPS)
    return fir.fir_bandpass(x, 100, 200, FS, NUM_TAPS)


def coefficients(kind):
    # Computed with scipy *outside* the encrypted kernel: public constants.
    if kind == "lowpass":
        return firwin(NUM_TAPS, 100, fs=FS, pass_zero="lowpass")
    if kind == "highpass":
        return firwin(NUM_TAPS, 200, fs=FS, pass_zero="highpass")
    return firwin(NUM_TAPS, [100, 200], fs=FS, pass_zero="bandpass")


def mlir(n, kind):
    """Encrypted kernel: y = sum_k h[k] * shift(x, k), as rotate + mul_const + add.
    Rotation is cyclic, so the plaintext constant is zero where samples wrapped."""
    h = coefficients(kind)
    k = Kernel(n, "fir")
    acc = None
    for tap, hk in enumerate(h):
        mask = np.zeros(n)
        mask[tap:] = hk
        term = k.mul_const(k.rotate(k.input, -tap), mask)
        acc = term if acc is None else k.add(acc, term)
    return k.build(acc)
