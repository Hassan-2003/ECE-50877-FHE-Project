"""Cross-correlation (matched filter) kernel spec for Scripts/run_kernel_test.py."""
import numpy as np

import cross_correlation as xc
from mlir_builder import Kernel

REF_LEN = 64
SIZES = [128, 256, 512]
TOL = 1e-3          # max abs error allowed under CKKS (f32 + approximation)

CASES = {"matched": dict()}

# Public reference (transmitted) signal; only the received signal is secret.
REFERENCE = np.random.default_rng(1).standard_normal(REF_LEN)


def make_input(n):
    """Received signal: noise with the reference buried at n // 4."""
    rng = np.random.default_rng(0)
    received = 0.5 * rng.standard_normal(n)
    d = n // 4
    received[d:d + REF_LEN] += REFERENCE
    return received


def plaintext(x):
    """Reference: the library implementation from cross_correlation.py."""
    return xc.matched_filter(x, REFERENCE)


def mlir(n):
    """Matched filter = FIR with taps h = reverse(reference):
    y[i] = sum_t h[t] * x[i-t], as rotate + mul_const + add. Rotation is cyclic, so
    the plaintext constant is zero where samples wrapped."""
    h = REFERENCE[::-1]
    k = Kernel(n, "xcorr")
    acc = None
    for tap, hk in enumerate(h):
        mask = np.zeros(n)
        mask[tap:] = hk
        term = k.mul_const(k.rotate(k.input, -tap), mask)
        acc = term if acc is None else k.add(acc, term)
    return k.build(acc)
