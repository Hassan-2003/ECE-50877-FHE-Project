"""Real FFT kernel spec for Scripts/run_kernel_test.py (plaintext reference: fft.py).

The DFT of a real length-n signal is computed as one public-matrix x secret-vector product
(multiplicative depth 1), with the matrix applied by the diagonal method and
baby-step giant-step (BSGS):

    y = sum_j d_j * rot(x, j),          d_j[i] = M[i, (i + j) % n]       (n rotations)
      = sum_a rot( sum_b rot(d_{ag+b}, -ag) * rot(x, b),  ag )         (j = a*g + b)

Rotating the public diagonals is free, so BSGS needs (g - 1) baby-step rotations, all of the
input x, plus (n/g - 1) giant-step rotations of different partial sums. HEIR hoists the
baby steps for OpenFHE/FIDESlib (one digit decomposition + EvalFastRotation), which suggested
a g above sqrt(n) might win. Measured (Scripts/run_sweep.sh, RTX 5070 Ti + Ryzen 7 9700X): the
balanced g ~ sqrt(n) is fastest or within noise in nearly every (backend, n); the exception
is OpenFHE at n=512, where g ~ 2 sqrt(n) is ~9% faster. Total rotations dominate. BSGS beats
the plain diagonal method by 1.5-6.3x (Lattigo n=512: 0.688 s -> 0.109 s). The CASES sweep g:

    bsgs1x  g ~ sqrt(n)        (balanced: equal baby/giant counts)
    bsgs2x  g ~ 2 sqrt(n)
    bsgs4x  g ~ 4 sqrt(n)
    diag    g = n              (plain diagonal method: n-1 hoisted rotations, no giant steps)

Output layout (n reals, FFTW "halfcomplex", orthonormal scaling so |y| stays O(1)):
    y[k]     = Re X[k],  k = 0 .. n/2
    y[n - k] = Im X[k],  k = 1 .. n/2 - 1
"""
import numpy as np

import fft
from mlir_builder import Kernel

SIZES = [128, 256, 512]
TOL = 1e-3          # depth 1 (one plaintext multiply per term), like the FIR kernel

CASES = {
    "bsgs1x": dict(g_mult=1),
    "bsgs2x": dict(g_mult=2),
    "bsgs4x": dict(g_mult=4),
    "diag": dict(g_mult=None),
}


def make_input(n):
    return np.random.default_rng(0).standard_normal(n)


def plaintext(x, g_mult=None):
    """Reference: fft.real_fft (halfcomplex layout); g_mult only changes the encrypted kernel."""
    return fft.real_fft(x)


def dft_matrix(n):
    """Real n x n matrix with dft_matrix(n) @ x == plaintext(x)."""
    j = np.arange(n)
    M = np.zeros((n, n))
    for k in range(n // 2 + 1):
        M[k] = np.cos(2 * np.pi * j * k / n)
    for k in range(1, n // 2):
        M[n - k] = -np.sin(2 * np.pi * j * k / n)
    return M / np.sqrt(n)


def baby_step(n, g_mult):
    """Baby-step size g (a power of two dividing n)."""
    if g_mult is None:
        return n
    g = 2 ** int(np.ceil(np.log2(n) / 2)) * g_mult
    return min(g, n)


def mlir(n, g_mult=None):
    M = dft_matrix(n)
    g = baby_step(n, g_mult)
    i = np.arange(n)
    k = Kernel(n, "fft")
    baby = [k.rotate(k.input, b) for b in range(g)]       # rotate(x, 0) is x itself
    acc = None
    for a in range(n // g):
        inner = None
        for b in range(g):
            diag = M[i, (i + a * g + b) % n]                 # d_j, j = a*g + b
            # rot(d_j, -a*g) as a public constant: Kernel.rotate(v, s)[i] = v[(i+s) % n]
            term = k.mul_const(baby[b], np.roll(diag, a * g))
            inner = term if inner is None else k.add(inner, term)
        part = k.rotate(inner, a * g)                        # giant step
        acc = part if acc is None else k.add(acc, part)
    return k.build(acc)
