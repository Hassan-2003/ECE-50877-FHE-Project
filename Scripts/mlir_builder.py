"""
Tiny builder that emits the MLIR HEIR compiles for packed-vector CKKS kernels.

HEIR's Python frontend only handles scalars (no tensor indexing / vector ops), so
vector kernels are written with this builder instead of raw MLIR:

    k = Kernel(n)                    # one secret input vector of length n
    x = k.input
    r = k.rotate(x, 3)               # r[i] = x[(i + 3) % n]   (cyclic)
    t = k.mul_const(r, [0.5] * n)    # multiply by a public (plaintext) vector
    y = k.add(t, x)                  # add two ciphertext vectors
    mlir = k.build(y)

Supported: add, sub, mul (ct*ct), mul_const (ct*plaintext vector), add_const, rotate.
Not supported: comparisons, division, argmax, data-dependent control flow.
"""
import numpy as np


class Kernel:
    def __init__(self, n, name="kernel"):
        self.n = n
        self.name = name
        self.lines = []
        self.count = 0
        self.input = "%x"

    def _v(self):
        self.count += 1
        return f"%v{self.count}"

    def _ty(self):
        return f"tensor<{self.n}xf32>"

    def _const_vec(self, vec):
        vec = np.broadcast_to(np.asarray(vec, dtype=np.float64), (self.n,))
        v = self._v()
        vals = ", ".join(f"{a:.9e}" for a in vec)
        self.lines.append(f"  {v} = arith.constant dense<[{vals}]> : {self._ty()}")
        return v

    def _binop(self, op, a, b):
        v = self._v()
        self.lines.append(f"  {v} = arith.{op} {a}, {b} : {self._ty()}")
        return v

    def add(self, a, b):
        return self._binop("addf", a, b)

    def sub(self, a, b):
        return self._binop("subf", a, b)

    def mul(self, a, b):
        return self._binop("mulf", a, b)

    def mul_const(self, a, vec):
        return self._binop("mulf", a, self._const_vec(vec))

    def add_const(self, a, vec):
        return self._binop("addf", a, self._const_vec(vec))

    def rotate(self, a, k):
        """result[i] = a[(i + k) % n]"""
        k %= self.n
        if k == 0:
            return a
        c, v = self._v(), self._v()
        self.lines.append(f"  {c} = arith.constant {k} : index")
        self.lines.append(
            f"  {v} = tensor_ext.rotate {a}, {c} : {self._ty()}, index")
        return v

    def build(self, out):
        head = (f"func.func @{self.name}(%x: {self._ty()} {{secret.secret}}) "
                f"-> {self._ty()} {{")
        return "\n".join([head, *self.lines,
                          f"  return {out} : {self._ty()}", "}"])
