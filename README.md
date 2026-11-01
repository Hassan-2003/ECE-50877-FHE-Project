# ECE-50877-FHE-Project

CKKS implementations of DSP kernels (FIR, cross-correlation, FFT, ...) compiled with
[HEIR](https://github.com/google/heir) to OpenFHE, compared against plaintext
implementations. See `FHE_Applied_Signal_Processing_Research_Plan.pdf`.

```
DSP_Kernels/   <name>_fhe.py  kernel specs (one per kernel) + plaintext implementations
Scripts/       run_kernel_test.py (runner), mlir_builder.py (kernel builder), setup_env.sh
Results/       <name>_results.csv written by the runner
env/           toolchain built by setup_env.sh (git-ignored)
```

## 1. Setup

```bash
bash Scripts/setup_env.sh      # once, ~10-20 min: installs heir-py, builds OpenFHE C++ into env/
source env/activate.sh         # every session
export MPLCONFIGDIR=$TMPDIR    # only if ~/.config is read-only
```

OpenFHE is built from source because the host glibc is too old for the PyPI wheel; do not
`pip install openfhe`. If `go` is on the PATH, setup also prefetches Lattigo (needs
proxy.golang.org); without it the Lattigo backend is skipped.

## 2. Running a kernel test

```bash
python Scripts/run_kernel_test.py fir                  # all cases, all SIZES, all backends
python Scripts/run_kernel_test.py fir --backends openfhe   # one library only
python Scripts/run_kernel_test.py fir --emit-fideslib out/ # write FIDESlib C++, no GPU needed
python Scripts/run_kernel_test.py fir --sizes 128 256  # override sizes
python Scripts/run_kernel_test.py fir xcorr            # several kernels
```

For each (case, size, backend) the runner: computes the plaintext reference, compiles your
kernel with HEIR (CKKS) to OpenFHE (C++), Lattigo (Go) and FIDESlib (CUDA), encrypts, evaluates, decrypts, then checks
`max|encrypted - plaintext| <= TOL`. It prints PASS/FAIL, exits non-zero on any failure,
and writes `Results/<name>_results.csv`:

| column | meaning |
|---|---|
| `backend` | `openfhe`, `lattigo` or `fideslib`; one row per backend. CKKS parameters are chosen by HEIR per backend |
| `plain_time_s`, `fhe_time_s`, `slowdown` | plaintext time, encrypted kernel eval time, ratio (fhe/plain) |
| `plain_accuracy_pct`, `fhe_accuracy_pct` | 100·(1 − ‖err‖/‖truth‖) vs the float64 plaintext result |
| `plain_max_err`, `fhe_max_err`, `tol`, `passed` | max abs error vs the float64 result; pass = `fhe_max_err <= tol` |
| `plain_mem_mb`, `fhe_mem_mb`, `mem_overhead_mb` | plaintext peak Python allocation; FHE memory (OpenFHE: peak RSS growth of the Python process over setup+encrypt+eval; Lattigo: peak RSS of the whole Go process); difference (MB = 2^20 bytes) |
| `encrypt_pct`, `decrypt_pct` | encryption / decryption time as a percentage of total execution time (`fhe_time_s` + encrypt + decrypt). Compile and key setup are not reported (assumed amortized) |
| `ops_total`, `op_<backend>_<Name>` | library primitives executed during the kernel, e.g. `op_openfhe_EvalMult`, `op_openfhe_MakeCKKSPackedPlaintext`, `op_lattigo_RotateNew`, `op_lattigo_Encode` (plaintext encodes). OpenFHE counts come from counters in the generated C++; Lattigo counts are static counts of the straight-line generated Go |

Expect ~30-90 s per case at n=128 (HEIR compile dominates).

### FIDESlib (GPU)

HEIR has no FIDESlib backend, so `Scripts/fideslib_emitter.py` translates the OpenFHE C++
HEIR generates into FIDESlib's OpenFHE-like API (`ModReduce` -> `Rescale`, GPU context
setup, a timed `main()`), and fails loudly if the kernel uses a primitive FIDESlib lacks.
`fideslib_backend.py` builds it with CMake and runs it. The backend runs only where an
NVIDIA GPU, `nvcc`, `cmake` and an installed FIDESlib (`$FIDESLIB_PREFIX`, default
`env/fideslib`) exist; otherwise the runner prints `SKIP backend fideslib`. Memory for this
backend is host peak RSS plus GPU memory in use. The emitted code has been syntax-checked
against FIDESlib's real headers, but not compiled or run on a GPU.

## 3. Adding a kernel

Create `DSP_Kernels/<name>_fhe.py` (copy `fir_fhe.py`). It must define:

| name | what |
|---|---|
| `SIZES` | list of input lengths to sweep |
| `CASES` | optional dict of variants, e.g. `{"lowpass": dict(kind="lowpass")}`; each dict is passed as `**params` |
| `TOL` | max absolute error allowed under CKKS |
| `make_input(n)` | deterministic float input of length `n` (numpy array) |
| `plaintext(x, **params)` | reference implementation. **Any library is fine here** (scipy, numpy, ...) |
| `mlir(n, **params)` | returns the MLIR string for the encrypted kernel (build it with `mlir_builder.Kernel`) |

Then run `python Scripts/run_kernel_test.py <name>`. No sweep script is needed; the
runner does the sweep.

### Writing the encrypted kernel with `Kernel`

```python
from mlir_builder import Kernel
k = Kernel(n, "mykernel")        # one secret input vector of length n
x = k.input
r = k.rotate(x, 3)               # r[i] = x[(i+3) % n]  -- cyclic!
t = k.mul_const(r, vec)          # ciphertext * public plaintext vector (length n or scalar)
y = k.add(t, x)                  # also: sub, mul (ct*ct), add_const
return k.build(y)
```

## 4. Restrictions on encrypted kernels

- Only add, sub, multiply (ct×ct or ×public constant) and cyclic rotate. Anything else
  (comparison, `abs`, `max`, division, `sqrt`, `if` on data) must become a polynomial
  approximation, mirrored in `plaintext()`.
- No library calls (`lfilter`, `np.convolve`, `np.fft`) on the secret input; constants that
  don't depend on it (taps, twiddles) can be computed with libraries.
- Rotation is cyclic: zero wrapped samples with a mask constant (see `fir_fhe.py`).
- One secret input vector of `n` floats, one output of length `n`, float32 only.
- Depth is chosen by HEIR with no bootstrapping; very deep kernels may fail, which is a
  result worth reporting. Cost grows with rotations × `n`.

## 5. Troubleshooting

- `HEIR Error ... Signature parsing failed` / `UnboundLocalError: numba_ir`: the real
  MLIR error is hidden. Dump your MLIR to a file and run
  `env/venv/lib/python3.12/site-packages/heir/heir-opt --canonicalize --mlir-to-ckks file.mlir`.
- `Python.h not found`: `source env/activate.sh` (sets `CPATH`).
- `ccache: Read-only file system`: `export CCACHE_DISABLE=1` (set by `activate.sh`).

## 6. Generating a kernel file with an AI model

If a student has a plaintext Python implementation (numpy/scipy), an AI model can turn it
into a `DSP_Kernels/<name>_fhe.py` for this runner. Paste the prompt below, then the
student's file, into the model. Always run the result with
`python Scripts/run_kernel_test.py <name> --sizes 128` and treat a FAIL as a bug in the
generated file, not in the runner.

````text
You are converting a student's plaintext Python signal-processing code into a kernel spec
for a CKKS (homomorphic encryption) test harness built on HEIR -> OpenFHE. Output ONE
complete Python file, named <name>_fhe.py, and nothing else (no prose, one code block).

WHAT THE HARNESS DOES:
For each (case, size) it calls make_input(n), computes the reference with plaintext(),
compiles the string returned by mlir() with HEIR, encrypts the input, evaluates, decrypts,
and passes if max|decrypted - plaintext| <= TOL.

THE FILE MUST DEFINE EXACTLY THESE NAMES:
- SIZES: list of ints, input lengths to sweep (default [128, 256, 512]; keep modest).
- CASES: dict {case_name: dict_of_params}; each dict is passed as **params to
  plaintext() and mlir(). Use {"default": {}} if the kernel has no variants.
- TOL: float, max absolute error allowed under CKKS. Use 1e-3 for shallow kernels
  (multiplicative depth <= ~3); raise it for deeper ones and say why in a comment.
- make_input(n): returns a deterministic float numpy array of length n, with values of
  modest magnitude (roughly within +/-3). Use np.random.default_rng(0) for random data.
- plaintext(x, **params): the reference implementation, returning a length-n array. Reuse
  the student's code (any library is fine here). It must accept float32 and float64
  input. If the encrypted kernel needs a polynomial approximation of something CKKS
  cannot do, the reference must use the SAME approximation.
- mlir(n, **params): returns the MLIR string, built with the Kernel builder below.

Imports: `import numpy as np`, `from mlir_builder import Kernel`, plus whatever the
reference needs. The student's reference functions may be imported (e.g. `import
fir_filters as fir`) if the file sits in DSP_Kernels/; otherwise inline them.

THE KERNEL BUILDER (SCRIPTS/MLIR_BUILDER.PY):
    k = Kernel(n, "name")      # ONE secret input vector of length n
    x = k.input
    k.rotate(a, s)             # r[i] = a[(i+s) % n]   CYCLIC, s may be negative
    k.add(a, b)  k.sub(a, b)   # ciphertext +/- ciphertext
    k.mul(a, b)                # ciphertext * ciphertext
    k.mul_const(a, vec)        # ciphertext * PUBLIC vector (length n, or a scalar)
    k.add_const(a, vec)        # ciphertext + PUBLIC vector
    return k.build(result)     # -> MLIR string
Every method works on whole packed vectors; there is no element indexing, no loops over
elements inside the encrypted program. Python loops in mlir() just emit more operations.

RULES FOR THE ENCRYPTED PART (VIOLATIONS MAKE HEIR FAIL OR GIVE WRONG RESULTS):
1. Only add, sub, mul, mul_const, add_const and rotate exist. Comparisons, abs, max/min,
   argmax, division, sqrt, exp, log, and data-dependent `if` are NOT available. Replace
   them with a polynomial approximation built from mul/mul_const/add_const (state the
   approximation and its valid input range in a comment), or by dividing by a public
   constant via mul_const(1/c). Division by a secret value is not possible.
2. No library calls on the secret data. np.convolve, scipy.signal.lfilter, np.fft etc.
   cannot be used inside mlir(). Re-express as rotate/multiply/add. Anything that does
   not depend on the secret input (filter taps, twiddle factors, reference signals,
   masks) may be computed with numpy/scipy in mlir() and passed as constants.
3. Rotation is cyclic. Where a shift would bring in wrapped-around samples that the
   plaintext version treats as zero (zero padding / causal filters), multiply by a mask
   that is 0 at those positions. Fold the mask into the coefficient vector when possible
   (see the FIR example).
4. One secret input vector of length n and one output vector of length n. If the kernel
   needs a different output size, pad or truncate in plaintext() so both are length n. A
   second input that is not secret is just a constant vector.
5. float32 only; CKKS is approximate (errors ~1e-7 for shallow circuits, growing with
   depth). Every ct*ct or ct*plaintext multiply consumes a level; keep the multiplicative
   depth as low as possible (e.g. sum independent products in a tree rather than chaining
   multiplies, and fold constants together before multiplying). The cost grows with
   (number of rotations x n).
6. Prefer log-step algorithms for reductions: summing all n entries is log2(n)
   rotate-and-add steps (rotate by 1, 2, 4, ...), not n steps. n is a power of two.
7. Do not write raw MLIR, do not add a main(), and do not run anything on import.

WORKED EXAMPLE (FIR FILTER; Y[I] = SUM_T H[T] * X[I-T], ZERO-PADDED):
```python
import numpy as np
from scipy.signal import firwin
import fir_filters as fir
from mlir_builder import Kernel

FS = 1000.0
NUM_TAPS = 101
SIZES = [128, 256, 512]
TOL = 1e-3

CASES = {"lowpass": dict(kind="lowpass")}

def make_input(n):
    return np.random.default_rng(0).standard_normal(n)

def plaintext(x, kind):
    return fir.fir_lowpass(x, 100, FS, NUM_TAPS)

def mlir(n, kind):
    h = firwin(NUM_TAPS, 100, fs=FS, pass_zero="lowpass")   # public constants
    k = Kernel(n, "fir")
    acc = None
    for tap, hk in enumerate(h):
        mask = np.zeros(n)
        mask[tap:] = hk                  # coefficient, zero where rotation wrapped
        term = k.mul_const(k.rotate(k.input, -tap), mask)
        acc = term if acc is None else k.add(acc, term)
    return k.build(acc)
```

PROCEDURE:
1. Read the student's code and identify the math in terms of shifts, elementwise
   weights, and sums.
2. List any operation that is not allowed (rule 1) and choose a polynomial approximation
   or a rewrite; put it in a comment, and mirror it in plaintext().
3. Write the file. Check before answering: names SIZES/CASES/TOL/make_input/plaintext/
   mlir all exist; plaintext() and mlir() agree exactly on the semantics at the edges
   (padding, wraparound, output length); no library call touches the secret input.
4. If the student's code cannot be expressed under these rules, say so in a single
   comment at the top of the file and implement the closest supported version, with the
   reference changed to match.
````
