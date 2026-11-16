# ECE-50877-FHE-Project

CKKS (homomorphic encryption) versions of DSP kernels (FIR, cross-correlation, FFT, SURF and HOG),
compiled with [HEIR](https://github.com/google/heir) to three libraries (OpenFHE and
Lattigo on the CPU, FIDESlib on an NVIDIA GPU) and compared against plaintext Python.
See `FHE_Applied_Signal_Processing_Research_Plan.pdf`.

```
DSP_Kernels/   <name>_fhe.py kernel files (+ the plaintext code they came from)
Scripts/       setup_env.sh, run_sweep.sh, run_kernel_test.py (runner) and backends
Results/       <name>_results.csv, written by the sweep
env/           toolchain built by setup_env.sh (git-ignored)
```

## 1. Setup (once)

On a node with internet (e.g. the cluster login node):

```bash
bash Scripts/setup_env.sh
```

This builds everything into `env/` (about 30-40 min): a Python venv with HEIR, OpenFHE,
Lattigo, and FIDESlib if CUDA is found. On Purdue Scholar it needs no modules or options.
On other machines you can set `CUDA_PATH=<cuda root>`, `FIDESLIB_ARCH=<sm list, e.g.
80-real>` or `SKIP_FIDESLIB=1`.

## 2. Converting a Python kernel

A kernel is one file, `DSP_Kernels/<name>_fhe.py`; `fir_fhe.py` and `xcorr_fhe.py` are
examples. To make one from plaintext numpy/scipy code, paste the prompt below into an AI
model, followed by the plaintext file. Save its answer as `DSP_Kernels/<name>_fhe.py` and
check it with `sbatch Scripts/run_sweep.sh <name> --sizes 128`. A FAIL means the generated
file is wrong, not the runner.

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

## 3. Running the sweep

```bash
sbatch Scripts/run_sweep.sh              # fir and xcorr, ~10 min on one A30
sbatch Scripts/run_sweep.sh <name>       # your kernel(s)
```

For every case and size in the kernel file, on all three libraries, the sweep computes the
plaintext reference, compiles the kernel with HEIR, encrypts, evaluates, decrypts and checks
the error. It writes `Results/<name>_results.csv`, replacing the previous one, and logs to
`Results/sweep_<jobid>.log`. Each line of the log reads `PASS` or `FAIL`. The job exits
non-zero on any FAIL, and it refuses to start if one of the libraries is missing.

It runs as one GPU job. The defaults are Purdue Scholar's (`-A scholar -p gpu
--gres=gpu:1 -C A30`); elsewhere use `sbatch -A <account> -p <gpu partition> ...`. Options
after the kernel names go to the runner, e.g. `--sizes 128`.

### FFT, SURF and HOG use the same runner

Each algorithm has a plaintext module and a matching `<name>_fhe.py` specification,
as in `fft.py` / `fft_fhe.py`. All use the same cases, backend selection and
`Results/<name>_results.csv` format:

```bash
sbatch Scripts/run_sweep.sh fft surf hog --sizes 256
# Or inside the configured environment:
python Scripts/run_kernel_test.py fft surf hog --sizes 256 --backends openfhe lattigo fideslib
```

SURF and HOG accept a row-major 16×16 grayscale image with intensities in [0,1].
Their seeded blob/square/noise image is identical across backends and algorithms.

| Kernel | Cases and output |
|---|---|
| SURF | Determinant and trace maps at sizes 9, 15 and 21; 256 row-major responses |
| HOG | Horizontal/vertical gradients, squared magnitude, polynomial histogram and descriptor |
| HOG descriptor layout | Four 8×8 cells × nine unsigned bins in slots 0:36; the remaining slots are zero |

SURF follows [Bay et al., Section 3](https://doi.org/10.1007/11744023_32) for the
box Hessian and [Shortell and Shokoufandeh, Section 5](https://doi.org/10.1049/iet-ifs.2019.0157)
for computing responses before decryption and comparing keypoints afterward.
Zero padding is used at image edges; size 21 therefore uses padding everywhere
on this small image. Candidate detection omits subpixel interpolation, further
octaves, orientation assignment and SURF descriptors.

HOG follows [Dalal and Triggs, Sections 6.2–6.4](https://doi.org/10.1109/CVPR.2005.177)
for gradients, orientation cells and normalization. Undefined normal derivatives
are zero at the image boundary. Our encrypted variant replaces magnitude/angle
voting with `(8/27)*mean(((gx*cos(theta)+gy*sin(theta))/2)^4)` per cell and bin.
It uses hard cells, regularized L2 normalization with epsilon=0.25 and a quadratic
inverse-square-root approximation; it omits spatial interpolation and L2-Hys
clipping. For [0,1] images, the squared histogram norm is at most 35/648, and the
normalization polynomial's relative error is below 0.98% on that domain.

`hog.classical_descriptor()` retains exact voting, spatial interpolation and
L2-Hys for comparison. Run `python DSP_Kernels/hog.py` to see algorithm and
normalization errors separately; the histogram/descriptor differences from
classical HOG are about 0.0111/0.228 on the fixed image. The runner's
`fhe_max_err` instead measures numerical error against the matching polynomial
reference. A CKKS PASS does not imply equivalence to full classical HOG.
Gamma correction, Gaussian block weighting and classification are omitted.

The runner and backend adapters are unchanged. The FIDESlib emitter additionally
maps adjacent multiply/relinearize pairs and level drops for SURF/HOG.
The SURF/HOG result files currently contain local Lattigo CPU measurements only.
Memory is not measured on macOS because the existing driver reads Linux `/proc`.
OpenFHE/FIDESlib execution
and comparable memory measurements require the configured Linux GPU node.

## 4. Results

One row per (backend, case, n), where `backend` is `openfhe`, `lattigo` or `fideslib`:

| column | meaning |
|---|---|
| `passed`, `fhe_max_err`, `tol` | PASS when the max absolute error of the decrypted output vs. the float64 plaintext result is <= the kernel's `TOL` |
| `fhe_accuracy_pct`, `plain_accuracy_pct` | 100·(1 − ‖err‖/‖truth‖) for the encrypted and the float32 plaintext run |
| `fhe_time_s` | time of the encrypted kernel evaluation only (including encoding its constant vectors). Compilation and key generation are excluded (paid once) |
| `plain_time_s`, `slowdown` | plaintext time; slowdown = `fhe_time_s / plain_time_s` |
| `encrypt_pct`, `decrypt_pct` | encryption / decryption time as a % of `fhe_time_s` + encrypt + decrypt |
| `fhe_mem_mb`, `plain_mem_mb`, `mem_overhead_mb` | memory (MB) of the FHE run, of the plaintext run, and their difference |
| `ops_total`, `op_<backend>_<Name>` | library operations executed by the kernel (rotations, multiplies, adds, rescales, plaintext encodes) |

How to read them:

- **Errors** are around 1e-7 for FIR and 1e-6 for xcorr on every backend, far below
  `TOL = 1e-3`. HEIR chooses the CKKS parameters per library, so the errors differ
  slightly. On the same input, FIDESlib and OpenFHE agree to about 4e-9.
- **Time hardly depends on `n`** up to 1024. HEIR packs the input into one 1024-slot
  ciphertext, so cost follows the number of operations (one rotation and multiply per FIR
  tap), not the signal length.
- **Memory is measured differently per backend**, so compare it within a backend, not across:
  - OpenFHE: growth of the Python process.
  - Lattigo: the whole Go process.
  - FIDESlib: host peak memory plus GPU memory (about 1.1 GB of it is GPU memory, mostly
    keys).

  The first row of a sweep reads high because the library warms up then.
- **`op_*` counts** show where the time goes; FIR uses 100 rotations and 101 multiplies.

Reference sweep (Scholar, NVIDIA A30 GPU node with 8 CPU cores), ranges over all cases and
sizes:

| kernel | backend | fhe_time_s | slowdown | fhe_mem_mb | max error |
|---|---|---|---|---|---|
| fir | openfhe | 0.49–0.57 | 914–1030× | ~253 | 1.2e-7 |
| fir | lattigo | 0.22 | 398–432× | ~137 | 2.8e-7 |
| fir | fideslib | 0.071–0.079 | 133–151× | ~1368 | 1.2e-7 |
| xcorr | openfhe | 0.36–0.37 | 1016–2123× | ~133 | 1.9e-6 |
| xcorr | lattigo | 0.14 | 382–797× | ~93 | 3.7e-6 |
| xcorr | fideslib | 0.042–0.047 | 129–262× | ~1330 | 1.9e-6 |

## Troubleshooting

- `HEIR Error ... Signature parsing failed`: the real MLIR error is hidden. Write
  `mlir(n)`'s output to `file.mlir` and run
  `env/venv/lib/python3.12/site-packages/heir/heir-opt --canonicalize --mlir-to-ckks file.mlir`.
- Running the runner by hand (on a compute node): `source env/activate.sh`, then
  `env/venv/bin/python Scripts/run_kernel_test.py <name> --backends openfhe --sizes 128`.
  `--emit-fideslib DIR` writes the generated GPU C++ without needing a GPU.
