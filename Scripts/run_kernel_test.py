"""
Correctness + metrics runner for CKKS kernels compiled with HEIR -> OpenFHE, Lattigo and FIDESlib.

    source env/activate.sh
    python Scripts/run_kernel_test.py fir            # uses DSP_Kernels/fir_fhe.py
    python Scripts/run_kernel_test.py fir xcorr --sizes 128 256

See README.md for the kernel file contract. Output: Results/<name>_results.csv with
plaintext/FHE time, plaintext/FHE accuracy, tolerance, pass, slowdown, memory
overhead, and a count of every dynamically executed OpenFHE primitive (op_* columns).

How primitives are counted: the C++ HEIR generates is rewritten before compilation so
each `cc->EvalXxx(...)`/`ModReduce`/`Encrypt`/... call first bumps a counter. The
counters are reset right before the kernel runs and read right after, so loops and
hoisted rotations are counted as actually executed.

How time is measured: every backend runs the encrypted kernel once untimed (warm-up), then
--repeats times (default 10); fhe_time_s is the median and fhe_time_std_s the standard
deviation. The plaintext reference takes only microseconds, so it is timed like `timeit`:
batches of many calls, 7 batches, median per-call time. slowdown = median / median.
"""
import argparse
import csv
import ctypes
import importlib
import os
import re
import shutil
import sys
import threading
import time
import tracemalloc

import numpy as np
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "DSP_Kernels"))
sys.path.insert(0, os.path.join(ROOT, "Scripts"))
import matplotlib
matplotlib.use("Agg")  # kernel files may import pyplot; stay headless

import fideslib_backend  # noqa: E402
import lattigo_backend  # noqa: E402
import heir  # noqa: E402
import heir.pipeline as heir_pipeline  # noqa: E402
from heir.backends.util import cpp_compiler  # noqa: E402

# ------------------------------------------------------- primitive counting
COUNTER_CODE = r"""
#include <map>
#include <string>
static std::map<std::string, long>& heir_counts() {
  static std::map<std::string, long> m; return m;
}
static inline void heir_cnt(const char* n) { heir_counts()[n]++; }
extern "C" void heir_cnt_reset() { heir_counts().clear(); }
extern "C" const char* heir_cnt_dump() {
  static std::string s; s.clear();
  for (auto& kv : heir_counts()) s += kv.first + "=" + std::to_string(kv.second) + ";";
  return s.c_str();
}
"""
# Calls on the crypto context that we treat as FHE primitives. Key generation
# (Eval*KeyGen) runs in setup, not in the kernel, and is excluded.
PRIM_RE = re.compile(
    r"\bcc->(?!Eval\w*KeyGen)"
    r"(Eval\w+|ModReduce\w*|Rescale\w*|Relinearize\w*|LevelReduce\w*|KeySwitch\w*"
    r"|MakeCKKSPackedPlaintext|Encrypt|Decrypt)\(")

_orig_compile = cpp_compiler.CppCompilerBackend.compile_to_shared_object
_kept_dirs = []


def _instrumenting_compile(self, cpp_source_filepath, *a, **kw):
    p = str(cpp_source_filepath)
    if p.endswith(".cpp") and not p.endswith("_bindings.cpp"):
        src = open(p).read()
        src = PRIM_RE.sub(lambda m: f'(heir_cnt("{m.group(1)}"), cc)->{m.group(1)}(', src)
        marker = "using namespace lbcrypto;"
        src = src.replace(marker, marker + "\n" + COUNTER_CODE, 1)
        open(p, "w").write(src)
        _kept_dirs.append(os.path.dirname(p))
    return _orig_compile(self, cpp_source_filepath, *a, **kw)


def _keep_dir(path, *a, **kw):
    pass  # the counters live in a .so in the workspace; keep it until we exit


cpp_compiler.CppCompilerBackend.compile_to_shared_object = _instrumenting_compile
heir_pipeline.shutil = type("shutil_keep", (), {"rmtree": staticmethod(_keep_dir),
                                                "__getattr__": lambda s, n: getattr(shutil, n)})()


def read_counts(so_path):
    lib = ctypes.CDLL(so_path)
    lib.heir_cnt_dump.restype = ctypes.c_char_p
    s = lib.heir_cnt_dump().decode()
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(";") if kv)}


def reset_counts(so_path):
    ctypes.CDLL(so_path).heir_cnt_reset()


# ------------------------------------------------------------------ metrics
class PeakRSS:
    """Samples process RSS in a thread; reports peak growth over the starting RSS."""

    def __enter__(self):
        self.proc = psutil.Process()
        self.base = self.peak = self.proc.memory_info().rss
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.002)

    def __exit__(self, *a):
        self.stop = True
        self.t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.delta = self.peak - self.base


_COUNTER = 0


def run_fhe(mlir_str, x, repeats=1):
    """Compile, then setup/encrypt/eval/decrypt. Returns (y, timings, op_counts, mem)."""
    # HEIR names the compiled module after the function; reusing a name in one
    # process silently returns the first module. Make each name unique.
    global _COUNTER
    _COUNTER += 1
    mlir_str = re.sub(r"func\.func @\w+", f"func.func @kernel_{_COUNTER}", mlir_str, count=1)

    n_dirs = len(_kept_dirs)
    t0 = time.perf_counter()
    fn = heir.compile(mlir_str=mlir_str, scheme="ckks")
    t_compile = time.perf_counter() - t0
    so_path = os.path.join(_kept_dirs[n_dirs], f"kernel_{_COUNTER}.so")

    with PeakRSS() as mem:
        t0 = time.perf_counter()
        fn.setup()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        ct = fn.encrypt_arg_0([float(v) for v in x])
        t_enc = time.perf_counter() - t0

        out = fn.eval(ct)                      # warm-up, untimed
        t_eval = []
        for r in range(repeats):
            if r == repeats - 1:
                reset_counts(so_path)          # count the ops of one run only
            t0 = time.perf_counter()
            out = fn.eval(ct)
            t_eval.append(time.perf_counter() - t0)
        counts = read_counts(so_path)

    t0 = time.perf_counter()
    y = np.array(fn.decrypt_result(out), dtype=float).ravel()[:len(x)]
    t_dec = time.perf_counter() - t0
    return y, dict(eval_s=t_eval, compile_s=t_compile, setup_s=t_setup,
                   encrypt_s=t_enc, decrypt_s=t_dec), counts, mem.delta


def time_plain(f, x, batches=7, min_batch_s=0.02):
    """Median per-call time of f(x), timeit-style: grow the batch until one batch takes at
    least min_batch_s (so timer resolution and one-off noise do not matter), then time
    `batches` batches."""
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            f(x)
        if time.perf_counter() - t0 >= min_batch_s:
            break
        number *= 2
    per_call = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(number):
            f(x)
        per_call.append((time.perf_counter() - t0) / number)
    return float(np.median(per_call))


def accuracy_pct(y, ref):
    return 100 * (1 - np.linalg.norm(y - ref) / np.linalg.norm(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="+", help="kernel names, e.g. fir")
    ap.add_argument("--sizes", nargs="+", type=int, help="override SIZES")
    ap.add_argument("--backends", nargs="+", choices=["openfhe", "lattigo", "fideslib"],
                    default=["openfhe", "lattigo", "fideslib"],
                    help="FHE libraries to run (default: all; unavailable ones are skipped)")
    ap.add_argument("--repeats", type=int, default=10,
                    help="timed runs of the encrypted kernel after one warm-up run; "
                         "fhe_time_s is their median (default: 10)")
    ap.add_argument("--emit-fideslib", metavar="DIR",
                    help="only write the generated FIDESlib C++ for every case/size to DIR "
                         "(no GPU needed) and exit")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    if args.emit_fideslib:
        os.makedirs(args.emit_fideslib, exist_ok=True)
        for kname in args.kernels:
            mod = importlib.import_module(f"{kname}_fhe")
            for case, params in getattr(mod, "CASES", {"default": {}}).items():
                for n in args.sizes or mod.SIZES:
                    path = os.path.join(args.emit_fideslib, f"{kname}_{case}_{n}.cpp")
                    with open(path, "w") as fh:
                        fh.write(fideslib_backend.emit_source(mod.mlir(n, **params)))
                    print("wrote", path)
        return

    failures = total = 0
    runners = {"openfhe": run_fhe, "lattigo": lattigo_backend.run_lattigo,
               "fideslib": fideslib_backend.run_fideslib}
    checks = {"lattigo": lattigo_backend.available, "fideslib": fideslib_backend.available}
    backends = []
    for b in args.backends:
        why = checks[b]() if b in checks else None
        if why:
            print(f"SKIP backend {b}: {why}")
        else:
            backends.append(b)
    for kname in args.kernels:
        mod = importlib.import_module(f"{kname}_fhe")
        rows = []
        for case, params in getattr(mod, "CASES", {"default": {}}).items():
            for n in args.sizes or mod.SIZES:
                x = mod.make_input(n)
                truth = np.asarray(mod.plaintext(x, **params), dtype=np.float64)

                # Plaintext run: float32 like the FHE kernel's precision; its accuracy is
                # measured against the float64 truth. Memory comes from one traced call,
                # time from untraced repeated calls (tracemalloc slows Python code down).
                x32 = x.astype(np.float32)
                tracemalloc.start()
                y_plain = np.asarray(mod.plaintext(x32, **params)).astype(np.float32)
                plain_mem = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                t_plain = time_plain(lambda v: mod.plaintext(v, **params), x32)

                for backend in backends:
                    total += 1
                    tag = f"{kname}/{case} n={n} [{backend}]"
                    try:
                        y, tm, counts, fhe_mem = runners[backend](mod.mlir(n, **params), x,
                                                                  args.repeats)
                        tm["fhe_time_s"] = float(np.median(tm["eval_s"]))
                    except BaseException as e:   # heir may SystemExit on compile errors
                        print(f"FAIL {tag}: {e!r}")
                        failures += 1
                        continue
                    err = float(np.max(np.abs(y - truth)))
                    # compile/setup are amortized and omitted; enc/dec are percentages
                    t_total = tm["fhe_time_s"] + tm["encrypt_s"] + tm["decrypt_s"]
                    ok = bool(err <= mod.TOL)
                    failures += not ok
                    rows.append(dict(
                        kernel=kname, backend=backend, case=case, n=n,
                        plain_time_s=t_plain, fhe_time_s=tm["fhe_time_s"],
                        fhe_time_std_s=float(np.std(tm["eval_s"])), repeats=args.repeats,
                        slowdown=tm["fhe_time_s"] / t_plain if t_plain else float("nan"),
                        encrypt_pct=100 * tm["encrypt_s"] / t_total,
                        decrypt_pct=100 * tm["decrypt_s"] / t_total,
                        plain_mem_mb=plain_mem / 2**20, fhe_mem_mb=fhe_mem / 2**20,
                        mem_overhead_mb=(fhe_mem - plain_mem) / 2**20,
                        plain_accuracy_pct=accuracy_pct(y_plain, truth),
                        fhe_accuracy_pct=accuracy_pct(y, truth),
                        plain_max_err=float(np.max(np.abs(y_plain - truth))),
                        fhe_max_err=err, tol=mod.TOL, passed=ok,
                        ops_total=sum(counts.values()),
                        _counts={f"{backend}_{k}": v for k, v in counts.items()}))
                    r = rows[-1]
                    print(f"{'PASS' if ok else 'FAIL'} {tag}: "
                          f"plain {t_plain*1e3:.3f} ms, fhe {tm['fhe_time_s']:.3f} "
                          f"± {r['fhe_time_std_s']:.3f} s, "
                          f"slowdown {r['slowdown']:.0f}x, "
                          f"acc plain/fhe {r['plain_accuracy_pct']:.4f}/"
                          f"{r['fhe_accuracy_pct']:.4f}%, err {err:.2e} (tol {mod.TOL}), "
                          f"mem +{r['mem_overhead_mb']:.1f} MB")
                    print("     ops: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if rows:
            rows.sort(key=lambda r: args.backends.index(r["backend"]))   # group by backend
            names = sorted({k for r in rows for k in r["_counts"]})
            for r in rows:
                c = r.pop("_counts")
                for k in names:
                    r[f"op_{k}"] = c.get(k, 0)
            os.makedirs(os.path.join(ROOT, "Results"), exist_ok=True)
            out = os.path.join(ROOT, "Results", f"{kname}_results.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {out}")
    for d in _kept_dirs:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{total - failures}/{total} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
