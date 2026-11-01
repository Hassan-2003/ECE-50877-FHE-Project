"""
FIDESlib (CUDA GPU CKKS) backend for run_kernel_test.py.

MLIR -> (heir-opt --mlir-to-ckks, --scheme-to-openfhe; heir-translate --emit-openfhe-pke)
-> fideslib_emitter -> CMake build against an installed FIDESlib -> run the binary, which
reports its own timings, peak host RSS and GPU memory.

Needs an NVIDIA GPU, nvcc, cmake, and FIDESlib installed (see FIDESLIB_PREFIX below). Without
them available() says why and the runner skips this backend; `--emit-fideslib DIR` still
writes the generated sources.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

import fideslib_emitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEIR_DIR = os.path.join(ROOT, "env", "venv", "lib", "python3.12", "site-packages", "heir")
# Install prefix of FIDESlib (contains lib/cmake/fideslib). Override with $FIDESLIB_PREFIX.
PREFIX = os.environ.get("FIDESLIB_PREFIX", os.path.join(ROOT, "env", "fideslib"))

CMAKE = """cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CXX_STANDARD 20)
project(kernel LANGUAGES CXX)
find_package(fideslib REQUIRED CONFIG)
set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
add_executable(kernel kernel.cpp)
target_include_directories(kernel PRIVATE ${FIDESLIB_INCLUDE_PATH})
target_link_directories(kernel PRIVATE ${FIDESLIB_LIBRARIES_PATH})
target_link_libraries(kernel PRIVATE ${FIDESLIB_LIBRARIES})
"""


def available():
    """Reason string if the backend cannot run here, else None."""
    for tool in ("nvcc", "cmake"):
        if not shutil.which(tool):
            return f"{tool} not found"
    if subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode:
        return "no NVIDIA GPU visible"
    if not os.path.isdir(PREFIX):
        return f"FIDESlib not installed at {PREFIX} (set FIDESLIB_PREFIX)"
    return None


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed:\n{(r.stdout + r.stderr)[-2500:]}")
    return r.stdout


def emit_source(mlir_str):
    """MLIR string -> complete FIDESlib C++ program text (needs no GPU)."""
    mlir_str = re.sub(r"func\.func @\w+", "func.func @kernel", mlir_str, count=1)
    opt, translate = (os.path.join(HEIR_DIR, b) for b in ("heir-opt", "heir-translate"))
    ckks = _run([opt, "--canonicalize", "--mlir-to-ckks"], input=mlir_str)
    ofhe = _run([opt, "--scheme-to-openfhe=entry-function=kernel"], input=ckks)
    cpp = _run([translate, "--emit-openfhe-pke"], input=ofhe)
    return fideslib_emitter.emit_fideslib(cpp, "kernel")


def run_fideslib(mlir_str, x):
    """Same contract as run_fhe: (y, timings, op_counts, mem_bytes)."""
    work = tempfile.mkdtemp(prefix="fideslib_")
    try:
        t0 = time.perf_counter()
        src = emit_source(mlir_str)
        open(os.path.join(work, "kernel.cpp"), "w").write(src)
        open(os.path.join(work, "CMakeLists.txt"), "w").write(CMAKE)
        _run(["cmake", "-S", work, "-B", os.path.join(work, "build"),
              f"-DCMAKE_PREFIX_PATH={PREFIX}", "-DCMAKE_BUILD_TYPE=Release"])
        _run(["cmake", "--build", os.path.join(work, "build"), "-j"])
        t_compile = time.perf_counter() - t0

        inp = os.path.join(work, "in.txt")
        with open(inp, "w") as fh:
            fh.write(f"{len(x)}\n" + " ".join(repr(float(v)) for v in x) + "\n")
        out = _run([os.path.join(work, "build", "kernel"), inp])
        res = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        y = np.array([float(v) for v in res["y"].split()], dtype=float)[:len(x)]
        tm = dict(fhe_time_s=float(res["fhe_time_s"]), compile_s=t_compile,
                  setup_s=float(res["setup_s"]), encrypt_s=float(res["encrypt_s"]),
                  decrypt_s=float(res["decrypt_s"]))
        counts = _count_ops(src)
        return y, tm, counts, int(res["peak_rss"]) + int(res["gpu_bytes"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _count_ops(src):
    """Static counts of cc-> calls in the generated kernel body (loops unrolled by HEIR's
    omp-for over rotations are counted per source occurrence, so rotation counts are lower
    bounds for kernels that use them)."""
    body = src.split("__preprocessing(", 1)[1].split("__encrypt__arg0(", 1)[0]
    counts = {}
    for name in re.findall(r"\bcc->(\w+)\(", body):
        if name != "GetRingDimension":
            counts[name] = counts.get(name, 0) + 1
    return counts
