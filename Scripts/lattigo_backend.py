"""
Lattigo (Go) backend for run_kernel_test.py.

HEIR lowers the same MLIR to Go code against Lattigo (heir-opt --scheme-to-lattigo,
heir-translate --emit-lattigo). We wrap that code in a small Go driver, build it, and run it
as a subprocess that does setup / encrypt / eval / decrypt and reports its own timings
and peak RSS. Needs `go` and the module cache that Scripts/setup_env.sh prefetches into
env/gopath (the runner builds offline).

Op counts are static counts of evaluator calls in the generated Go: that code is
straight-line, so static equals executed.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVDIR = os.path.join(ROOT, "env")
GOENV = {"GOPATH": os.path.join(ENVDIR, "gopath"), "GOCACHE": os.path.join(ENVDIR, "gocache"),
         "GOFLAGS": "-mod=mod", "GOPROXY": "off", "GOTOOLCHAIN": "local",
         "CGO_ENABLED": "0"}
TEMPLATE = os.path.join(ENVDIR, "golattigo")      # go.mod / go.sum made by setup_env.sh
HEIR_DIR = os.path.join(ENVDIR, "venv", "lib", "python3.12", "site-packages", "heir")

GO_DRIVER = r"""package main

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"
)

func peakRSS() int64 {
	b, _ := os.ReadFile("/proc/self/status")
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "VmHWM:") {
			f := strings.Fields(l)
			kb, _ := strconv.ParseInt(f[1], 10, 64)
			return kb * 1024
		}
	}
	return 0
}

func main() {
	var in struct{ X []float32 }
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		panic(err)
	}
	t0 := time.Now()
	ev, param, enc, encryptor, decryptor := Kernel__configure()
	setup := time.Since(t0).Seconds()

	t0 = time.Now()
	ct := Kernel__encrypt__arg0(ev, param, enc, encryptor, in.X)
	tEnc := time.Since(t0).Seconds()

	t0 = time.Now()
	out := Kernel(ev, param, enc, ct)
	tEval := time.Since(t0).Seconds()

	t0 = time.Now()
	y := Kernel__decrypt__result0(ev, param, enc, decryptor, out)
	tDec := time.Since(t0).Seconds()

	json.NewEncoder(os.Stdout).Encode(map[string]any{
		"y": y, "setup_s": setup, "encrypt_s": tEnc, "fhe_time_s": tEval,
		"decrypt_s": tDec, "peak_rss": peakRSS()})
}
"""


def available():
    """Reason string if the backend cannot run here, else None."""
    if not shutil.which("go"):
        return "go not found"
    if not os.path.exists(os.path.join(TEMPLATE, "go.sum")):
        return "Lattigo not prefetched (run Scripts/setup_env.sh)"
    return None


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def _count_ops(go_src):
    body = go_src.split("func Kernel__preprocessing", 1)[1].split("func Kernel(", 1)[0]
    counts = {}
    for name in re.findall(r"\bevaluator\.(\w+)\(", body):
        counts[name] = counts.get(name, 0) + 1
    enc = len(re.findall(r"\bencoder\.Encode\(", body))
    if enc:
        counts["Encode"] = enc
    return counts


def run_lattigo(mlir_str, x):
    """Same contract as run_fhe: (y, timings, op_counts, peak_rss_bytes)."""
    mlir_str = re.sub(r"func\.func @\w+", "func.func @kernel", mlir_str, count=1)
    work = tempfile.mkdtemp(prefix="lattigo_")
    try:
        t0 = time.perf_counter()
        with open(os.path.join(work, "k.mlir"), "w") as fh:
            fh.write(mlir_str)
        opt, translate = (os.path.join(HEIR_DIR, b) for b in ("heir-opt", "heir-translate"))
        ckks = _run([opt, "--canonicalize", "--mlir-to-ckks", os.path.join(work, "k.mlir")])
        lat = _run([opt, "--scheme-to-lattigo"], input=ckks)
        go_src = _run([translate, "--emit-lattigo"], input=lat)
        with open(os.path.join(work, "kernel.go"), "w") as fh:
            fh.write(go_src)
        with open(os.path.join(work, "main.go"), "w") as fh:
            fh.write(GO_DRIVER)
        for f in ("go.mod", "go.sum"):
            shutil.copy(os.path.join(TEMPLATE, f), work)
        env = {**os.environ, **GOENV}
        _run(["go", "build", "-o", "kernel_bin", "."], cwd=work, env=env)
        t_compile = time.perf_counter() - t0

        out = _run([os.path.join(work, "kernel_bin")], cwd=work, env=env,
                   input=json.dumps({"X": [float(v) for v in x]}))
        res = json.loads(out)
        y = np.array(res["y"], dtype=float).ravel()[:len(x)]
        tm = dict(fhe_time_s=res["fhe_time_s"], compile_s=t_compile, setup_s=res["setup_s"],
                  encrypt_s=res["encrypt_s"], decrypt_s=res["decrypt_s"])
        return y, tm, _count_ops(go_src), int(res["peak_rss"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
