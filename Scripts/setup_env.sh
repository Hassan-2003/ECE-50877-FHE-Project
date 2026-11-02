#!/usr/bin/env bash
# One-time setup: installs HEIR (prebuilt wheel), builds the OpenFHE C++ library it
# generates code against, prefetches Lattigo (Go) and, where CUDA exists, builds FIDESlib
# (GPU), all into ./env. Works on glibc 2.28 hosts. Run it on a node with internet.
# Usage: bash Scripts/setup_env.sh      then: source env/activate.sh
# Optional: CUDA_PATH=<cuda root>, FIDESLIB_ARCH="80-real" (GPUs to build for), SKIP_FIDESLIB=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVDIR="$ROOT/env"
SRC="$ENVDIR/src"
PREFIX="$ENVDIR/openfhe"
VENV="$ENVDIR/venv"
OPENFHE_TAG="${OPENFHE_TAG:-v1.6.1}"
# Python 3.12: the system one if present, else Purdue Scholar's anaconda/2024.10-py312
# interpreter. Only used to create the venv; the anaconda module is not loaded, so its
# CPATH / CMAKE_PREFIX_PATH do not leak conda libraries into the C++ builds.
if [ -x /usr/bin/python3.12 ]; then DEFPY=/usr/bin/python3.12
else DEFPY=/apps/external/anaconda/2024.10/bin/python3.12; fi
PYTHON="${PYTHON:-$DEFPY}"
JOBS="${JOBS:-8}"

# Newer compiler (OpenFHE needs C++17); still links against the system glibc.
# COMPILER_SETUP is replayed by env/activate.sh. On Purdue Scholar the default Lmod
# gcc/14.1.0 module puts its own lib64 on LIBRARY_PATH, so the toolset would link against
# that newer libstdc++.so (GLIBCXX_3.4.32), which the conda python3.12 cannot load; unload it.
COMPILER_SETUP=":"
if [ -f /opt/rh/gcc-toolset-15/enable ]; then
    COMPILER_SETUP="source /opt/rh/gcc-toolset-15/enable"
fi
if type module >/dev/null 2>&1; then
    COMPILER_SETUP="module unload openmpi gcc; $COMPILER_SETUP"
fi
eval "$COMPILER_SETUP"
export CCACHE_DISABLE=1   # /usr/lib64/ccache wrappers need a writable ~/.ccache
echo "compiler:$(g++ --version | head -1)"

# Python venv: test deps + HEIR compiler (python + openfhe extras)
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt" "heir-py[python,openfhe]"

# OpenFHE C++ library (headers + libs for HEIR-generated code)
mkdir -p "$SRC"
[ -d "$SRC/openfhe-development" ] || git clone --depth 1 --branch "$OPENFHE_TAG" \
    https://github.com/openfheorg/openfhe-development.git "$SRC/openfhe-development"
cmake -S "$SRC/openfhe-development" -B "$SRC/build-openfhe" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build "$SRC/build-openfhe" -j"$JOBS"
cmake --install "$SRC/build-openfhe"

# Lattigo (Go) backend: prefetch the module into env/ so the runner can build offline
if command -v go >/dev/null; then
    GOLAT="$ENVDIR/golattigo"
    mkdir -p "$GOLAT"
    cat > "$GOLAT/main.go" <<'GOEOF'
package main

import (
	_ "github.com/tuneinsight/lattigo/v6/core/rlwe"
	_ "github.com/tuneinsight/lattigo/v6/ring"
	_ "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {}
GOEOF
    (cd "$GOLAT" && export GOPATH="$ENVDIR/gopath" GOCACHE="$ENVDIR/gocache" \
        GOFLAGS=-mod=mod GOTOOLCHAIN=local CGO_ENABLED=0 &&
        { [ -f go.mod ] || go mod init kernel; } &&
        go get github.com/tuneinsight/lattigo/v6@v6.2.0 && go mod tidy && go build -o /dev/null .)
else
    echo "go not found: skipping the Lattigo backend (OpenFHE only)"
fi

# FIDESlib (CUDA GPU backend). It installs its own patched OpenFHE 1.6.1 (static) under
# env/fideslib/openfhe, separate from env/openfhe above. CUDA: $CUDA_PATH, else nvcc on
# PATH, else Purdue Scholar's /apps/rocky9/cuda/12.6. FIDESLIB_ARCH defaults to the
# V100 (70), A30/A100 (80) and A40 (86); "auto" builds for the local GPU only.
FIDES="$ENVDIR/fideslib"
CUDA_PATH="${CUDA_PATH:-}"
if [ -z "$CUDA_PATH" ] && command -v nvcc >/dev/null; then
    CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
elif [ -z "$CUDA_PATH" ] && [ -x /apps/rocky9/cuda/12.6/bin/nvcc ]; then
    CUDA_PATH=/apps/rocky9/cuda/12.6
fi
if [ "${SKIP_FIDESLIB:-0}" = 1 ] || [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
    echo "no CUDA (or SKIP_FIDESLIB=1): skipping the FIDESlib backend"
    CUDA_PATH=""
elif [ -f "$FIDES/share/fideslib/cmake/fideslibConfig.cmake" ]; then
    echo "FIDESlib already installed in $FIDES (delete it to rebuild)"
else
    [ -d "$SRC/FIDESlib" ] || git clone https://github.com/CAPS-UMU/FIDESlib "$SRC/FIDESlib"
    # FIDESlib builds with whatever `g++` is on PATH, and CUDA 12.x rejects gcc-toolset-15
    # (gcc > 13), so drop the toolset and use the system gcc. Its OpenFHE installer is found
    # relative to the working directory, so configure from inside build/.
    (PATH="$CUDA_PATH/bin:$(tr : '\n' <<<"$PATH" | grep -v gcc-toolset | paste -sd:)"
     export JOBS
     echo "FIDESlib compiler:$(g++ --version | head -1), CUDA: $CUDA_PATH"
     mkdir -p "$SRC/FIDESlib/build" && cd "$SRC/FIDESlib/build" &&
     cmake .. -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_INSTALL_OPENFHE=ON -DCUDA_PATH="$CUDA_PATH" \
         -DFIDESLIB_INSTALL_PREFIX="$FIDES" -DOPENFHE_INSTALL_PREFIX="$FIDES/openfhe" \
         "-DFIDESLIB_ARCH=${FIDESLIB_ARCH:-70-real;80-real;86-real}" \
         -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF &&
     cmake --build . -j"$JOBS" && cmake --build . --target install -j"$JOBS")
fi

PYINC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
cat > "$ENVDIR/activate.sh" <<EOF
source "$VENV/bin/activate"
$COMPILER_SETUP
export OPENFHE_PREFIX="$PREFIX"
export LD_LIBRARY_PATH="$PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export OPENFHE_INCLUDE_DIR="$PREFIX/include/openfhe:$PREFIX/include/openfhe/pke:$PREFIX/include/openfhe/core:$PREFIX/include/openfhe/binfhe"
export OPENFHE_LIB_DIR="$PREFIX/lib"
export OPENFHE_LINK_LIBS="OPENFHEpke:OPENFHEbinfhe:OPENFHEcore"
export CPATH="$PYINC:\${CPATH:-}"   # Python.h if python3.12-devel is missing
export CCACHE_DISABLE=1
EOF
[ -z "$CUDA_PATH" ] || echo "export PATH=\"$CUDA_PATH/bin:\$PATH\"   # nvcc for the FIDESlib backend" \
    >> "$ENVDIR/activate.sh"
echo "done. next: source env/activate.sh"
