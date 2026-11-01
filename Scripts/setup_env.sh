#!/usr/bin/env bash
# One-time setup: installs HEIR (prebuilt wheel), builds the OpenFHE C++ library it
# generates code against, and prefetches Lattigo (Go), all into ./env. Works on glibc 2.28 hosts.
# Usage: bash Scripts/setup_env.sh      then: source env/activate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVDIR="$ROOT/env"
SRC="$ENVDIR/src"
PREFIX="$ENVDIR/openfhe"
VENV="$ENVDIR/venv"
OPENFHE_TAG="${OPENFHE_TAG:-v1.6.1}"
PYTHON="${PYTHON:-/usr/bin/python3.12}"
JOBS="${JOBS:-8}"

# Newer compiler (OpenFHE needs C++17); still links against the system glibc.
if [ -f /opt/rh/gcc-toolset-15/enable ]; then
    source /opt/rh/gcc-toolset-15/enable
fi
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

cat > "$ENVDIR/activate.sh" <<EOF
source "$VENV/bin/activate"
source /opt/rh/gcc-toolset-15/enable
export OPENFHE_PREFIX="$PREFIX"
export LD_LIBRARY_PATH="$PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export OPENFHE_INCLUDE_DIR="$PREFIX/include/openfhe:$PREFIX/include/openfhe/pke:$PREFIX/include/openfhe/core:$PREFIX/include/openfhe/binfhe"
export OPENFHE_LIB_DIR="$PREFIX/lib"
export OPENFHE_LINK_LIBS="OPENFHEpke:OPENFHEbinfhe:OPENFHEcore"
export CPATH="/package/python3/3.12.3/include/python3.12:\${CPATH:-}"   # Python.h if python3.12-devel is missing
export CCACHE_DISABLE=1
EOF
echo "done. next: source env/activate.sh"
