#!/usr/bin/env bash
# Full sweep: every case x SIZES of each kernel on openfhe, lattigo and fideslib, on one GPU
# node, writing Results/<kernel>_results.csv. Refuses to start if any backend would be skipped.
#
#   sbatch Scripts/run_sweep.sh                 # fir and xcorr (~10 min on an A30)
#   sbatch Scripts/run_sweep.sh fir             # one kernel
#   sbatch Scripts/run_sweep.sh fir --sizes 128 # extra args go to run_kernel_test.py
#   bash Scripts/run_sweep.sh fir               # inside an existing GPU allocation (salloc/srun)
#
# The #SBATCH lines are Purdue Scholar defaults; override on the command line, e.g.
#   sbatch -A <account> -p <gpu-partition> -C <feature> Scripts/run_sweep.sh
#SBATCH -J fhe-sweep
#SBATCH -A scholar -p gpu --gres=gpu:1 -C A30
#SBATCH -N 1 -n 1 -c 8 --mem=48G -t 01:00:00
#SBATCH -o Results/sweep_%j.log
set -uo pipefail

# Under sbatch this file runs from a spool copy, so locate the repo from the submit dir.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || exit 1
[ -f env/activate.sh ] || { echo "env/ missing: run Scripts/setup_env.sh first"; exit 1; }
source env/activate.sh
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl.${SLURM_JOB_ID:-$$}"

[ $# -gt 0 ] && [[ "$1" != -* ]] || set -- fir xcorr "$@"
BACKENDS=(openfhe lattigo fideslib)

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -1

# The runner skips unavailable backends; for a full sweep that is an error.
"$ROOT/env/venv/bin/python" - <<'EOF' || exit 1
import sys
sys.path.insert(0, "Scripts")
import fideslib_backend, lattigo_backend
bad = {b: f() for b, f in [("lattigo", lattigo_backend.available),
                           ("fideslib", fideslib_backend.available)] if f()}
for b, why in bad.items():
    print(f"backend {b} unavailable: {why}")
sys.exit(1 if bad else 0)
EOF

"$ROOT/env/venv/bin/python" Scripts/run_kernel_test.py "$@" --backends "${BACKENDS[@]}"
rc=$?
echo "SWEEP_EXIT=$rc"
exit $rc
