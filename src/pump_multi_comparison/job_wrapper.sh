#!/bin/bash -l
# ---------------------------------------------------------------------------
# Generic Slurm job wrapper for the multi-pump comparison pipeline.
#
# Handles: module loading, venv activation, PYTHONPATH, GPU validation.
# Usage:  sbatch [slurm flags] job_wrapper.sh <python command ...>
#
# The orchestrator passes the concrete python command as arguments, e.g.:
#   sbatch ... job_wrapper.sh python threshold_search.py --config ...
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

module purge
module load common/python/3.13.2
module load common/compilers/gcc/13.2.0
module load gpu/cuda/12.1

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -c "import cupy; print(f'CuPy OK  -- CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
python -c "import h5py;  print(f'h5py OK  -- HDF5 {h5py.version.hdf5_version}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd "$PROJECT_ROOT/src/pump_multi_comparison"
exec "$@"
