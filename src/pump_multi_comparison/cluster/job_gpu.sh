#!/bin/bash -l
# =============================================================================
# cluster/job_gpu.sh — GPU job wrapper for Rysy at ICM UW.
#
# Loads CUDA modules, activates the Python environment, validates that a GPU
# is present, sets PYTHONPATH, and executes the command passed by sbatch.
#
# Resource model: always requests --gres=gpu:N.  Used by:
#   - threshold_search  (Job 1)
#   - scenario array    (Job 2)
#
# Usage (called by submit.sh via sbatch):
#   sbatch [slurm flags] cluster/job_gpu.sh python -m pipeline.stages.gpu.threshold_search ...
# =============================================================================
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$_THIS/../.." && pwd)"
fi
PUMP_DIR="$PROJECT_ROOT/src/pump_multi_comparison"

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

# Both PROJECT_ROOT (for polarism) and PUMP_DIR (for pipeline package) must be importable.
export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${PYTHONPATH:-}"

# Validate environment.
python -c "import cupy; print(f'CuPy OK  -- CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
python -c "import h5py;  print(f'h5py OK  -- HDF5 {h5py.version.hdf5_version}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd "$PUMP_DIR"
exec "$@"
