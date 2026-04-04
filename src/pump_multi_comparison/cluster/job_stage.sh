#!/bin/bash -l
# =============================================================================
# cluster/job_stage.sh — Non-GPU stage wrapper for Rysy at ICM UW.
#
# PURPOSE AND NAMING
# ------------------
# This wrapper is called "job_stage.sh" rather than "job_cpu.sh" to avoid
# implying that these jobs run on dedicated CPU-only hardware.
#
# RESOURCE MODEL (Rysy-specific)
# ------------------------------
# Rysy (ICM UW) is documented as a GPU cluster.  Its public Slurm examples
# always use --gres=gpu:...  and there is no confirmed dedicated CPU
# partition.  Non-GPU stages (visualization, finalize) that use this wrapper
# may therefore still run on GPU-equipped nodes — they just do not request
# a GPU allocation.
#
# submit.sh handles partition selection:
#   - If CPU_PARTITION is set in slurm.env and differs from SLURM_PARTITION,
#     stage jobs target that partition.
#   - Otherwise they fall back to SLURM_PARTITION (the GPU partition) with a
#     clear WARNING during submission.
#
# WHAT THIS WRAPPER DOES
# ----------------------
# Loads Python (no CUDA), activates the venv, sets PYTHONPATH.
# Does NOT load gpu/cuda modules — CUDA drivers are not needed for
# visualization or finalize stages.
#
# Usage (called by submit.sh via sbatch):
#   sbatch [slurm flags] cluster/job_stage.sh python -m pipeline.stages.cpu.visualize ...
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
# Note: gpu/cuda/12.1 is intentionally NOT loaded here.

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${PYTHONPATH:-}"

python -c "import h5py;      print(f'h5py OK      -- HDF5 {h5py.version.hdf5_version}')"
python -c "import matplotlib; print(f'matplotlib OK -- {matplotlib.__version__}')"
echo ""

cd "$PUMP_DIR"
exec "$@"
