#!/bin/bash -l
set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$_THIS/../.." && pwd)"
fi
PUMP_DIR="$PROJECT_ROOT/src/pump_multi_comparison"
DOT_DIR="$PROJECT_ROOT/src/dot_response_fit"

cd "$PROJECT_ROOT"

module purge
module load common/python/3.13.2
module load common/compilers/gcc/13.2.0

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${DOT_DIR}:${PYTHONPATH:-}"

python -c "import h5py;      print(f'h5py OK      -- HDF5 {h5py.version.hdf5_version}')"
python -c "import matplotlib; print(f'matplotlib OK -- {matplotlib.__version__}')"
echo ""

cd "$PUMP_DIR"
exec "$@"
