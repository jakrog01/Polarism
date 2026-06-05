#!/bin/bash -l
set -euo pipefail

_pick_scratch_root() {
    local job_id="" candidate=""
    for job_id in "${SLURM_JOBID:-}" "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}"; do
        [[ -z "$job_id" ]] && continue
        candidate="/scratch/${job_id}"
        [[ -d "$candidate" ]] && { printf "%s\n" "$candidate"; return 0; }
    done
    for candidate in "${SLURM_TMPDIR:-}" "${TMPDIR:-}"; do
        if [[ -n "$candidate" && -d "$candidate" && "$(realpath "$candidate")" != "/tmp" ]]; then
            printf "%s\n" "$candidate"; return 0
        fi
    done
    return 1
}

if [[ "${ICM_RYSY_NVME:-0}" == "1" ]]; then
    if ! _SCRATCH_ROOT="$(_pick_scratch_root)"; then
        echo "ERROR: could not determine NVMe scratch root." >&2
        exit 1
    fi
    export SCRATCH="$_SCRATCH_ROOT"
    export POLARITON_SCRATCH_ID="$(basename "$(realpath "$_SCRATCH_ROOT")")"
fi

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$_THIS/../../.." && pwd)"
fi
MNIST_SPACETIME_DIR="$PROJECT_ROOT/src/mnist_spacetime_v1"
MNIST_COMMON_DIR="$PROJECT_ROOT/src/mnist_common"

cd "$PROJECT_ROOT"

_load_first_module() {
    local candidate
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if module load "$candidate" >/dev/null 2>&1; then
            echo "Loaded module: $candidate"
            return 0
        fi
    done
    return 1
}

module purge >/dev/null 2>&1 || true
_load_first_module "${RYSY_PYTHON_MODULE:-}" common/python/3.13.2 python/3.13.2 python/3.13 python || true
_load_first_module "${RYSY_OPENSSL_MODULE:-}" common/libs/openssl/1.1.1 openssl/1.1.1 OpenSSL/1.1.1 openssl || true
_load_first_module "${RYSY_GCC_MODULE:-}" common/compilers/gcc/13.2.0 gcc/13.2.0 GCC/13.2.0 gcc || true
_load_first_module "${RYSY_CUDA_MODULE:-}" gpu/cuda/12.1 cuda/12.1 CUDA/12.1 cuda || true

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
else
    echo "ERROR: virtualenv not found in $PROJECT_ROOT" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${MNIST_SPACETIME_DIR}:${MNIST_COMMON_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" >/dev/null 2>&1 || true

python -c "import cupy; print(f'CuPy OK  -- CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
python -c "import numpy; print(f'NumPy OK -- {numpy.__version__}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
[[ -n "${SCRATCH:-}" ]] && echo "SCRATCH: ${SCRATCH}"
echo ""

if [[ -n "${SCRATCH:-}" ]]; then
    if [[ ! -d "$SCRATCH" ]]; then
        echo "ERROR: SCRATCH dir missing: $SCRATCH" >&2
        exit 1
    fi
    _probe="$SCRATCH/.scratch_probe_$$"
    if ! touch "$_probe" 2>/dev/null; then
        echo "ERROR: SCRATCH not writable: $SCRATCH" >&2
        exit 1
    fi
    rm -f "$_probe"
fi

cd "$MNIST_SPACETIME_DIR"
exec "$@"
