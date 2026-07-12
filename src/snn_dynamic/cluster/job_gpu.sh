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
            if [[ "$candidate" == /scratch/* ]]; then
                local ok=0
                for job_id in "${SLURM_JOBID:-}" "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}"; do
                    [[ -n "$job_id" && "$(basename "$(realpath "$candidate")")" == "$job_id" ]] && { ok=1; break; }
                done
                [[ "$ok" -ne 1 ]] && continue
            fi
            printf "%s\n" "$candidate"; return 0
        fi
    done
    return 1
}

if [[ "${ICM_RYSY_NVME:-0}" == "1" ]]; then
    if ! _SCRATCH_ROOT="$(_pick_scratch_root)"; then
        echo "ERROR: could not determine NVMe scratch root." >&2; exit 1
    fi
    export SCRATCH="$_SCRATCH_ROOT"
    export POLARITON_SCRATCH_ID="$(basename "$(realpath "$_SCRATCH_ROOT")")"
fi

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "$_THIS/../../.." && pwd -P)"
fi

if ! cd "$PROJECT_ROOT" 2>/dev/null; then
    echo "ERROR: PROJECT_ROOT not accessible from compute node: $PROJECT_ROOT" >&2
    echo "  Retrying via physical path (readlink -f) ..." >&2
    _PROJECT_ROOT_PHYS="$(readlink -f "$PROJECT_ROOT" 2>/dev/null || true)"
    if [[ -n "$_PROJECT_ROOT_PHYS" && -d "$_PROJECT_ROOT_PHYS" ]]; then
        PROJECT_ROOT="$_PROJECT_ROOT_PHYS"
        cd "$PROJECT_ROOT"
    else
        echo "  readlink -f produced: '${_PROJECT_ROOT_PHYS:-<empty>}'" >&2
        exit 1
    fi
fi
PROJECT_ROOT="$(pwd -P)"
SNN_DYNAMIC_DIR="$PROJECT_ROOT/src/snn_dynamic"
MNIST_COMMON_DIR="$PROJECT_ROOT/src/mnist_common"

_load_first_module() {
    local candidate
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if module load "$candidate" >/dev/null 2>&1; then
            echo "Loaded module: $candidate"; return 0
        fi
    done
    return 1
}

module purge >/dev/null 2>&1 || true
_load_first_module "${RYSY_PYTHON_MODULE:-}"  common/python/3.13.2 python/3.13.2 python/3.13 python || true
_load_first_module "${RYSY_OPENSSL_MODULE:-}" common/libs/openssl/1.1.1 openssl/1.1.1 OpenSSL/1.1.1 openssl || true
_load_first_module "${RYSY_GCC_MODULE:-}"     common/compilers/gcc/13.2.0 gcc/13.2.0 GCC/13.2.0 gcc || true
_load_first_module "${RYSY_CUDA_MODULE:-}"    gpu/cuda/12.1 cuda/12.1 CUDA/12.1 cuda || true

_VENV_ACTIVATE=""
for _cand in \
    "$PROJECT_ROOT/.venv/bin/activate" \
    "$PROJECT_ROOT/venv/bin/activate" \
    "${SNN_DYNAMIC_VENV:-}" \
    "$HOME/polaritonSNN/PolaritonSNN/.venv/bin/activate" \
    "$HOME/.venv/bin/activate"; do
    [[ -z "$_cand" ]] && continue
    if [[ -f "$_cand" ]]; then _VENV_ACTIVATE="$_cand"; break; fi
done

if [[ -z "$_VENV_ACTIVATE" ]]; then
    echo "ERROR: virtualenv not found." >&2
    echo "  PROJECT_ROOT     : $PROJECT_ROOT" >&2
    echo "  pwd -P           : $(pwd -P)" >&2
    echo "  hostname         : $(hostname 2>/dev/null || echo '?')" >&2
    echo "  Looked for:" >&2
    echo "    $PROJECT_ROOT/.venv/bin/activate" >&2
    echo "    $PROJECT_ROOT/venv/bin/activate" >&2
    echo "    \$SNN_DYNAMIC_VENV = ${SNN_DYNAMIC_VENV:-<unset>}" >&2
    echo "    $HOME/polaritonSNN/PolaritonSNN/.venv/bin/activate" >&2
    echo "    $HOME/.venv/bin/activate" >&2
    echo "  Directory listing of $PROJECT_ROOT:" >&2
    ls -la "$PROJECT_ROOT" 2>&1 | head -30 >&2 || true
    if [[ -d "$PROJECT_ROOT/.venv" ]]; then
        echo "  Directory listing of $PROJECT_ROOT/.venv:" >&2
        ls -la "$PROJECT_ROOT/.venv" 2>&1 | head -20 >&2 || true
    fi
    exit 1
fi

echo "Activating venv: $_VENV_ACTIVATE"
source "$_VENV_ACTIVATE"

export PYTHONPATH="${PROJECT_ROOT}:${SNN_DYNAMIC_DIR}:${MNIST_COMMON_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" >/dev/null 2>&1 || true

python -c "import cupy; print(f'CuPy OK  -- CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
python -c "import numpy; print(f'NumPy OK -- {numpy.__version__}')"
python -c "import sklearn; print(f'sklearn OK -- {sklearn.__version__}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
[[ -n "${SCRATCH:-}" ]] && echo "SCRATCH: ${SCRATCH}"
echo ""

if [[ -n "${SCRATCH:-}" ]]; then
    if [[ ! -d "$SCRATCH" ]]; then
        echo "ERROR: SCRATCH dir missing: $SCRATCH" >&2; exit 1
    fi
    _probe="$SCRATCH/.scratch_probe_$$"
    if ! touch "$_probe" 2>/dev/null; then
        echo "ERROR: SCRATCH not writable: $SCRATCH" >&2; exit 1
    fi
    rm -f "$_probe"
fi

cd "$SNN_DYNAMIC_DIR"
exec "$@"
