#!/bin/bash -l
set -euo pipefail

_pick_scratch_root() {
    local job_id="${SLURM_JOB_ID:-}"
    local candidate=""

    if [[ -n "$job_id" ]]; then
        candidate="/scratch/${job_id}"
        if [[ -d "$candidate" ]]; then
            printf "%s\n" "$candidate"
            return 0
        fi
    fi

    for candidate in "${SLURM_TMPDIR:-}" "${TMPDIR:-}"; do
        if [[ -n "$candidate" && -d "$candidate" && "$(realpath "$candidate")" != "/tmp" ]]; then
            printf "%s\n" "$candidate"
            return 0
        fi
    done

    mapfile -t _user_scratch_dirs < <(
        find /scratch -mindepth 1 -maxdepth 1 -type d -user "$USER" 2>/dev/null | sort
    )
    if [[ ${#_user_scratch_dirs[@]} -eq 1 ]]; then
        printf "%s\n" "${_user_scratch_dirs[0]}"
        return 0
    fi

    return 1
}

if [[ "${ICM_RYSY_NVME:-0}" == "1" ]]; then
    if ! _SCRATCH_ROOT="$(_pick_scratch_root)"; then
        echo "ERROR: could not determine a usable NVMe scratch root." >&2
        echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}" >&2
        echo "  SLURM_JOBID=${SLURM_JOBID:-<unset>}" >&2
        echo "  SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>}" >&2
        echo "  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}" >&2
        echo "  SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}" >&2
        echo "  TMPDIR=${TMPDIR:-<unset>}" >&2
        echo "  /scratch contents:" >&2
        ls -la /scratch/ 2>&1 | head -20 >&2 || true
        exit 1
    fi
    export SCRATCH="$_SCRATCH_ROOT"
fi

_LOCAL_FFMPEG="$HOME/tools/ffmpeg/8.1/bin/ffmpeg"
if [[ -z "${FFMPEG_BIN:-}" && -x "$_LOCAL_FFMPEG" ]]; then
    export FFMPEG_BIN="$_LOCAL_FFMPEG"
fi

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
module load gpu/cuda/12.1

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${DOT_DIR}:${PYTHONPATH:-}"

python -c "import cupy; print(f'CuPy OK  -- CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
python -c "import h5py;  print(f'h5py OK  -- HDF5 {h5py.version.hdf5_version}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
[[ -n "${SCRATCH:-}" ]] && echo "SCRATCH: ${SCRATCH}"
[[ -n "${FFMPEG_BIN:-}" ]] && echo "FFMPEG_BIN: ${FFMPEG_BIN}"
[[ -n "${RENDER_ENCODER:-}" ]] && echo "RENDER_ENCODER: ${RENDER_ENCODER}"
echo ""

if [[ -n "${SCRATCH:-}" ]]; then
    if [[ ! -d "$SCRATCH" ]]; then
        echo "ERROR: SCRATCH dir missing at job start (NVMe mount may have hidden it): $SCRATCH" >&2
        ls -la /scratch/ 2>&1 | head -10 >&2 || true
        exit 1
    fi
    _probe="$SCRATCH/.scratch_probe_$$"
    if ! touch "$_probe" 2>/dev/null; then
        echo "ERROR: SCRATCH dir is not writable at job start: $SCRATCH" >&2
        df -h "$SCRATCH" 2>&1 >&2 || true
        exit 1
    fi
    rm -f "$_probe"
fi

cd "$PUMP_DIR"
exec "$@"
