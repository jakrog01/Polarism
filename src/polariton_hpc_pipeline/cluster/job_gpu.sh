#!/bin/bash -l
set -euo pipefail

if [[ "${POLARITON_ARRAY_DUMMY_MAX:-0}" == "1" \
      && -n "${SLURM_ARRAY_TASK_ID:-}" \
      && ( ( -n "${SLURM_ARRAY_TASK_MAX:-}" && "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MAX}" ) \
           || ( -n "${POLARITON_ARRAY_DUMMY_INDEX:-}" && "${SLURM_ARRAY_TASK_ID}" == "${POLARITON_ARRAY_DUMMY_INDEX}" ) ) ]]; then
    echo "Dummy Slurm array placeholder task ${SLURM_ARRAY_TASK_ID}; exiting before NVMe scratch setup."
    exit 0
fi

_pick_scratch_root() {
    local job_id=""
    local candidate=""

    for job_id in "${SLURM_JOBID:-}" "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}"; do
        [[ -z "$job_id" ]] && continue
        candidate="/scratch/${job_id}"
        if [[ -d "$candidate" ]]; then
            printf "%s\n" "$candidate"
            return 0
        fi
    done

    for candidate in "${SLURM_TMPDIR:-}" "${TMPDIR:-}"; do
        if [[ -n "$candidate" && -d "$candidate" && "$(realpath "$candidate")" != "/tmp" ]]; then
            local base
            base="$(basename "$(realpath "$candidate")")"
            if [[ "$candidate" == /scratch/* ]]; then
                local matches_job_id=0
                for job_id in "${SLURM_JOBID:-}" "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}"; do
                    if [[ -n "$job_id" && "$base" == "$job_id" ]]; then
                        matches_job_id=1
                        break
                    fi
                done
                if [[ "$matches_job_id" -ne 1 ]]; then
                    continue
                fi
            fi
            printf "%s\n" "$candidate"
            return 0
        fi
    done

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
    export POLARITON_SCRATCH_ID="$(basename "$(realpath "$_SCRATCH_ROOT")")"
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
HPC_PIPELINE_DIR="$PROJECT_ROOT/src/polariton_hpc_pipeline"
DOT_DIR="$PROJECT_ROOT/src/dot_response_fit"

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
    echo "ERROR: project virtualenv not found in $PROJECT_ROOT" >&2
    echo "Expected .venv/bin/activate or venv/bin/activate." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${HPC_PIPELINE_DIR}:${DOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" >/dev/null 2>&1 || true

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

cd "$HPC_PIPELINE_DIR"
exec "$@"
