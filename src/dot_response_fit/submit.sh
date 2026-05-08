#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --wait)       WAIT_FOR_COMPLETION=1; shift ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --config=*)   CONFIG="${1#--config=}"; shift ;;
        --runs-dir)   RUNS_BASE_DIR="$2"; shift 2 ;;
        --runs-dir=*) RUNS_BASE_DIR="${1#--runs-dir=}"; shift ;;
        *) [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOT_DIR="$SCRIPT_DIR"
PUMP_DIR="$PROJECT_ROOT/src/pump_multi_comparison"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

echo "========================================"
echo " Dot-Response Fit Pipeline"
echo " Rysy-only  |  inline render  |  run from a Rysy login node"
echo "========================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2; exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    module load common/python/3.13.2 >/dev/null 2>&1 || true
fi
for venv_path in "$PROJECT_ROOT/.venv/bin/activate" "$PROJECT_ROOT/venv/bin/activate"; do
    [[ -f "$venv_path" ]] && { source "$venv_path" 2>/dev/null || true; break; }
done
export PYTHONPATH="${PROJECT_ROOT}:${DOT_DIR}:${PUMP_DIR}:${PYTHONPATH:-}"

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* ]]; then
    echo "ERROR: submit.sh must be run on a Rysy login node." >&2
    echo "Current host: $HOSTNAME_SHORT" >&2
    exit 1
fi

echo "  Config    : $CONFIG"
echo "  slurm.env : $SLURM_ENV"
echo "  Host      : $HOSTNAME_SHORT"
echo ""

echo "  Validating config ..."
python3 -m dot_response_fit.config.validator --config "$CONFIG" --slurm-env "$SLURM_ENV" --check-files || {
    echo "Aborting: fix the validation errors above." >&2; exit 1
}
echo ""

mapfile -t _PARSED < <(python3 - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("fit", {}).get("max_runtime_minutes", 120))
mnist = cfg.get("mnist", {})
if mnist.get("sample_indices") is not None:
    n = len(mnist["sample_indices"])
    source = "sample_indices"
elif mnist.get("sample_index") is not None:
    n = 1
    source = "sample_index"
else:
    n = mnist.get("n_images", 1)
    source = "n_images"
print(n)
print(source)
for sc in cfg.get("scenarios", []):
    print(sc["name"])
PYEOF
)
FIT_MINUTES="${_PARSED[0]}"
EFFECTIVE_N_IMAGES="${_PARSED[1]}"
_N_IMAGES_SOURCE="${_PARSED[2]}"
SCENARIOS=("${_PARSED[@]:3}")
N_SCENARIOS="${#SCENARIOS[@]}"
FIT_HOURS=$((FIT_MINUTES / 60))
FIT_REMAINDER=$((FIT_MINUTES % 60))
FIT_TIME_DERIVED=$(printf "%02d:%02d:00" "$FIT_HOURS" "$FIT_REMAINDER")

if [[ "$EFFECTIVE_N_IMAGES" -lt 1 ]]; then
    echo "ERROR: effective number of images is $EFFECTIVE_N_IMAGES (from mnist.$_N_IMAGES_SOURCE)." >&2
    echo "  Check mnist.sample_indices (must be non-empty), mnist.sample_index, or mnist.n_images in config." >&2
    exit 1
fi
echo "  Images    : $EFFECTIVE_N_IMAGES  (from mnist.$_N_IMAGES_SOURCE)"
echo "  Scenarios : $N_SCENARIOS"
for sc in "${SCENARIOS[@]}"; do echo "    - $sc"; done
echo ""

set -a
source "$SLURM_ENV"
set +a

_LOCAL_FFMPEG="$HOME/tools/ffmpeg/8.1/bin/ffmpeg"
if [[ -z "${FFMPEG_BIN:-}" && -x "$_LOCAL_FFMPEG" ]]; then
    export FFMPEG_BIN="$_LOCAL_FFMPEG"
fi
if [[ -n "${FFMPEG_BIN:-}" && -z "${RENDER_ENCODER:-}" ]]; then
    if "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -q 'h264_nvenc'; then
        export RENDER_ENCODER="h264_nvenc"
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -q 'libx264rgb'; then
        export RENDER_ENCODER="libx264rgb"
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -q 'libx264'; then
        export RENDER_ENCODER="libx264"
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -qE ' ffv1 '; then
        export RENDER_ENCODER="ffv1"
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -qE ' png '; then
        export RENDER_ENCODER="png"
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -qE '^[[:space:]]*V.*[[:space:]]mpeg4[[:space:]]'; then
        export RENDER_ENCODER="mpeg4"
    fi
fi

for var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS SLURM_TIME NVME_GB \
           TETYDA_RUNS_BASE MAX_CONCURRENT_SCENARIOS \
           FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME \
           PREPARE_REF_MEM PREPARE_REF_CPUS PREPARE_REF_TIME; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: slurm.env missing required variable: $var" >&2; exit 1
    fi
done

FIT_TIME="${FIT_TIME:-$FIT_TIME_DERIVED}"
SCENARIO_TIME="${SCENARIO_TIME:-$SLURM_TIME}"

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
MAX_CONCURRENT="$MAX_CONCURRENT_SCENARIOS"

echo "  Rysy  GPU : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
echo "  Repo root : $PROJECT_ROOT"
echo "  Tetyda    : $TETYDA_RUNS_BASE"
echo "  Max concurrent GPU image jobs: $MAX_CONCURRENT"
[[ -n "${FFMPEG_BIN:-}" ]] && echo "  FFmpeg    : $FFMPEG_BIN"
[[ -n "${RENDER_ENCODER:-}" ]] && echo "  Encoder   : $RENDER_ENCODER"
echo ""

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$RUNS_BASE_DIR"
    RUN_DIR=$(python3 - "$RUNS_BASE_DIR" "$CONFIG" <<'PYEOF'
import sys, os, hashlib
from datetime import datetime
base, cfg_path = sys.argv[1], sys.argv[2]
chash = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:8]
run_dir = os.path.join(base, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{chash}")
os.makedirs(run_dir, exist_ok=False)
print(run_dir)
PYEOF
)
else
    RUN_DIR="${TETYDA_RUNS_BASE}/DRY_$(date +%Y%m%d_%H%M%S)"
    echo "  [DRY RUN] run dir would be: $RUN_DIR"
fi
RUN_NAME="$(basename "$RUN_DIR")"

if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Run dir : $RUN_DIR"

    cp "$CONFIG" "$RUN_DIR/config.yaml"

    python3 -c "
import json
with open('${RUN_DIR}/scenario_index.json', 'w') as f:
    json.dump($(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${SCENARIOS[@]}"), f, indent=2)
"
    python3 -c "
from dot_response_fit.manifest.io import init_manifest
init_manifest(
    '${RUN_DIR}',
    '${RUN_DIR}/config.yaml',
    $(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${SCENARIOS[@]}"),
)
"
    echo "  Manifest initialised."
    echo ""
fi

LOGS_DIR="$RUN_DIR/logs"
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$LOGS_DIR"
fi

_RYSY_QOS_FLAG=""
[[ -n "${SLURM_QOS:-}" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

_rysy_sbatch() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
        "$@"
}

_rysy_sbatch_nvme() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1" \
        "$@"
}

_dependency_id() {
    local job_id="$1"
    printf "%s" "${job_id%%;*}"
}

_print_slurm_diagnostics() {
    local jobs_csv="$1"
    echo " Status  : sacct -j ${jobs_csv} \\"
    echo "           --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList,ReqTRES%50,AllocTRES%70,MaxRSS,MaxVMSize"
    echo " Queue   : squeue -j ${jobs_csv} -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
    echo " Starts  : squeue --start -j ${jobs_csv}"
    echo " Failures: grep -RniE \"error|exception|traceback|cuda|cupy|out.of.memory|oom|timeout|killed|scratch|nvme|no space|hdf5|permission|failed\" \"$LOGS_DIR\" | tail -300"
}

_wait_rysy() {
    local job_id="$1"
    local label="$2"
    local log_prefix="${3:-$2}"
    echo "  Polling Rysy for $label (job ${job_id}) ..."
    while squeue -j "$job_id" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
    local states
    states=$(sacct -j "$job_id" \
        --format=State --noheader --parsable2 2>/dev/null \
        | grep -v '^$' | sort -u | tr '\n' ' ' | xargs)
    echo "  $label final states: $states"
    if echo "$states" | grep -qE '(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)'; then
        echo "ERROR: $label (job $job_id) failed. States: $states" >&2
        if [[ -n "${LOGS_DIR:-}" ]]; then
            for log_file in "${LOGS_DIR}/${log_prefix}_${job_id}.out" "${LOGS_DIR}/${log_prefix}_${job_id}.err"; do
                if [[ -f "$log_file" ]]; then
                    echo "" >&2
                    echo "---- tail: $log_file ----" >&2
                    tail -80 "$log_file" >&2 || true
                fi
            done
        fi
        return 1
    fi
    if ! echo "$states" | grep -q 'COMPLETED'; then
        echo "ERROR: $label (job $job_id) did not complete. States: $states" >&2
        return 1
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[1] prepare_reference  (Rysy CPU)  [DRY RUN]"
    PREPARE_REF_JOB="DRY_1"
else
    PREPARE_REF_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$PREPARE_REF_MEM" \
        --cpus-per-task="$PREPARE_REF_CPUS" \
        --time="$PREPARE_REF_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --job-name="drf_ref_${RUN_NAME}" \
        --output="${LOGS_DIR}/prepare_reference_%j.out" \
        --error="${LOGS_DIR}/prepare_reference_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_stage.sh" \
            python -m dot_response_fit.stages.cpu.prepare_reference \
                --config "$RUN_DIR/config.yaml" \
                --run-dir "$RUN_DIR")
    echo "[1] prepare_reference  -> Rysy job $PREPARE_REF_JOB  (time=${PREPARE_REF_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$PREPARE_REF_JOB" "prepare_reference" "prepare_reference"
    fi
fi

PREPARE_REF_DEP_ID="$(_dependency_id "$PREPARE_REF_JOB")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[2] fit_dot_size    (afterok:${PREPARE_REF_DEP_ID}, GPU, time=${FIT_TIME})  [DRY RUN]"
    FIT_JOB="DRY_2"
else
    FIT_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$FIT_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${PREPARE_REF_DEP_ID}" \
        --job-name="drf_fit_${RUN_NAME}" \
        --output="${LOGS_DIR}/fit_%j.out" \
        --error="${LOGS_DIR}/fit_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_gpu.sh" \
            python -m dot_response_fit.stages.gpu.fit_dot_size \
                --run-dir "$RUN_DIR")
    echo "[2] fit_dot_size    -> Rysy job $FIT_JOB  (afterok:${PREPARE_REF_DEP_ID}, time=${FIT_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$FIT_JOB" "fit_dot_size" "fit"
    fi
fi

IMAGE_ARRAY_SPEC="0-$((EFFECTIVE_N_IMAGES - 1))%${MAX_CONCURRENT}"
FIT_DEP_ID="$(_dependency_id "$FIT_JOB")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[3] run_scenario    (afterok:${FIT_DEP_ID}, GPU, array=${IMAGE_ARRAY_SPEC}, simulate+render)  [DRY RUN]"
    SCENARIO_JOB="DRY_3"
else
    SCENARIO_JOB=$(_rysy_sbatch_nvme \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$SCENARIO_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${FIT_DEP_ID}" \
        --array="$IMAGE_ARRAY_SPEC" \
        --job-name="drf_sc_${RUN_NAME}" \
        --output="${LOGS_DIR}/scenario_%A_%a.out" \
        --error="${LOGS_DIR}/scenario_%A_%a.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_gpu.sh" \
            python -m dot_response_fit.stages.gpu.run_scenario \
                --run-dir "$RUN_DIR")
    echo "[3] run_scenario    -> Rysy job $SCENARIO_JOB  (afterok:${FIT_DEP_ID}, array=${IMAGE_ARRAY_SPEC}, time=${SCENARIO_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$SCENARIO_JOB" "run_scenario_array" "scenario"
    fi
fi

SCENARIO_DEP_ID="$(_dependency_id "$SCENARIO_JOB")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[4] finalize        (afterok:${SCENARIO_DEP_ID}, Rysy)  [DRY RUN]"
    FINALIZE_JOB="DRY_4"
else
    FINALIZE_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$FINALIZE_MEM" \
        --cpus-per-task="$FINALIZE_CPUS" \
        --time="$FINALIZE_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${SCENARIO_DEP_ID}" \
        --job-name="drf_fin_${RUN_NAME}" \
        --output="${LOGS_DIR}/finalize_%j.out" \
        --error="${LOGS_DIR}/finalize_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_stage.sh" \
            python -m dot_response_fit.stages.cpu.finalize \
                --run-dir "$RUN_DIR")
    echo "[4] finalize        -> Rysy job $FINALIZE_JOB  (afterok:${SCENARIO_DEP_ID})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$FINALIZE_JOB" "finalize" "finalize"
    fi
fi

echo ""
echo "========================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo " Dry run complete."
    echo " Planned Slurm dependencies:"
    echo "   prepare_reference -> fit_dot_size -> run_scenario image array -> finalize"
elif [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo " All jobs complete."
else
    echo " Pipeline submitted."
    echo " Slurm dependencies will run stages in order:"
    echo "   prepare_reference -> fit_dot_size -> run_scenario image array -> finalize"
    echo " All jobs queued."
fi
echo ""
echo " Run dir : $RUN_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    ALL_JOB_IDS="${PREPARE_REF_DEP_ID},${FIT_DEP_ID},${SCENARIO_DEP_ID},$(_dependency_id "$FINALIZE_JOB")"
    echo " Logs    : $LOGS_DIR"
    echo " Jobs    : prepare=$PREPARE_REF_JOB  fit=$FIT_JOB  images=$SCENARIO_JOB  finalize=$FINALIZE_JOB"
    _print_slurm_diagnostics "$ALL_JOB_IDS"
    echo " Cancel  : scancel $PREPARE_REF_JOB $FIT_JOB $SCENARIO_JOB $FINALIZE_JOB"
fi
echo "========================================"
