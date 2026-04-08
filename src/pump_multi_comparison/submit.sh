#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --wait)          WAIT_FOR_COMPLETION=1; shift ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --config=*)      CONFIG="${1#--config=}"; shift ;;
        --runs-dir)      RUNS_BASE_DIR="$2"; shift 2 ;;
        --runs-dir=*)    RUNS_BASE_DIR="${1#--runs-dir=}"; shift ;;
        *) [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PUMP_DIR="$SCRIPT_DIR"
CLUSTER_DIR="$SCRIPT_DIR/cluster"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

echo "========================================"
echo " Polariton Multi-Pump Pipeline"
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
export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${PYTHONPATH:-}"

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
python3 -m pipeline.config.validator --config "$CONFIG" --slurm-env "$SLURM_ENV" || {
    echo "Aborting: fix the validation errors above." >&2; exit 1
}
echo ""

mapfile -t _PARSED < <(python3 - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["global"]["threshold_search"]["max_runtime_minutes"])
for sc in cfg["scenarios"]:
    print(sc["name"])
PYEOF
)
THRESHOLD_MINUTES="${_PARSED[0]}"
SCENARIOS=("${_PARSED[@]:1}")
N_SCENARIOS="${#SCENARIOS[@]}"
THRESHOLD_HOURS=$((THRESHOLD_MINUTES / 60))
THRESHOLD_REMAINDER_MINUTES=$((THRESHOLD_MINUTES % 60))
THRESHOLD_TIME_DERIVED=$(printf "%02d:%02d:00" "$THRESHOLD_HOURS" "$THRESHOLD_REMAINDER_MINUTES")

echo "  Scenarios ($N_SCENARIOS):"
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
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -qE ' mpeg4 '; then
        export RENDER_ENCODER="mpeg4"
    fi
fi

for var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS SLURM_TIME NVME_GB \
           TETYDA_RUNS_BASE MAX_CONCURRENT_SCENARIOS \
           FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: slurm.env missing required variable: $var" >&2; exit 1
    fi
done

THRESHOLD_TIME="${THRESHOLD_TIME:-$THRESHOLD_TIME_DERIVED}"
SCENARIO_TIME="${SCENARIO_TIME:-$SLURM_TIME}"

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
MAX_CONCURRENT="$MAX_CONCURRENT_SCENARIOS"

echo "  Rysy  GPU : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
echo "  Repo root : $PROJECT_ROOT"
echo "  Tetyda    : $TETYDA_RUNS_BASE"
echo "  Max concurrent GPU scenarios: $MAX_CONCURRENT"
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
from pipeline.manifest.io import init_manifest
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

_wait_rysy() {
    local job_id="$1"
    local label="$2"
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
        return 1
    fi
    if ! echo "$states" | grep -q 'COMPLETED'; then
        echo "ERROR: $label (job $job_id) did not complete. States: $states" >&2
        return 1
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[1] threshold_search  (time=$THRESHOLD_TIME)  [DRY RUN]"
    THRESHOLD_JOB="DRY_1"
else
    THRESHOLD_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$THRESHOLD_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --job-name="pol_th_${RUN_NAME}" \
        --output="${LOGS_DIR}/threshold_%j.out" \
        --error="${LOGS_DIR}/threshold_%j.err" \
        "$CLUSTER_DIR/job_gpu.sh" \
            python -m pipeline.stages.gpu.threshold_search \
                --config  "$RUN_DIR/config.yaml" \
                --run-dir "$RUN_DIR")
    echo "[1] threshold_search  -> Rysy job $THRESHOLD_JOB  (time=$THRESHOLD_TIME)"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$THRESHOLD_JOB" "threshold_search"
    fi
fi

SCENARIO_ARRAY_SPEC="0-$((N_SCENARIOS - 1))%${MAX_CONCURRENT}"
THRESHOLD_DEP_ID="$(_dependency_id "$THRESHOLD_JOB")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[2] scenario array    (afterok:${THRESHOLD_DEP_ID}, array=${SCENARIO_ARRAY_SPEC}, simulate+render)  [DRY RUN]"
    SCENARIO_ARRAY_JOB="DRY_2"
else
    SCENARIO_ARRAY_JOB=$(_rysy_sbatch_nvme \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$SCENARIO_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${THRESHOLD_DEP_ID}" \
        --array="$SCENARIO_ARRAY_SPEC" \
        --job-name="pol_sc_${RUN_NAME}" \
        --output="${LOGS_DIR}/scenario_%A_%a.out" \
        --error="${LOGS_DIR}/scenario_%A_%a.err" \
        "$CLUSTER_DIR/job_gpu.sh" \
            python -m pipeline.stages.gpu.run_scenario \
                --run-dir "$RUN_DIR")
    echo "[2] scenario array    -> Rysy job $SCENARIO_ARRAY_JOB  (afterok:${THRESHOLD_DEP_ID}, array=${SCENARIO_ARRAY_SPEC}, time=${SCENARIO_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$SCENARIO_ARRAY_JOB" "scenario_array"
    fi
fi

SCENARIO_DEP_ID="$(_dependency_id "$SCENARIO_ARRAY_JOB")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[3] finalize          (afterok:${SCENARIO_DEP_ID}, Rysy)  [DRY RUN]"
    FINALIZE_JOB="DRY_3"
else
    FINALIZE_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$FINALIZE_MEM" \
        --cpus-per-task="$FINALIZE_CPUS" \
        --time="$FINALIZE_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${SCENARIO_DEP_ID}" \
        --job-name="pol_fin_${RUN_NAME}" \
        --output="${LOGS_DIR}/finalize_%j.out" \
        --error="${LOGS_DIR}/finalize_%j.err" \
        "$CLUSTER_DIR/job_stage.sh" \
            python -m pipeline.stages.cpu.finalize \
                --run-dir "$RUN_DIR")
    echo "[3] finalize          -> Rysy job $FINALIZE_JOB  (afterok:${SCENARIO_DEP_ID})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$FINALIZE_JOB" "finalize"
    fi
fi

echo ""
echo "========================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo " Dry run complete."
    echo " Planned Slurm dependencies:"
    echo "   threshold_search -> scenario array -> finalize"
elif [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo " All jobs complete."
else
    echo " Pipeline submitted."
    echo " Slurm dependencies will run stages in order:"
    echo "   threshold_search -> scenario array -> finalize"
    echo " All jobs queued."
fi
echo ""
echo " Run dir : $RUN_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    echo " Logs    : $LOGS_DIR"
    echo " Jobs    : threshold=$THRESHOLD_JOB  scenarios=$SCENARIO_ARRAY_JOB  finalize=$FINALIZE_JOB"
    echo " Monitor : squeue -u \$USER"
    echo " Cancel  : scancel $THRESHOLD_JOB $SCENARIO_ARRAY_JOB $FINALIZE_JOB"
fi
echo "========================================"
