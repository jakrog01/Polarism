#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG=""
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
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
python3 -m dot_response_fit.config.validator --config "$CONFIG" --slurm-env "$SLURM_ENV" || {
    echo "Aborting: fix the validation errors above." >&2; exit 1
}
echo ""

mapfile -t _PARSED < <(python3 - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("fit", {}).get("max_runtime_minutes", 120))
for sc in cfg.get("scenarios", []):
    print(sc["name"])
PYEOF
)
FIT_MINUTES="${_PARSED[0]}"
SCENARIOS=("${_PARSED[@]:1}")
N_SCENARIOS="${#SCENARIOS[@]}"
FIT_HOURS=$((FIT_MINUTES / 60))
FIT_REMAINDER=$((FIT_MINUTES % 60))
FIT_TIME_DERIVED=$(printf "%02d:%02d:00" "$FIT_HOURS" "$FIT_REMAINDER")

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
    elif "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -qE '^[[:space:]]*V.*[[:space:]]mpeg4[[:space:]]'; then
        export RENDER_ENCODER="mpeg4"
    fi
fi

for var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS SLURM_TIME NVME_GB \
           TETYDA_RUNS_BASE MAX_CONCURRENT_SCENARIOS \
           FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME \
           TIME_RESPONSE_MEM TIME_RESPONSE_CPUS TIME_RESPONSE_TIME; do
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
    echo "[1] time_response   (Rysy CPU)  [DRY RUN]"
    TIME_RESPONSE_JOB="DRY_1"
else
    TIME_RESPONSE_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$TIME_RESPONSE_MEM" \
        --cpus-per-task="$TIME_RESPONSE_CPUS" \
        --time="$TIME_RESPONSE_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --job-name="drf_tr_${RUN_NAME}" \
        --output="${LOGS_DIR}/time_response_%j.out" \
        --error="${LOGS_DIR}/time_response_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_stage.sh" \
            python -m dot_response_fit.stages.cpu.time_response \
                --config "$RUN_DIR/config.yaml" \
                --run-dir "$RUN_DIR")
    echo "[1] time_response   -> Rysy job $TIME_RESPONSE_JOB"
    _wait_rysy "$TIME_RESPONSE_JOB" "time_response"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[2] fit_dot_size    (GPU, time=${FIT_TIME})  [DRY RUN]"
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
        --job-name="drf_fit_${RUN_NAME}" \
        --output="${LOGS_DIR}/fit_%j.out" \
        --error="${LOGS_DIR}/fit_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_gpu.sh" \
            python -m dot_response_fit.stages.gpu.fit_dot_size \
                --run-dir "$RUN_DIR")
    echo "[2] fit_dot_size    -> Rysy job $FIT_JOB  (time=${FIT_TIME})"
    _wait_rysy "$FIT_JOB" "fit_dot_size"
fi

SCENARIO_ARRAY_SPEC="0-$((N_SCENARIOS - 1))%${MAX_CONCURRENT}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[3] run_scenario    (GPU, array=${SCENARIO_ARRAY_SPEC}, simulate+render)  [DRY RUN]"
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
        --array="$SCENARIO_ARRAY_SPEC" \
        --job-name="drf_sc_${RUN_NAME}" \
        --output="${LOGS_DIR}/scenario_%A_%a.out" \
        --error="${LOGS_DIR}/scenario_%A_%a.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_gpu.sh" \
            python -m dot_response_fit.stages.gpu.run_scenario \
                --run-dir "$RUN_DIR")
    echo "[3] run_scenario    -> Rysy job $SCENARIO_JOB  (array=${SCENARIO_ARRAY_SPEC}, time=${SCENARIO_TIME})"
    _wait_rysy "$SCENARIO_JOB" "run_scenario_array"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[4] finalize        (Rysy)  [DRY RUN]"
    FINALIZE_JOB="DRY_4"
else
    FINALIZE_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$FINALIZE_MEM" \
        --cpus-per-task="$FINALIZE_CPUS" \
        --time="$FINALIZE_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --job-name="drf_fin_${RUN_NAME}" \
        --output="${LOGS_DIR}/finalize_%j.out" \
        --error="${LOGS_DIR}/finalize_%j.err" \
        "$PROJECT_ROOT/src/pump_multi_comparison/cluster/job_stage.sh" \
            python -m dot_response_fit.stages.cpu.finalize \
                --run-dir "$RUN_DIR")
    echo "[4] finalize        -> Rysy job $FINALIZE_JOB"
    _wait_rysy "$FINALIZE_JOB" "finalize"
fi

echo ""
echo "========================================"
echo " All jobs complete."
echo ""
echo " Run dir : $RUN_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    echo " Logs    : $LOGS_DIR"
    echo " Monitor : squeue -u \$USER"
fi
echo "========================================"
