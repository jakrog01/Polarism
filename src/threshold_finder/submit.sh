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
THRESHOLD_DIR="$SCRIPT_DIR"
CLUSTER_DIR="$SCRIPT_DIR/cluster"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

echo "========================================"
echo " Threshold-Finder  |  scalar response sweep"
echo " Rysy-only  |  run from a Rysy login node"
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
export PYTHONPATH="${PROJECT_ROOT}:${THRESHOLD_DIR}:${PYTHONPATH:-}"

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: submit.sh must be run on a Rysy login node." >&2
    echo "Current host: $HOSTNAME_SHORT" >&2
    exit 1
fi
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 1 ]]; then
    echo "  [DRY RUN] Not on Rysy (host: $HOSTNAME_SHORT) — Slurm calls will be skipped."
fi

echo "  Config    : $CONFIG"
echo "  slurm.env : $SLURM_ENV"
echo "  Host      : $HOSTNAME_SHORT"
echo ""

echo "  Validating config ..."
python3 -m threshold_finder.config.validator --config "$CONFIG" --slurm-env "$SLURM_ENV" || {
    echo "Aborting: fix the validation errors above." >&2; exit 1
}
echo ""

mapfile -t _PARSED < <(python3 - "$CONFIG" "$PROJECT_ROOT" "$THRESHOLD_DIR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[3])
from threshold_finder.config.loader import load_config, get_sweep_config, generate_sweep_points
cfg = load_config(sys.argv[1])
sweep = get_sweep_config(cfg)
points = generate_sweep_points(cfg)
print(len(points))
print(int(sweep.get("max_concurrent", 4)))
PYEOF
)
N_POWERS="${_PARSED[0]}"
MAX_CONCURRENT="${_PARSED[1]}"
if [[ "$N_POWERS" -lt 1 ]]; then
    echo "ERROR: no power values to submit." >&2
    exit 1
fi
POWER_LAST_INDEX=$((N_POWERS - 1))
if [[ "$N_POWERS" -gt 1 ]]; then
    POWER_ARRAY_ENABLED=1
    POWER_ARRAY_SPEC="0-$((N_POWERS - 2))%${MAX_CONCURRENT}"
else
    POWER_ARRAY_ENABLED=0
    POWER_ARRAY_SPEC=""
fi

set -a
source "$SLURM_ENV"
set +a

_RYSY_QOS_FLAG=""
[[ -n "${SLURM_QOS:-}" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

GPU_TIME="${SWEEP_TIME:-${SLURM_TIME}}"

for var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS \
           TETYDA_RUNS_BASE FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: slurm.env missing required variable: $var" >&2; exit 1
    fi
done

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"

echo "  Points    : $N_POWERS  (array $POWER_ARRAY_SPEC)"
echo "  GPU time  : $GPU_TIME"
echo "  Rysy GPU  : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS"
echo "  Repo root : $PROJECT_ROOT"
echo "  Tetyda    : $RUNS_BASE_DIR"
echo ""

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$RUNS_BASE_DIR"
    RUN_DIR=$(python3 - "$RUNS_BASE_DIR" "$CONFIG" <<'PYEOF'
import sys, os, hashlib
from datetime import datetime
base, cfg_path = sys.argv[1], sys.argv[2]
chash = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:8]
run_dir = os.path.join(base, f"tf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{chash}")
os.makedirs(run_dir, exist_ok=False)
print(run_dir)
PYEOF
)
else
    RUN_DIR="${RUNS_BASE_DIR}/tf_DRY_$(date +%Y%m%d_%H%M%S)"
    echo "  [DRY RUN] run dir would be: $RUN_DIR"
fi
RUN_NAME="$(basename "$RUN_DIR")"

if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Run dir : $RUN_DIR"

    cp "$CONFIG" "$RUN_DIR/config.yaml"

    python3 - "$RUN_DIR" "$CONFIG" "$PROJECT_ROOT" "$THRESHOLD_DIR" <<'PYEOF'
import sys, json
sys.path.insert(0, sys.argv[3])
sys.path.insert(0, sys.argv[4])
from threshold_finder.config.loader import load_config, generate_sweep_points
run_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = load_config(cfg_path)
values = generate_sweep_points(cfg)
with open(f"{run_dir}/power_index.json", "w") as f:
    json.dump(values, f, indent=2)
print(f"  power_index.json: {len(values)} points")
PYEOF

    python3 - "$RUN_DIR" "$CONFIG" "$PROJECT_ROOT" "$THRESHOLD_DIR" <<'PYEOF'
import sys, json
sys.path.insert(0, sys.argv[3])
sys.path.insert(0, sys.argv[4])
from threshold_finder.manifest.io import init_manifest
run_dir, cfg_path = sys.argv[1], sys.argv[2]
with open(f"{run_dir}/power_index.json") as f:
    n = len(json.load(f))
init_manifest(run_dir, cfg_path, n)
print("  Manifest initialised.")
PYEOF

    mkdir -p "$RUN_DIR/powers"
    echo ""
fi

LOGS_DIR="$RUN_DIR/logs"
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$LOGS_DIR"
fi

_rysy_sbatch() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
        "$@"
}

_dependency_id() {
    printf "%s" "${1%%;*}"
}

_print_slurm_diagnostics() {
    local jobs_csv="$1"
    echo " Status  : sacct -j ${jobs_csv} \\"
    echo "           --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList,ReqTRES%50,AllocTRES%70,MaxRSS,MaxVMSize"
    echo " Queue   : squeue -j ${jobs_csv} -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
    echo " Starts  : squeue --start -j ${jobs_csv}"
    echo " Failures: grep -RniE \"error|exception|traceback|cuda|cupy|nan|inf|diverged\" \"$LOGS_DIR\" | tail -200"
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

if [[ $POWER_ARRAY_ENABLED -eq 0 ]]; then
    SWEEP_ARRAY_JOB=""
elif [[ $DRY_RUN -eq 1 ]]; then
    echo "[1a] scalar sweep array  (array=${POWER_ARRAY_SPEC}, time=${GPU_TIME})  [DRY RUN]"
    SWEEP_ARRAY_JOB="DRY_1"
else
    SWEEP_ARRAY_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$GPU_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --array="$POWER_ARRAY_SPEC" \
        --job-name="tf_sw_${RUN_NAME}" \
        --output="${LOGS_DIR}/sweep_%A_%a.out" \
        --error="${LOGS_DIR}/sweep_%A_%a.err" \
        "$CLUSTER_DIR/job_gpu.sh" \
            python -m threshold_finder.stages.gpu.run_power \
                --run-dir "$RUN_DIR")
    echo "[1a] scalar sweep array  -> Rysy job $SWEEP_ARRAY_JOB  (array=${POWER_ARRAY_SPEC}, time=${GPU_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$SWEEP_ARRAY_JOB" "sweep_array"
    fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[1b] scalar sweep last   (index=${POWER_LAST_INDEX}, singleton, time=${GPU_TIME})  [DRY RUN]"
    SWEEP_LAST_JOB="DRY_1B"
else
    SWEEP_LAST_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$GPU_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --job-name="tf_sw_last_${RUN_NAME}" \
        --output="${LOGS_DIR}/sweep_last_%j.out" \
        --error="${LOGS_DIR}/sweep_last_%j.err" \
        "$CLUSTER_DIR/job_gpu.sh" \
            python -m threshold_finder.stages.gpu.run_power \
                --run-dir "$RUN_DIR" \
                --power-index "$POWER_LAST_INDEX")
    echo "[1b] scalar sweep last   -> Rysy job $SWEEP_LAST_JOB  (index=${POWER_LAST_INDEX}, time=${GPU_TIME})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$SWEEP_LAST_JOB" "sweep_last"
    fi
fi

SWEEP_DEP_IDS=()
if [[ -n "${SWEEP_ARRAY_JOB:-}" ]]; then
    SWEEP_DEP_IDS+=("$(_dependency_id "$SWEEP_ARRAY_JOB")")
fi
SWEEP_DEP_IDS+=("$(_dependency_id "$SWEEP_LAST_JOB")")
SWEEP_DEP_ID="$(IFS=:; echo "${SWEEP_DEP_IDS[*]}")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[2] finalize           (afterok:${SWEEP_DEP_ID})  [DRY RUN]"
    FINALIZE_JOB="DRY_2"
else
    FINALIZE_JOB=$(_rysy_sbatch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$FINALIZE_MEM" \
        --cpus-per-task="$FINALIZE_CPUS" \
        --time="$FINALIZE_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --dependency="afterok:${SWEEP_DEP_ID}" \
        --job-name="tf_fin_${RUN_NAME}" \
        --output="${LOGS_DIR}/finalize_%j.out" \
        --error="${LOGS_DIR}/finalize_%j.err" \
        "$CLUSTER_DIR/job_stage.sh" \
            python -m threshold_finder.stages.cpu.finalize \
                --run-dir "$RUN_DIR")
    echo "[2] finalize           -> Rysy job $FINALIZE_JOB  (afterok:${SWEEP_DEP_ID})"
    if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
        _wait_rysy "$FINALIZE_JOB" "finalize"
    fi
fi

echo ""
echo "========================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo " Dry run complete."
    echo " Planned Slurm dependencies:"
    echo "   sweep_array + last singleton -> finalize"
elif [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo " All jobs complete."
else
    echo " Pipeline submitted."
    echo " Slurm dependencies:  sweep_array + last singleton -> finalize"
fi
echo ""
echo " Run dir : $RUN_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    ALL_JOB_IDS="${SWEEP_DEP_ID//:/,},$(_dependency_id "$FINALIZE_JOB")"
    echo " Logs    : $LOGS_DIR"
    if [[ -n "${SWEEP_ARRAY_JOB:-}" ]]; then
        echo " Jobs    : sweep=$SWEEP_ARRAY_JOB  last=$SWEEP_LAST_JOB  finalize=$FINALIZE_JOB"
    else
        echo " Jobs    : last=$SWEEP_LAST_JOB  finalize=$FINALIZE_JOB"
    fi
    _print_slurm_diagnostics "$ALL_JOB_IDS"
    echo " Cancel  : scancel ${SWEEP_ARRAY_JOB:-} $SWEEP_LAST_JOB $FINALIZE_JOB"
fi
echo "========================================"
