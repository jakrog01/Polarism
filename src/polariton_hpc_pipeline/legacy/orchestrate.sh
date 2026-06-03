#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *)         CONFIG="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

source "$PROJECT_ROOT/slurm.env"

module load common/python/3.13.2 2>/dev/null || true
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
elif [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || true
fi

mapfile -t PARSED < <(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
ts = cfg['global']['threshold_search']
print(ts['max_runtime_minutes'])
for sc in cfg['scenarios']:
    print(sc['name'])
" "$CONFIG")

THRESHOLD_MINUTES="${PARSED[0]}"
SCENARIOS=("${PARSED[@]:1}")
THRESHOLD_TIME=$(printf "00:%02d:00" "$THRESHOLD_MINUTES")

mkdir -p "$SCRIPT_DIR/logs"

COMMON=(
    --account="$SLURM_ACCOUNT"
    --partition="$SLURM_PARTITION"
    --qos="$SLURM_QS"
    --mem="$SLURM_MEM"
    --gres=gpu:"$SLURM_GPUS"
    --cpus-per-task="$SLURM_CPUS"
)

echo "========================================"
echo " Polariton Multi-Pump Orchestrator"
echo "========================================"
echo "  Config    : $CONFIG"
echo "  Scenarios : ${#SCENARIOS[@]}"
for sc in "${SCENARIOS[@]}"; do
    echo "    - $sc"
done
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[1] threshold_search  (time=$THRESHOLD_TIME)  [DRY RUN]"
    PREV_ID="DRY_0"
else
    PREV_ID=$(sbatch --parsable \
        "${COMMON[@]}" \
        --time="$THRESHOLD_TIME" \
        --job-name=polSNN_threshold \
        --output="$SCRIPT_DIR/logs/threshold_%j.out" \
        --error="$SCRIPT_DIR/logs/threshold_%j.err" \
        "$SCRIPT_DIR/job_wrapper.sh" \
        python threshold_search.py --config "$CONFIG")
    echo "[1] threshold_search  -> $PREV_ID  (time=$THRESHOLD_TIME)"
fi

ALL_IDS="$PREV_ID"

N=2
for SCENARIO in "${SCENARIOS[@]}"; do
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[$N] $SCENARIO  (dep=afterok:$PREV_ID)  [DRY RUN]"
        PREV_ID="DRY_$N"
    else
        JOB_ID=$(sbatch --parsable \
            "${COMMON[@]}" \
            --time="$SLURM_TIME" \
            --dependency="afterok:$PREV_ID" \
            --job-name="polSNN_${SCENARIO}" \
            --output="$SCRIPT_DIR/logs/${SCENARIO}_%j.out" \
            --error="$SCRIPT_DIR/logs/${SCENARIO}_%j.err" \
            "$SCRIPT_DIR/job_wrapper.sh" \
            python run_scenario.py --config "$CONFIG" --scenario "$SCENARIO")
        echo "[$N] $SCENARIO  -> $JOB_ID  (dep=afterok:$PREV_ID)"
        PREV_ID="$JOB_ID"
    fi
    ALL_IDS="$ALL_IDS $PREV_ID"
    N=$((N + 1))
done

echo ""
echo "All jobs queued."
IDS_CSV="$(echo "$ALL_IDS" | xargs | tr ' ' ',')"
echo "  Status  : sacct -j $IDS_CSV --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList"
echo "  Queue   : squeue -j $IDS_CSV -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
echo "  Starts  : squeue --start -j $IDS_CSV"
if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Cancel  : scancel $ALL_IDS"
fi
