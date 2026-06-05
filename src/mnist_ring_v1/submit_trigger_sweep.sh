#!/usr/bin/env bash
# Trigger-fraction sweep for mnist_ring_v1.
#
# Submits one independent pipeline run per sweep point. Each point uses
# the same balanced 200-image split (10 train + 10 test per class) with seed=42.
# sigma_time_trigger_ps=0.1 (fast trigger); sigma_time_ring_ps=1.5 (unchanged).
#
# Usage:
#   bash submit_trigger_sweep.sh [--runs-dir <dir>] [--dry-run] [--wait]
#
# Access-node safe: no Python, no module load.
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1;                          shift ;;
        --wait)        WAIT_FOR_COMPLETION=1;              shift ;;
        --runs-dir)    RUNS_BASE_DIR="$2";                 shift 2 ;;
        --runs-dir=*)  RUNS_BASE_DIR="${1#--runs-dir=}";   shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLURM_ENV="$PROJECT_ROOT/slurm.env"
BASE_CONFIG="$SCRIPT_DIR/config_trigger_sweep.yaml"
SUBMIT_SH="$SCRIPT_DIR/submit.sh"
CLUSTER_DIR="$SCRIPT_DIR/cluster"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: base config not found: $BASE_CONFIG" >&2; exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2; exit 1
fi

source "$SLURM_ENV"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"

SWEEP_FRACTIONS="0.3 0.4 0.5 0.6 0.8"

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${RUNS_BASE_DIR}/${TS}_trigger_sweep"
CONFIGS_DIR="${SWEEP_DIR}/configs"
SWEEP_RUNS_DIR="${SWEEP_DIR}/runs"
SWEEP_LOG="${SWEEP_DIR}/sweep_run_dirs.txt"

echo "========================================================"
echo " mnist_ring_v1 trigger sweep"
echo "========================================================"
echo "  Sweep fractions : $SWEEP_FRACTIONS"
echo "  Base config     : $BASE_CONFIG"
echo "  Sweep dir       : $SWEEP_DIR"
echo "  Runs dir        : $SWEEP_RUNS_DIR"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would create:"
    echo "  $CONFIGS_DIR/"
    for FRAC in $SWEEP_FRACTIONS; do
        echo "    config_trig${FRAC}.yaml"
    done
    echo "  Then call submit.sh for each point with:"
    echo "    --pilot-train-per-class 10 --pilot-test-per-class 10"
    echo ""
    echo "  Then submit finalize_sweep (CPU) after all points finish."
    echo ""
    bash "$SUBMIT_SH" --dry-run --config "$BASE_CONFIG" --runs-dir "$SWEEP_RUNS_DIR/dry"
    exit 0
fi

mkdir -p "$CONFIGS_DIR" "$SWEEP_RUNS_DIR"
echo "# trigger_fraction -> run_dir" > "$SWEEP_LOG"

LAST_FINALIZE_JOBS=""

for FRAC in $SWEEP_FRACTIONS; do
    POINT_CONFIG="${CONFIGS_DIR}/config_trig${FRAC}.yaml"
    sed "s/P_TRIGGER_FRACTION_PLACEHOLDER/${FRAC}/g" "$BASE_CONFIG" > "$POINT_CONFIG"

    POINT_RUNS_DIR="${SWEEP_RUNS_DIR}/trig${FRAC}"

    echo "--- Submitting trigger_fraction=${FRAC} ---"
    bash "$SUBMIT_SH" \
        --config "$POINT_CONFIG" \
        --runs-dir "$POINT_RUNS_DIR" \
        --pilot-train-per-class 10 \
        --pilot-test-per-class 10

    POINT_RUN_DIR="$(ls -d "${POINT_RUNS_DIR}/"*"_mnist_ring_v1" 2>/dev/null | tail -1 || echo "")"
    if [[ -n "$POINT_RUN_DIR" ]]; then
        echo "${FRAC} -> $POINT_RUN_DIR" >> "$SWEEP_LOG"
    fi
done

echo ""
echo "========================================================"
echo " All sweep points submitted."
echo "  Sweep dir  : $SWEEP_DIR"
echo "  Run dirs   : $SWEEP_LOG"
echo ""
echo " After all finalize jobs complete, run sweep analysis:"
echo "  python -m mnist_ring_v1.stages.cpu.finalize_sweep \\"
echo "    --sweep-log $SWEEP_LOG \\"
echo "    --out-dir $SWEEP_DIR"
echo "========================================================"

if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo ""
    echo " [--wait] Sweep submission complete. Monitor with:"
    echo "  tail -f $SWEEP_DIR/runs/*/logs/finalize_*.out"
fi
