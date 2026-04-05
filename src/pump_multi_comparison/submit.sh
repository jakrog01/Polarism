#!/usr/bin/env bash
# =============================================================================
# submit.sh — PRODUCTION ENTRY POINT for the pump_multi_comparison pipeline.
#
# Run this script from the login node.  It validates config + Slurm env,
# creates a unique run directory, copies a config snapshot, writes the
# scenario index and manifest, submits the full DAG via Slurm dependencies,
# then exits immediately.  No compute allocation is held here.
#
# DAG:
#   [1] threshold_search  (GPU, single)
#         ↓ afterok
#   [2] scenario array    (GPU, --array=0-(N-1)%K, bounded fan-out)
#         ↓ afterok (ALL tasks must succeed)
#   [3] visualize array   (non-GPU stage, --array=0-(N-1))
#         ↓
#   [4] finalize          (non-GPU stage, single)
#
# Concurrency is capped at MAX_CONCURRENT_SCENARIOS (slurm.env, default 1).
# Scenario temp directories are job-unique:
#   $SCRATCH/polariton/<run>/<SLURM_JOB_ID>_<ARRAY_TASK_ID>/<scenario>/
#
# STAGE-SPECIFIC RESOURCE OVERRIDES
# ------------------------------------
# All overrides are optional.  Fallback precedence per stage:
#   threshold : THRESHOLD_{PARTITION,MEM,CPUS,GPUS} -> GPU scenario defaults
#               walltime always = config max_runtime_minutes (THRESHOLD_TIME)
#   visualize : VIZ_{PARTITION,MEM,CPUS,TIME}       -> CPU_* -> GPU defaults
#   finalize  : FINALIZE_{PARTITION,MEM,CPUS,TIME}  -> CPU_* -> GPU defaults
#
# If VIZ_PARTITION / FINALIZE_PARTITION resolve to SLURM_PARTITION (the GPU
# partition), a WARNING is printed; those stages still run but occupy GPU nodes
# without a GPU allocation.  See cluster/README.md for details.
#
# Usage:
#   bash submit.sh [--config /path/to/config.yaml] [--runs-dir /path] [--dry-run]
# =============================================================================
set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
CONFIG=""
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
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

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-$PUMP_DIR/runs}"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

echo "========================================"
echo " Polariton Multi-Pump Pipeline"
echo " Entry point: submit.sh"
echo "========================================"

# ── Preflight checks ──────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2
    printf '  Required variables:\n    SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS SLURM_TIME\n' >&2
    printf '  Optional:\n    SLURM_QOS MAX_CONCURRENT_SCENARIOS\n' >&2
    printf '    CPU_PARTITION CPU_MEM CPU_CPUS CPU_TIME\n' >&2
    printf '    THRESHOLD_{PARTITION,MEM,CPUS,GPUS}\n' >&2
    printf '    VIZ_{PARTITION,MEM,CPUS,TIME}\n' >&2
    printf '    FINALIZE_{PARTITION,MEM,CPUS,TIME}  (see cluster/README.md)\n' >&2
    exit 1
fi

# Activate Python environment on the login node only if no Python is already available.
if ! command -v python3 >/dev/null 2>&1; then
    module load common/python/3.13.2 >/dev/null 2>&1 || true
fi
for venv_path in "$PROJECT_ROOT/.venv/bin/activate" "$PROJECT_ROOT/venv/bin/activate"; do
    [[ -f "$venv_path" ]] && { source "$venv_path" 2>/dev/null || true; break; }
done
export PYTHONPATH="${PROJECT_ROOT}:${PUMP_DIR}:${PYTHONPATH:-}"

echo "  Config    : $CONFIG"
echo "  slurm.env : $SLURM_ENV"
echo ""

# Validate config + slurm env before touching the filesystem.
echo "  Validating config ..."
python3 -m pipeline.config.validator --config "$CONFIG" --slurm-env "$SLURM_ENV" || {
    echo "Aborting: fix the validation errors above." >&2; exit 1
}
echo ""

# ── Parse scenario names + threshold timeout from config ──────────────────────
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
THRESHOLD_TIME=$(printf "%02d:%02d:00" "$THRESHOLD_HOURS" "$THRESHOLD_REMAINDER_MINUTES")

echo "  Scenarios ($N_SCENARIOS):"
for sc in "${SCENARIOS[@]}"; do echo "    - $sc"; done
echo ""

# ── Load Slurm env ────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
#
# Export everything defined in slurm.env so variables such as SCRATCH are
# visible inside the submitted jobs, not only in this login-shell process.
set -a
source "$SLURM_ENV"
set +a

for var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS SLURM_TIME; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: slurm.env missing required variable: $var" >&2; exit 1
    fi
done

# Bounded scenario fan-out (default 1 — safest for shared scratch and GPU memory).
MAX_CONCURRENT="${MAX_CONCURRENT_SCENARIOS:-1}"

# ── Per-stage resource resolution ─────────────────────────────────────────────
# Fallback precedence (all variables are optional in slurm.env):
#   threshold : THRESHOLD_*   -> GPU scenario defaults
#               (time always comes from config max_runtime_minutes)
#   visualize : VIZ_*         -> CPU_*  -> GPU scenario defaults
#   finalize  : FINALIZE_*    -> CPU_*  -> GPU scenario defaults
#
# GPU scenario defaults (SLURM_*) are always required and serve as the
# ultimate fallback for every stage.

_TH_PARTITION="${THRESHOLD_PARTITION:-$SLURM_PARTITION}"
_TH_MEM="${THRESHOLD_MEM:-$SLURM_MEM}"
_TH_CPUS="${THRESHOLD_CPUS:-$SLURM_CPUS}"
_TH_GPUS="${THRESHOLD_GPUS:-$SLURM_GPUS}"

_VIZ_PARTITION="${VIZ_PARTITION:-${CPU_PARTITION:-$SLURM_PARTITION}}"
_VIZ_MEM="${VIZ_MEM:-${CPU_MEM:-$SLURM_MEM}}"
_VIZ_CPUS="${VIZ_CPUS:-${CPU_CPUS:-$SLURM_CPUS}}"
_VIZ_TIME="${VIZ_TIME:-${CPU_TIME:-$SLURM_TIME}}"

_FIN_PARTITION="${FINALIZE_PARTITION:-${CPU_PARTITION:-$SLURM_PARTITION}}"
_FIN_MEM="${FINALIZE_MEM:-${CPU_MEM:-$SLURM_MEM}}"
_FIN_CPUS="${FINALIZE_CPUS:-${CPU_CPUS:-$SLURM_CPUS}}"
_FIN_TIME="${FINALIZE_TIME:-${CPU_TIME:-$SLURM_TIME}}"

_cpu_fallback_active=0
for _stage_part in "$_VIZ_PARTITION" "$_FIN_PARTITION"; do
    if [[ "$_stage_part" == "$SLURM_PARTITION" ]]; then
        _cpu_fallback_active=1
    fi
done
if [[ $_cpu_fallback_active -eq 1 ]]; then
    echo "  WARNING: one or more CPU stages (visualize, finalize) resolved to the"
    echo "     GPU partition ($SLURM_PARTITION) because no stage-specific or"
    echo "     CPU_PARTITION override is set.  They will occupy a GPU node slot"
    echo "     without allocating a GPU.  Set CPU_PARTITION, VIZ_PARTITION, or"
    echo "     FINALIZE_PARTITION in slurm.env to suppress this warning."
fi
echo "  threshold : partition=$_TH_PARTITION  mem=$_TH_MEM  cpus=$_TH_CPUS  gpus=$_TH_GPUS  time=$THRESHOLD_TIME"
echo "  visualize : partition=$_VIZ_PARTITION  mem=$_VIZ_MEM  cpus=$_VIZ_CPUS  time=$_VIZ_TIME"
echo "  finalize  : partition=$_FIN_PARTITION  mem=$_FIN_MEM  cpus=$_FIN_CPUS  time=$_FIN_TIME"
echo "  Max concurrent GPU scenarios : $MAX_CONCURRENT"
echo ""

# ── Create run directory ──────────────────────────────────────────────────────
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
RUN_NAME="$(basename "$RUN_DIR")"
echo "  Run dir  : $RUN_DIR"

# Snapshot config (jobs read from run_dir, never from the live source file).
cp "$CONFIG" "$RUN_DIR/config.yaml"

# Write scenario_index.json.
python3 -c "
import json
with open('${RUN_DIR}/scenario_index.json', 'w') as f:
    json.dump($(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${SCENARIOS[@]}"), f, indent=2)
"

# Initialise the run manifest.
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

# ── Slurm flag sets ───────────────────────────────────────────────────────────
LOGS_DIR="$RUN_DIR/logs"
mkdir -p "$LOGS_DIR"

_QOS_FLAG=()
[[ -n "${SLURM_QOS:-}" ]] && _QOS_FLAG=(--qos="$SLURM_QOS")

GPU_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --partition="$SLURM_PARTITION"
    --mem="$SLURM_MEM"
    --gres="gpu:${SLURM_GPUS}"
    --cpus-per-task="$SLURM_CPUS"
    --time="$SLURM_TIME"
    "${_QOS_FLAG[@]}"
)

THRESHOLD_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --partition="$_TH_PARTITION"
    --mem="$_TH_MEM"
    --gres="gpu:${_TH_GPUS}"
    --cpus-per-task="$_TH_CPUS"
    "${_QOS_FLAG[@]}"
)

VIZ_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --partition="$_VIZ_PARTITION"
    --mem="$_VIZ_MEM"
    --cpus-per-task="$_VIZ_CPUS"
    --time="$_VIZ_TIME"
    "${_QOS_FLAG[@]}"
)

FINALIZE_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --partition="$_FIN_PARTITION"
    --mem="$_FIN_MEM"
    --cpus-per-task="$_FIN_CPUS"
    --time="$_FIN_TIME"
    "${_QOS_FLAG[@]}"
)

# ── Job 1: Threshold search (GPU) ─────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[1] threshold_search  (time=$THRESHOLD_TIME)  [DRY RUN]"
    THRESHOLD_JOB="DRY_1"
else
    THRESHOLD_JOB=$(sbatch --parsable \
        "${THRESHOLD_FLAGS[@]}" \
        --time="$THRESHOLD_TIME" \
        --job-name="pol_th_${RUN_NAME}" \
        --output="${LOGS_DIR}/threshold_%j.out" \
        --error="${LOGS_DIR}/threshold_%j.err" \
        "$SCRIPT_DIR/cluster/job_gpu.sh" \
            python -m pipeline.stages.gpu.threshold_search \
                --config  "$RUN_DIR/config.yaml" \
                --run-dir "$RUN_DIR")
    echo "[1] threshold_search  -> job $THRESHOLD_JOB  (time=$THRESHOLD_TIME)"
fi

# ── Job 2: Scenario array (GPU, bounded fan-out) ──────────────────────────────
# $SLURM_ARRAY_TASK_ID is read by run_scenario.py to resolve the scenario name.
SCENARIO_ARRAY_SPEC="0-$((N_SCENARIOS - 1))%${MAX_CONCURRENT}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[2] scenario array    (array=${SCENARIO_ARRAY_SPEC}, dep=afterok:${THRESHOLD_JOB})  [DRY RUN]"
    SCENARIO_ARRAY_JOB="DRY_2"
else
    SCENARIO_ARRAY_JOB=$(sbatch --parsable \
        "${GPU_FLAGS[@]}" \
        --array="$SCENARIO_ARRAY_SPEC" \
        --dependency="afterok:${THRESHOLD_JOB}" \
        --job-name="pol_sc_${RUN_NAME}" \
        --output="${LOGS_DIR}/scenario_%A_%a.out" \
        --error="${LOGS_DIR}/scenario_%A_%a.err" \
        "$SCRIPT_DIR/cluster/job_gpu.sh" \
            python -m pipeline.stages.gpu.run_scenario \
                --run-dir "$RUN_DIR")
    echo "[2] scenario array    -> job $SCENARIO_ARRAY_JOB  (array=${SCENARIO_ARRAY_SPEC}, dep=afterok:${THRESHOLD_JOB})"
fi

# ── Job 3: Visualize array (non-GPU stage, per scenario) ─────────────────────
# afterok:ARRAY_JOB_ID means ALL array tasks must succeed.
VIZ_ARRAY_SPEC="0-$((N_SCENARIOS - 1))"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[3] visualize array   (array=${VIZ_ARRAY_SPEC}, dep=afterok:${SCENARIO_ARRAY_JOB})  [DRY RUN]"
    VIZ_ARRAY_JOB="DRY_3"
else
    VIZ_ARRAY_JOB=$(sbatch --parsable \
        "${VIZ_FLAGS[@]}" \
        --array="$VIZ_ARRAY_SPEC" \
        --dependency="afterok:${SCENARIO_ARRAY_JOB}" \
        --job-name="pol_viz_${RUN_NAME}" \
        --output="${LOGS_DIR}/viz_%A_%a.out" \
        --error="${LOGS_DIR}/viz_%A_%a.err" \
        "$SCRIPT_DIR/cluster/job_stage.sh" \
            python -m pipeline.stages.cpu.visualize \
                --run-dir "$RUN_DIR")
    echo "[3] visualize array   -> job $VIZ_ARRAY_JOB  (array=${VIZ_ARRAY_SPEC}, dep=afterok:${SCENARIO_ARRAY_JOB})"
fi

# ── Job 4: Finalize (non-GPU stage, after all scenarios) ─────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[4] finalize          (dep=afterok:${SCENARIO_ARRAY_JOB})  [DRY RUN]"
    FINALIZE_JOB="DRY_4"
else
    FINALIZE_JOB=$(sbatch --parsable \
        "${FINALIZE_FLAGS[@]}" \
        --dependency="afterok:${SCENARIO_ARRAY_JOB}" \
        --job-name="pol_fin_${RUN_NAME}" \
        --output="${LOGS_DIR}/finalize_%j.out" \
        --error="${LOGS_DIR}/finalize_%j.err" \
        "$SCRIPT_DIR/cluster/job_stage.sh" \
            python -m pipeline.stages.cpu.finalize \
                --run-dir "$RUN_DIR")
    echo "[4] finalize          -> job $FINALIZE_JOB  (dep=afterok:${SCENARIO_ARRAY_JOB})"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " All jobs queued."
echo ""
echo " Run dir : $RUN_DIR"
echo " Monitor : squeue -u \$USER"
echo " Logs    : $LOGS_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    echo " Cancel  : scancel $THRESHOLD_JOB $SCENARIO_ARRAY_JOB $VIZ_ARRAY_JOB $FINALIZE_JOB"
fi
echo "========================================"
