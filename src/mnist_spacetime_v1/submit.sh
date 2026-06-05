#!/usr/bin/env bash
# mnist_spacetime_v1 Slurm/Rysy orchestrator - access-node safe.
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --wait)         WAIT_FOR_COMPLETION=1; shift ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --config=*)     CONFIG="${1#--config=}"; shift ;;
        --runs-dir)     RUNS_BASE_DIR="$2"; shift 2 ;;
        --runs-dir=*)   RUNS_BASE_DIR="${1#--runs-dir=}"; shift ;;
        *)              [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_DIR="$SCRIPT_DIR/cluster"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

echo "========================================================"
echo " mnist_spacetime_v1  |  scenario campaign"
echo "========================================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2
    exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: submit.sh must be run on a Rysy login/access node." >&2
    echo "  Use --dry-run to test locally." >&2
    exit 1
fi
[[ "$HOSTNAME_SHORT" != rysy* ]] && echo "  [DRY RUN] Not on Rysy - Slurm calls skipped."

source "$SLURM_ENV"

for _var in SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME SLURM_MEM SLURM_GPUS SLURM_CPUS \
            NVME_GB TETYDA_RUNS_BASE FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME; do
    if [[ -z "${!_var:-}" ]]; then
        echo "ERROR: slurm.env missing: $_var" >&2
        exit 1
    fi
done

SLURM_QOS="${SLURM_QOS:-}"
_RYSY_QOS_FLAG=""
[[ -n "$SLURM_QOS" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
TS="$(date +%Y%m%d_%H%M%S)"
HASH="$(sha256sum "$CONFIG" | cut -c1-8)"
RUN_DIR="${RUNS_BASE_DIR}/${TS}_${HASH}_mnist_spacetime_v1"
LOGS_DIR="$RUN_DIR/logs"

echo "  Config    : $CONFIG"
echo "  slurm.env : $SLURM_ENV"
echo "  Run dir   : $RUN_DIR"
echo "  GPU       : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
[[ -n "$SLURM_QOS" ]] && echo "  QoS       : $SLURM_QOS"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  [1] scenario campaign  GPU singleton"
    echo "  [2] finalize           CPU singleton afterok:GPU"
    echo ""
    echo "  No Python is executed on the access node."
    exit 0
fi

mkdir -p "$RUN_DIR" "$LOGS_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"

_rysy_sbatch_gpu() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1" "$@"
}

_rysy_sbatch_cpu() {
    sbatch --parsable --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" "$@"
}

_dependency_id() { printf "%s" "${1%%;*}"; }

_wait_rysy() {
    local job_id="$1" label="$2"
    echo "  Polling for $label (job $job_id) ..."
    while squeue -j "$job_id" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
    local states
    states=$(sacct -j "$job_id" --format=State --noheader --parsable2 2>/dev/null \
        | grep -v '^$' | sort -u | tr '\n' ' ' | xargs)
    echo "  $label states: $states"
    if echo "$states" | grep -qE '(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)'; then
        echo "ERROR: $label failed. States: $states" >&2
        return 1
    fi
    if ! echo "$states" | grep -q 'COMPLETED'; then
        echo "ERROR: $label did not complete. States: $states" >&2
        return 1
    fi
}

_print_slurm_diagnostics() {
    local jobs_csv="$1"
    echo " Status  : sacct -j ${jobs_csv} \\"
    echo "           --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList,ReqTRES%50,AllocTRES%70,MaxRSS"
    echo " Queue   : squeue -j ${jobs_csv} -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
    echo " Starts  : squeue --start -j ${jobs_csv}"
    echo " Failures: grep -RniE \"error|exception|traceback|cuda|cupy|oom|timeout|killed|scratch|nvme|no space|permission|failed\" \"$LOGS_DIR\" | tail -300"
}

echo "[1] scenario campaign ..."
GPU_JOB=$(_rysy_sbatch_gpu \
    --job-name=mnist_space_gpu \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
    --cpus-per-task="$SLURM_CPUS" --time="$SLURM_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/campaign_%j.out" --error="${LOGS_DIR}/campaign_%j.err" \
    "$CLUSTER_DIR/job_gpu.sh" \
    python3 -m mnist_spacetime_v1.stages.gpu.run_campaign --run-dir "$RUN_DIR")
echo "  [1] campaign -> $GPU_JOB"
GPU_DEP="$(_dependency_id "$GPU_JOB")"

echo "[2] finalize ..."
FIN_JOB=$(_rysy_sbatch_cpu \
    --job-name=mnist_space_finalize \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$FINALIZE_MEM" --cpus-per-task="$FINALIZE_CPUS" --time="$FINALIZE_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/finalize_%j.out" --error="${LOGS_DIR}/finalize_%j.err" \
    --dependency="afterok:${GPU_DEP}" \
    "$CLUSTER_DIR/job_stage.sh" \
    python3 -m mnist_spacetime_v1.stages.cpu.finalize --run-dir "$RUN_DIR")
echo "  [2] finalize -> $FIN_JOB  (afterok:${GPU_DEP})"
FIN_DEP="$(_dependency_id "$FIN_JOB")"

echo ""
echo "========================================================"
echo " All jobs submitted."
echo "  Run dir : $RUN_DIR"
echo "  Logs    : $LOGS_DIR"
echo ""
_print_slurm_diagnostics "${GPU_DEP},${FIN_DEP}"
echo ""
echo " Results  : $RUN_DIR/results_summary_spacetime.json"
echo " CSV      : $RUN_DIR/summary_table.csv"
echo " Plots    : $RUN_DIR/plots/"
echo "========================================================"

if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo ""
    echo " [--wait] Polling finalize job $FIN_DEP ..."
    _wait_rysy "$FIN_DEP" "finalize" || exit 1
    echo " Run complete: $RUN_DIR/results_summary_spacetime.json"
fi
