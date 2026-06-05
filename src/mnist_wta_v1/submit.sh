#!/usr/bin/env bash
# mnist_wta_v1 Slurm/Rysy orchestrator — access-node safe.
#
# Usage:
#   bash submit.sh [--config <cfg>] [--runs-dir <dir>] [--radius R]
#                  [--pilot] [--pilot-train-per-class N] [--pilot-test-per-class N]
#                  [--dry-run] [--wait]
#
# Sweep geometry: sweeps over ring_radius_um values specified in config.yaml.
# For a single run, pass --radius to fix the radius.
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""
PILOT=0
PILOT_TRAIN_PER_CLASS=""
PILOT_TEST_PER_CLASS=""
RADIUS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                  DRY_RUN=1;                              shift ;;
        --wait)                     WAIT_FOR_COMPLETION=1;                  shift ;;
        --pilot)                    PILOT=1;                                shift ;;
        --pilot-train-per-class)    PILOT_TRAIN_PER_CLASS="$2";            shift 2 ;;
        --pilot-train-per-class=*)  PILOT_TRAIN_PER_CLASS="${1#*=}";       shift ;;
        --pilot-test-per-class)     PILOT_TEST_PER_CLASS="$2";             shift 2 ;;
        --pilot-test-per-class=*)   PILOT_TEST_PER_CLASS="${1#*=}";        shift ;;
        --config)                   CONFIG="$2";                            shift 2 ;;
        --config=*)                 CONFIG="${1#--config=}";                shift ;;
        --runs-dir)                 RUNS_BASE_DIR="$2";                     shift 2 ;;
        --runs-dir=*)               RUNS_BASE_DIR="${1#--runs-dir=}";       shift ;;
        --radius)                   RADIUS_OVERRIDE="$2";                   shift 2 ;;
        --radius=*)                 RADIUS_OVERRIDE="${1#--radius=}";       shift ;;
        *) [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

[[ -n "$PILOT_TRAIN_PER_CLASS" || -n "$PILOT_TEST_PER_CLASS" ]] && PILOT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MNIST_WTA_DIR="$SCRIPT_DIR"
CLUSTER_DIR="$SCRIPT_DIR/cluster"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

echo "========================================================"
echo " mnist_wta_v1  |  Polariton WTA Classifier"
echo "========================================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2; exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: submit.sh must be run on a Rysy login/access node." >&2
    echo "  Use --dry-run to test locally." >&2
    exit 1
fi
[[ "$HOSTNAME_SHORT" != rysy* ]] && echo "  [DRY RUN] Not on Rysy — Slurm calls skipped."

source "$SLURM_ENV"

for _var in SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME SLURM_MEM SLURM_GPUS SLURM_CPUS \
            NVME_GB TETYDA_RUNS_BASE MAX_CONCURRENT_SCENARIOS \
            FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME; do
    if [[ -z "${!_var:-}" ]]; then
        echo "ERROR: slurm.env missing: $_var" >&2; exit 1
    fi
done
SLURM_QOS="${SLURM_QOS:-}"
_RYSY_QOS_FLAG=""
[[ -n "$SLURM_QOS" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

_yaml_scalar() {
    local file="$1" key="$2"
    grep -m1 "^[[:space:]]*${key}[[:space:]]*:" "$file" \
        | sed -E "s/^[^:]+:[[:space:]]*//" \
        | sed -E "s/[[:space:]]*#.*//"     \
        | sed -E "s/^['\"]|['\"]$//"       \
        | xargs
}

# ── MNIST dataset: path + check + optional download ─────────────────────────
_MNIST_PATH_RAW="$(_yaml_scalar "$CONFIG" "data_path")"
_MNIST_PATH="${_MNIST_PATH_RAW/#\~/$HOME}"
_MNIST_URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_MNIST_MIN_BYTES=10000000

_need_download=0
if [[ ! -f "$_MNIST_PATH" ]]; then
    _need_download=1
    echo "  MNIST not found: $_MNIST_PATH"
else
    _fsize="$(stat -c%s "$_MNIST_PATH" 2>/dev/null || echo 0)"
    if [[ "$_fsize" -lt "$_MNIST_MIN_BYTES" ]]; then
        _need_download=1
        echo "  MNIST too small (${_fsize}B < 10MB): $_MNIST_PATH"
    else
        echo "  MNIST OK: $_MNIST_PATH  (${_fsize}B)"
    fi
fi

if [[ $_need_download -eq 1 && $DRY_RUN -eq 0 ]]; then
    echo "  Downloading MNIST from $_MNIST_URL ..."
    mkdir -p "$(dirname "$_MNIST_PATH")"
    _MNIST_TMP="${_MNIST_PATH}.download_$$"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "$_MNIST_TMP" "$_MNIST_URL" \
            || { echo "ERROR: curl download failed" >&2; rm -f "$_MNIST_TMP"; exit 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$_MNIST_TMP" "$_MNIST_URL" \
            || { echo "ERROR: wget download failed" >&2; rm -f "$_MNIST_TMP"; exit 1; }
    else
        echo "ERROR: neither curl nor wget available." >&2
        echo "  Download manually: curl -L -o $_MNIST_PATH $_MNIST_URL" >&2
        exit 1
    fi
    mv "$_MNIST_TMP" "$_MNIST_PATH"
    echo "  MNIST ready: $_MNIST_PATH"
fi

_BATCH_SIZE="$(_yaml_scalar "$CONFIG" "batch_size")"

if [[ $PILOT -eq 1 ]]; then
    _TR_PC="${PILOT_TRAIN_PER_CLASS:-25}"
    _TE_PC="${PILOT_TEST_PER_CLASS:-5}"
    N_TOTAL=$(( (_TR_PC + _TE_PC) * 10 ))
else
    _N_TRAIN="$(_yaml_scalar "$CONFIG" "n_train")"
    _N_TEST="$(_yaml_scalar  "$CONFIG" "n_test")"
    N_TOTAL=$(( _N_TRAIN + _N_TEST ))
fi

N_BATCHES=$(( (N_TOTAL + _BATCH_SIZE - 1) / _BATCH_SIZE ))
BATCH_LAST_INDEX=$(( N_BATCHES - 1 ))
export POLARITON_ARRAY_DUMMY_INDEX="$BATCH_LAST_INDEX"

if [[ "$N_BATCHES" -gt 1 ]]; then
    ARRAY_ENABLED=1
    ARRAY_SPEC="0-${BATCH_LAST_INDEX}%${MAX_CONCURRENT_SCENARIOS}"
else
    ARRAY_ENABLED=0
    ARRAY_SPEC=""
fi

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
TS="$(date +%Y%m%d_%H%M%S)"
HASH="$(sha256sum "$CONFIG" | cut -c1-8)"
RADIUS_TAG="${RADIUS_OVERRIDE:+_R${RADIUS_OVERRIDE}}"
RUN_DIR="${RUNS_BASE_DIR}/${TS}_${HASH}_mnist_wta_v1${RADIUS_TAG}"
LOGS_DIR="$RUN_DIR/logs"

echo "  Config      : $CONFIG"
echo "  slurm.env   : $SLURM_ENV"
echo "  Run dir     : $RUN_DIR"
echo "  N_TOTAL     : $N_TOTAL  batch_size=$_BATCH_SIZE"
echo "  N_BATCHES   : $N_BATCHES  BATCH_LAST=${BATCH_LAST_INDEX}"
[[ -n "$RADIUS_OVERRIDE" ]] && echo "  Radius      : $RADIUS_OVERRIDE μm"

_PREPARE_FLAGS=""
if [[ $PILOT -eq 1 ]]; then
    _PREPARE_FLAGS="--pilot"
    [[ -n "$PILOT_TRAIN_PER_CLASS" ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-train-per-class ${PILOT_TRAIN_PER_CLASS}"
    [[ -n "$PILOT_TEST_PER_CLASS"  ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-test-per-class ${PILOT_TEST_PER_CLASS}"
fi
[[ -n "$RADIUS_OVERRIDE" ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --radius ${RADIUS_OVERRIDE}"

_RADIUS_FLAG=""
[[ -n "$RADIUS_OVERRIDE" ]] && _RADIUS_FLAG="--radius ${RADIUS_OVERRIDE}"

if [[ $DRY_RUN -eq 1 ]]; then
    [[ $_need_download -eq 1 ]] && echo "  MNIST        : MISSING — would download before first sbatch"
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  [0] prepare_run  CPU  (dataset index, PCA, score model)"
    echo "  [1] calibrate    GPU  (single-spot P_th + threshold_cond)  afterok:PREPARE"
    [[ $ARRAY_ENABLED -eq 1 ]] && \
        echo "  [2a] batch array GPU  array=$ARRAY_SPEC  afterok:PREPARE+CALIB"
    echo "  [2b] batch_last  GPU  idx=$BATCH_LAST_INDEX  afterok:PREPARE+CALIB"
    echo "  [3]  finalize    CPU  afterok:ARRAY+LAST"
    echo ""
    echo "  Slurm tasks: $N_BATCHES"
    exit 0
fi

mkdir -p "$RUN_DIR" "$LOGS_DIR"

_rysy_sbatch_gpu() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1,POLARITON_ARRAY_DUMMY_MAX=1" "$@"
}
_rysy_sbatch_cpu() {
    sbatch --parsable --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" "$@"
}
_dependency_id() { printf "%s" "${1%%;*}"; }

echo "[0] prepare_run ..."
PREPARE_JOB=$(_rysy_sbatch_cpu \
    --job-name=wta_prepare \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$FINALIZE_MEM" --cpus-per-task="$FINALIZE_CPUS" --time="$FINALIZE_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/prepare_%j.out" --error="${LOGS_DIR}/prepare_%j.err" \
    "$CLUSTER_DIR/job_stage.sh" \
    python3 -m mnist_wta_v1.stages.cpu.prepare_run \
        --config "$CONFIG" --run-dir "$RUN_DIR" ${_PREPARE_FLAGS})
echo "  [0] prepare_run -> $PREPARE_JOB"
PREPARE_DEP="$(_dependency_id "$PREPARE_JOB")"

echo "[1] calibrate_wta ..."
CALIB_JOB=$(_rysy_sbatch_gpu \
    --job-name=wta_calib \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
    --cpus-per-task="$SLURM_CPUS" --time="$SLURM_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/calib_%j.out" --error="${LOGS_DIR}/calib_%j.err" \
    --dependency="afterok:${PREPARE_DEP}" \
    "$CLUSTER_DIR/job_gpu.sh" \
    python3 -m mnist_wta_v1.stages.gpu.calibrate_wta \
        --run-dir "$RUN_DIR" ${_RADIUS_FLAG})
echo "  [1] calibrate -> $CALIB_JOB  (afterok:${PREPARE_DEP})"
CALIB_DEP="$(_dependency_id "$CALIB_JOB")"

ARRAY_JOB=""
if [[ $ARRAY_ENABLED -eq 1 ]]; then
    echo "[2a] GPU batch array ($ARRAY_SPEC) ..."
    ARRAY_JOB=$(_rysy_sbatch_gpu \
        --job-name=wta_batch \
        --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
        --cpus-per-task="$SLURM_CPUS" --time="$SLURM_TIME" \
        --array="${ARRAY_SPEC}" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --output="${LOGS_DIR}/batch_%A_%a.out" --error="${LOGS_DIR}/batch_%A_%a.err" \
        --dependency="afterok:${PREPARE_DEP}:${CALIB_DEP}" \
        "$CLUSTER_DIR/job_gpu.sh" \
        python3 -m mnist_wta_v1.stages.gpu.run_batch --run-dir "$RUN_DIR" ${_RADIUS_FLAG})
    echo "  [2a] array -> $ARRAY_JOB"
fi

echo "[2b] batch_last singleton ..."
LAST_JOB=$(_rysy_sbatch_gpu \
    --job-name=wta_batch_last \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
    --cpus-per-task="$SLURM_CPUS" --time="$SLURM_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/batch_last_%j.out" --error="${LOGS_DIR}/batch_last_%j.err" \
    --dependency="afterok:${PREPARE_DEP}:${CALIB_DEP}" \
    "$CLUSTER_DIR/job_gpu.sh" \
    python3 -m mnist_wta_v1.stages.gpu.run_batch --run-dir "$RUN_DIR" --batch-index "$BATCH_LAST_INDEX" ${_RADIUS_FLAG})
echo "  [2b] batch_last -> $LAST_JOB"

if [[ $ARRAY_ENABLED -eq 1 ]]; then
    _FIN_DEP="afterok:$(_dependency_id "$ARRAY_JOB"):$(_dependency_id "$LAST_JOB")"
else
    _FIN_DEP="afterok:$(_dependency_id "$LAST_JOB")"
fi

echo "[3] finalize ..."
FIN_JOB=$(_rysy_sbatch_cpu \
    --job-name=wta_finalize \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mem="$FINALIZE_MEM" --cpus-per-task="$FINALIZE_CPUS" --time="$FINALIZE_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/finalize_%j.out" --error="${LOGS_DIR}/finalize_%j.err" \
    --dependency="$_FIN_DEP" \
    "$CLUSTER_DIR/job_stage.sh" \
    python3 -m mnist_wta_v1.stages.cpu.finalize --run-dir "$RUN_DIR")
echo "  [3] finalize -> $FIN_JOB  (${_FIN_DEP})"

echo ""
echo "========================================================"
echo " All jobs submitted."
echo "  Run dir : $RUN_DIR"
echo "  Logs    : $LOGS_DIR"
echo " Results  : $RUN_DIR/results_summary_wta.json"
echo "========================================================"

if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo ""
    echo " [--wait] Polling finalize job $FIN_JOB ..."
    while squeue -j "$FIN_JOB" --noheader 2>/dev/null | grep -q .; do sleep 60; done
    echo " Run complete: $RUN_DIR/results_summary_wta.json"
fi
