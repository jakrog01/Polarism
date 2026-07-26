#!/usr/bin/env bash
# mnist_digits_polariton_snn_dynamic Slurm/Rysy orchestrator - access-node safe.
#
# Usage:
#   bash submit.sh [--config <cfg>] [--runs-dir <dir>] [--dry-run] [--wait]
#
# Submits three sequential GPU jobs that run the full mnist_digits_polariton_snn_dynamic pipeline
# end-to-end (encode -> reservoir simulation -> feature collection -> linear readout).
set -euo pipefail

DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""
CHAIN_RUNS=3
CHAIN_TIME="48:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --wait)         WAIT_FOR_COMPLETION=1; shift ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --config=*)     CONFIG="${1#--config=}"; shift ;;
        --runs-dir)     RUNS_BASE_DIR="$2"; shift 2 ;;
        --runs-dir=*)   RUNS_BASE_DIR="${1#--runs-dir=}"; shift ;;
        --chain-runs)   CHAIN_RUNS="$2"; shift 2 ;;
        --chain-runs=*) CHAIN_RUNS="${1#--chain-runs=}"; shift ;;
        --chain-time)   CHAIN_TIME="$2"; shift 2 ;;
        --chain-time=*) CHAIN_TIME="${1#--chain-time=}"; shift ;;
        *)              [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
CLUSTER_DIR="$SCRIPT_DIR/cluster"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

echo "========================================================"
echo " mnist_digits_polariton_snn_dynamic  |  polariton reservoir MNIST classifier"
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
            NVME_GB TETYDA_RUNS_BASE; do
    if [[ -z "${!_var:-}" ]]; then
        echo "ERROR: slurm.env missing: $_var" >&2
        exit 1
    fi
done

SLURM_QOS="${SLURM_QOS:-}"
_RYSY_QOS_FLAG=""
[[ -n "$SLURM_QOS" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

if ! [[ "$CHAIN_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --chain-runs must be a positive integer, got $CHAIN_RUNS" >&2
    exit 1
fi

_yaml_scalar() {
    local file="$1" key="$2"
    grep -m1 "^[[:space:]]*${key}[[:space:]]*:" "$file" \
        | sed -E "s/^[^:]+:[[:space:]]*//" \
        | sed -E "s/[[:space:]]*#.*//"     \
        | sed -E "s/^['\"]//; s/['\"]\$//" \
        | xargs
}

_MNIST_PATH_RAW="$(_yaml_scalar "$CONFIG" "path")"
_MNIST_PATH="${_MNIST_PATH_RAW/#\~/$HOME}"
_MNIST_URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_MNIST_MIN_BYTES=10000000

_need_download=0
if [[ -z "$_MNIST_PATH" ]]; then
    echo "WARN: could not parse data.path from $CONFIG; skipping MNIST prefetch."
elif [[ ! -f "$_MNIST_PATH" ]]; then
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

RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
TS="$(date +%Y%m%d_%H%M%S)"
HASH="$(sha256sum "$CONFIG" | cut -c1-8)"
RUN_DIR="${RUNS_BASE_DIR}/${TS}_${HASH}_mnist_digits_polariton_snn_dynamic"
LOGS_DIR="$RUN_DIR/logs"

echo "  Config    : $CONFIG"
echo "  slurm.env : $SLURM_ENV"
echo "  Run dir   : $RUN_DIR"
echo "  Chain     : ${CHAIN_RUNS} sequential GPU runs, time=${CHAIN_TIME} each"
echo "  GPU       : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
[[ -n "$SLURM_QOS" ]] && echo "  QoS       : $SLURM_QOS"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    for ((i = 1; i <= CHAIN_RUNS; i++)); do
        if [[ $i -eq 1 ]]; then
            echo "  [$i] mnist_digits_polariton_snn_dynamic pipeline  GPU  time=${CHAIN_TIME}"
        else
            echo "  [$i] mnist_digits_polariton_snn_dynamic pipeline  GPU  time=${CHAIN_TIME}  dependency=afterok:job[$((i - 1))]"
        fi
    done
    echo ""
    echo "  No Python is executed on the access node."
    exit 0
fi

mkdir -p "$RUN_DIR" "$LOGS_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"

_POLARISM_CFG_RAW="$(_yaml_scalar "$CONFIG" "polarism_config_path")"
if [[ -n "$_POLARISM_CFG_RAW" ]]; then
    if [[ "$_POLARISM_CFG_RAW" == /* ]]; then
        _POLARISM_CFG="$_POLARISM_CFG_RAW"
    else
        _POLARISM_CFG="$(cd "$(dirname "$CONFIG")" && cd "$(dirname "$_POLARISM_CFG_RAW")" && pwd)/$(basename "$_POLARISM_CFG_RAW")"
    fi
    [[ -f "$_POLARISM_CFG" ]] && cp "$_POLARISM_CFG" "$RUN_DIR/polarism_base.yaml"
fi

_rysy_sbatch_gpu() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1" "$@"
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

GPU_DEPS=()
PREV_DEP=""
for ((i = 1; i <= CHAIN_RUNS; i++)); do
    RUN_INDEX="$(printf "%02d" "$i")"
    RUN_OUTPUT_DIR="${RUN_DIR}/run_${RUN_INDEX}"
    mkdir -p "$RUN_OUTPUT_DIR"
    DEP_ARGS=()
    if [[ -n "$PREV_DEP" ]]; then
        DEP_ARGS=(--dependency="afterok:${PREV_DEP}")
    fi
    echo "[$i] mnist_digits_polariton_snn_dynamic pipeline ..."
    GPU_JOB=$(_rysy_sbatch_gpu \
        --job-name="mnist_digits_polariton_snn_dynamic_${RUN_INDEX}" \
        --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
        --cpus-per-task="$SLURM_CPUS" --time="$CHAIN_TIME" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} \
        --output="${LOGS_DIR}/pipeline_${RUN_INDEX}_%j.out" \
        --error="${LOGS_DIR}/pipeline_${RUN_INDEX}_%j.err" \
        "$CLUSTER_DIR/job_gpu.sh" \
        python3 -m mnist_digits_polariton_snn_dynamic.pipeline --config "$CONFIG" --output-dir "$RUN_OUTPUT_DIR")
    GPU_DEP="$(_dependency_id "$GPU_JOB")"
    GPU_DEPS+=("$GPU_DEP")
    PREV_DEP="$GPU_DEP"
    echo "  [$i] pipeline -> $GPU_JOB"
done
JOBS_CSV="$(IFS=,; printf "%s" "${GPU_DEPS[*]}")"

echo ""
echo "========================================================"
echo " Job submitted."
echo "  Run dir : $RUN_DIR"
echo "  Logs    : $LOGS_DIR"
echo ""
_print_slurm_diagnostics "${JOBS_CSV}"
echo ""
for ((i = 1; i <= CHAIN_RUNS; i++)); do
    RUN_INDEX="$(printf "%02d" "$i")"
    echo " Run $RUN_INDEX results  : ${RUN_DIR}/run_${RUN_INDEX}/report.json"
    echo " Run $RUN_INDEX features : ${RUN_DIR}/run_${RUN_INDEX}/features.npy"
    echo " Run $RUN_INDEX labels   : ${RUN_DIR}/run_${RUN_INDEX}/labels.npy"
    echo " Run $RUN_INDEX metadata : ${RUN_DIR}/run_${RUN_INDEX}/run_metadata.json"
done
echo "========================================================"

if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo ""
    for ((i = 0; i < ${#GPU_DEPS[@]}; i++)); do
        label="pipeline run $(printf "%02d" "$((i + 1))")"
        echo " [--wait] Polling ${label} job ${GPU_DEPS[$i]} ..."
        _wait_rysy "${GPU_DEPS[$i]}" "$label" || exit 1
    done
    echo " Run chain complete: $RUN_DIR"
fi
