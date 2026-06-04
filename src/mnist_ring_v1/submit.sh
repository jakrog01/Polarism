#!/usr/bin/env bash
# mnist_ring_v1 Slurm/Rysy orchestrator — access-node safe, zero Python/module
#
# Usage:
#   bash submit.sh [--config <cfg>] [--runs-dir <dir>]
#                  [--pilot] [--pilot-n N]
#                  [--pilot-mode fast|atlas|accuracy]
#                  [--pilot-train-per-class N] [--pilot-test-per-class N]
#                  [--dry-run] [--wait]
#
# Pilot presets (--pilot-mode):
#   fast     :  5+5/class  = 100 imgs
#   atlas    : 25+5/class  = 300 imgs  (default)
#   accuracy : 100+20/class = 1200 imgs
#
# Access-node guarantees:
#   - no module load
#   - no source .venv
#   - no python3 / numpy
#   - only: bash builtins, awk, sed, grep, sha256sum, curl/wget, mkdir, date
set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
WAIT_FOR_COMPLETION=0
CONFIG=""
RUNS_BASE_DIR=""
PILOT=0
PILOT_N=""
PILOT_MODE=""
PILOT_TRAIN_PER_CLASS=""
PILOT_TEST_PER_CLASS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)                    DRY_RUN=1;                              shift ;;
        --wait)                       WAIT_FOR_COMPLETION=1;                  shift ;;
        --pilot)                      PILOT=1;                                shift ;;
        --pilot-n)                    PILOT_N="$2";                           shift 2 ;;
        --pilot-n=*)                  PILOT_N="${1#--pilot-n=}";              shift ;;
        --pilot-mode)                 PILOT_MODE="$2";                        shift 2 ;;
        --pilot-mode=*)               PILOT_MODE="${1#--pilot-mode=}";        shift ;;
        --pilot-train-per-class)      PILOT_TRAIN_PER_CLASS="$2";            shift 2 ;;
        --pilot-train-per-class=*)    PILOT_TRAIN_PER_CLASS="${1#--pilot-train-per-class=}"; shift ;;
        --pilot-test-per-class)       PILOT_TEST_PER_CLASS="$2";             shift 2 ;;
        --pilot-test-per-class=*)     PILOT_TEST_PER_CLASS="${1#--pilot-test-per-class=}";  shift ;;
        --config)                     CONFIG="$2";                            shift 2 ;;
        --config=*)                   CONFIG="${1#--config=}";                shift ;;
        --runs-dir)                   RUNS_BASE_DIR="$2";                     shift 2 ;;
        --runs-dir=*)                 RUNS_BASE_DIR="${1#--runs-dir=}";       shift ;;
        *) [[ -z "$CONFIG" ]] && CONFIG="$1"; shift ;;
    esac
done

# pilot mode implies --pilot flag
[[ -n "$PILOT_MODE" || -n "$PILOT_TRAIN_PER_CLASS" || -n "$PILOT_TEST_PER_CLASS" ]] && PILOT=1

# ── Paths (no Python, no module) ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MNIST_RING_DIR="$SCRIPT_DIR"
CLUSTER_DIR="$SCRIPT_DIR/cluster"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

echo "========================================================"
echo " mnist_ring_v1  |  MNIST Polariton Classifier"
echo "========================================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2
    echo "  (same slurm.env used by polariton_hpc_pipeline)" >&2
    exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: submit.sh must be run on a Rysy login/access node." >&2
    echo "  Current host: $HOSTNAME_SHORT" >&2
    echo "  Use --dry-run to test locally." >&2
    exit 1
fi
[[ "$HOSTNAME_SHORT" != rysy* ]] && \
    echo "  [DRY RUN] Not on Rysy (host: $HOSTNAME_SHORT) — Slurm calls skipped."

# ── slurm.env: load + validate ────────────────────────────────────────────────
source "$SLURM_ENV"

for _var in SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME SLURM_MEM SLURM_GPUS SLURM_CPUS \
            NVME_GB TETYDA_RUNS_BASE MAX_CONCURRENT_SCENARIOS \
            FINALIZE_MEM FINALIZE_CPUS FINALIZE_TIME; do
    if [[ -z "${!_var:-}" ]]; then
        echo "ERROR: slurm.env missing: $_var" >&2; exit 1
    fi
done
SLURM_QOS="${SLURM_QOS:-}"

# ── Shell YAML scalar reader ───────────────────────────────────────────────────
# Reads the first occurrence of "  key: value" in a YAML file.
# Strips leading/trailing whitespace and inline comments; no eval, no Python.
_yaml_scalar() {
    local file="$1" key="$2"
    grep -m1 "^[[:space:]]*${key}[[:space:]]*:" "$file" \
        | sed -E "s/^[^:]+:[[:space:]]*//" \
        | sed -E "s/[[:space:]]*#.*//"     \
        | sed -E "s/^['\"]|['\"]$//"       \
        | xargs
}

# ── MNIST dataset: path + check + optional download ──────────────────────────
_MNIST_PATH_RAW="$(_yaml_scalar "$CONFIG" "data_path")"
# Expand leading ~ to $HOME without eval
_MNIST_PATH="${_MNIST_PATH_RAW/#\~/$HOME}"
_MNIST_URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_MNIST_MIN_BYTES=10000000  # 10 MB — full MNIST npz is ~11 MB

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
        echo "ERROR: neither curl nor wget available on this node." >&2
        echo "  Download manually: curl -L -o $_MNIST_PATH $_MNIST_URL" >&2
        exit 1
    fi
    mv "$_MNIST_TMP" "$_MNIST_PATH"
    echo "  MNIST ready: $_MNIST_PATH"
fi

# ── N_BATCHES (shellside integer arithmetic, no Python) ───────────────────────
_BATCH_SIZE="$(_yaml_scalar "$CONFIG" "batch_size")"

# Mirror the priority logic from prepare_run.py:
#  1. explicit per-class args
#  2. --pilot-n without --pilot-mode
#  3. --pilot-mode preset
#  4. config n_train+n_test
if [[ $PILOT -eq 1 ]]; then
    if [[ -n "$PILOT_TRAIN_PER_CLASS" || -n "$PILOT_TEST_PER_CLASS" ]]; then
        _TR_PC="${PILOT_TRAIN_PER_CLASS:-25}"
        _TE_PC="${PILOT_TEST_PER_CLASS:-0}"
        N_TOTAL=$(( (_TR_PC + _TE_PC) * 10 ))
    elif [[ -n "$PILOT_N" && -z "$PILOT_MODE" ]]; then
        N_TOTAL=$PILOT_N
    else
        case "${PILOT_MODE:-atlas}" in
            fast)     N_TOTAL=100  ;;   #  5+5  /class
            atlas)    N_TOTAL=300  ;;   # 25+5  /class
            accuracy) N_TOTAL=1200 ;;   # 100+20/class
            *)        N_TOTAL=300  ;;
        esac
    fi
else
    _N_TRAIN="$(_yaml_scalar "$CONFIG" "n_train")"
    _N_TEST="$(_yaml_scalar  "$CONFIG" "n_test")"
    N_TOTAL=$(( _N_TRAIN + _N_TEST ))
fi

N_BATCHES=$(( (N_TOTAL + _BATCH_SIZE - 1) / _BATCH_SIZE ))
BATCH_LAST_INDEX=$(( N_BATCHES - 1 ))

# Rysy/Slurm executes the max array index as a parent placeholder without
# per-task NVMe scratch. Submit that index as a separate singleton below.
export POLARITON_ARRAY_DUMMY_INDEX="$BATCH_LAST_INDEX"
if [[ "$N_BATCHES" -gt 1 ]]; then
    ARRAY_ENABLED=1
    ARRAY_SPEC="0-${BATCH_LAST_INDEX}%${MAX_CONCURRENT_SCENARIOS}"
else
    ARRAY_ENABLED=0
    ARRAY_SPEC=""
fi

# ── Run dir (name only; mkdir happens after dry-run check) ────────────────────
RUNS_BASE_DIR="${RUNS_BASE_DIR:-$TETYDA_RUNS_BASE}"
TS="$(date +%Y%m%d_%H%M%S)"
HASH="$(sha256sum "$CONFIG" | cut -c1-8)"
RUN_DIR="${RUNS_BASE_DIR}/${TS}_${HASH}_mnist_ring_v1"
LOGS_DIR="$RUN_DIR/logs"

# ── QoS flag ──────────────────────────────────────────────────────────────────
_RYSY_QOS_FLAG=""
[[ -n "$SLURM_QOS" ]] && _RYSY_QOS_FLAG="--qos=${SLURM_QOS}"

echo ""
echo "  Config       : $CONFIG"
echo "  slurm.env    : $SLURM_ENV"
echo "  Runs base    : $RUNS_BASE_DIR"
echo "  Run dir      : $RUN_DIR"
echo "  Host         : $HOSTNAME_SHORT"
echo "  GPU          : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
[[ -n "$SLURM_QOS" ]] && echo "  QoS          : $SLURM_QOS"
if [[ $PILOT -eq 1 ]]; then
    _PILOT_DESC="PILOT"
    [[ -n "$PILOT_MODE" ]]             && _PILOT_DESC="${_PILOT_DESC}/${PILOT_MODE}"
    [[ -n "$PILOT_N" && -z "$PILOT_MODE" ]] && _PILOT_DESC="${_PILOT_DESC}/pilot-n=${PILOT_N}"
    [[ -n "$PILOT_TRAIN_PER_CLASS" ]]  && _PILOT_DESC="${_PILOT_DESC}/tr=${PILOT_TRAIN_PER_CLASS}/class"
    [[ -n "$PILOT_TEST_PER_CLASS" ]]   && _PILOT_DESC="${_PILOT_DESC}+te=${PILOT_TEST_PER_CLASS}/class"
    echo "  Mode         : ${_PILOT_DESC}"
fi
echo "  N_TOTAL      : $N_TOTAL  batch_size=$_BATCH_SIZE"
echo "  N_BATCHES    : $N_BATCHES  BATCH_LAST=${BATCH_LAST_INDEX}"
[[ $ARRAY_ENABLED -eq 1 ]] && echo "  Array spec   : $ARRAY_SPEC  (task $BATCH_LAST_INDEX = dummy)"

# ── Dry-run: print plan and exit (zero Python, zero Slurm) ────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    [[ $_need_download -eq 1 ]] && echo "  MNIST        : MISSING — would download before first sbatch"
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  [0] prepare_run  CPU singleton            afterok: (none)"
    echo "  [1] calibration  GPU singleton            afterok: PREPARE"
    if [[ $ARRAY_ENABLED -eq 1 ]]; then
        echo "  [2a] batch array GPU  array=$ARRAY_SPEC  afterok: PREPARE+CALIB"
    else
        echo "  [2a] batch array skipped (N_BATCHES=1)"
    fi
    echo "  [2b] batch_last  GPU singleton idx=$BATCH_LAST_INDEX  afterok: PREPARE+CALIB"
    echo "  [3]  finalize    CPU singleton            afterok: ARRAY+LAST (or just LAST)"
    echo ""
    echo "  NVMe: --gres=gpu:${SLURM_GPUS},nvme:${NVME_GB}  (GPU jobs)"
    echo "  QoS:  ${_RYSY_QOS_FLAG:-(none)}"
    echo "  Logs: $LOGS_DIR"
    echo ""
    # Empirical: V100 1024² rk4-cuda 200ps → ~26 s/img  (pilot run 2026-06-04)
    _SECS_PER_IMG=26
    _BATCH_WALL=$(( _BATCH_SIZE * _SECS_PER_IMG ))
    _N_WAVES=$(( (N_BATCHES + MAX_CONCURRENT_SCENARIOS - 1) / MAX_CONCURRENT_SCENARIOS ))
    _SWEEP_IMGS=21   # ~8 ring + ~7 trig + ~6 assisted pts
    _GUARD_IMGS=120  # 6 fractions × 20 guard samples; stop_time T_max+35 ≈ 135ps → ~18s/img
    _GUARD_WALL=$(( _GUARD_IMGS * 18 ))
    echo "  Walltime estimate at MAX_CONCURRENT_SCENARIOS=${MAX_CONCURRENT_SCENARIOS}:"
    echo "    prepare  : ~5 min  (PCA + dataset index, CPU)"
    echo "    calib    : sweep ${_SWEEP_IMGS} pts × ~${_SECS_PER_IMG} s ≈ $(( _SWEEP_IMGS * _SECS_PER_IMG / 60 )) min"
    echo "               + guard ${_GUARD_IMGS} runs × ~18 s ≈ $(( _GUARD_WALL / 60 )) min"
    echo "               calib total ≈ $(( (_SWEEP_IMGS * _SECS_PER_IMG + _GUARD_WALL) / 60 )) min"
    echo "    batches  : N_BATCHES=$N_BATCHES  ${_N_WAVES} waves × $_BATCH_SIZE imgs × ~${_SECS_PER_IMG} s"
    echo "               each batch ≈ $(( _BATCH_WALL / 60 )) min  |  total batches ≈ $(( _N_WAVES * _BATCH_WALL / 60 )) min"
    echo "    time_limit per job: $SLURM_TIME"
    exit 0
fi

# ── Create run dir (only for real submit) ────────────────────────────────────
mkdir -p "$RUN_DIR" "$LOGS_DIR"

# ── sbatch helpers ────────────────────────────────────────────────────────────
_rysy_sbatch_gpu() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1,POLARITON_ARRAY_DUMMY_MAX=1" \
        "$@"
}

_rysy_sbatch_cpu() {
    sbatch --parsable \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
        "$@"
}

_dependency_id() { printf "%s" "${1%%;*}"; }

_wait_rysy() {
    local job_id="$1" label="$2"
    echo "  Polling for $label (job $job_id) ..."
    while squeue -j "$job_id" --noheader 2>/dev/null | grep -q .; do sleep 60; done
    local states
    states=$(sacct -j "$job_id" --format=State --noheader --parsable2 2>/dev/null \
        | grep -v '^$' | sort -u | tr '\n' ' ' | xargs)
    echo "  $label states: $states"
    if echo "$states" | grep -qE '(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)'; then
        echo "ERROR: $label failed. States: $states" >&2; return 1
    fi
    if ! echo "$states" | grep -q 'COMPLETED'; then
        echo "ERROR: $label did not complete. States: $states" >&2; return 1
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

# ── Build prepare flags ───────────────────────────────────────────────────────
_PREPARE_FLAGS=""
if [[ $PILOT -eq 1 ]]; then
    _PREPARE_FLAGS="--pilot"
    [[ -n "$PILOT_N"                ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-n ${PILOT_N}"
    [[ -n "$PILOT_MODE"             ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-mode ${PILOT_MODE}"
    [[ -n "$PILOT_TRAIN_PER_CLASS"  ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-train-per-class ${PILOT_TRAIN_PER_CLASS}"
    [[ -n "$PILOT_TEST_PER_CLASS"   ]] && _PREPARE_FLAGS="${_PREPARE_FLAGS} --pilot-test-per-class ${PILOT_TEST_PER_CLASS}"
fi

# ── [0] prepare_run (CPU Slurm job, no dependency) ───────────────────────────
echo "[0] prepare_run (CPU job) ..."
PREPARE_JOB=$(_rysy_sbatch_cpu \
    --job-name=mnist_prepare \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --mem="$FINALIZE_MEM" \
    --cpus-per-task="$FINALIZE_CPUS" \
    --time="$FINALIZE_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/prepare_%j.out" \
    --error="${LOGS_DIR}/prepare_%j.err" \
    "$CLUSTER_DIR/job_stage.sh" \
    python3 -m mnist_ring_v1.config.prepare_run \
        --config "$CONFIG" --run-dir "$RUN_DIR" ${_PREPARE_FLAGS})
echo "  [0] prepare_run  -> Rysy job $PREPARE_JOB"
PREPARE_DEP="$(_dependency_id "$PREPARE_JOB")"

# ── [1] Calibration (GPU, afterok:PREPARE) ────────────────────────────────────
echo "[1] calibration (GPU singleton) ..."
CALIB_JOB=$(_rysy_sbatch_gpu \
    --job-name=mnist_calib \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --mem="$SLURM_MEM" \
    --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
    --cpus-per-task="$SLURM_CPUS" \
    --time="$SLURM_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/calib_%j.out" \
    --error="${LOGS_DIR}/calib_%j.err" \
    --dependency="afterok:${PREPARE_DEP}" \
    "$CLUSTER_DIR/job_gpu.sh" \
    python3 -m mnist_ring_v1.stages.gpu.calibrate \
        --run-dir "$RUN_DIR" --mode all)
echo "  [1] calibration  -> Rysy job $CALIB_JOB  (afterok:${PREPARE_DEP})"
CALIB_DEP="$(_dependency_id "$CALIB_JOB")"

# ── [2a] GPU batch array (afterok:PREPARE:CALIB) ─────────────────────────────
ARRAY_JOB=""
if [[ $ARRAY_ENABLED -eq 1 ]]; then
    echo "[2a] GPU batch array ($ARRAY_SPEC, task $BATCH_LAST_INDEX = dummy) ..."
    ARRAY_JOB=$(_rysy_sbatch_gpu \
        --job-name=mnist_batch \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --mem="$SLURM_MEM" \
        --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$SLURM_TIME" \
        --array="${ARRAY_SPEC}" \
        ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
        --output="${LOGS_DIR}/batch_%A_%a.out" \
        --error="${LOGS_DIR}/batch_%A_%a.err" \
        --dependency="afterok:${PREPARE_DEP}:${CALIB_DEP}" \
        "$CLUSTER_DIR/job_gpu.sh" \
        python3 -m mnist_ring_v1.stages.gpu.run_batch \
            --run-dir "$RUN_DIR")
    echo "  [2a] batch array -> Rysy job $ARRAY_JOB  (afterok:${PREPARE_DEP}:${CALIB_DEP})"
else
    echo "[2a] batch array: skipped (N_BATCHES=1)"
fi

# ── [2b] batch_last singleton (afterok:PREPARE:CALIB) ────────────────────────
echo "[2b] batch_last singleton (index=$BATCH_LAST_INDEX) ..."
LAST_JOB=$(_rysy_sbatch_gpu \
    --job-name=mnist_batch_last \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --mem="$SLURM_MEM" \
    --gres="gpu:${SLURM_GPUS},nvme:${NVME_GB}" \
    --cpus-per-task="$SLURM_CPUS" \
    --time="$SLURM_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/batch_last_%j.out" \
    --error="${LOGS_DIR}/batch_last_%j.err" \
    --dependency="afterok:${PREPARE_DEP}:${CALIB_DEP}" \
    "$CLUSTER_DIR/job_gpu.sh" \
    python3 -m mnist_ring_v1.stages.gpu.run_batch \
        --run-dir "$RUN_DIR" --batch-index "$BATCH_LAST_INDEX")
echo "  [2b] batch_last  -> Rysy job $LAST_JOB  (afterok:${PREPARE_DEP}:${CALIB_DEP})"

# ── [3] Finalize (CPU, afterok:ARRAY+LAST or afterok:LAST) ───────────────────
if [[ $ARRAY_ENABLED -eq 1 ]]; then
    _FIN_DEP="afterok:$(_dependency_id "$ARRAY_JOB"):$(_dependency_id "$LAST_JOB")"
else
    _FIN_DEP="afterok:$(_dependency_id "$LAST_JOB")"
fi

echo "[3] finalize (CPU, partition=$SLURM_PARTITION) ..."
FIN_JOB=$(_rysy_sbatch_cpu \
    --job-name=mnist_finalize \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --mem="$FINALIZE_MEM" \
    --cpus-per-task="$FINALIZE_CPUS" \
    --time="$FINALIZE_TIME" \
    ${_RYSY_QOS_FLAG:+"$_RYSY_QOS_FLAG"} \
    --output="${LOGS_DIR}/finalize_%j.out" \
    --error="${LOGS_DIR}/finalize_%j.err" \
    --dependency="$_FIN_DEP" \
    "$CLUSTER_DIR/job_stage.sh" \
    python3 -m mnist_ring_v1.stages.cpu.finalize \
        --run-dir "$RUN_DIR")
echo "  [3] finalize     -> Rysy job $FIN_JOB  (${_FIN_DEP})"

# ── Summary ───────────────────────────────────────────────────────────────────
_ALL_JOBS="${PREPARE_JOB},${CALIB_JOB}${ARRAY_JOB:+,$ARRAY_JOB},${LAST_JOB},${FIN_JOB}"

echo ""
echo "========================================================"
echo " All jobs submitted."
echo "  Run dir   : $RUN_DIR"
echo "  Logs      : $LOGS_DIR"
echo "  N batches : $N_BATCHES  (LAST=$BATCH_LAST_INDEX)"
echo ""
_print_slurm_diagnostics "$_ALL_JOBS"
echo ""
echo " Results   : $RUN_DIR/results_summary.json"
echo " Plots     : $RUN_DIR/plots/"
echo "========================================================"

# --wait: submit entire DAG first (done above), then poll only final job
if [[ $WAIT_FOR_COMPLETION -eq 1 ]]; then
    echo ""
    echo " [--wait] Polling until finalize completes ..."
    _wait_rysy "$FIN_JOB" "finalize" || exit 1
    echo " Run complete: $RUN_DIR/results_summary.json"
fi
