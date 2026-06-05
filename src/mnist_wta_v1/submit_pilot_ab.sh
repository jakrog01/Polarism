#!/usr/bin/env bash
# Submit WTA pilot A (R=8) and B (R=15) sequentially on Rysy.
#
# Default sizes: 20 train/class + 10 test/class = 300 images per run.
# Override with: --train-per-class N  --test-per-class N
#
# Access-node safe: delegates MNIST check/download and Slurm submission to submit.sh.
#
# Usage:
#   bash submit_pilot_ab.sh [--train-per-class N] [--test-per-class N] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SH="$SCRIPT_DIR/submit.sh"

if [[ ! -f "$SUBMIT_SH" ]]; then
    echo "ERROR: submit.sh not found: $SUBMIT_SH" >&2
    exit 1
fi

TRAIN_PER_CLASS=20
TEST_PER_CLASS=10
DRY_RUN=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-per-class)      TRAIN_PER_CLASS="$2"; shift 2 ;;
        --train-per-class=*)    TRAIN_PER_CLASS="${1#*=}"; shift ;;
        --test-per-class)       TEST_PER_CLASS="$2";  shift 2 ;;
        --test-per-class=*)     TEST_PER_CLASS="${1#*=}";  shift ;;
        --dry-run)              DRY_RUN=1; PASSTHROUGH+=("$1"); shift ;;
        --wait)
            echo "ERROR: --wait is not supported by submit_pilot_ab.sh (would serialize A and B)." >&2
            echo "Monitor the two printed run directories independently." >&2
            exit 1
            ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

N_TOTAL=$(( (TRAIN_PER_CLASS + TEST_PER_CLASS) * 10 ))

_TMP_FILES=()
_cleanup() {
    local f
    for f in "${_TMP_FILES[@]}"; do
        [[ -f "$f" ]] && rm -f "$f"
    done
}
trap _cleanup EXIT

_dependency_id() { printf "%s" "${1%%;*}"; }

_extract_run_dir() {
    local file="$1" line
    line="$(grep -m1 "Run dir" "$file" || true)"
    [[ -z "$line" ]] && return 0
    printf "%s" "$line" | sed -E 's/.*:[[:space:]]*//' | xargs
}

_extract_job() {
    local file="$1" pattern="$2" line
    line="$(grep -m1 "$pattern" "$file" || true)"
    [[ -z "$line" ]] && return 0
    printf "%s" "$line" | sed -E 's/.*->[[:space:]]*([^[:space:]]+).*/\1/' | xargs
}

_jobs_from_log() {
    local file="$1" id jobs=()
    for pattern in "\\[0\\] prepare_run" "\\[1\\] calibrate" "\\[2a\\] array" "\\[2b\\] batch_last" "\\[3\\] finalize"; do
        id="$(_extract_job "$file" "$pattern")"
        [[ -n "$id" ]] && jobs+=("$(_dependency_id "$id")")
    done
    local IFS=,
    printf "%s" "${jobs[*]}"
}

_print_combined_slurm_diagnostics() {
    local jobs_csv="$1" logs_a="$2" logs_b="$3"
    echo " Status  : sacct -j ${jobs_csv} \\"
    echo "           --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList,ReqTRES%50,AllocTRES%70,MaxRSS"
    echo " Queue   : squeue -j ${jobs_csv} -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
    echo " Starts  : squeue --start -j ${jobs_csv}"
    echo " Fail A  : grep -RniE \"error|exception|traceback|cuda|cupy|oom|timeout|killed|scratch|nvme|no space|permission|failed|SCORE GUARD\" \"$logs_a\" | tail -300"
    echo " Fail B  : grep -RniE \"error|exception|traceback|cuda|cupy|oom|timeout|killed|scratch|nvme|no space|permission|failed|SCORE GUARD\" \"$logs_b\" | tail -300"
}

_submit_one() {
    local label="$1" radius="$2" log_file="$3"
    echo "[$label] Submitting R=$radius μm ..."
    bash "$SUBMIT_SH" "${COMMON_FLAGS[@]}" --radius "$radius" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" | tee "$log_file"
}

echo "========================================================"
echo " mnist_wta_v1  |  Pilot A/B submit"
echo "========================================================"
echo "  Pilot A: R=8.0  μm  (stronger coupling)"
echo "  Pilot B: R=15.0 μm  (weak-coupling control)"
echo "  train/class: $TRAIN_PER_CLASS  test/class: $TEST_PER_CLASS  → N=$N_TOTAL per run"
echo "  alpha=0.7 beta=0.5  (from config.yaml)"
echo ""

COMMON_FLAGS=(
    --pilot
    --pilot-train-per-class "$TRAIN_PER_CLASS"
    --pilot-test-per-class  "$TEST_PER_CLASS"
)

A_LOG="$(mktemp /tmp/mnist_wta_pilot_A.XXXXXX)"
B_LOG="$(mktemp /tmp/mnist_wta_pilot_B.XXXXXX)"
_TMP_FILES+=("$A_LOG" "$B_LOG")

_submit_one "A" "8.0" "$A_LOG"

echo ""
_submit_one "B" "15.0" "$B_LOG"

A_RUN_DIR="$(_extract_run_dir "$A_LOG")"
B_RUN_DIR="$(_extract_run_dir "$B_LOG")"
A_JOBS="$(_jobs_from_log "$A_LOG")"
B_JOBS="$(_jobs_from_log "$B_LOG")"
A_LOGS_DIR="${A_RUN_DIR:+${A_RUN_DIR}/logs}"
B_LOGS_DIR="${B_RUN_DIR:+${B_RUN_DIR}/logs}"

ALL_JOBS="$A_JOBS"
if [[ -n "$B_JOBS" ]]; then
    [[ -n "$ALL_JOBS" ]] && ALL_JOBS="${ALL_JOBS},"
    ALL_JOBS="${ALL_JOBS}${B_JOBS}"
fi

echo ""
echo "========================================================"
echo " Submitted pilot A (R=8) and B (R=15)."
[[ -n "$A_RUN_DIR" ]] && echo "  Run A    : $A_RUN_DIR"
[[ -n "$B_RUN_DIR" ]] && echo "  Run B    : $B_RUN_DIR"
if [[ $DRY_RUN -eq 0 && -n "$ALL_JOBS" ]]; then
    echo ""
    _print_combined_slurm_diagnostics "$ALL_JOBS" "$A_LOGS_DIR" "$B_LOGS_DIR"
fi
if [[ $DRY_RUN -eq 0 && -n "$A_RUN_DIR" && -n "$B_RUN_DIR" ]]; then
    echo ""
    echo " Audit    : PYTHONPATH=src/mnist_common:src/mnist_wta_v1 python3 -m mnist_wta_v1.stages.cpu.pilot_audit \\"
    echo "              --run-a \"$A_RUN_DIR\" \\"
    echo "              --run-b \"$B_RUN_DIR\" \\"
    echo "              --out-dir \"$A_RUN_DIR/pilot_audit_ab\""
fi
echo "========================================================"
