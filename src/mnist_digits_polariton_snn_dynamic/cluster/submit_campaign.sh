#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
WITH_CALIBRATE=0
MANIFEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --with-calibrate) WITH_CALIBRATE=1; shift ;;
        *) [[ -z "$MANIFEST" ]] && MANIFEST="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
PACKAGE_DIR="$PROJECT_ROOT/src/mnist_digits_polariton_snn_dynamic"
SLURM_ENV="$PROJECT_ROOT/slurm.env"

if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: usage: bash cluster/submit_campaign.sh [--dry-run] [--with-calibrate] <manifest.yaml>" >&2
    exit 1
fi
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$SLURM_ENV" ]]; then
    echo "ERROR: slurm.env not found: $SLURM_ENV" >&2
    exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" != rysy* && $DRY_RUN -eq 0 ]]; then
    echo "ERROR: submit_campaign.sh must be run on a Rysy login/access node." >&2
    echo "  Use --dry-run to inspect locally." >&2
    exit 1
fi

source "$SLURM_ENV"

for _var in SLURM_ACCOUNT SLURM_PARTITION SLURM_MEM SLURM_GPUS SLURM_CPUS NVME_GB TETYDA_RUNS_BASE; do
    if [[ -z "${!_var:-}" ]]; then
        echo "ERROR: slurm.env missing: $_var" >&2
        exit 1
    fi
done

SLURM_QOS="${SLURM_QOS:-}"
SNN_MIX_CAL_TIME="${SNN_MIX_CAL_TIME:-06:00:00}"
SNN_MIX_CAL_NVME="${SNN_MIX_CAL_NVME:-$NVME_GB}"
SNN_MIX_FULL_TIME="${SNN_MIX_FULL_TIME:-48:00:00}"
SNN_MIX_FULL_NVME="${SNN_MIX_FULL_NVME:-$NVME_GB}"
FINALIZE_TIME="${FINALIZE_TIME:-00:10:00}"
FINALIZE_MEM="${FINALIZE_MEM:-8G}"
FINALIZE_CPUS="${FINALIZE_CPUS:-4}"
SNN_THRESHOLD_MEM="${SNN_THRESHOLD_MEM:-4G}"
SNN_THRESHOLD_CPUS="${SNN_THRESHOLD_CPUS:-1}"
SNN_THRESHOLD_TIME="${SNN_THRESHOLD_TIME:-00:10:00}"
QOS_ARGS=()
[[ -n "$SLURM_QOS" ]] && QOS_ARGS=(--qos="$SLURM_QOS")

CAMPAIGN_NAME="$(
    python3 - "$MANIFEST" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    data = yaml.safe_load(stream)
if not isinstance(data, dict):
    sys.exit(f"ERROR: manifest YAML is empty or malformed: {sys.argv[1]!r} -> {type(data).__name__}")
print(data["campaign_name"])
PY
)"

BASELINE_SCENARIO_ID="$(
    python3 - "$MANIFEST" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    data = yaml.safe_load(stream)
if not isinstance(data, dict):
    sys.exit(f"ERROR: manifest YAML is empty or malformed: {sys.argv[1]!r} -> {type(data).__name__}")
scenarios = data["scenarios"]
print(data.get("baseline_scenario_id") or scenarios[0]["id"])
PY
)"

SCENARIOS_TSV="$(
    python3 - "$MANIFEST" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    data = yaml.safe_load(stream)
if not isinstance(data, dict):
    sys.exit(f"ERROR: manifest YAML is empty or malformed: {sys.argv[1]!r} -> {type(data).__name__}")
for index, scenario in enumerate(data["scenarios"]):
    print(f'{index}\t{scenario["id"]}')
PY
)"

N_SCENARIOS_TOTAL="$(printf "%s\n" "$SCENARIOS_TSV" | grep -c $'\t' || true)"

TS="$(date +%Y%m%d_%H%M%S)"
if [[ $DRY_RUN -eq 1 ]]; then
    CAMP_DIR="${PACKAGE_DIR}/dry_run/${TS}_${CAMPAIGN_NAME}"
else
    CAMP_DIR="${TETYDA_RUNS_BASE}/${TS}_${CAMPAIGN_NAME}"
fi
LOGS_DIR="$CAMP_DIR/logs"
ROWS_JSONL="$CAMP_DIR/submission_manifest.rows.jsonl"

mkdir -p "$CAMP_DIR" "$LOGS_DIR"
: > "$ROWS_JSONL"

echo "========================================"
echo " MNIST-digits polariton SNN dynamic campaign"
echo " Rysy-only  |  per-scenario threshold -> run -> finalize"
echo "========================================"
echo "  Campaign  : $CAMPAIGN_NAME"
echo "  Manifest  : $MANIFEST"
echo "  slurm.env : $SLURM_ENV"
echo "  Host      : $HOSTNAME_SHORT"
echo "  Baseline  : $BASELINE_SCENARIO_ID"
echo "  Scenarios : $N_SCENARIOS_TOTAL"
echo "  Ordering  : sequential (each threshold afterok previous finalize)"
echo ""
echo "  Rysy GPU  : partition=$SLURM_PARTITION  mem=$SLURM_MEM  gpus=$SLURM_GPUS  nvme=${NVME_GB}G"
echo "  Cal time  : $SNN_MIX_CAL_TIME     (nvme=${SNN_MIX_CAL_NVME}G)"
echo "  Thr CPU   : mem=$SNN_THRESHOLD_MEM  cpus=$SNN_THRESHOLD_CPUS  time=$SNN_THRESHOLD_TIME"
echo "  Run time  : $SNN_MIX_FULL_TIME   (nvme=${SNN_MIX_FULL_NVME}G)"
echo "  Fin CPU   : mem=$FINALIZE_MEM  cpus=$FINALIZE_CPUS  time=$FINALIZE_TIME"
echo "  Repo root : $PROJECT_ROOT"
echo "  Camp dir  : $CAMP_DIR"
echo "  Logs      : $LOGS_DIR"
echo "========================================"
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [DRY RUN] No sbatch calls; showing planned stage layout."
    echo ""
fi

_dependency_id() { printf "%s" "${1%%;*}"; }

_record_submission() {
    local scenario_id="$1" stage="$2" job_id="$3" dependency="$4" wall_time="$5"
    python3 - "$ROWS_JSONL" "$scenario_id" "$stage" "$job_id" "$dependency" "$wall_time" <<'PY'
import json
import sys
path, scenario_id, stage, job_id, dependency, wall_time = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "scenario_id": scenario_id,
        "stage": stage,
        "job_id": job_id,
        "dependency": dependency or None,
        "wall_time": wall_time,
    }, sort_keys=True) + "\n")
PY
}

CAL_JOB_IDS=()
THRESHOLD_JOB_IDS=()
RUN_JOB_IDS=()
FINAL_JOB_IDS=()
SCENARIO_IDS=()
PREVIOUS_FINAL_JOB=""
STAGE_INDEX=0

while IFS=$'\t' read -r INDEX SCENARIO_ID; do
    [[ -z "$SCENARIO_ID" ]] && continue
    SCENARIO_IDS+=("$SCENARIO_ID")
    STAGE_INDEX=$((STAGE_INDEX + 1))
    STAGE_TAG="[${STAGE_INDEX}] ${SCENARIO_ID}"
    THRESHOLD_DEP_ARGS=()
    [[ -n "$PREVIOUS_FINAL_JOB" ]] && THRESHOLD_DEP_ARGS=(--dependency="afterok:${PREVIOUS_FINAL_JOB}")
    THRESHOLD_DEP_LABEL="none"
    [[ -n "$PREVIOUS_FINAL_JOB" ]] && THRESHOLD_DEP_LABEL="afterok:${PREVIOUS_FINAL_JOB}"
    if [[ $DRY_RUN -eq 1 ]]; then
        THRESHOLD_JOB="DRYRUN_${SCENARIO_ID}_threshold"
        echo "${STAGE_TAG}/threshold -> [DRY RUN] job $THRESHOLD_JOB  (dep=${THRESHOLD_DEP_LABEL}, time=${SNN_THRESHOLD_TIME})"
    else
        THRESHOLD_JOB_RAW=$(
            sbatch --parsable \
                --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
                --job-name="mnist_digits_${SCENARIO_ID}_threshold" \
                --account="$SLURM_ACCOUNT" --partition="${SLURM_CPU_PARTITION:-$SLURM_PARTITION}" \
                --mem="$SNN_THRESHOLD_MEM" --cpus-per-task="$SNN_THRESHOLD_CPUS" --time="$SNN_THRESHOLD_TIME" \
                ${QOS_ARGS[@]+"${QOS_ARGS[@]}"} ${THRESHOLD_DEP_ARGS[@]+"${THRESHOLD_DEP_ARGS[@]}"} \
                --output="${LOGS_DIR}/${SCENARIO_ID}_threshold_%j.out" \
                --error="${LOGS_DIR}/${SCENARIO_ID}_threshold_%j.err" \
                "$SCRIPT_DIR/job_cpu.sh" \
                python3 -m mnist_digits_polariton_snn_dynamic.scenarios.find_threshold \
                    --manifest "$MANIFEST" \
                    --scenario-id "$SCENARIO_ID" \
                    --campaign-output-dir "$CAMP_DIR"
        )
        THRESHOLD_JOB="$(_dependency_id "$THRESHOLD_JOB_RAW")"
        echo "${STAGE_TAG}/threshold -> Rysy job $THRESHOLD_JOB  (dep=${THRESHOLD_DEP_LABEL}, time=${SNN_THRESHOLD_TIME})"
    fi
    THRESHOLD_JOB_IDS+=("$THRESHOLD_JOB")
    _record_submission "$SCENARIO_ID" "threshold" "$THRESHOLD_JOB" "$THRESHOLD_DEP_LABEL" "$SNN_THRESHOLD_TIME"

    RUN_DEP="$THRESHOLD_JOB"
    POWER_SOURCE="threshold"
    if [[ $WITH_CALIBRATE -eq 1 ]]; then
        CAL_DEP_ARGS=(--dependency="afterok:${THRESHOLD_JOB}")
        if [[ $DRY_RUN -eq 1 ]]; then
            CAL_JOB="DRYRUN_${SCENARIO_ID}_calibrate"
            echo "${STAGE_TAG}/calibrate -> [DRY RUN] job $CAL_JOB  (dep=afterok:${THRESHOLD_JOB}, time=${SNN_MIX_CAL_TIME})"
        else
            CAL_JOB_RAW=$(sbatch --parsable --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1" --job-name="mnist_digits_${SCENARIO_ID}_cal" --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${SNN_MIX_CAL_NVME}" --cpus-per-task="$SLURM_CPUS" --time="$SNN_MIX_CAL_TIME" ${QOS_ARGS[@]+"${QOS_ARGS[@]}"} "${CAL_DEP_ARGS[@]}" --output="${LOGS_DIR}/${SCENARIO_ID}_calibrate_%j.out" --error="${LOGS_DIR}/${SCENARIO_ID}_calibrate_%j.err" "$SCRIPT_DIR/job_gpu.sh" python3 -m mnist_digits_polariton_snn_dynamic.scenarios.calibrate_scenario --manifest "$MANIFEST" --scenario-id "$SCENARIO_ID" --campaign-output-dir "$CAMP_DIR")
            CAL_JOB="$(_dependency_id "$CAL_JOB_RAW")"
            echo "${STAGE_TAG}/calibrate -> Rysy job $CAL_JOB  (dep=afterok:${THRESHOLD_JOB}, time=${SNN_MIX_CAL_TIME})"
        fi
        CAL_JOB_IDS+=("$CAL_JOB")
        _record_submission "$SCENARIO_ID" "calibrate" "$CAL_JOB" "afterok:${THRESHOLD_JOB}" "$SNN_MIX_CAL_TIME"
        RUN_DEP="$CAL_JOB"
        POWER_SOURCE="calibration"
    fi
    RUN_DEP_ARGS=(--dependency="afterok:${RUN_DEP}")
    if [[ $DRY_RUN -eq 1 ]]; then
        RUN_JOB="DRYRUN_${SCENARIO_ID}_run"
        echo "${STAGE_TAG}/run      -> [DRY RUN] job $RUN_JOB  (dep=afterok:${RUN_DEP}, time=${SNN_MIX_FULL_TIME})"
    else
        RUN_JOB_RAW=$(
            sbatch --parsable \
                --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ICM_RYSY_NVME=1" \
                --job-name="mnist_digits_${SCENARIO_ID}_run" \
                --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
                --mem="$SLURM_MEM" --gres="gpu:${SLURM_GPUS},nvme:${SNN_MIX_FULL_NVME}" \
                --cpus-per-task="$SLURM_CPUS" --time="$SNN_MIX_FULL_TIME" \
                ${QOS_ARGS[@]+"${QOS_ARGS[@]}"} "${RUN_DEP_ARGS[@]}" \
                --output="${LOGS_DIR}/${SCENARIO_ID}_run_%j.out" \
                --error="${LOGS_DIR}/${SCENARIO_ID}_run_%j.err" \
                "$SCRIPT_DIR/job_gpu.sh" \
                python3 -m mnist_digits_polariton_snn_dynamic.scenarios.run_campaign \
                    --manifest "$MANIFEST" \
                    --scenario-id "$SCENARIO_ID" \
                    --campaign-output-dir "$CAMP_DIR" \
                    --power-source "$POWER_SOURCE"
        )
        RUN_JOB="$(_dependency_id "$RUN_JOB_RAW")"
        echo "${STAGE_TAG}/run      -> Rysy job $RUN_JOB  (dep=afterok:${RUN_DEP}, time=${SNN_MIX_FULL_TIME})"
    fi
    RUN_JOB_IDS+=("$RUN_JOB")
    _record_submission "$SCENARIO_ID" "run" "$RUN_JOB" "afterok:${RUN_DEP}" "$SNN_MIX_FULL_TIME"

    FINAL_DEP="$RUN_JOB"
    FINAL_DEP_ARGS=(--dependency="afterok:${FINAL_DEP}")
    if [[ $DRY_RUN -eq 1 ]]; then
        FINAL_JOB="DRYRUN_${SCENARIO_ID}_finalize"
        echo "${STAGE_TAG}/finalize -> [DRY RUN] job $FINAL_JOB  (dep=afterok:${FINAL_DEP}, time=${FINALIZE_TIME})"
    else
        FINAL_JOB_RAW=$(
            sbatch --parsable \
                --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
                --job-name="mnist_digits_${SCENARIO_ID}_fin" \
                --account="$SLURM_ACCOUNT" --partition="${SLURM_CPU_PARTITION:-$SLURM_PARTITION}" \
                --mem="$FINALIZE_MEM" --cpus-per-task="$FINALIZE_CPUS" --time="$FINALIZE_TIME" \
                ${QOS_ARGS[@]+"${QOS_ARGS[@]}"} "${FINAL_DEP_ARGS[@]}" \
                --output="${LOGS_DIR}/${SCENARIO_ID}_finalize_%j.out" \
                --error="${LOGS_DIR}/${SCENARIO_ID}_finalize_%j.err" \
                "$SCRIPT_DIR/job_cpu.sh" \
                python3 -m mnist_digits_polariton_snn_dynamic.scenarios.finalize_scenario \
                    --manifest "$MANIFEST" \
                    --scenario-id "$SCENARIO_ID" \
                    --campaign-output-dir "$CAMP_DIR"
        )
        FINAL_JOB="$(_dependency_id "$FINAL_JOB_RAW")"
        echo "${STAGE_TAG}/finalize -> Rysy job $FINAL_JOB  (dep=afterok:${FINAL_DEP}, time=${FINALIZE_TIME})"
    fi
    FINAL_JOB_IDS+=("$FINAL_JOB")
    _record_submission "$SCENARIO_ID" "finalize" "$FINAL_JOB" "afterok:${FINAL_DEP}" "$FINALIZE_TIME"
    PREVIOUS_FINAL_JOB="$FINAL_JOB"
    echo ""
done <<< "$SCENARIOS_TSV"

python3 - "$MANIFEST" "$CAMP_DIR" "$ROWS_JSONL" "$BASELINE_SCENARIO_ID" <<'PY'
import json
import sys
from pathlib import Path
manifest, camp_dir, rows_path, baseline_id = sys.argv[1:]
rows = [
    json.loads(line)
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
out = {
    "manifest": str(Path(manifest).resolve()),
    "campaign_output_dir": str(Path(camp_dir).resolve()),
    "baseline_scenario_id": baseline_id,
    "scenario_ordering": "sequential_after_previous_finalize",
    "submissions": rows,
}
Path(camp_dir, "submission_manifest.json").write_text(
    json.dumps(out, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

echo "========================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo " Dry run complete. No Slurm jobs were submitted."
else
    echo " Campaign submitted. Slurm afterok chain will run stages in order:"
    echo "   scenario_i/threshold -> scenario_i/run -> scenario_i/finalize -> scenario_{i+1}/threshold ..."
fi
echo "========================================"
echo " Camp dir : $CAMP_DIR"
echo " Logs     : $LOGS_DIR"
echo " Manifest : $CAMP_DIR/submission_manifest.json"

if [[ $DRY_RUN -eq 0 ]]; then
    ALL_JOB_IDS_CSV="$(printf "%s," "${THRESHOLD_JOB_IDS[@]}" "${CAL_JOB_IDS[@]}" "${RUN_JOB_IDS[@]}" "${FINAL_JOB_IDS[@]}" | sed 's/,$//')"
    ALL_JOB_IDS_SPACE="$(printf "%s " "${THRESHOLD_JOB_IDS[@]}" "${CAL_JOB_IDS[@]}" "${RUN_JOB_IDS[@]}" "${FINAL_JOB_IDS[@]}" | sed 's/ $//')"
    echo ""
    echo " Jobs per scenario (threshold / calibrate / run / finalize):"
    for i in "${!SCENARIO_IDS[@]}"; do
        CAL_DISPLAY="-"
        [[ $WITH_CALIBRATE -eq 1 ]] && CAL_DISPLAY="${CAL_JOB_IDS[$i]}"
        printf "   %-40s thr=%s  cal=%s  run=%s  fin=%s\n" "${SCENARIO_IDS[$i]}" "${THRESHOLD_JOB_IDS[$i]}" "$CAL_DISPLAY" "${RUN_JOB_IDS[$i]}" "${FINAL_JOB_IDS[$i]}"
    done
    echo ""
    echo " Status  : sacct -j ${ALL_JOB_IDS_CSV} \\"
    echo "           --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,NodeList,ReqTRES%50,AllocTRES%70,MaxRSS,MaxVMSize"
    echo " Queue   : squeue -j ${ALL_JOB_IDS_CSV} -o \"%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R\""
    echo " Starts  : squeue --start -j ${ALL_JOB_IDS_CSV}"
    echo " Failures: grep -RniE \"error|exception|traceback|cuda|cupy|out.of.memory|oom|timeout|killed|scratch|nvme|no space|hdf5|permission|failed\" \"$LOGS_DIR\" | tail -300"
    echo " Skipped : grep -RniE \"scenario_skipped|SKIP:\" \"$LOGS_DIR\" \"$CAMP_DIR\"/*/{spike_threshold,calibration}.json 2>/dev/null | tail -100"
    echo " Cancel  : scancel ${ALL_JOB_IDS_SPACE}"
fi
echo "========================================"
