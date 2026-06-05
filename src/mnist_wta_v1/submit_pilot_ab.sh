#!/usr/bin/env bash
# Submit the two first WTA pilot controls in parallel:
#   A: R=8 um   (stronger coupling)
#   B: R=15 um  (weak-coupling control)
#
# Access-node safe: delegates MNIST check/download and Slurm submission to submit.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SH="$SCRIPT_DIR/submit.sh"

if [[ ! -f "$SUBMIT_SH" ]]; then
    echo "ERROR: submit.sh not found: $SUBMIT_SH" >&2
    exit 1
fi

for arg in "$@"; do
    if [[ "$arg" == "--wait" ]]; then
        echo "ERROR: --wait is not supported by submit_pilot_ab.sh because it would serialize A and B." >&2
        echo "Submit without --wait and monitor the two printed run directories." >&2
        exit 1
    fi
done

echo "========================================================"
echo " mnist_wta_v1  |  Pilot A/B submit"
echo "========================================================"
echo "  Pilot A: R=8.0  alpha=0.7 beta=0.5  N=100"
echo "  Pilot B: R=15.0 alpha=0.7 beta=0.5  N=100"
echo ""

COMMON_FLAGS=(
    --pilot
    --pilot-train-per-class 5
    --pilot-test-per-class 5
)

echo "[A] Submitting R=8.0 um ..."
bash "$SUBMIT_SH" "${COMMON_FLAGS[@]}" --radius 8.0 "$@"

echo ""
echo "[B] Submitting R=15.0 um ..."
bash "$SUBMIT_SH" "${COMMON_FLAGS[@]}" --radius 15.0 "$@"

echo ""
echo "Submitted pilot A/B. Use the run directories printed above for monitoring."
echo "After both finalize jobs complete, compare them with:"
echo "  python -m mnist_wta_v1.stages.cpu.compare_pilots \\"
echo "    --run-a <R8_run_dir> --run-b <R15_run_dir> --out-dir <report_dir>"
