#!/bin/bash -l
set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "$_THIS/../../.." && pwd -P)"
fi

if ! cd "$PROJECT_ROOT" 2>/dev/null; then
    echo "ERROR: PROJECT_ROOT not accessible from compute node: $PROJECT_ROOT" >&2
    _PROJECT_ROOT_PHYS="$(readlink -f "$PROJECT_ROOT" 2>/dev/null || true)"
    if [[ -n "$_PROJECT_ROOT_PHYS" && -d "$_PROJECT_ROOT_PHYS" ]]; then
        PROJECT_ROOT="$_PROJECT_ROOT_PHYS"
        cd "$PROJECT_ROOT"
    else
        exit 1
    fi
fi
PROJECT_ROOT="$(pwd -P)"
SNN_DYNAMIC_DIR="$PROJECT_ROOT/src/mnist_digits_polariton_snn_dynamic"

_load_first_module() {
    local candidate
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if module load "$candidate" >/dev/null 2>&1; then
            echo "Loaded module: $candidate"; return 0
        fi
    done
    return 1
}

module purge >/dev/null 2>&1 || true
_load_first_module "${RYSY_PYTHON_MODULE:-}" common/python/3.13.2 python/3.13.2 python/3.13 python || true
_load_first_module "${RYSY_OPENSSL_MODULE:-}" common/libs/openssl/1.1.1 openssl/1.1.1 OpenSSL/1.1.1 openssl || true
_load_first_module "${RYSY_GCC_MODULE:-}" common/compilers/gcc/13.2.0 gcc/13.2.0 GCC/13.2.0 gcc || true

_VENV_ACTIVATE=""
for _cand in \
    "$PROJECT_ROOT/.venv/bin/activate" \
    "$PROJECT_ROOT/venv/bin/activate" \
    "${SNN_DYNAMIC_VENV:-}" \
    "$HOME/polaritonSNN/PolaritonSNN/.venv/bin/activate" \
    "$HOME/.venv/bin/activate"; do
    [[ -z "$_cand" ]] && continue
    if [[ -f "$_cand" ]]; then _VENV_ACTIVATE="$_cand"; break; fi
done

if [[ -z "$_VENV_ACTIVATE" ]]; then
    echo "ERROR: virtualenv not found." >&2
    exit 1
fi

source "$_VENV_ACTIVATE"
export PYTHONPATH="${PROJECT_ROOT}:${SNN_DYNAMIC_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" >/dev/null 2>&1 || true

cd "$SNN_DYNAMIC_DIR"
exec "$@"
