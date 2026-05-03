#!/bin/bash -l
set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    _THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$_THIS/../.." && pwd)"
fi
THRESHOLD_DIR="$PROJECT_ROOT/src/threshold_finder"

cd "$PROJECT_ROOT"

_load_first_module() {
    local candidate
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if module load "$candidate" >/dev/null 2>&1; then
            echo "Loaded module: $candidate"
            return 0
        fi
    done
    return 1
}

module purge >/dev/null 2>&1 || true
_load_first_module "${RYSY_PYTHON_MODULE:-}" common/python/3.13.2 python/3.13.2 python/3.13 python || true
_load_first_module "${RYSY_OPENSSL_MODULE:-}" common/libs/openssl/1.1.1 openssl/1.1.1 OpenSSL/1.1.1 openssl || true
_load_first_module "${RYSY_GCC_MODULE:-}" common/compilers/gcc/13.2.0 gcc/13.2.0 GCC/13.2.0 gcc || true

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
else
    echo "ERROR: project virtualenv not found in $PROJECT_ROOT" >&2
    echo "Expected .venv/bin/activate or venv/bin/activate." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${THRESHOLD_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" >/dev/null 2>&1 || true

python -c "import matplotlib; print(f'matplotlib OK -- {matplotlib.__version__}')"
echo ""

cd "$THRESHOLD_DIR"
exec "$@"
