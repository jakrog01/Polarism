"""Package entry point for presentation snapshot plotting."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "plot_presentation_snapshots.py"
)
_SPEC = importlib.util.spec_from_file_location("_presentation_snapshot_script", _SOURCE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load presentation snapshot script from {_SOURCE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


if __name__ == "__main__":
    main()
