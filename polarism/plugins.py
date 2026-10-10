"""User plugin discovery from a local ``polarism_plugins`` folder."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PLUGIN_PACKAGE = "polarism_plugins"


def load_plugins(root: str | Path | None = None) -> list[str]:
    """Import every module under ``<root>/polarism_plugins`` and return their names.

    Plugin modules register their components with the usual registry
    decorators (``register_laser``, ``register_potential``, ...) on import.
    ``root`` defaults to the current working directory.
    """
    root = Path.cwd() if root is None else Path(root)
    root = root.resolve()
    if not (root / PLUGIN_PACKAGE).is_dir():
        return []

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    loaded = []
    for path in sorted((root / PLUGIN_PACKAGE).rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        importlib.import_module(name)
        loaded.append(name)
    return loaded
