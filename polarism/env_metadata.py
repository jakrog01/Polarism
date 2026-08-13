"""Portable environment metadata for reproducible Polarism outputs."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from polarism.compute_engine import CUPY_AVAILABLE, cp


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator and key.strip() == "model name":
                    return value.strip() or None
    except Exception:
        pass
    try:
        return platform.processor() or None
    except Exception:
        return None


def _gpu_properties() -> dict[str, Any] | None:
    if not CUPY_AVAILABLE:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        return cp.cuda.runtime.getDeviceProperties(0)
    except Exception:
        return None


def _cuda_runtime_version() -> str | None:
    if not CUPY_AVAILABLE:
        return None
    try:
        version = int(cp.cuda.runtime.runtimeGetVersion())
        return f"{version // 1000}.{version % 1000 // 10}.{version % 10}"
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=root,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def collect_env_metadata() -> dict[str, object]:
    """Collect stable hardware, runtime, and source metadata without raising."""
    properties = _gpu_properties()
    gpu_name: str | None = None
    gpu_memory: int | None = None
    if properties is not None:
        try:
            name = properties["name"]
            gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            gpu_memory = int(properties["totalGlobalMem"])
        except Exception:
            gpu_name = None
            gpu_memory = None
    return {
        "cpu_model": _cpu_model(),
        "cpu_core_count": os.cpu_count(),
        "gpu_device_name": gpu_name,
        "gpu_total_memory_bytes": gpu_memory,
        "cuda_runtime_version": _cuda_runtime_version(),
        "cupy_version": getattr(cp, "__version__", None) if CUPY_AVAILABLE else None,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "os_release": platform.platform(),
        "git_commit": _git_commit(),
    }


def attach_environment(meta: dict) -> dict:
    """Return a shallow copy of metadata enriched with environment details."""
    return {**meta, "environment": collect_env_metadata()}


def _latex_value(value: object) -> str:
    return ("n/a" if value is None else str(value)).replace("_", "\\_")


def write_hardware_tex(path: Path) -> None:
    """Write a LaTeX hardware table to *path*."""
    environment = collect_env_metadata()
    memory = environment["gpu_total_memory_bytes"]
    gpu_memory = None if memory is None else f"{int(memory) / 1e9:.3g} GB"
    lines = [
        "\\begin{tabular}{ll}",
        f"CPU & {_latex_value(environment['cpu_model'])} ({_latex_value(environment['cpu_core_count'])} cores) \\\\",
        f"GPU & {_latex_value(environment['gpu_device_name'])} ({_latex_value(gpu_memory)}) \\\\",
        f"CUDA & {_latex_value(environment['cuda_runtime_version'])}, CuPy {_latex_value(environment['cupy_version'])} \\\\",
        f"Python & {_latex_value(environment['python_version'])}, NumPy {_latex_value(environment['numpy_version'])} \\\\",
        f"OS & {_latex_value(environment['os_release'])} \\\\",
        f"Commit & \\texttt{{{_latex_value(environment['git_commit'])}}} \\\\",
        "\\end{tabular}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
