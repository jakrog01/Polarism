"""Built-in deterministic initial conditions."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from polarism.initial_condition.initial_condition_registry import register_initial_condition


@register_initial_condition("zero")
def zero(grid, cfg, xp, cdtype):
    """Return an all-zero field."""
    return xp.zeros(grid.shape, dtype=cdtype)


@register_initial_condition("plane-wave")
def plane_wave(grid, cfg, xp, cdtype):
    """Return ``amplitude * exp(i (kx x + ky y))``; 1D grids use ``kx`` only."""
    phase = cfg.kx * grid.X
    if grid.ndim == 2:
        phase = phase + cfg.ky * grid.Y
    return cfg.amplitude * xp.exp(1j * phase)


@register_initial_condition("vortex")
def vortex(grid, cfg, xp, cdtype):
    """Return ``amplitude * tanh(r / core_width) * exp(i charge theta)`` about (x0, y0)."""
    dx = grid.X - cfg.x0
    dy = grid.Y - cfg.y0
    radius = xp.hypot(dx, dy)
    return (cfg.amplitude * xp.tanh(radius / cfg.core_width)
            * xp.exp(1j * cfg.charge * xp.arctan2(dy, dx)))


def _read_hdf5(path: Path, frame: int):
    """Read a field from a Polarism results file or a checkpoint v2."""
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("checkpoint_version") == 2:
            field = h5["field"]
            return field[0] if field.shape[1] == 2 else field[0, 0]
        if "fields/psi" not in h5:
            raise ValueError(
                f"initial_condition.path {str(path)!r} is neither a checkpoint v2 nor a "
                "results file with a saved psi field (set result.save_psi)"
            )
        dataset = h5["fields/psi"]
        count = dataset.shape[0]
        if not -count <= frame < count:
            raise ValueError(
                f"initial_condition.frame={frame} is outside the {count} saved frames of {str(path)!r}"
            )
        index = frame if frame >= 0 else count + frame
        if "member" in dataset.attrs["dims"]:
            return dataset[index, 0] if dataset.shape[2] == 2 else dataset[index, 0, 0]
        return dataset[index] if dataset.shape[1] == 2 else dataset[index, 0]


@register_initial_condition("file")
def from_file(grid, cfg, xp, cdtype):
    """Load a field from a ``.npy`` array, a results ``.h5`` file or a checkpoint v2."""
    path = Path(cfg.path)
    if not path.is_file():
        raise FileNotFoundError(f"initial_condition.path {str(path)!r} does not exist")
    if path.suffix == ".npy":
        psi = np.load(path)
    elif path.suffix in (".h5", ".hdf5"):
        psi = _read_hdf5(path, cfg.frame)
    else:
        raise ValueError(
            f"initial_condition.path {str(path)!r} must be a .npy, .h5 or .hdf5 file"
        )
    if tuple(psi.shape) not in (tuple(grid.shape), (2, *grid.shape)):
        raise ValueError(
            f"initial_condition.path {str(path)!r} holds a field of shape {tuple(psi.shape)}, "
            f"but the grid shape is {tuple(grid.shape)}; interpolation is not supported"
        )
    return xp.asarray(psi, dtype=cdtype)
