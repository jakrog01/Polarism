from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios" / "pitch_sigma_sweep"
BASE_CONFIG = ROOT / "polarism_base.yaml"


@dataclass(frozen=True, slots=True)
class DiscretizationResult:
    name: str
    sigma_um: float
    pts_per_sigma: float
    center_integral: float
    analytic_integral: float
    center_rel_error: float
    lattice_min_rel_error: float
    lattice_max_rel_error: float
    lattice_mean_rel_error: float


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _scenario_paths() -> list[Path]:
    return sorted(SCENARIO_DIR.glob("pitch*_sigma*.yaml"))


def _grid_coordinates(nx: int, ny: int, lx: float, ly: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    dx = lx / nx
    dy = ly / ny
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) * 0.5) * np.float64(dx)
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) * 0.5) * np.float64(dy)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    return x_grid, y_grid, dx, dy


def _gaussian_integral(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dx: float,
    dy: float,
    sigma_um: float,
    x0_um: float,
    y0_um: float,
) -> float:
    sigma_sq = np.float64(sigma_um) * np.float64(sigma_um)
    r_sq = (x_grid - np.float64(x0_um)) ** 2 + (y_grid - np.float64(y0_um)) ** 2
    raw = np.exp(np.float64(-0.5) * r_sq / sigma_sq, dtype=np.float64)
    return float(np.sum(raw, dtype=np.float64) * np.float64(dx) * np.float64(dy))


def _lattice_offsets(n_side: int, pitch_um: float) -> np.ndarray:
    return (
        np.arange(n_side, dtype=np.float64) - np.float64(n_side - 1) / np.float64(2.0)
    ) * np.float64(pitch_um)


def evaluate() -> tuple[float, float, list[DiscretizationResult]]:
    base = _load_yaml(BASE_CONFIG)
    grid = base.get("grid")
    if not isinstance(grid, dict):
        raise ValueError(f"{BASE_CONFIG} must contain a grid mapping")
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    lx = float(grid["lx"])
    ly = float(grid["ly"])
    x_grid, y_grid, dx, dy = _grid_coordinates(nx, ny, lx, ly)
    results: list[DiscretizationResult] = []
    for scenario_path in _scenario_paths():
        scenario = _load_yaml(scenario_path)
        geometry = scenario.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"{scenario_path} must contain a geometry mapping")
        sigma_um = float(geometry["sigma_space_um"])
        pitch_um = float(geometry["pitch_um"])
        n_side = int(geometry["n_side"])
        center_x_um = float(geometry["center_x_um"])
        center_y_um = float(geometry["center_y_um"])
        analytic = float(2.0 * math.pi * sigma_um * sigma_um)
        center_integral = _gaussian_integral(
            x_grid,
            y_grid,
            dx,
            dy,
            sigma_um,
            center_x_um,
            center_y_um,
        )
        offsets = _lattice_offsets(n_side, pitch_um)
        rel_errors = np.asarray(
            [
                (
                    _gaussian_integral(
                        x_grid,
                        y_grid,
                        dx,
                        dy,
                        sigma_um,
                        center_x_um + float(x_offset),
                        center_y_um + float(y_offset),
                    )
                    - analytic
                )
                / analytic
                for x_offset in offsets
                for y_offset in offsets
            ],
            dtype=np.float64,
        )
        results.append(
            DiscretizationResult(
                name=scenario_path.stem,
                sigma_um=sigma_um,
                pts_per_sigma=sigma_um / dx,
                center_integral=center_integral,
                analytic_integral=analytic,
                center_rel_error=(center_integral - analytic) / analytic,
                lattice_min_rel_error=float(np.min(rel_errors)),
                lattice_max_rel_error=float(np.max(rel_errors)),
                lattice_mean_rel_error=float(np.mean(rel_errors)),
            )
        )
    return dx, dy, sorted(results, key=lambda item: item.pts_per_sigma, reverse=True)


def main() -> int:
    dx, dy, results = evaluate()
    print(f"dx_um={dx:.8f} dy_um={dy:.8f}")
    print(
        "variant,sigma_um,pts_per_sigma,center_integral,analytic_integral,"
        "center_rel_error,lattice_min_rel_error,lattice_max_rel_error,lattice_mean_rel_error"
    )
    for result in results:
        print(
            f"{result.name},{result.sigma_um:.16g},{result.pts_per_sigma:.8f},"
            f"{result.center_integral:.16e},{result.analytic_integral:.16e},"
            f"{result.center_rel_error:.16e},{result.lattice_min_rel_error:.16e},"
            f"{result.lattice_max_rel_error:.16e},{result.lattice_mean_rel_error:.16e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
