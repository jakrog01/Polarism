"""Spatial convergence artefacts with a sigma=0.9 μm broadband packet.

The broadband width keeps every solver below the 1e-2 error ceiling while
leaving a measurable spectral error on the coarsest grid. The intermediate
192 grid supplies a third asymptotic point after rejecting nx=64 while keeping
the existing 1024 reference. This gives a reference-to-finest ratio of four
without the memory and runtime cost of a 2048-square reference for every case.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests._helpers import initial_packet, rel_l2_err
from tests.unit.conftest import small_config

SOLVERS = (
    "rk4-fdm",
    "rk4-fdm-fused",
    "ip-rk4",
    "split-step-fft",
    "ifrk4-fft-cuda",
    "etd-rk2",
)
SPECTRAL_SOLVERS = {"ip-rk4", "ifrk4-fft-cuda", "split-step-fft", "etd-rk2"}
CASES = {
    "bandlimited": {"sigma": 4.0, "kx": 3.0},
    "broadband": {"sigma": 0.9, "kx": 6.0},
}
TEST_GRIDS = (64, 128, 192, 256)
REFERENCE_GRID = 1024


def _fourier_restrict(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(field))
    y0 = (field.shape[0] - shape[0]) // 2
    x0 = (field.shape[1] - shape[1]) // 2
    cropped = spectrum[y0:y0 + shape[0], x0:x0 + shape[1]]
    source_modes_y = np.fft.fftshift(np.fft.fftfreq(field.shape[0])) * field.shape[0]
    source_modes_x = np.fft.fftshift(np.fft.fftfreq(field.shape[1])) * field.shape[1]
    modes_y = source_modes_y[y0:y0 + shape[0]]
    modes_x = source_modes_x[x0:x0 + shape[1]]
    phase = np.exp(
        1j
        * np.pi
        * (
            modes_y[:, None] * (1.0 / shape[0] - 1.0 / field.shape[0])
            + modes_x[None, :] * (1.0 / shape[1] - 1.0 / field.shape[1])
        )
    )
    return np.fft.ifft2(np.fft.ifftshift(cropped * phase)) * (np.prod(shape) / field.size)


def _local_slopes(entries: list[dict[str, float]]) -> list[dict[str, float]]:
    return [
        {
            "nx_from": int(coarse["nx"]),
            "nx_to": int(fine["nx"]),
            "slope": float(
                np.log(coarse["error"] / fine["error"])
                / np.log(coarse["dx"] / fine["dx"])
            ),
        }
        for coarse, fine in zip(entries, entries[1:])
    ]


def _run_case(solver: str, case: str) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    use_gpu = solver == "ifrk4-fft-cuda"
    compute_engine.configure(ComputeEngineParameters(use_gpu=use_gpu))
    fields: dict[int, np.ndarray] = {}
    parameters = CASES[case]
    for nx in (*TEST_GRIDS, REFERENCE_GRID):
        cfg = small_config(
            grid__nx=nx,
            grid__ny=nx,
            grid__lx=40.0,
            grid__ly=40.0,
            solver__method=solver,
            solver__dt=1e-5,
            solver__total_time=2e-5,
            physics__g_C=5e-3,
            physics__gamma_C=0.0,
            physics__R=0.0,
            compute_engine=ComputeEngineParameters(use_gpu=use_gpu),
        )
        controller = SimulationController(cfg)
        packet = initial_packet(controller.grid, **parameters)
        controller.state.psi[:] = packet / np.sqrt(controller.grid.dx * controller.grid.dy)
        controller.run()
        fields[nx] = compute_engine.to_cpu(controller.state.psi)
    reference = fields[REFERENCE_GRID]
    entries = [
        {
            "nx": nx,
            "dx": 40.0 / nx,
            "error": rel_l2_err(
                np.abs(fields[nx]) ** 2,
                _fourier_restrict(np.abs(reference) ** 2, fields[nx].shape),
            ),
        }
        for nx in TEST_GRIDS
    ]
    return entries, _local_slopes(entries)


@pytest.mark.slow
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", tuple(CASES))
def test_space_convergence_artifact(solver: str, case: str) -> None:
    entries, local_slopes = _run_case(solver, case)
    path = Path("artifacts/convergence")
    path.mkdir(parents=True, exist_ok=True)
    (path / f"space_{solver}_{case}.json").write_text(
        json.dumps(
            {
                "entries": entries,
                "local_slopes": local_slopes,
                "reference_nx": REFERENCE_GRID,
            }
        )
    )
    errors = np.asarray([entry["error"] for entry in entries])
    assert np.isfinite(errors).all()
    assert np.all(np.diff(errors) <= 0.0)
    if case == "bandlimited" and solver in SPECTRAL_SOLVERS:
        assert float(np.max(errors)) < 1e-8
    if case == "broadband":
        assert float(np.max(errors)) < 1e-2
