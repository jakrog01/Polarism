"""Tests for dot-response-fit configuration builders."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from polarism.compute_engine import compute_engine

DOT_RESPONSE_SRC = Path(__file__).resolve().parents[1] / "src" / "dot_response_fit"
if str(DOT_RESPONSE_SRC) not in sys.path:
    sys.path.insert(0, str(DOT_RESPONSE_SRC))

from dot_response_fit.config.builder import build_mnist_lasers


class _Grid:
    def __init__(self) -> None:
        x = np.linspace(-5.0, 5.0, 8)
        y = np.linspace(-5.0, 5.0, 8)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")


def test_mnist_lasers_use_fixed_spatial_dot_position() -> None:
    """MNIST pixels must encode amplitude/time, not spatial pump location."""
    compute_engine.xp = np
    compute_engine.use_gpu = False

    encoded_events = {
        "sigma_time": 2.0,
        "amplitudes": [1.0, 2.0, 3.0],
        "delays": [0.0, 10.0, 20.0],
        "x_positions": [-4.0, 0.0, 4.0],
        "y_positions": [3.0, 0.0, -3.0],
        "separation": 10.0,
    }
    global_cfg = {"laser_defaults": {"x0": 1.25, "y0": -2.5, "cutoff_sigma": 3.0}}

    lasers = build_mnist_lasers(encoded_events, 5.0, global_cfg, _Grid())

    assert [laser.P0 for laser in lasers] == encoded_events["amplitudes"]
    assert [laser.delay for laser in lasers] == encoded_events["delays"]
    assert {laser.x0 for laser in lasers} == {1.25}
    assert {laser.y0 for laser in lasers} == {-2.5}

