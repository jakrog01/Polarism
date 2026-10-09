"""Shared pytest fixtures and options."""
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters


def pytest_addoption(parser):
    """Add custom pytest command-line options."""
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Run benchmarks on GPU (requires CuPy)",
    )


@pytest.fixture(scope="session")
def use_gpu(request):
    """Return whether GPU tests are enabled."""
    return request.config.getoption("--use-gpu", default=False)


@pytest.fixture(autouse=True)
def configure_backend(request, use_gpu):
    """Configure the compute backend for a test."""
    is_benchmark = "phoenix_benchmark" in str(request.fspath)
    is_gpu_test = request.node.get_closest_marker("gpu") is not None

    if use_gpu and (is_benchmark or is_gpu_test):
        engine_cfg = ComputeEngineParameters(use_gpu=True)
        compute_engine.configure(engine_cfg)
    else:
        compute_engine.xp = np
        compute_engine.use_gpu = False

    yield


@pytest.fixture()
def output_root():
    """Return the root directory for test outputs."""
    root = Path("tests/test_results")
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def output_dir(request, output_root):
    """Return the output directory for the current test."""
    name = request.node.name
    d = output_root / name
    d.mkdir(parents=True, exist_ok=True)
    return d
