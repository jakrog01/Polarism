from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters


def pytest_addoption(parser):
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Run benchmarks on GPU (requires CuPy)",
    )


@pytest.fixture(scope="session")
def use_gpu(request):
    return request.config.getoption("--use-gpu", default=True)


@pytest.fixture(autouse=True)
def configure_backend(request, use_gpu):
    is_benchmark = "phoenix_benchmark" in str(request.fspath)

    if use_gpu and is_benchmark:
        engine_cfg = ComputeEngineParameters(use_gpu=True)
        compute_engine.configure(engine_cfg)
    else:
        compute_engine.xp = np
        compute_engine.use_gpu = False

    yield


@pytest.fixture()
def output_root():
    root = Path("tests/test_results")
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def output_dir(request, output_root):
    name = request.node.name
    d = output_root / name
    d.mkdir(parents=True, exist_ok=True)
    return d
