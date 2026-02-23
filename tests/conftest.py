from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine


@pytest.fixture(autouse=True)
def force_numpy_backend():
    compute_engine.xp = np
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
