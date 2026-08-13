from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_digits_polariton_snn_dynamic.encoding.base import LinearPixelEncoder
from mnist_digits_polariton_snn_dynamic.simulation.geometry import GridLatticeGeometry


def _old_49_positions(cx: float, cy: float, pitch: float) -> np.ndarray:
    offsets = (np.arange(7, dtype=np.float64) - 3.0) * np.float64(pitch)
    xs, ys = np.meshgrid(
        offsets + np.float64(cx),
        offsets + np.float64(cy),
        indexing="xy",
    )
    return np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64, copy=False)


def _old_encode_powers(images: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    return pmin + images.reshape(images.shape[0], 49) * (pmax - pmin)


old_pos = _old_49_positions(0.0, 0.0, 12.0)
new_pos = GridLatticeGeometry(7, 12.0, 2.0, 0.0, 0.0).positions_um
assert np.array_equal(old_pos, new_pos), (
    f"positions bit-exact broken: max abs diff = {np.abs(old_pos - new_pos).max()}"
)

rng = np.random.default_rng(0)
x = rng.random((5, 7, 7))
old_p = _old_encode_powers(x, 0.0, 800.0)
new_p = LinearPixelEncoder(7, 0.0, 800.0).encode(x)
assert np.array_equal(old_p, new_p), (
    f"encoder bit-exact broken: max abs diff = {np.abs(old_p - new_p).max()}"
)

print("OK: positions + encoder bit-exact for n_side=7, pitch=12, sigma=2, power=(0,800)")
