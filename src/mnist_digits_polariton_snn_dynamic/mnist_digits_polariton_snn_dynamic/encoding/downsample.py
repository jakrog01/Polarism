"""MNIST loading and square block-average downsampling."""
from __future__ import annotations

import numpy as np

from mnist_digits_polariton_snn_dynamic.encoding.mnist import load_mnist, select_balanced_subset


def load_and_downsample(
    data_path: str,
    n_samples: int | None,
    seed: int,
    target_side: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST and downsample images to square block averages.

    Parameters
    ----------
    data_path
        Path to an ``mnist.npz`` file.
    n_samples
        Number of balanced training samples to load, or ``None`` for full train.
    seed
        Random seed used for balanced sampling.
    target_side
        Output image side length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Images with shape ``(N, target_side, target_side)`` float64 in ``[0, 1]``
        and labels with shape ``(N,)`` int64.
    """
    side = int(target_side)
    if side <= 0:
        raise ValueError(f"target_side must be positive, got {target_side}")
    if 28 % side != 0:
        raise ValueError(f"target_side must divide 28, got {target_side}")
    x_train, y_train, _, _ = load_mnist(data_path, test_fraction=0.0, seed=seed)
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive or null")
        _, x_train, y_train = select_balanced_subset(
            x_train, y_train, n_samples, seed
        )
    factor = 28 // side
    x = np.asarray(x_train, dtype=np.float64).reshape(-1, 28, 28)
    x = x.reshape(-1, side, factor, side, factor).mean(axis=(2, 4), dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.dtype != np.float64:
        raise TypeError(f"Downsampled images must be float64, got {x.dtype}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Downsampled images contain non-finite values")
    if np.min(x) < 0.0 or np.max(x) > 1.0:
        raise ValueError("Downsampled images must be in [0, 1]")
    y = np.asarray(y_train, dtype=np.int64)
    return x, y
