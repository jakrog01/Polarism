"""Shared MNIST loading and balanced subset selection."""
from __future__ import annotations

import os

import numpy as np


def load_mnist(
    data_path: str,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from a Keras-layout .npz file.

    If the file has only a train split (no x_test/y_test keys), the last
    ``test_fraction`` of each class is held out as the test set.

    Returns
    -------
    x_train, y_train, x_test, y_test
        Float64 images in [0, 1], shape (N, 784).
    """
    path = os.path.expanduser(data_path)
    data = np.load(path)

    for tr_img, tr_lbl, te_img, te_lbl in (
        ("x_train", "y_train", "x_test", "y_test"),
        ("training_images", "training_labels", "test_images", "test_labels"),
    ):
        if tr_img in data and te_img in data:
            x_tr = data[tr_img].astype(np.float64) / 255.0
            y_tr = data[tr_lbl].astype(np.int64)
            x_te = data[te_img].astype(np.float64) / 255.0
            y_te = data[te_lbl].astype(np.int64)
            return x_tr.reshape(len(x_tr), -1), y_tr, x_te.reshape(len(x_te), -1), y_te

    for img_key, lbl_key in (("x_train", "y_train"), ("images", "labels")):
        if img_key in data:
            x_all = data[img_key].astype(np.float64) / 255.0
            y_all = data[lbl_key].astype(np.int64)
            x_all = x_all.reshape(len(x_all), -1)
            tr_idx, te_idx = _stratified_split(y_all, test_fraction, seed)
            return x_all[tr_idx], y_all[tr_idx], x_all[te_idx], y_all[te_idx]

    raise ValueError(f"Cannot find image/label arrays in {path}. Keys: {list(data.keys())}")


def _stratified_split(
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(labels):
        pool = np.where(labels == cls)[0]
        rng.shuffle(pool)
        n_test = max(1, int(len(pool) * test_fraction))
        test_idx.extend(pool[:n_test].tolist())
        train_idx.extend(pool[n_test:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def select_balanced_subset(
    x: np.ndarray,
    y: np.ndarray,
    n_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (indices, x_subset, y_subset) with balanced class representation.

    Draws floor(n_total/10) samples per class; last class may get one extra
    if n_total is not divisible by 10.

    Returns
    -------
    indices, x_subset, y_subset
    """
    rng = np.random.default_rng(seed)
    per_class = n_total // 10
    remainder = n_total - per_class * 10

    all_indices: list[int] = []
    for cls in range(10):
        pool = np.where(y == cls)[0]
        n = per_class + (1 if cls < remainder else 0)
        if len(pool) < n:
            raise ValueError(
                f"Not enough samples for class {cls}: need {n}, have {len(pool)}"
            )
        chosen = rng.choice(pool, size=n, replace=False)
        all_indices.extend(chosen.tolist())

    idx = np.array(all_indices, dtype=np.int64)
    rng.shuffle(idx)
    return idx, x[idx], y[idx]
