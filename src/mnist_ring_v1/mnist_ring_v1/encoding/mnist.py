"""MNIST loading, train/test split, and projection model fitting."""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def load_mnist(
    data_path: str,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from a Keras-layout .npz file.

    If the file contains only a train split (no x_test), the last
    ``test_fraction`` of each class is held out as the test set.

    Returns
    -------
    x_train, y_train, x_test, y_test
        Float64 images in [0,1], shape (N, 784).
    """
    path = os.path.expanduser(data_path)
    data = np.load(path)

    for train_img_key, train_lbl_key, test_img_key, test_lbl_key in (
        ("x_train", "y_train", "x_test", "y_test"),
        ("training_images", "training_labels", "test_images", "test_labels"),
    ):
        if train_img_key in data and test_img_key in data:
            x_tr = data[train_img_key].astype(np.float64) / 255.0
            y_tr = data[train_lbl_key].astype(np.int64)
            x_te = data[test_img_key].astype(np.float64) / 255.0
            y_te = data[test_lbl_key].astype(np.int64)
            return x_tr.reshape(len(x_tr), -1), y_tr, x_te.reshape(len(x_te), -1), y_te

    for img_key, lbl_key in (("x_train", "y_train"), ("images", "labels")):
        if img_key in data:
            x_all = data[img_key].astype(np.float64) / 255.0
            y_all = data[lbl_key].astype(np.int64)
            x_all = x_all.reshape(len(x_all), -1)
            tr_idx, te_idx = _train_test_split_indices(y_all, test_fraction, seed)
            return x_all[tr_idx], y_all[tr_idx], x_all[te_idx], y_all[te_idx]

    raise ValueError(f"Cannot find image/label arrays in {path}. Keys: {list(data.keys())}")


def _train_test_split_indices(
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split; returns (train_indices, test_indices)."""
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
    images: np.ndarray,
    labels: np.ndarray,
    n_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return balanced subset indices, images, labels.

    Draws floor(n_total/10) samples per class; last class may get one extra
    if n_total is not divisible by 10.

    Returns
    -------
    indices, subset_images, subset_labels
    """
    rng = np.random.default_rng(seed)
    per_class = n_total // 10
    remainder = n_total - per_class * 10

    all_indices: list[int] = []
    for cls in range(10):
        pool = np.where(labels == cls)[0]
        n = per_class + (1 if cls < remainder else 0)
        if len(pool) < n:
            raise ValueError(
                f"Class {cls} has only {len(pool)} samples in the dataset "
                f"but {n} are required (requesting {n_total} total balanced). "
                f"Use a smaller n_total or a larger dataset."
            )
        chosen = rng.choice(pool, size=n, replace=False)
        all_indices.extend(chosen.tolist())

    idx = np.array(all_indices, dtype=np.int64)
    rng.shuffle(idx)
    return idx, images[idx], labels[idx]
