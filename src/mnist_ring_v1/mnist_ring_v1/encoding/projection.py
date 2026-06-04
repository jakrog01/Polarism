"""Projection models: PCA, Gabor bank, random projection.

All models expose a common interface:
  .fit(X_train) -> self
  .transform(X) -> np.ndarray  shape (N, n_components)
  .save(path)
  .load(path) -> cls  (classmethod)

Only PCA is implemented for Phase 1.
"""
from __future__ import annotations

import numpy as np


class PCAProjection:
    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "PCAProjection":
        self.mean_ = X_train.mean(axis=0)
        centered = X_train - self.mean_
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]
        self.explained_variance_ = (s[:k] ** 2) / max(len(X_train) - 1, 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAProjection not fitted")
        return (X - self.mean_) @ self.components_.T

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean=self.mean_,
            components=self.components_,
            explained_variance=self.explained_variance_,
            n_components=np.array([self.n_components]),
        )

    @classmethod
    def load(cls, path: str) -> "PCAProjection":
        data = np.load(path)
        obj = cls(n_components=int(data["n_components"][0]))
        obj.mean_ = data["mean"]
        obj.components_ = data["components"]
        obj.explained_variance_ = data["explained_variance"]
        return obj


def fit_train_normalizer(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature min/max from training features for [0,1] normalization.

    Returns
    -------
    f_min, f_max : arrays of shape (n_components,)
    """
    f_min = features.min(axis=0)
    f_max = features.max(axis=0)
    return f_min, f_max


def apply_normalizer(
    features: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize features to [0,1] using train-set statistics."""
    span = np.maximum(f_max - f_min, eps)
    return np.clip((features - f_min) / span, 0.0, 1.0)
