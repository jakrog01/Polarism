"""Offline linear classifier: fit on PCA features, export per-class scores."""
from __future__ import annotations

import numpy as np


def _fit_ridge_numpy(
    X_scaled: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy fallback: one-vs-rest Ridge via normal equations. Returns (W, b, classes)."""
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_feat = X_scaled.shape[1]
    X_b = np.hstack([X_scaled, np.ones((len(X_scaled), 1))])
    Y_oh = (y_train[:, None] == classes[None, :]).astype(np.float64) * 2.0 - 1.0
    A = X_b.T @ X_b + alpha * np.eye(n_feat + 1)
    A[-1, -1] = 0.0
    W_b = np.linalg.solve(A, X_b.T @ Y_oh)
    W = W_b[:-1].T
    b = W_b[-1]
    return W, b, classes


def fit_score_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Fit a Ridge classifier and return a score model dict.

    Parameters
    ----------
    X_train
        PCA-projected + normalised features, shape (n_train, n_features).
    y_train
        Integer class labels, shape (n_train,).
    alpha
        Ridge regularisation parameter.

    Returns
    -------
    score_model dict with keys:
        W          : (n_classes, n_features)
        b          : (n_classes,)
        scaler_mean: (n_features,)
        scaler_std : (n_features,)
        classes    : (n_classes,)
        train_accuracy : float
    """
    scaler_mean = X_train.mean(axis=0)
    scaler_std = np.where(X_train.std(axis=0) > 1e-8, X_train.std(axis=0), 1.0)
    X_scaled = (X_train - scaler_mean) / scaler_std

    try:
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_scaled, y_train)
        W = clf.coef_.astype(np.float64)
        b = clf.intercept_.astype(np.float64)
        classes = clf.classes_.astype(np.int64)
    except ImportError:
        W, b, classes = _fit_ridge_numpy(X_scaled, y_train, alpha)

    scores_tr = X_scaled @ W.T + b
    y_pred_tr = classes[np.argmax(scores_tr, axis=1)]
    train_acc = float(np.mean(y_pred_tr == y_train))

    return {
        "W": W,
        "b": b,
        "scaler_mean": scaler_mean.astype(np.float64),
        "scaler_std": scaler_std.astype(np.float64),
        "classes": classes.astype(np.int64),
        "train_accuracy": train_acc,
    }


def save_score_model(model: dict, path: str) -> None:
    """Save score model to .npz."""
    np.savez_compressed(
        path,
        W=model["W"],
        b=model["b"],
        scaler_mean=model["scaler_mean"],
        scaler_std=model["scaler_std"],
        classes=model["classes"],
        train_accuracy=np.array([model["train_accuracy"]]),
    )


def load_score_model(path: str) -> dict:
    """Load score model from .npz."""
    data = np.load(path)
    return {
        "W": data["W"],
        "b": data["b"],
        "scaler_mean": data["scaler_mean"],
        "scaler_std": data["scaler_std"],
        "classes": data["classes"],
        "train_accuracy": float(data["train_accuracy"][0]),
    }


def predict_scores(
    model: dict,
    X: np.ndarray,
) -> np.ndarray:
    """Compute raw decision-function scores for each class.

    Parameters
    ----------
    model
        Score model from fit_score_model or load_score_model.
    X
        PCA-projected + normalised features, shape (n_samples, n_features).

    Returns
    -------
    scores : (n_samples, n_classes)
    """
    std = np.where(model["scaler_std"] > 1e-8, model["scaler_std"], 1.0)
    X_sc = (X - model["scaler_mean"]) / std
    return X_sc @ model["W"].T + model["b"]
