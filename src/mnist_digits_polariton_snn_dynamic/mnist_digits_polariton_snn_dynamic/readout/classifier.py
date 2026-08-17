"""Linear classifier readout for dynamic SNN features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mnist_digits_polariton_snn_dynamic.readout.features import FeatureBundle


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Classifier metrics for a stratified train/test readout split.

    Attributes
    ----------
    accuracy_train
        Accuracy on the training split.
    accuracy_test
        Accuracy on the held-out test split.
    confusion_test
        Confusion matrix over digit labels, shape ``(10, 10)`` int64.
    confusion_test_normalized
        Row-normalized confusion matrix over digit labels, shape ``(10, 10)`` float64.
    per_class_precision_test
        Per-class precision scores, shape ``(10,)`` float64.
    per_class_recall_test
        Per-class recall scores, shape ``(10,)`` float64.
    per_class_f1_test
        Per-class F1 scores, shape ``(10,)`` float64.
    n_train
        Number of training samples.
    n_test
        Number of test samples.
    test_indices
        Original sample indices assigned to the test split.
    predictions_test
        Predicted integer labels aligned with ``test_indices``.
    """

    accuracy_train: float
    accuracy_test: float
    confusion_test: np.ndarray
    confusion_test_normalized: np.ndarray
    per_class_precision_test: np.ndarray
    per_class_recall_test: np.ndarray
    per_class_f1_test: np.ndarray
    n_train: int
    n_test: int
    test_indices: np.ndarray
    predictions_test: np.ndarray


def train_and_evaluate(
    bundle: FeatureBundle,
    seed: int,
    test_fraction: float = 0.20,
    C: float = 1.0,
    max_iter: int = 2000,
) -> ClassificationReport:
    """Train multinomial logistic regression and compute metrics.

    Parameters
    ----------
    bundle
        Feature matrix and labels.
    seed
        Random seed for the stratified split.
    test_fraction
        Fraction of samples held out for testing.
    C
        Logistic-regression inverse regularization strength.
    max_iter
        Maximum solver iterations.

    Returns
    -------
    ClassificationReport
        Accuracy, confusion matrix, per-class F1, and split metadata.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    indices = np.arange(bundle.labels.shape[0], dtype=np.int64)
    x_train, x_test, y_train, y_test, _, test_idx = train_test_split(
        bundle.features,
        bundle.labels,
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=bundle.labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            n_jobs=-1,
        ),
    )
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    labels = np.arange(10, dtype=np.int64)
    cm_counts = np.asarray(
        confusion_matrix(y_test, y_pred_test, labels=labels), dtype=np.int64
    )
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=row_sums > 0,
    )
    return ClassificationReport(
        accuracy_train=float(accuracy_score(y_train, y_pred_train)),
        accuracy_test=float(accuracy_score(y_test, y_pred_test)),
        confusion_test=cm_counts,
        confusion_test_normalized=np.asarray(cm_normalized, dtype=np.float64),
        per_class_precision_test=np.asarray(
            precision_score(
                y_test,
                y_pred_test,
                average=None,
                labels=labels,
                zero_division=0.0,
            ),
            dtype=np.float64,
        ),
        per_class_recall_test=np.asarray(
            recall_score(
                y_test,
                y_pred_test,
                average=None,
                labels=labels,
                zero_division=0.0,
            ),
            dtype=np.float64,
        ),
        per_class_f1_test=np.asarray(
            f1_score(
                y_test,
                y_pred_test,
                average=None,
                labels=labels,
                zero_division=0.0,
            ),
            dtype=np.float64,
        ),
        n_train=int(y_train.shape[0]),
        n_test=int(y_test.shape[0]),
        test_indices=np.asarray(test_idx, dtype=np.int64),
        predictions_test=np.asarray(y_pred_test, dtype=np.int64),
    )
