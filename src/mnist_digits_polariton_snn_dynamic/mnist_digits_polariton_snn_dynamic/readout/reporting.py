"""Headless reporting plots for dynamic SNN readout metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_aggregate(
    cm_counts: np.ndarray,
    cm_normalized: np.ndarray,
    out_path: str | Path,
) -> None:
    """Plot aggregate confusion counts and row-normalized recall matrix.

    Parameters
    ----------
    cm_counts
        Integer confusion matrix with shape ``(10, 10)``.
    cm_normalized
        Row-normalized confusion matrix with shape ``(10, 10)``.
    out_path
        Destination PNG path.
    """
    counts = _as_matrix(cm_counts, np.int64)
    normalized = _as_matrix(cm_normalized, np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)
    _draw_matrix(
        axes[0],
        counts,
        "Counts",
        "d",
        cmap="Blues",
        vmax=float(np.max(counts)) if counts.size else 1.0,
    )
    _draw_matrix(
        axes[1],
        normalized,
        "Row-normalized",
        ".0%",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    out_path: str | Path,
    overall_accuracy: float | None = None,
) -> None:
    """Plot grouped per-class precision, recall, and F1 bars.

    Parameters
    ----------
    precision, recall, f1
        Per-class metric vectors with shape ``(10,)``.
    out_path
        Destination PNG path.
    overall_accuracy
        Optional held-out accuracy shown as a dashed horizontal line.
    """
    p = _as_vector(precision)
    r = _as_vector(recall)
    f = _as_vector(f1)
    classes = np.arange(10)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.bar(classes - width, p, width=width, label="Precision")
    ax.bar(classes, r, width=width, label="Recall")
    ax.bar(classes + width, f, width=width, label="F1")
    if overall_accuracy is not None:
        ax.axhline(
            float(overall_accuracy),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Accuracy",
        )
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(classes)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_confusion_rows(
    cm_normalized: np.ndarray,
    out_path: str | Path,
) -> None:
    """Plot each true class's normalized predicted-label distribution.

    Parameters
    ----------
    cm_normalized
        Row-normalized confusion matrix with shape ``(10, 10)``.
    out_path
        Destination PNG path.
    """
    normalized = _as_matrix(cm_normalized, np.float64)
    classes = np.arange(10)
    fig, axes = plt.subplots(5, 2, figsize=(10.0, 12.0), constrained_layout=True)
    for true_class, ax in enumerate(axes.ravel()):
        colors = np.full(10, "#4c78a8", dtype=object)
        colors[true_class] = "#f58518"
        ax.bar(classes, normalized[true_class], color=colors)
        ax.set_title(f"True {true_class}", fontsize=9)
        ax.set_xticks(classes)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.2)
        if true_class % 2 == 0:
            ax.set_ylabel("Fraction")
        if true_class >= 8:
            ax.set_xlabel("Predicted class")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_field_snapshots_grid(
    rho: np.ndarray,
    times_ps: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    out_path: str | Path,
    lattice_positions_um: np.ndarray | None = None,
) -> None:
    """Plot full-field density snapshots by time and source sample.

    Parameters
    ----------
    rho
        Float64 density frames with shape ``(M, Ny, Nx)``.
    times_ps, sample_indices, labels
        Per-frame metadata vectors with shape ``(M,)``.
    x_um, y_um
        Spatial coordinate vectors for the frame axes.
    out_path
        Destination PNG path.
    lattice_positions_um
        Optional pump-center positions with shape ``(n_spots, 2)``.
    """
    frames = np.asarray(rho, dtype=np.float64)
    times = np.asarray(times_ps, dtype=np.float64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    frame_labels = np.asarray(labels, dtype=np.int64)
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"rho must have non-empty shape (M, Ny, Nx), got {frames.shape}")
    if times.shape != (frames.shape[0],) or samples.shape != (frames.shape[0],):
        raise ValueError("times_ps and sample_indices must align with rho frames")
    if frame_labels.shape != (frames.shape[0],):
        raise ValueError("labels must align with rho frames")
    if x.shape != (frames.shape[2],) or y.shape != (frames.shape[1],):
        raise ValueError("x_um and y_um must align with rho spatial dimensions")
    positions = None
    if lattice_positions_um is not None:
        positions = np.asarray(lattice_positions_um, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("lattice_positions_um must have shape (n_spots, 2)")
    unique_times = np.unique(times)
    unique_samples = np.unique(samples)
    vmax = float(np.percentile(frames, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(frames)) if frames.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    fig, axes = plt.subplots(
        unique_times.size,
        unique_samples.size,
        figsize=(4.0 * unique_samples.size, 3.6 * unique_times.size),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    for row, time_ps in enumerate(unique_times):
        for col, sample_index in enumerate(unique_samples):
            ax = axes[row, col]
            matches = np.flatnonzero((times == time_ps) & (samples == sample_index))
            if matches.size != 1:
                raise ValueError(
                    "Each field-snapshot time and sample pair must occur exactly once"
                )
            frame_index = int(matches[0])
            image = ax.imshow(
                frames[frame_index],
                extent=extent,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                aspect="equal",
            )
            if positions is not None:
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    s=18.0,
                    facecolors="none",
                    edgecolors="white",
                    alpha=0.4,
                )
            ax.set_title(f"sample={sample_index}, label={frame_labels[frame_index]}, t={time_ps:g} ps")
            if col == 0:
                ax.set_ylabel("y [um]")
            if row == unique_times.size - 1:
                ax.set_xlabel("x [um]")
    if image is None:
        raise RuntimeError("No field snapshots available for plotting")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="|psi|^2 [a.u.]")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    fmt: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    threshold = 0.5 * float(np.nanmax(matrix)) if np.size(matrix) else 0.0
    for row in range(10):
        for col in range(10):
            value = matrix[row, col]
            text_color = "white" if float(value) > threshold else "black"
            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )


def _as_matrix(array: np.ndarray, dtype: type[np.integer] | type[np.floating]) -> np.ndarray:
    matrix = np.asarray(array, dtype=dtype)
    if matrix.shape != (10, 10):
        raise ValueError(f"Expected matrix shape (10, 10), got {matrix.shape}")
    return matrix


def _as_vector(array: np.ndarray) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float64)
    if vector.shape != (10,):
        raise ValueError(f"Expected vector shape (10,), got {vector.shape}")
    return vector
