"""Spatial heatmap reporting for saved dynamic SNN traces."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def generate_trace_heatmaps(
    output_dir: str | Path,
    n_side: int | None = None,
    samples_per_label: int = 3,
) -> list[Path]:
    """Generate per-sample spatial heatmaps from saved trace arrays.

    Parameters
    ----------
    output_dir
        Scenario output directory containing ``traces_psi.npy`` and ``labels.npy``.
    n_side
        Optional lattice side length. If omitted, it is inferred from the spot count.
    samples_per_label
        Number of first-seen samples selected for every unique label.

    Returns
    -------
    list[Path]
        PNG paths written to ``output_dir``.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    traces = np.asarray(np.load(out_dir / "traces_psi.npy"), dtype=np.float64)
    labels = np.asarray(np.load(out_dir / "labels.npy"), dtype=np.int64)
    side = _resolve_n_side(traces, n_side)
    selected = _select_samples(labels, int(samples_per_label))
    written = []
    for sample_idx, label, sample_rank in selected:
        frame_indices = _frame_indices(traces[sample_idx, :, : side * side])
        path = out_dir / f"heatmap_label{label}_sample{sample_rank}.png"
        _plot_sample_heatmap(
            traces[sample_idx, :, : side * side],
            label,
            sample_idx,
            frame_indices,
            side,
            path,
        )
        written.append(path)
    return written


def _resolve_n_side(traces: np.ndarray, n_side: int | None) -> int:
    if traces.ndim != 3:
        raise ValueError(
            f"Expected traces_psi shape (N, T, channels), got {traces.shape}"
        )
    n_spots = int(traces.shape[2] - 1)
    if n_spots <= 0:
        raise ValueError(
            f"Expected at least one spot channel plus CAP channel, got {traces.shape[2]}"
        )
    if n_side is not None:
        side = int(n_side)
        if side * side != n_spots:
            raise ValueError(f"n_side={side} is incompatible with {n_spots} spot channels")
        return side
    side = int(round(np.sqrt(n_spots)))
    if side * side != n_spots:
        raise ValueError(f"Cannot infer square lattice side from {n_spots} spot channels")
    return side


def _select_samples(
    labels: np.ndarray,
    samples_per_label: int,
) -> list[tuple[int, int, int]]:
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (N,), got {labels.shape}")
    if samples_per_label <= 0:
        raise ValueError(f"samples_per_label must be positive, got {samples_per_label}")
    selected = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)[:samples_per_label]
        for sample_rank, sample_idx in enumerate(indices):
            selected.append((int(sample_idx), int(label), int(sample_rank)))
    return selected


def _frame_indices(sample_traces: np.ndarray) -> tuple[int, int, int]:
    density = np.sum(sample_traces, axis=1, dtype=np.float64)
    return 0, int(np.argmax(density)), int(sample_traces.shape[0] - 1)


def _plot_sample_heatmap(
    sample_traces: np.ndarray,
    label: int,
    sample_idx: int,
    frame_indices: tuple[int, int, int],
    n_side: int,
    out_path: Path,
) -> None:
    titles = ("t=0", "t_peak", "t_final")
    frames = [sample_traces[index].reshape(n_side, n_side) for index in frame_indices]
    vmax = max(float(np.nanmax(frame)) for frame in frames)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), constrained_layout=True)
    for ax, title, frame, frame_idx in zip(
        axes,
        titles,
        frames,
        frame_indices,
        strict=True,
    ):
        image = ax.imshow(
            frame,
            cmap="magma",
            vmin=0.0,
            vmax=vmax if vmax > 0.0 else None,
        )
        ax.set_title(f"{title} ({frame_idx})")
        ax.set_xticks(np.arange(n_side))
        ax.set_yticks(np.arange(n_side))
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"label={label}, sample_index={sample_idx}", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
