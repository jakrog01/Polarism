"""Feature collection for dynamic SNN simulations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mnist_digits_polariton_snn_dynamic.config.loader import ReadoutConfig
from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources
from mnist_digits_polariton_snn_dynamic.simulation.runner_batched import simulate_batch
from mnist_digits_polariton_snn_dynamic.simulation.runner import simulate_one_image


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Simulation features aligned with integer labels.

    Attributes
    ----------
    features
        Float64 classifier feature matrix with shape ``(N, D)``.
    labels
        Int64 labels with shape ``(N,)``.
    traces_psi
        Condensate density traces with shape ``(N, T_out, n_spots + 1)``.
    traces_n_active
        Active reservoir traces with shape ``(N, T_out, n_spots + 1)``, or ``None``.
    traces_n_inactive
        Inactive reservoir traces with shape ``(N, T_out, n_spots + 1)``, or ``None``.
    trace_times_ps
        Recorded trace times in picoseconds with shape ``(T_out,)``.
    feature_shape
        Per-sample feature shape before flattening.
    """

    features: np.ndarray
    labels: np.ndarray
    traces_psi: np.ndarray
    traces_n_active: np.ndarray | None
    traces_n_inactive: np.ndarray | None
    trace_times_ps: np.ndarray
    feature_shape: tuple[int, ...]


def collect_features(
    resources: SharedSimResources,
    powers: np.ndarray,
    labels: np.ndarray,
    readout: ReadoutConfig,
) -> FeatureBundle:
    """Collect dynamic trace features for each sample.

    Parameters
    ----------
    resources
        Shared simulation resources.
    powers
        Float64 encoded pump powers with shape ``(N, n_spots)``.
    labels
        Int64 labels with shape ``(N,)``.
    readout
        Dynamic trace recording and feature construction settings.

    Returns
    -------
    FeatureBundle
        Feature matrix, raw traces, aligned labels, and trace times.
    """
    p = np.asarray(powers, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != resources.n_spots:
        raise ValueError(f"powers shape must be (N, {resources.n_spots}), got {p.shape}")
    if p.shape[0] == 0:
        raise ValueError("powers must contain at least one sample")
    if y.shape != (p.shape[0],):
        raise ValueError(f"labels shape must be ({p.shape[0]},), got {y.shape}")
    if readout.feature_mode not in {"raw", "summary", "both"}:
        raise ValueError(
            "readout.feature_mode must be one of 'raw', 'summary', or 'both', "
            f"got {readout.feature_mode!r}"
        )
    if readout.batch_size > 1:
        traces = simulate_batch(
            resources,
            p,
            readout.warmup_ps,
            readout.stride_steps,
            readout.record_reservoir,
            readout.batch_size,
        )
        feature0 = _build_feature_vector(
            traces.psi[0],
            traces.n_active[0] if traces.n_active is not None else None,
            traces.n_inactive[0] if traces.n_inactive is not None else None,
            traces.times_ps,
            readout.feature_mode,
        )
        features = np.empty((p.shape[0], feature0.size), dtype=np.float64)
        features[0] = feature0
        for i in range(1, p.shape[0]):
            features[i] = _build_feature_vector(
                traces.psi[i],
                traces.n_active[i] if traces.n_active is not None else None,
                traces.n_inactive[i] if traces.n_inactive is not None else None,
                traces.times_ps,
                readout.feature_mode,
            )
        return FeatureBundle(
            features=features,
            labels=y,
            traces_psi=traces.psi,
            traces_n_active=traces.n_active,
            traces_n_inactive=traces.n_inactive,
            trace_times_ps=traces.times_ps,
            feature_shape=(feature0.size,),
        )
    first = simulate_one_image(
        resources,
        p[0],
        readout.warmup_ps,
        readout.stride_steps,
        readout.record_reservoir,
    )
    traces_psi = np.empty((p.shape[0], *first.psi.shape), dtype=np.float64)
    traces_psi[0] = first.psi
    traces_n_active = (
        np.empty((p.shape[0], *first.n_active.shape), dtype=np.float64)
        if first.n_active is not None
        else None
    )
    traces_n_inactive = (
        np.empty((p.shape[0], *first.n_inactive.shape), dtype=np.float64)
        if first.n_inactive is not None
        else None
    )
    if traces_n_active is not None and first.n_active is not None:
        traces_n_active[0] = first.n_active
    if traces_n_inactive is not None and first.n_inactive is not None:
        traces_n_inactive[0] = first.n_inactive
    feature0 = _build_feature_vector(
        first.psi,
        first.n_active,
        first.n_inactive,
        first.times_ps,
        readout.feature_mode,
    )
    features = np.empty((p.shape[0], feature0.size), dtype=np.float64)
    features[0] = feature0
    for i in range(p.shape[0]):
        if i == 0:
            continue
        traces = simulate_one_image(
            resources,
            p[i],
            readout.warmup_ps,
            readout.stride_steps,
            readout.record_reservoir,
        )
        traces_psi[i] = traces.psi
        if traces_n_active is not None and traces.n_active is not None:
            traces_n_active[i] = traces.n_active
        if traces_n_inactive is not None and traces.n_inactive is not None:
            traces_n_inactive[i] = traces.n_inactive
        features[i] = _build_feature_vector(
            traces.psi,
            traces.n_active,
            traces.n_inactive,
            traces.times_ps,
            readout.feature_mode,
        )
    return FeatureBundle(
        features=features,
        labels=y,
        traces_psi=traces_psi,
        traces_n_active=traces_n_active,
        traces_n_inactive=traces_n_inactive,
        trace_times_ps=first.times_ps,
        feature_shape=(feature0.size,),
    )


def _build_feature_vector(
    psi: np.ndarray,
    n_active: np.ndarray | None,
    n_inactive: np.ndarray | None,
    times_ps: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    traces = _feature_traces(psi, n_active, n_inactive)
    raw = traces.reshape(-1)
    summary = _summarize_traces(traces, times_ps).reshape(-1)
    if feature_mode == "raw":
        return np.ascontiguousarray(raw, dtype=np.float64)
    if feature_mode == "summary":
        return np.ascontiguousarray(summary, dtype=np.float64)
    return np.ascontiguousarray(np.concatenate((summary, raw)), dtype=np.float64)


def _feature_traces(
    psi: np.ndarray,
    n_active: np.ndarray | None,
    n_inactive: np.ndarray | None,
) -> np.ndarray:
    arrays = [np.asarray(psi, dtype=np.float64)]
    if n_active is not None:
        arrays.append(np.asarray(n_active, dtype=np.float64))
    if n_inactive is not None:
        arrays.append(np.asarray(n_inactive, dtype=np.float64))
    return np.concatenate(arrays, axis=1)


def _summarize_traces(traces: np.ndarray, times_ps: np.ndarray) -> np.ndarray:
    x = np.asarray(traces, dtype=np.float64)
    t = np.asarray(times_ps, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"traces must have shape (T, K), got {x.shape}")
    if t.shape != (x.shape[0],):
        raise ValueError(f"times_ps shape must be ({x.shape[0]},), got {t.shape}")
    peak_idx = np.argmax(x, axis=0)
    channels = np.arange(x.shape[1])
    peak = x[peak_idx, channels]
    t_peak = t[peak_idx]
    area = np.trapezoid(x, t, axis=0)
    post_peak = t[:, None] >= t_peak[None, :]
    below = x <= (peak[None, :] / np.e)
    candidates = post_peak & below
    first_below_idx = np.argmax(candidates, axis=0)
    has_below = np.any(candidates, axis=0)
    tau = np.where(has_below, t[first_below_idx] - t_peak, t[-1] - t_peak)
    return np.stack((peak, t_peak, area, tau), axis=1)
