"""Far-field readout: extract psi_k feature vector from final psi state."""
from __future__ import annotations

import math

import numpy as np


def extract_kspace_feature(
    psi: "np.ndarray | Any",
    xp: Any,
    crop_size: int = 64,
    normalization: str = "log_total_power",
) -> np.ndarray:
    """Compute log-power far-field feature from complex wavefunction.

    Parameters
    ----------
    psi
        Complex wavefunction on GPU (CuPy) or CPU (NumPy), shape (ny, nx).
    xp
        Compute backend (numpy or cupy).
    crop_size
        After fftshift, crop to (crop_size x crop_size) centered at DC.
    normalization
        'log_total_power': log10(P / sum(P) + 1e-10), values in [-10, 0].

    Returns
    -------
    feature : np.ndarray of shape (crop_size**2,), CPU float64.
    """
    psi_k = xp.fft.fftshift(xp.fft.fft2(psi))
    power = xp.abs(psi_k) ** 2

    ny, nx = power.shape
    cy, cx = ny // 2, nx // 2
    half = crop_size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    crop = power[y0:y1, x0:x1]

    if normalization == "log_total_power":
        total = float(xp.sum(crop))
        if total < 1e-40:
            total = 1.0
        feat = xp.log10(crop / total + 1e-10)
    else:
        feat = crop

    if hasattr(feat, "get"):
        feat_cpu = feat.get().astype(np.float64).ravel()
    else:
        feat_cpu = np.asarray(feat, dtype=np.float64).ravel()

    return feat_cpu


def extract_scalar_features(sidecar: dict) -> dict[str, float]:
    """Extract readout-relevant scalars from a sidecar dict.

    Sidecar dict is {name: np.ndarray(time-series)}.  We take the value at
    argmax(psi_sq_max) as a snapshot diagnostic (not used as primary readout).
    """
    psi_sq = sidecar.get("psi_sq_max", np.array([0.0]))
    t_peak_idx = int(np.argmax(psi_sq))

    def _at(key: str, default: float = 0.0) -> float:
        arr = sidecar.get(key)
        if arr is None or len(arr) == 0:
            return default
        idx = min(t_peak_idx, len(arr) - 1)
        return float(arr[idx])

    return {
        "psi_sq_max": float(np.max(psi_sq)),
        "k_peak_um": _at("k_peak_um"),
        "k_centroid_um": _at("k_centroid_um"),
        "high_k_frac_0p8_nyq": _at("high_k_frac_0p8_nyq"),
        "k0_frac": _at("k0_frac"),
        "t_peak_ps": float(sidecar["time"][t_peak_idx]) if "time" in sidecar and len(sidecar["time"]) > 0 else 0.0,
    }
