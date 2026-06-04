"""Time-to-First-Spike encoding for the ring spots.

Convention: t_i = delay field in Polarism (Gaussian peak fires at
delay + cutoff_sigma * sigma_time ≈ delay + 4.5 ps).

Mapping: t_i = T_max * (1 - clip(f_i, 0, 1))
  - f_i = 1  -> t_i = 0  (earliest firing, brightest feature)
  - f_i = 0  -> t_i = T_max (latest firing)
"""
from __future__ import annotations

import numpy as np


def features_to_delays(
    features_normalized: np.ndarray,
    T_max: float,
) -> np.ndarray:
    """Convert normalized features in [0,1] to Polarism delay values (ps).

    Parameters
    ----------
    features_normalized
        Array of shape (n_features,) with values in [0, 1].
    T_max
        Maximum delay window in ps.  Corresponds to the latest ring firing.

    Returns
    -------
    delays : array of shape (n_features,)  in ps.
    """
    f = np.clip(features_normalized, 0.0, 1.0)
    return T_max * (1.0 - f)


def trigger_delay(T_max: float, trigger_offset_ps: float = 0.0) -> float:
    """Return the trigger (central spot) delay.

    The trigger fires after all ring spots: delay = T_max + trigger_offset.
    Default offset is 0 so the trigger fires immediately after the latest ring spot.
    """
    return T_max + trigger_offset_ps
