"""Per-image amplitude encoding: map classifier scores to pump powers."""
from __future__ import annotations

import numpy as np


def normalize_scores_per_image(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize scores to [0, 1] per image.

    Parameters
    ----------
    scores
        Raw decision-function scores, shape (n_classes,) for one image.
    eps
        Minimum range to avoid division by near-zero.

    Returns
    -------
    norm_scores : (n_classes,) in [0, 1]
    """
    s_min = scores.min()
    s_max = scores.max()
    rng = s_max - s_min
    return (scores - s_min) / max(rng, eps)


def scores_to_powers(
    norm_scores: np.ndarray,
    alpha: float,
    beta: float,
    P_th_single: float,
) -> np.ndarray:
    """Convert normalised scores to pump powers.

    P_c = (alpha + beta * norm_score_c) * P_th_single

    Parameters
    ----------
    norm_scores
        Per-class normalised scores in [0, 1], shape (n_classes,).
    alpha, beta
        Amplitude encoding parameters.
        Resulting P_c ∈ [alpha, alpha+beta] × P_th.
    P_th_single
        Single-spot condensation threshold from calibration.

    Returns
    -------
    powers : (n_classes,) pump powers
    """
    return (alpha + beta * np.clip(norm_scores, 0.0, 1.0)) * P_th_single
