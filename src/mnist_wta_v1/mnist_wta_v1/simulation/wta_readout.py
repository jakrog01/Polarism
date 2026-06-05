"""WTA readout: per-class ROI integrals → winner / margin / flags."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_wta_readout(
    psi_sq: Any,
    class_masks: list[Any],
    xp: Any,
    threshold_cond: float,
) -> dict[str, Any]:
    """Compute winner-takes-all readout from condensate density.

    Parameters
    ----------
    psi_sq
        |ψ|² field (GPU or CPU array).
    class_masks
        List of boolean masks, one per class (same device as psi_sq).
    xp
        Array module (numpy or cupy).
    threshold_cond
        Minimum ROI integral for a class to be considered condensed.
        Derived from calibration blank runs.

    Returns
    -------
    dict with keys:
        class_roi_integrals : list[float]  (n_classes)
        winner              : int
        margin              : float  (I_winner / I_second)
        no_cond             : bool
        multi_cond          : bool  (I_second / I_winner > 0.5)
        power_per_class     : not set here (set by caller from encoding)
    """
    n_classes = len(class_masks)
    integrals = [float(xp.sum(psi_sq * mask)) for mask in class_masks]

    sorted_desc = sorted(integrals, reverse=True)
    winner = int(np.argmax(integrals))
    i_winner = sorted_desc[0]
    i_second = sorted_desc[1] if n_classes > 1 else 0.0

    margin = i_winner / max(i_second, 1e-30)
    no_cond = i_winner < threshold_cond
    multi_cond = (i_second / max(i_winner, 1e-30)) > 0.5

    return {
        "class_roi_integrals": integrals,
        "winner": winner,
        "margin": float(margin),
        "no_cond": bool(no_cond),
        "multi_cond": bool(multi_cond),
    }
