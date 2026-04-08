"""Separation encoder: pixel intensity → inter-pulse separation, fixed amplitude."""
from __future__ import annotations

import numpy as np


class SeparationEncoder:
    """Encode image intensities as inter-pulse time separations at fixed amplitude.

    All pulses share the same peak amplitude *max_amp*.  Each pixel controls
    the gap from the previous pulse centre: a pixel of 0 yields
    *min_separation*, a pixel of 1 yields *max_separation*.  Centres are
    accumulated cumulatively so the sequence is time-ordered.

    Parameters
    ----------
    max_amp : float
        Peak amplitude shared by every pulse.
    pulse_width : float
        FWHM of each Gaussian pulse (ps).  Stored for downstream use.
    min_separation : float
        Minimum inter-pulse interval (ps), for a pixel value of 0.
    max_separation : float
        Maximum inter-pulse interval (ps), for a pixel value of 1.
    """

    def __init__(
        self,
        max_amp: float,
        pulse_width: float,
        min_separation: float,
        max_separation: float,
    ) -> None:
        """Initialise the separation encoder."""
        self.max_amp = max_amp
        self.pulse_width = pulse_width
        self.min_separation = min_separation
        self.max_separation = max_separation

    def encode(self, image_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode *image_flat* into amplitudes and centres.

        Parameters
        ----------
        image_flat : np.ndarray, shape (N,)
            Normalised pixel values in ``[0, 1]``.

        Returns
        -------
        amps : np.ndarray, shape (N,)
            Constant amplitude array, all equal to *max_amp*.
        centers : np.ndarray, shape (N,)
            Cumulative pulse centre times (ps).
        """
        amps = np.full(len(image_flat), self.max_amp, dtype=np.float64)
        separations = (
            self.min_separation
            + image_flat * (self.max_separation - self.min_separation)
        )
        centers = np.zeros(len(image_flat), dtype=np.float64)
        if len(image_flat) > 1:
            centers[1:] = np.cumsum(separations[:-1])
        return amps, centers
