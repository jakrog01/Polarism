"""Amplitude encoder: pixel intensity → pulse amplitude, fixed spacing."""
from __future__ import annotations

import numpy as np


class AmplitudeEncoder:
    """Encode image intensities as pulse amplitudes with uniform time spacing.

    Each pixel maps to a Gaussian pulse whose peak amplitude is linearly
    scaled between *min_amp* and *max_amp*.  Pulse centres are placed at
    fixed multiples of *separation*.

    Parameters
    ----------
    max_amp : float
        Peak amplitude for a pixel value of 1.
    pulse_width : float
        FWHM of each Gaussian pulse (ps).  Stored for downstream use by the
        pump generator; not used during encoding itself.
    separation : float
        Time gap between consecutive pulse centres (ps).
    min_amp : float, optional
        Peak amplitude for a pixel value of 0.  Defaults to 0.
    """

    def __init__(
        self,
        max_amp: float,
        pulse_width: float,
        separation: float,
        min_amp: float = 0.0,
    ) -> None:
        """Initialise the amplitude encoder."""
        self.max_amp = max_amp
        self.min_amp = min_amp
        self.pulse_width = pulse_width
        self.separation = separation

    def encode(self, image_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode *image_flat* into amplitudes and centres.

        Parameters
        ----------
        image_flat : np.ndarray, shape (N,)
            Normalised pixel values in ``[0, 1]``.

        Returns
        -------
        amps : np.ndarray, shape (N,)
            Pulse amplitudes in ``[min_amp, max_amp]``.
        centers : np.ndarray, shape (N,)
            Pulse centre times (ps), starting at 0.
        """
        amps = self.min_amp + image_flat * (self.max_amp - self.min_amp)
        centers = np.arange(len(image_flat), dtype=np.float64) * self.separation
        return amps, centers
