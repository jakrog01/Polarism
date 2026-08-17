"""Abstract encoder protocol."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class AbstractEncoder(Protocol):
    """Encoder that maps a flattened image vector to pulse parameters.

    An encoder turns a 1-D array of normalised pixel values (in ``[0, 1]``)
    into two arrays of the same length:

    * **amplitudes** — peak power of each Gaussian pulse.
    * **centers** — temporal centre of each Gaussian pulse (ps).

    Both arrays are consumed downstream to build the pump interpolant.
    """

    def encode(
        self, image_flat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a flattened image into pulse amplitudes and centres.

        Parameters
        ----------
        image_flat : np.ndarray, shape (N,)
            Normalised input values in ``[0, 1]``.

        Returns
        -------
        amps : np.ndarray, shape (N,)
            Pulse amplitudes.
        centers : np.ndarray, shape (N,)
            Pulse centre times (ps).
        """
        ...
