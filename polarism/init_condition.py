"""Initial wavefunction generation for the polariton GPE."""
from __future__ import annotations

from typing import Optional


def make_initial_psi(
    xp,
    ny: int,
    nx: int,
    eps: float,
    mode: str = "legacy_positive_uniform",
    dx: float = 1.0,
    dy: float = 1.0,
    k_cutoff_um: Optional[float] = None,
    seed: Optional[int] = None,
    cdtype=None,
    rdtype=None,
):
    """Generate the initial wavefunction for a polariton GPE simulation.

    Parameters
    ----------
    xp : module
        Compute backend (numpy or cupy).
    ny, nx : int
        Grid dimensions (rows, columns).
    eps : float
        Target RMS amplitude of the initial wavefunction.
    mode : str
        Initialisation mode.  One of:

        ``'legacy_positive_uniform'``
            Uniform [0, 1) real and imaginary parts, scaled by *eps*.
            Reproduces the pre-refactor behaviour exactly.

        ``'complex_gaussian_zero_mean'``
            Zero-mean complex Gaussian, RMS-normalised to *eps*.

        ``'filtered_complex_gaussian'``
            Zero-mean complex Gaussian low-pass filtered in k-space to
            ``|k| <= k_cutoff_um``, then mean-subtracted and RMS-renormalised
            to *eps*.  Requires *k_cutoff_um*.
    dx, dy : float
        Grid spacings in micrometres (used for the k-space filter).
    k_cutoff_um : float or None
        Low-pass cutoff ``|k|`` in rad/um.  Required when
        ``mode='filtered_complex_gaussian'``; ignored otherwise.
    seed : int or None
        RNG seed.  ``None`` → non-deterministic.
    cdtype : dtype or None
        Target complex dtype.  Defaults to ``xp.complex128``.
    rdtype : dtype or None
        Working real dtype.  Defaults to ``xp.float64``.

    Returns
    -------
    psi : array, shape (ny, nx), dtype cdtype
    """
    if cdtype is None:
        cdtype = xp.complex128
    if rdtype is None:
        rdtype = xp.float64

    rng = xp.random.default_rng(seed)

    if mode == "legacy_positive_uniform":
        psi = eps * (
            rng.random((ny, nx), dtype=rdtype)
            + 1j * rng.random((ny, nx), dtype=rdtype)
        )
        return psi.astype(cdtype)

    if mode == "complex_gaussian_zero_mean":
        re = rng.standard_normal((ny, nx)).astype(rdtype)
        im = rng.standard_normal((ny, nx)).astype(rdtype)
        psi = re + 1j * im
        psi -= xp.mean(psi)
        rms = float(xp.sqrt(xp.mean(xp.abs(psi) ** 2)))
        if rms < 1e-30:
            rms = 1.0
        return ((eps / rms) * psi).astype(cdtype)

    if mode == "filtered_complex_gaussian":
        if k_cutoff_um is None:
            raise ValueError(
                "init_mode='filtered_complex_gaussian' requires init_k_cutoff_um to be set."
            )
        re = rng.standard_normal((ny, nx)).astype(rdtype)
        im = rng.standard_normal((ny, nx)).astype(rdtype)
        psi = (re + 1j * im).astype(xp.complex128)

        kx = 2.0 * xp.pi * xp.fft.fftfreq(nx, d=dx)
        ky = 2.0 * xp.pi * xp.fft.fftfreq(ny, d=dy)
        KX, KY = xp.meshgrid(kx, ky)
        k_r = xp.sqrt(KX ** 2 + KY ** 2)

        psi_k = xp.fft.fft2(psi)
        n_modes_inside = int(xp.sum(k_r <= k_cutoff_um))
        if n_modes_inside <= 1:
            raise ValueError(
                f"init_k_cutoff_um={k_cutoff_um} admits only the DC bin on this grid "
                f"(dx={dx}, dy={dy}). Increase init_k_cutoff_um or use a finer grid."
            )
        psi_k = psi_k * (k_r <= k_cutoff_um)
        psi = xp.fft.ifft2(psi_k)

        psi -= xp.mean(psi)
        rms = float(xp.sqrt(xp.mean(xp.abs(psi) ** 2)))
        if rms < 1e-30:
            raise ValueError(
                f"filtered_complex_gaussian produced a near-zero wavefunction after "
                f"low-pass filtering (rms={rms:.3g}). init_k_cutoff_um={k_cutoff_um} "
                "is too small for this grid — increase it or use a finer grid."
            )
        return ((eps / rms) * psi).astype(cdtype)

    raise ValueError(
        f"Unknown init_mode '{mode}'. "
        "Valid values: 'legacy_positive_uniform', 'complex_gaussian_zero_mean', "
        "'filtered_complex_gaussian'."
    )
