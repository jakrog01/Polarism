"""Compatibility name for the classic four-stage Lawson RK solver."""
from __future__ import annotations

from polarism.solver.ifrk_fft_solver import IFRKFFTCudaSolver
from polarism.solver.solver_registry import register_solver


@register_solver("ifrk4-fft-cuda")
class IFRK4FFTCudaSolver(IFRKFFTCudaSolver):
    """Use the RK4 tableau regardless of the generic solver's setting."""
    tableau_name = "rk4"
