"""Time-only response model subpackage.

Provides a scalar ODE (no spatial grid) that models the polariton condensate
response to a pulsed pump.  The model matches the quadratic-double reservoir
used in the 2-D solver but collapses the spatial dimensions to a single pump
amplitude.

State order throughout this subpackage: ``(nR, nI, nC)``
---------------------------------------------------------
* ``nR`` — active reservoir density
* ``nI`` — inactive reservoir density (directly fed by pump)
* ``nC`` — condensate density (analogue of ``|ψ|²``)

Public API
----------
time_only
    Core ODE: RHS, pulse generator, single-amplitude solver.
sweep
    Amplitude sweep over many pump strengths; integral computation.
"""
from polarism.response import time_only, sweep

__all__ = ["time_only", "sweep"]
