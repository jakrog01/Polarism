"""Re-export of ``pipeline.simulation.core`` for backward compatibility.

The dot-response-fit pipeline now delegates full spatial simulation to
the shared kernel in ``src/polariton_hpc_pipeline/pipeline/simulation/core.py``.
This avoids maintaining two diverging copies of the simulation loop.

HDF5 field naming follows ``pipeline.simulation.core``:
  * ``psi``  — condensate wavefunction (complex128)
  * ``nA``   — active reservoir density (float64)
  * ``nI``   — inactive reservoir density (float64)
  * ``Pump`` — total pump field (float64)
"""
from pipeline.simulation.core import run_simulation_from_config  # noqa: F401

__all__ = ["run_simulation_from_config"]
