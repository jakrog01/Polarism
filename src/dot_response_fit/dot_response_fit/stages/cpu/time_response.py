"""REMOVED from pipeline — use prepare_reference instead.

This stage swept a range of pulse amplitudes through the time-only ODE and
compared the resulting amplitude-response curve to 2-D spatial simulations.
That approach has been replaced by :mod:`dot_response_fit.stages.cpu.prepare_reference`,
which encodes an actual MNIST digit into temporal pulses and uses those pulses
as both the reference ODE trace and the input to the 2-D spatial simulation.

This file is kept only for historical reference.  ``main()`` raises
``RuntimeError`` immediately.
"""
from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "time_response has been removed from the pipeline.  "
        "Use dot_response_fit.stages.cpu.prepare_reference (Stage 1) instead."
    )


if __name__ == "__main__":
    print(
        "ERROR: time_response is no longer part of the pipeline.  "
        "Use prepare_reference (Stage 1) instead.",
        file=sys.stderr,
    )
    sys.exit(1)
