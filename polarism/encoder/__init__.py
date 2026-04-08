"""Pulse encoder subpackage.

Encoders map a flattened image or signal vector onto pulse amplitudes and
pulse centre times, ready for use as pump envelopes in the ODE time-only
response model.

Public API
----------
AmplitudeEncoder
    Maps pixel intensities to pulse amplitudes with fixed spacing.
SeparationEncoder
    Maps pixel intensities to inter-pulse separations with fixed amplitude.
create_encoder
    Factory that constructs an encoder from a config dict.
"""
from polarism.encoder.amplitude_encoder import AmplitudeEncoder
from polarism.encoder.separation_encoder import SeparationEncoder
from polarism.encoder.create_encoder import create_encoder

__all__ = [
    "AmplitudeEncoder",
    "SeparationEncoder",
    "create_encoder",
]
