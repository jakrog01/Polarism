"""Pulse encoder subpackage.

Encoders map a flattened image or signal vector onto pulse amplitudes and
pulse centre times, ready for use as pump envelopes.

Public API
----------
AmplitudeEncoder
    Maps pixel intensities to pulse amplitudes with fixed spacing.
"""
from polarism.encoder.abstract_encoder import AbstractEncoder
from polarism.encoder.amplitude_encoder import AmplitudeEncoder

__all__ = [
    "AbstractEncoder",
    "AmplitudeEncoder",
]
