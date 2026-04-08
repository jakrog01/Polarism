"""Encoder factory."""
from __future__ import annotations

from typing import Any

from polarism.encoder.abstract_encoder import AbstractEncoder
from polarism.encoder.amplitude_encoder import AmplitudeEncoder
from polarism.encoder.separation_encoder import SeparationEncoder


def create_encoder(cfg: dict[str, Any]) -> AbstractEncoder:
    """Construct an encoder from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``"name"`` (``"amplitude"`` or ``"separation"``) and
        ``"params"`` (keyword arguments forwarded to the encoder constructor).

    Returns
    -------
    AbstractEncoder
        Concrete encoder instance.

    Raises
    ------
    ValueError
        If *cfg["name"]* is not a known encoder type.
    """
    name: str = cfg["name"]
    params: dict[str, Any] = cfg.get("params", {})
    if name == "amplitude":
        return AmplitudeEncoder(**params)
    if name == "separation":
        return SeparationEncoder(**params)
    raise ValueError(f"Unknown encoder type: {name!r}")
