"""Small data classes for plotted results."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Results2D:
    """Describe one 2D field view."""
    name: str
    cmap: str
    clim: tuple | None = None


@dataclass
class ResultScalar:
    """Describe one scalar result."""
    name: str
    color: str = "red"


@dataclass
class ResultScalarGroup:
    """Describe a grouped scalar result."""
    name: str
    labels: list[str]
