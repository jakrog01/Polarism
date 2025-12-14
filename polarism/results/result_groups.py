from dataclasses import dataclass

@dataclass
class Results2D:
    name: str
    cmap: str
    clim: tuple | None = None

@dataclass
class ResultScalar:
    name: str
    color: str = "red"

@dataclass
class ResultScalarGroup:
    name: str
    labels: list[str] 