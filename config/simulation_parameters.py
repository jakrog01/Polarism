from dataclasses import dataclass, field

@dataclass
class GridParameters:
    nx: int = 256
    ny: int = 256
    lx: float = 100.0
    ly: float = 100.0

@dataclass
class Config:
    grid: GridParameters = field(default_factory=GridParameters)