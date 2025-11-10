from dataclasses import dataclass, field

@dataclass
class GridParameters:
    nx: int = 256
    ny: int = 256
    lx: float = 100.0
    ly: float = 100.0

@dataclass
class AbsorptionParameters:
    profile_type: str = "sin2"
    strength: float = 1.0
    absorption: str = "absorbtion-mask"
    mask_width_percent: float = 0.1

@dataclass
class PhysicsConstants:
    hbar: float = 1.0
    dt: float = 1e-3

@dataclass
class Config:
    grid: GridParameters = field(default_factory=GridParameters)
    boundry_condition: AbsorptionParameters = field(default_factory=AbsorptionParameters)
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)