from dataclasses import dataclass, field

@dataclass
class GridParameters:
    nx: int = 256
    ny: int = 256
    lx: float = 100.0
    ly: float = 100.0

@dataclass
class PhysicsConstants:
    hbar: float = 1.0
    m_eff: float = 1.0
    R: float = 0.01
    gamma_R: float = 0.1
    gamma_C: float = 0.05
    g_C: float = 1.0
    g_R: float = 0.5

@dataclass
class BoundaryConditionParameters:
    profile_type: str = "sin2"
    strength: float = 1.0
    absorption: str = "no-absorption"
    mask_width_percent: float = 0.1

@dataclass
class PotentialParameters:
    potential_type: str = "zero"

@dataclass
class LaserParameters:
    laser_count = 1
    laser_type: str = "continuous-gaussian"
    wavelength: float = 800.0
    intensity: float = 1e6
    P0: float = 1.0
    Pmax: float = 100.0
    x0: float = 0.0
    y0: float = 0.0
    sigma_space: float = 1e-2
    pulse_duration: float = 3e-2

@dataclass
class ReservoirParameters:
    reservoir_type: str = "single"

@dataclass
class SolverParameters:
    total_time: float = 100.0
    dt: float = 1e-3
    method: str = "split-step-fft"

@dataclass
class Config:
    grid: GridParameters = field(default_factory=GridParameters)
    boundry_condition: BoundaryConditionParameters = field(default_factory=BoundaryConditionParameters)
    potential: PotentialParameters = field(default_factory=PotentialParameters)
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    laser: LaserParameters = field(default_factory=LaserParameters)
    reservoir: ReservoirParameters = field(default_factory=ReservoirParameters)
    solver: SolverParameters = field(default_factory=SolverParameters)
