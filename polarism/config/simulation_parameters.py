"""Simulation config dataclasses."""
from dataclasses import dataclass, field


@dataclass
class GridParameters:
    """Store grid size and box-length settings."""
    nx: int = 256
    ny: int = 256
    lx: float = 100.0
    ly: float = 100.0
    grid_type: str = "periodic"



@dataclass
class PhysicsConstants:
    """Store the physics constants for the model."""
    hbar: float = 0.6582119514
    m_eff: float = 0.284
    gamma_R: float = 0.005
    gamma_C: float = 0.083
    g_C: float = 0.001
    g_R: float = 0.002
    gamma_I: float = 0.001
    gamma_A: float = 0.005
    R: float = 0.02
    R_IA: float = 5e-2
    R_AI: float = 2.5e-3
    init_eps: float = 1e-3


@dataclass
class BoundaryConditionParameters:
    """Store boundary absorption settings."""
    profile_type: str = "sin2"
    strength: float = 1.0
    absorption: str = "no-absorption"
    mask_width_percent: float = 0.2


@dataclass
class PotentialParameters:
    """Store potential settings."""
    expose_results: bool = True
    potential_type: str = "zero"
    x1: float = -2.0
    y1: float = 0.0
    x2: float = 2.0
    y2: float = 0.0
    V1: float = -2.0e-3
    V2: float = -2.2e-3
    w1: float = 1.5
    w2: float = 1.5
    order: float = 2.0


@dataclass
class LaserParameters:
    """Store laser settings."""
    expose_results: bool = True
    mode: str = "single"
    config_file: str = "lasers_setup.yaml"
    laser_type: str = "pulse-gaussian"
    P0: float = 0.4
    Pmax: float = 1.2
    x0: float = 0.0
    y0: float = 0.0
    sigma_space: float = 15.0
    sigma_time: float = 0.1
    pulse_separation: float = 0.6
    cutoff_sigma: float = 3.0
    delay: float = 0.0
    n_pulses: int = 0


@dataclass
class ReservoirParameters:
    """Store reservoir settings."""
    expose_results: bool = True
    reservoir_type: str = "double"


@dataclass
class SolverParameters:
    """Store solver settings."""
    total_time: float = 5.0
    dt: float = 1e-3
    method: str = "rk4-fdm"
    precision: str = "double"


@dataclass
class ResultParameters:
    """Store result-output settings."""
    real_time_view: bool = False
    real_time_refresh_interval: float = 0.1
    save_results: bool = False
    save_hdf5: bool = False
    save_json: bool = False
    save_npy: bool = False
    save_interval: int = 1
    batch_size: int = 1000
    output_directory: str = "simulation_results"


@dataclass
class ComputeEngineParameters:
    """Store backend settings."""
    use_gpu: bool = False
    gpu_device: int = 0


@dataclass
class Config:
    """Store the full simulation configuration."""
    grid: GridParameters = field(default_factory=GridParameters)
    boundary_condition: BoundaryConditionParameters = field(
        default_factory=BoundaryConditionParameters
    )
    potential: PotentialParameters = field(default_factory=PotentialParameters)
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    laser: LaserParameters = field(default_factory=LaserParameters)
    reservoir: ReservoirParameters = field(default_factory=ReservoirParameters)
    solver: SolverParameters = field(default_factory=SolverParameters)
    result: ResultParameters = field(default_factory=ResultParameters)
    compute_engine: ComputeEngineParameters = field(
        default_factory=ComputeEngineParameters
    )
