from dataclasses import dataclass, field


@dataclass
class GridParameters:
    nx: int = 256
    ny: int = 256
    lx: float = 150.0
    ly: float = 150.0


@dataclass
class PhysicsConstants:
    hbar: float = 1.0
    m_eff: float = 1.0
    R: float = 0.5
    gamma_R: float = 0.05
    gamma_C: float = 1.0
    g_C: float = 0.1
    g_R: float = 0.1
    gamma_I: float = 0.01
    gamma_A: float = 0.05
    R_IA: float = 0.05
    R_AI: float = 0.005


@dataclass
class BoundaryConditionParameters:
    profile_type: str = "sin2"
    strength: float = 1.0
    absorption: str = "no-absorption"
    mask_width_percent: float = 0.1


@dataclass
class PotentialParameters:
    expose_results: bool = True
    potential_type: str = "zero"


@dataclass
class LaserParameters:
    expose_results: bool = True
    mode: str = "multiple"
    config_file: str = "lasers_setup.yaml"
    laser_type: str = "pulse-gaussian"
    P0: float = 10.0
    Pmax: float = 20.0
    x0: float = 0.0
    y0: float = 0.0
    sigma_space: float = 10.0
    sigma_time: float = 0.5
    pulse_separation: float = 6.0
    cutoff_sigma: float = 3.0


@dataclass
class ReservoirParameters:
    expose_results: bool = True
    reservoir_type: str = "double"


@dataclass
class SolverParameters:
    total_time: float = 10.0
    dt: float = 1e-3
    method: str = "split-step-fft"


@dataclass
class ResultParameters:
    real_time_view: bool = True
    real_time_refresh_interval: float = 0.1
    save_results: bool = False
    save_hdf5: bool = False
    save_json: bool = False
    save_npy: bool = False
    batch_size: int = 1000
    output_directory: str = "simulation_results"

@dataclass
class ComputeEngineParameters:
    use_gpu: bool = False
    gpu_device: int = 0


@dataclass
class Config:
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
    compute_engine: ComputeEngineParameters = field(default_factory=ComputeEngineParameters)
