"""Typed YAML loading for the dynamic SNN pipeline."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    ComputeEngineParameters,
    Config,
    GridParameters,
    LaserParameters,
    PhysicsConstants,
    PotentialParameters,
    ReservoirParameters,
    ResultParameters,
    SolverParameters,
)
from polarism.config.validation import validate_config

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """MNIST dataset selection.

    Attributes
    ----------
    path
        Path to an ``mnist.npz`` file.
    n_samples
        Number of balanced training samples, or ``None`` for the full train split.
    seed
        Random seed for balanced sampling and downstream splitting.
    """

    path: str
    n_samples: int | None
    seed: int


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """7x7 pump-lattice geometry in micrometers.

    Attributes
    ----------
    center_x_um, center_y_um
        Lattice center coordinates.
    pitch_um
        Nearest-neighbor lattice spacing.
    sigma_space_um
        Gaussian spatial width of each pump spot.
    """

    center_x_um: float
    center_y_um: float
    pitch_um: float
    sigma_space_um: float


@dataclass(frozen=True, slots=True)
class PulseConfig:
    """Shared pulse-Gaussian temporal envelope settings.

    Attributes
    ----------
    sigma_time
        Gaussian temporal width in picoseconds.
    pulse_separation
        Center-to-center pulse separation in picoseconds.
    n_pulses
        Number of pulses emitted by every lattice spot.
    cutoff_sigma
        Temporal cutoff measured in Gaussian widths.
    power_definition
        Polarism pulse-strength interpretation.
    """

    sigma_time: float
    pulse_separation: float
    n_pulses: int
    cutoff_sigma: float
    power_definition: str


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """Linear pixel-to-power mapping bounds.

    Attributes
    ----------
    power_min, power_max
        Powers assigned to image values 0 and 1 respectively.
    """

    power_min: float
    power_max: float


@dataclass(frozen=True, slots=True)
class ReadoutConfig:
    """Dynamic density trace readout settings.

    Attributes
    ----------
    warmup_ps
        Simulation time before trace recording starts.
    stride_steps
        Step interval between recorded trace frames.
    mask_radius_um
        Radius of each per-spot disk readout mask.
    record_reservoir
        Whether active and inactive reservoir traces are recorded.
    feature_mode
        Feature construction mode: ``"raw"``, ``"summary"``, or ``"both"``.
    batch_size
        Number of samples simulated together on the same GPU grid.
    """

    warmup_ps: float
    stride_steps: int
    mask_radius_um: float
    record_reservoir: bool
    feature_mode: str
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Logistic-regression readout hyperparameters.

    Attributes
    ----------
    test_fraction
        Stratified test-split fraction.
    C
        Inverse regularization strength.
    max_iter
        Maximum L-BFGS iterations.
    """

    test_fraction: float
    C: float
    max_iter: int


@dataclass(frozen=True, slots=True)
class SNNDynamicConfig:
    """Top-level configuration for the dynamic SNN pipeline.

    Attributes
    ----------
    polarism_config_path
        Path to the polarism YAML backing the reservoir.
    data, geometry, pulse, encoding, readout, classifier
        Nested typed sections.
    output_dir
        Directory for features/labels/report artifacts.
    """

    polarism_config_path: str
    data: DataConfig
    geometry: GeometryConfig
    pulse: PulseConfig
    encoding: EncodingConfig
    readout: ReadoutConfig
    classifier: ClassifierConfig
    output_dir: str


_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "polarism_config_path",
        "data",
        "geometry",
        "pulse",
        "encoding",
        "readout",
        "classifier",
        "output_dir",
    }
)

_POLARISM_KEYS: frozenset[str] = frozenset(
    {
        "grid",
        "boundary_condition",
        "potential",
        "physics",
        "laser",
        "reservoir",
        "solver",
        "result",
        "compute_engine",
    }
)


def load_snn_dynamic_config(path: str) -> SNNDynamicConfig:
    """Load the dynamic SNN pipeline YAML into frozen dataclasses.

    Parameters
    ----------
    path
        Path to the SNN dynamic YAML file.

    Returns
    -------
    SNNDynamicConfig
        Parsed immutable configuration.
    """
    raw = _load_yaml(path)
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in {path}: {sorted(unknown)}"
        )
    base = Path(path).expanduser().resolve().parent
    polarism_path = _resolve_path(base, str(raw["polarism_config_path"]))
    return SNNDynamicConfig(
        polarism_config_path=str(polarism_path),
        data=_make_dataclass(DataConfig, raw["data"]),
        geometry=_make_dataclass(GeometryConfig, raw["geometry"]),
        pulse=_make_dataclass(PulseConfig, raw["pulse"]),
        encoding=_make_dataclass(EncodingConfig, raw["encoding"]),
        readout=_make_dataclass(ReadoutConfig, raw["readout"]),
        classifier=_make_dataclass(ClassifierConfig, raw["classifier"]),
        output_dir=str(Path(str(raw["output_dir"])).expanduser()),
    )


def load_polarism_config(path: str) -> Config:
    """Load a Polarism YAML file into a validated Config.

    Parameters
    ----------
    path
        Path to the Polarism YAML file.

    Returns
    -------
    Config
        Validated simulation configuration.
    """
    raw = _load_yaml(path)
    unknown = set(raw) - _POLARISM_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in {path}: {sorted(unknown)}"
        )
    cfg = Config(
        grid=_make_dataclass(GridParameters, raw.get("grid", {})),
        boundary_condition=_make_dataclass(
            BoundaryConditionParameters,
            raw.get("boundary_condition", {}),
        ),
        potential=_make_dataclass(PotentialParameters, raw.get("potential", {})),
        physics=_make_dataclass(PhysicsConstants, raw.get("physics", {})),
        laser=_make_dataclass(LaserParameters, raw.get("laser", {})),
        reservoir=_make_dataclass(ReservoirParameters, raw.get("reservoir", {})),
        solver=_make_dataclass(SolverParameters, raw.get("solver", {})),
        result=_make_dataclass(ResultParameters, raw.get("result", {})),
        compute_engine=_make_dataclass(
            ComputeEngineParameters,
            raw.get("compute_engine", {}),
        ),
    )
    validate_config(cfg)
    return cfg


def _load_yaml(path: str) -> dict[str, Any]:
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _make_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    valid = {field.name for field in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}"
        )
    item = cls(**values)
    if isinstance(item, ReadoutConfig) and item.batch_size <= 0:
        raise ValueError(f"ReadoutConfig.batch_size must be positive, got {item.batch_size}")
    return item


def _resolve_path(base: Path, raw_path: str) -> Path:
    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (base / expanded).resolve()
