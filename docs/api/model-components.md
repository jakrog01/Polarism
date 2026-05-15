# Model Components

This page describes the main interchangeable building blocks used by the simulation controller.

## Grid interface

The grid layer is defined by the `SimulationGrid2D` protocol. Concrete grids must provide:

- domain sizes and resolution: `nx`, `ny`, `lx`, `ly`
- real-space coordinates: `X`, `Y`
- spacings: `dx`, `dy`
- spectral coordinates: `kx`, `ky`, `KX`, `KY`, `k_squared`

### Supported grid types

| `grid_type` | Meaning |
| --- | --- |
| `periodic` | periodic spatial topology, natural for FFT-based methods |
| `closed-interval` | bounded interval with finite-difference-oriented treatment |

## Boundary-condition wrapper

`BoundaryCondition` wraps the boundary absorption strategy around each solver step.

### Supported absorption strategies

| `absorption` | Meaning |
| --- | --- |
| `no-absorption` | no damping near the boundary |
| `mask` | multiplicative damping mask applied near the edges |
| `cap` | complex absorbing potential added near the edges |

## Potential factory

Potentials are chosen by `cfg.potential.potential_type` and built as spatial fields on the active grid.

### Supported potential types

| `potential_type` | Meaning |
| --- | --- |
| `zero` | no external potential |
| `double-well-supergaussian` | structured two-well profile with configurable centers, amplitudes, widths, and order |

## Laser interface and factory

Pump profiles share the `AbstractLaser` contract:

- accept `LaserParameters` and grid coordinates
- compute a pump field through spatial, temporal, and amplitude factors
- return the full pump power field at time `t`

### Supported laser types

| `laser_type` | Meaning |
| --- | --- |
| `uniform` | homogeneous continuous pump |
| `continuous-gaussian` | localized Gaussian continuous pump |
| `continuous-exp` | localized continuous pump with exponential radial decay |
| `pulse-gaussian` | repeated Gaussian pulses with Gaussian spatial profile; the first peak is phase-shifted after `delay`, and pulse peaks ramp from `P0` toward `Pmax` |

### Laser composition modes

| `mode` | Meaning |
| --- | --- |
| `single` | build one pump from `cfg.laser` |
| `multiple` | build a list of pumps from an external YAML file |

## Reservoir interface and factory

Reservoir models share the `AbstractReservoir` contract:

- store the current reservoir state
- expose the active density that couples to the condensate
- compute reservoir derivatives
- advance the reservoir in time
- optionally expose result nodes for storage and visualization

### Supported reservoir types

| `reservoir_type` | Meaning |
| --- | --- |
| `single` | one effective active reservoir |
| `double` | inactive plus active reservoir with transfer between them |
| `quadratic-double` | inactive reservoir directly fed by the pump and active reservoir fed by quadratic transfer `kappa*nI^2` |

In `quadratic-double`, the active density exposed to the condensate is `nR`.
The inactive density `nI` is useful for pulsed excitation because it can retain
memory of recent pulses before feeding the active reservoir.
