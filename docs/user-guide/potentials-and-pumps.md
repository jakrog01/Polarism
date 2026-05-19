# Potentials and Pumps

Polarism separates the external potential, pump profile, and reservoir model so you can change one layer without rewriting the others.

The `polarism` package owns the reusable physics components described on this page. The `src/pump_multi_comparison/` tree is an example workflow that assembles these components for batch studies on Slurm.

## Available potentials

The current potential factory exposes these `potential_type` values:

- `zero`
- `double-well-supergaussian`

Example:

```python
cfg.potential.potential_type = "double-well-supergaussian"
cfg.potential.x1 = -2.0
cfg.potential.y1 = 0.0
cfg.potential.x2 = 2.0
cfg.potential.y2 = 0.0
cfg.potential.V1 = -2.0e-3
cfg.potential.V2 = -2.2e-3
cfg.potential.w1 = 1.5
cfg.potential.w2 = 1.5
cfg.potential.order = 2.0
```

## Available laser types

The laser registry currently includes:

- `uniform`
- `continuous-gaussian`
- `continuous-exp`
- `pulse-gaussian`

All pump profiles follow the same high-level structure used by the shared `AbstractLaser` interface:

$$
P(x, y, t) =
\begin{cases}
0, & t < t_{\mathrm{delay}}, \\
A(t - t_{\mathrm{delay}})\, S(x, y)\, T(t - t_{\mathrm{delay}}), & t \ge t_{\mathrm{delay}}.
\end{cases}
$$

Here `P0`, `Pmax`, `sigma_space`, `sigma_time`, `pulse_separation`, and `delay` are the configuration fields on `cfg.laser`.
For `pulse-gaussian`, the implementation also defines a pulse-center offset
\(\phi = \texttt{cutoff\_sigma} \cdot \sigma_{\mathrm{time}}\), so the first
pulse peak occurs at \(t = t_{\mathrm{delay}} + \phi\) rather than exactly at
`delay`.

### Laser type summary

| `laser_type` | Meaning | Implemented profile |
| --- | --- | --- |
| `uniform` | Spatially constant continuous pump | \(S(x, y) = 1,\; A(t) = P_0,\; T(t) = 1\) |
| `continuous-gaussian` | Continuous Gaussian spot centered at \((x_0, y_0)\) | \(S(x, y) = \exp[-((x-x_0)^2 + (y-y_0)^2)/(2 \sigma_{\mathrm{space}}^2)]\) |
| `continuous-exp` | Continuous radially decaying pump with exponential tail | \(S(x, y) = \exp[-r / w^2],\; r = \sqrt{(x-x_0)^2 + (y-y_0)^2},\; w = \texttt{sigma\_space}\) |
| `pulse-gaussian` | Repeated Gaussian pulses with Gaussian spot and stepwise strength ramp up to `Pmax` | \(S(x, y) = \exp[-((x-x_0)^2 + (y-y_0)^2)/(2 \sigma_{\mathrm{space}}^2)]\) and \(T(t) = \exp[-(t-(\phi+n\Delta t))^2/(2 \sigma_{\mathrm{time}}^2)]\) near each pulse center \(\phi+n\Delta t\), where \(\phi = \texttt{cutoff\_sigma} \cdot \sigma_{\mathrm{time}}\) |

`continuous-exp` follows the current implementation exactly, including the `\exp(-r / w^2)` spatial form.

For `pulse-gaussian`, the base strength of each pulse ramps from `P0` (first pulse)
toward `Pmax`. The first pulse peaks at `delay + cutoff_sigma * sigma_time`;
at `t = delay` the envelope is at about `exp(-4.5) ≈ 0.011` of its peak value.
The physical meaning of `P0` depends on `power_definition`: it is the local peak
source density at the Gaussian centre when `power_definition: peak_amplitude`,
and the integrated per-pulse dose when `power_definition: pulse_energy`
(see the section below).

### `power_definition` — peak amplitude vs pulse energy

`pulse-gaussian` supports two interpretations of `power` controlled by the
`power_definition` field in `laser_defaults` (or per-laser override):

```yaml
laser_defaults:
  power_definition: pulse_energy   # recommended for geometry/sigma_space sweeps
```

| Value | Meaning of `P0` / `power` | When to use |
|---|---|---|
| `peak_amplitude` | Local peak source density at the Gaussian centre (legacy default) | Fixed geometry, single spot size |
| `pulse_energy` | Integrated pump dose delivered per pulse over the full domain | Any sweep over `sigma_space`, or when comparing scenarios with different spot sizes |

**Why this matters for sweeps.** With `peak_amplitude`, changing `sigma_space` while keeping `power` fixed
scales the total dose injected per pulse as \(\propto \sigma_{\mathrm{space}}^2\). A run with
`sigma_space = 3.5 µm` injects ~22× more total dose than one with `sigma_space = 0.75 µm` at the same `P0`,
which is the root cause of over-expanded condensates observed in spot-size sweeps.

With `pulse_energy`, the normalisation is:

$$
P(x, y, t) = A(n) \cdot \frac{e^{-r^2/(2\sigma_s^2)}}{\mathcal{I}_s} \cdot \frac{e^{-\delta t^2/(2\sigma_t^2)}}{\mathcal{I}_t}
$$

where

$$
\mathcal{I}_s = \sum_{\mathrm{grid}} e^{-r^2/(2\sigma_s^2)}\, \Delta x\,\Delta y, \qquad
\mathcal{I}_t = \sqrt{2\pi}\,\sigma_t\,\mathrm{erf}\!\left(\frac{c_\sigma}{\sqrt{2}}\right),
$$

\(c_\sigma = \texttt{cutoff\_sigma}\), and \(A(n) = P_0\) for the first pulse (ramping toward `Pmax`).
This guarantees:

$$
\int\!\!\int P(x,y,t)\,dx\,dy\,dt \approx P_0 \quad \text{per pulse.}
$$

Increasing `sigma_space` lowers the local peak density while conserving the total pulse dose. The
`P_max` scalar in sidecar files reflects this decrease; the `P_area_integral` time trace is independent
of spot size for the same `pulse_energy` and temporal pulse shape.

The pipeline also records `P_cumulative_area_time_integral`, a stepwise Riemann
sum of `P_area_integral * dt`.  It is the scalar to use when checking the total
delivered dose in long pulse trains or when comparing campaigns with different
scalar output strides.

**Migration note.** Old configs without `power_definition` default to `peak_amplitude` and are unaffected.
All production configs in `src/pump_multi_comparison/` are updated to `pulse_energy`.

For a single pump, configure the values directly on `cfg.laser`:

```python
cfg.laser.mode = "single"
cfg.laser.laser_type = "continuous-gaussian"
cfg.laser.P0 = 0.4
cfg.laser.x0 = 0.0
cfg.laser.y0 = 0.0
cfg.laser.sigma_space = 15.0
```

For multiple pumps, point at a YAML file:

```python
cfg.laser.mode = "multiple"
cfg.laser.config_file = "lasers_setup.yaml"
```

The YAML file should define a `lasers:` list where each entry names a `laser_type` plus its parameters.

## Reservoir models

Supported reservoir models:

- `single`
- `double`

Example:

```python
cfg.reservoir.reservoir_type = "double"
```

The condensate couples back only to the active reservoir density. In the single model that density is `n_R`; in the double model it is `n_A`.

### Single reservoir

Use `single` when one effective exciton reservoir is enough for the question you are studying.

The implemented evolution is:

$$
\frac{\partial n_R}{\partial t}
= P(x, y, t) - \left(\gamma_R + R |\psi|^2\right) n_R.
$$

This is the minimal driven-dissipative model: pump fills the reservoir, `\gamma_R` depletes it, and stimulated scattering proportional to \(R |\psi|^2 n_R\) transfers population into the condensate channel.

### Double reservoir

Use `double` when you want to distinguish an inactive reservoir `n_I` fed directly by the pump from an active reservoir `n_A` that couples to the condensate.

The implemented evolution is:

$$
\frac{\partial n_I}{\partial t}
= P(x, y, t) - \left(\gamma_I + R_{IA}\right) n_I + R_{AI} n_A,
$$

$$
\frac{\partial n_A}{\partial t}
= R_{IA} n_I - \left(\gamma_A + R_{AI} + R |\psi|^2\right) n_A.
$$

In this model, `R_IA` transfers density from the inactive to the active reservoir, `R_AI` allows reverse transfer, and only `n_A` enters the condensate equation.

Choose the reservoir model based on the physical process you are trying to resolve, then confirm the solver choice remains compatible and the timestep remains stable for the chosen rates.

## Compatibility warnings

The repository includes explicit compatibility checks:

- Spectral solvers warn when used with non-zero external potentials.
- Spectral solvers also warn on `closed-interval` grids, where FDM-based methods are usually the safer baseline.
- CUDA solvers warn when no GPU backend is active.

Treat these warnings as numerical guidance, not cosmetic log noise.
