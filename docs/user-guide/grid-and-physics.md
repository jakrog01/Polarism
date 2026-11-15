# Grid and physics

Set the dimension, box size and grid topology before choosing a solver. The `physics` block sets the coefficients in the condensate and reservoir equations.

```yaml
grid:
  ndim: 2
  nx: 32
  ny: 32
  lx: 20.0
  ly: 20.0
  grid_type: periodic
physics:
  gamma_C: 0.083
  g_C: 0.001
```

This grid has 32 by 32 samples over 20 µm by 20 µm. The example sets a condensate decay rate of 0.083 ps⁻¹ and a same-spin interaction coefficient of 0.001 meV µm².

## Grid coordinates and edges

`grid.ndim` is `1` or `2`. A one-dimensional grid uses `nx` and `lx`; `ny` and `ly` do not affect it. A two-dimensional field has array shape `(ny, nx)`, with x as the last axis. Box lengths are in µm.

| `grid_type` | Spacing | Edge behavior |
| --- | --- | --- |
| `periodic` | `dx = lx/nx`; `dy = ly/ny` in 2D | Opposite edges wrap together |
| `closed-interval` | `dx = lx/(nx−1)`; `dy = ly/(ny−1)` in 2D | Endpoints included; finite differences mirror adjacent interior values |

Periodic grids are required by the built-in FFT solvers. A closed-interval grid needs at least two samples on each active axis and an FDM solver. Use [boundary absorption](potentials-and-boundaries.md) when outgoing waves should decay near an edge. For solver compatibility, see [Solvers](../reference/solvers.md).

## Physics coefficients

The table gives the roles and two-dimensional units of the main coefficients. In one dimension, density has units µm⁻¹, interaction coefficients have units meV µm, and scattering coefficients have units µm ps⁻¹.

| Coefficient | Role | 2D units |
| --- | --- | --- |
| `hbar`, ħ | Converts energy to angular frequency | meV ps |
| `m_eff` | Effective mass in the parabolic kinetic term | meV ps² µm⁻² |
| `gamma_C`, γ_C | Condensate loss | ps⁻¹ |
| `gamma_R`, γ_R | Single or quadratic-double active-reservoir loss | ps⁻¹ |
| `gamma_I`, γ_I; `gamma_A`, γ_A | Inactive and double-model active loss | ps⁻¹ |
| `g_C`, g_C | Same-spin condensate interaction | meV µm² |
| `g_R`, g_R; `g_I`, g_I | Active and inactive reservoir shifts | meV µm² |
| `R` | Stimulated reservoir-to-condensate scattering | µm² ps⁻¹ |
| `R_IA`, `R_AI` | Inactive-to-active and reverse transfer | ps⁻¹ |
| `kappa`, κ | Quadratic inactive-to-active transfer | µm² ps⁻¹ |
| `reservoir_diffusion_I`, `reservoir_diffusion_A`, `reservoir_diffusion_R` | Reservoir transport | µm² ps⁻¹ |

`physics.dispersion` selects a registered kinetic energy relation; `parabolic` is the default. `kinetic_relaxation_eta` enables reservoir-dependent kinetic relaxation, while the diffusion coefficients enable reservoir transport. They are zero by default. Check the solver's support before using them and test convergence in grid spacing and time step. The [physics model](../explanation/physics.md) gives the equations; [Configuration reference](../reference/configuration.md) gives all defaults and limits.
