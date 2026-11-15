# Pumps and reservoirs

Use `laser` for an incoherent pump `P` and `reservoir` for the exciton population it feeds. A two-dimensional continuous pump has units µm⁻² ps⁻¹.

```yaml
laser:
  laser_type: continuous-gaussian
  P0: 0.02
  sigma_space: 4.0
reservoir:
  reservoir_type: single
```

This gives a Gaussian pump with centre strength 0.02 µm⁻² ps⁻¹ and width 4 µm. It feeds a single active reservoir `n_R`.

## Choose a laser

| `laser_type` | Pump shape |
| --- | --- |
| `uniform` | Constant in space and time after `delay` |
| `continuous-gaussian` | Gaussian spot with width `sigma_space` |
| `continuous-exp-length` | Radial exponential with 1/e decay length `sigma_space` |
| `continuous-exp` | Legacy exponential `exp(-r/sigma_space²)` |
| `pulse-gaussian` | Repeated Gaussian pulses with spatial width `sigma_space` and temporal width `sigma_time` |
| `profile` | Registered [spatial profile and temporal envelope](profiles-and-envelopes.md) |

`x0` and `y0` shift the built-in laser centre in µm; `delay` shifts its start in ps. The pulse laser's first peak occurs at `delay + cutoff_sigma × sigma_time`. `n_pulses: 0` leaves its pulse train unlimited, and `Pmax` caps its pulse strength. For a profile laser, `P0` multiplies its spatial profile and envelope.

`power_definition: peak_amplitude` interprets `P0` as local peak pump density. For `pulse-gaussian`, `power_definition: pulse_energy` normalizes one pulse over the grid and its time envelope, so `P0` is the integrated dose of that pulse. In 2D it is then the number of injected excitons per pulse, a dimensionless number. Use it when comparing pulse widths or spot sizes at fixed dose. The profile laser requires `peak_amplitude`. See [Configuration reference](../reference/configuration.md) for the related fields.

To add pumps, set `laser.mode` to `multiple` and `laser.config_file` to a YAML file whose `lasers` list contains one mapping per pump, each with `laser_type` and its own parameters. The pump fields add. Entries can each have a `profile` and `envelope` mapping.

## Choose a reservoir

| `reservoir_type` | Populations | Use |
| --- | --- | --- |
| `single` | Active `n_R` | One effective exciton reservoir |
| `double` | Inactive `n_I` and active `n_A` | Separate pump capture and condensate feeding |
| `quadratic-double` | Inactive `n_I` and active `n_R` | Nonlinear feeding proportional to κ`n_I²` |
| `none` | No reservoir | Coherent driving or free decay |

The default is `double`. For production studies use `quadratic-double`; keep `double` as a sanity-check and reference model. In `single`, the pump fills `n_R` directly. In `double`, the pump fills `n_I`, transfer at `R_IA` feeds `n_A`, and `R_AI` transfers back. In `quadratic-double`, κ`n_I²` feeds `n_R`. The active density supplies the gain and reservoir blueshift in the condensate equation. See [Physics](../explanation/physics.md) for the equations.

For spinor runs, the default is one reservoir population per circular component. Set `reservoir.components: 1` to share one population between the two fields. See [Spin and polarization](spin-and-polarization.md).
