# Initial conditions and noise

The starting field is the deterministic `initial_condition` plus `noise.initial` when noise is enabled. Use a fixed seed when comparing runs.

```yaml
initial_condition:
  type: vortex
  amplitude: 0.1
  charge: 1
  core_width: 2.0
noise:
  enabled: false
```

On a 2D grid, this starts a vortex with core width 2 µm and field amplitude 0.1 µm⁻¹. Disabling noise leaves this deterministic field intact.

## Choose a deterministic field

| `initial_condition.type` | Starting field |
| --- | --- |
| `zero` | No deterministic contribution; the default |
| `plane-wave` | `amplitude × exp[i(kx x + ky y)]` |
| `vortex` | Vortex with integer `charge`, centre `(x0, y0)` and `core_width`; 2D only |
| `file` | Field read from a `.npy`, results `.h5` or checkpoint file |

`amplitude` has units µm⁻¹ in 2D or µm⁻¹ᐟ² in 1D. Positions and core width are in µm; `kx` and `ky` are in rad µm⁻¹. A file field must match the grid shape exactly. For a results file, save `psi` with `result.save_psi: true`; `initial_condition.frame` selects a saved frame, with −1 meaning the last. The file's amplitude is used directly. See [Checkpoint and restart](checkpoint-and-restart.md) for continuing a full state.

## Choose initial noise

`noise.initial` accepts `none`, `complex_gaussian_zero_mean`, `filtered_complex_gaussian` and `legacy_positive_uniform`. The zero-mean Gaussian is the default, scaled to RMS `noise.initial_amplitude` in µm⁻¹ for 2D or µm⁻¹ᐟ² for 1D. The filtered mode removes Fourier modes above `initial_k_cutoff_um` in rad µm⁻¹, then restores the requested RMS; the cutoff must admit a nonzero mode. The legacy mode has a positive mean and seeds a uniform mode, so use it only to reproduce earlier runs. Set `noise.initial: none` to start with only the deterministic field, or `noise.enabled: false` to disable both initial and per-step noise.

`noise.seed` accepts a nonnegative 64-bit integer. With `null`, Polarism draws a seed from operating-system entropy and records it for reproduction. The initial random field is drawn on the host, giving the same seeded field on CPU and GPU. For an ensemble, vary the seed in a [parameter sweep](parameter-sweeps.md).

## Add stochastic Wigner increments

Set `noise.stochastic: wigner` to add a complex random increment after each solver step. For each real and imaginary part its scale is `sqrt[s (γ_C + R n_active) dt / (4 ΔV)]`, where `s` is `noise.stochastic_strength`, `dt` is in ps, and `ΔV` is a cell length in µm for 1D or area in µm² for 2D. `n_active` is the active reservoir density in µm⁻¹ or µm⁻². The result has field-amplitude units.

Wigner increments use Philox counters keyed by seed, cell and step. The [physics page](../explanation/physics.md) explains the approximation and omitted corrections. Adaptive stepping requires stochastic noise off; see [Solvers and time stepping](solvers-and-time-stepping.md). The full set of noise defaults and limits is in [Configuration reference](../reference/configuration.md).
