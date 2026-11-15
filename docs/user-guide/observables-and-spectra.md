# Observables and spectra

Accumulators sample during a run and save time-averaged momentum, correlation, or spectral results without storing every field frame.

## Accumulate momentum and density moments

```yaml
result:
  save_results: true
  formats: [hdf5]
  accumulators:
    momentum-distribution: {start: 0.0, every: 1.0}
    g2: {start: 0.0, every: 1.0}
```

`start`, `stop`, and `every` are in ps and align with `solver.dt`. Sampling includes the initial state when `start` is zero. Without `every`, an accumulator uses `result.save_every`, or one solver step if that is unset. Saved accumulated values record sample count and first and last times in ps. They can be read from `results.accumulated` and their metadata from `results.accumulated_meta`; the exact layout is in [Results format](../reference/results-format.md).

`momentum-distribution` writes FFT-shifted `momentum_distribution` in particles per mode and `g1`, the normalized first-order spatial correlation. Summing `momentum_distribution` over modes gives the mean particle number. Its `kx` and `ky` coordinates are in µm⁻¹. `g1` is dimensionless and centered so zero separation is at the array center.

`g2` writes `density_mean` in µm⁻², `density_squared_mean` in µm⁻⁴, and their ratio `g2`. In 1D the density units are µm⁻¹ and µm⁻². With Wigner noise, the accumulators remove the symmetric-ordering vacuum terms using `noise.stochastic_strength` and cell volume ΔV: `momentum_distribution` subtracts `s/2` particles per mode, `density_mean` subtracts `s/(2ΔV)`, and `density_squared_mean` subtracts `2(s/ΔV)⟨|ψ|²⟩ − s²/(2ΔV²)` from the raw fourth moment. These are observable corrections; the stochastic evolution itself omits the full truncated-Wigner drift corrections described in [Physics](../explanation/physics.md).

## Combine a noise ensemble

Sweep `noise.seed` as described in [Parameter sweeps](parameter-sweeps.md). Compute `SweepResults.mean("momentum_distribution")`, `mean("density_mean")`, and `mean("density_squared_mean")`. The means weight each member by its sample count. Derive ensemble ratios from those linear means with `polarism.analysis.g1(n_k)` and `polarism.analysis.g2(mean, squared_mean)`. Averaging saved `g1` or `g2` ratios is not equivalent. For a 1D batched momentum array, pass `ndim=1` to `g1`.

## Accumulate energy spectra

```yaml
grid:
  nx: 64
  ny: 64
  lx: 60.0
  ly: 60.0
laser:
  laser_type: continuous-gaussian
  P0: 0.3
  sigma_space: 6.0
solver:
  method: ifrk-fft-cuda
  tableau: dop853
  adaptive: true
  dt: 0.1
  total_time: 150.0
result:
  save_results: true
  formats: [hdf5]
  save_every: 1.0
  accumulators:
    spectrum-k: {every: 1.0, segment: 64}
    spectrum-x: {every: 1.0, segment: 64, y: 0.0}
```

`spectrum-k` samples the `ky = 0` momentum line in 2D, or the full momentum axis in 1D. `spectrum-x` samples the real-space row nearest `y` µm in 2D, or the full x axis in 1D. Omit `y` in 1D. `segment` is an even sample count, defaults to 1024, and requires at least one full segment. The estimator uses a periodic Hann window and 50% overlap. It warns and emits no spectrum if the run ends too soon.

The energy spacing is `2πħ/(segment × every)` meV; the window spans about `±πħ/every` meV around the drive energy, or zero without a drive. `spectrum_k` is in particles per mode per meV and `spectrum_x` is in µm⁻ᵈ meV⁻¹. The corresponding energy arrays are `spectrum_k_energy` and `spectrum_x_energy`, in meV. Select a member before interpreting a batched spectrum. Spectra are not Wigner corrected; vacuum noise follows the bare dispersion.

To estimate a chemical potential μ, sum the spectral power over position or wave number and components, then take the energy at its largest peak. This is a peak estimate, not an equilibrium chemical-potential solver. See [examples/spectra_and_correlations.py](https://github.com/jakrog01/Polarism/blob/master/examples/spectra_and_correlations.py) for a pumped-spot analysis with an `E(k)` plot, bare dispersion, μ marker, `n(k)`, and `|g1|`.
