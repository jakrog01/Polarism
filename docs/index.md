# Polarism

Polarism simulates driven-dissipative exciton-polariton condensates in one and two dimensions. You describe the sample, pump and numerics in Python or YAML; Polarism integrates the generalized Gross-Pitaevskii equation coupled to reservoir rate equations on a CPU or an NVIDIA GPU.

## Highlights

- **Physics:** scalar ψ or spinor (ψ+, ψ−) fields with TE-TM splitting, Zeeman splitting and cross-spin interaction; single, double, quadratic-double or no reservoir; incoherent pumps from spatial profiles, lattices and temporal envelopes; a coherent drive; static and time-dependent potentials; truncated-Wigner noise.
- **Solvers:** finite-difference RK4, split-step FFT, ETD-RK2 and interaction-picture RK solvers up to Verner 9, with adaptive time stepping for the embedded tableaus. The same configuration runs on NumPy, CuPy or fused CUDA kernels.
- **Observables:** density, Stokes parameters, currents, vortices, and accumulated n(k), g1, g2 and energy spectra E(k) and E(x).
- **Workflows:** batched parameter sweeps, checkpoint and restart (including SLURM time limits), asynchronous HDF5, NPY and JSON output, and a lazy result reader.
- **Extensible:** registries for lasers, potentials, reservoirs, terms, dispersions, solvers, observables, accumulators and storage formats.

## A first run

```python
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = 64
cfg.grid.ny = 64
cfg.grid.lx = 60.0
cfg.grid.ly = 60.0
cfg.laser.laser_type = "continuous-gaussian"
cfg.laser.P0 = 0.3
cfg.laser.sigma_space = 6.0
cfg.solver.method = "ifrk-fft-cuda"
cfg.solver.tableau = "dop853"
cfg.solver.adaptive = True
cfg.solver.dt = 0.1
cfg.solver.total_time = 300.0
controller = ps.SimulationController(cfg)
controller.run()
```

Lengths are in µm, times in ps and energies in meV. This run forms a condensate under the pump spot in a few seconds on a CPU; the [quickstart](getting-started/quickstart.md) saves and plots it.

## Where to go next

- **New to Polarism:** [install it](getting-started/installation.md) and follow the [quickstart](getting-started/quickstart.md).
- **Setting up a study:** the [user guide](user-guide/configuration.md) walks through configuration, physics, solvers, output and sweeps.
- **Looking up a key or a solver:** the [configuration reference](reference/configuration.md) and the [solver reference](reference/solvers.md).
- **Checking the model:** the [physics page](explanation/physics.md) gives the equations, units and references.
- **Adding your own physics:** the [plugin guide](plugins/index.md).

Release notes for each version are on the [GitHub releases page](https://github.com/jakrog01/Polarism/releases).
