# Parameter sweeps

A sweep applies ordered parameter overrides to one configuration. Compatible members advance together as a batch on the selected CPU or GPU.

## Run a sweep

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
cfg.solver.total_time = 150.0
parameters = {"laser.P0": [0.2, 0.3, 0.4]}
sweep = ps.Sweep(cfg, parameters)
final_states = sweep.run()
print(len(final_states))
```

This runs three pump strengths in parameter insertion order. `mode="product"` makes all combinations of multiple parameter lists; `mode="zip"` pairs values by position and requires equal-length lists. `sweep.members` gives the ordered override dictionaries. Without storage, `run()` returns final `psi` and `reservoir` arrays for each member.

To save results, set `cfg.result.save_results = True` and `cfg.result.formats = ("hdf5",)` before constructing the sweep. `run()` then returns `SweepResults`; use `result.member(i)` for one member and `result.fields["density"]` for a combined lazy view. `result.parameters` and `result.parameter_table` map indices to overrides. Close results or use a context manager. See [Results format](../reference/results-format.md) for the sweep manifest and dataset axes.

## Choose overrides and batch size

Paths such as `physics.g_C`, `laser.P0`, `noise.seed`, and `initial_condition.amplitude` may vary within a batch. Grid, solver, reservoir, boundary, result, checkpoint, and sweep settings form separate groups. `physics.spin`, `physics.dispersion`, `noise.enabled`, `noise.stochastic`, and `initial_condition.type` also form separate groups. A key under `terms.<name>` must already exist. With multiple lasers, use zero-based paths such as `lasers[1].delay`. These paths write into `laser.entry_overrides` on each member's config. `sweep.groups` shows the grouping.

On CPU, each group runs in one batch unless `sweep.batch_size` or `sweep.max_batch` limits it. On GPU, Polarism estimates memory use, limits the batch by `sweep.memory_fraction`, and times candidate batch sizes. Set `sweep.batch_size` to a fixed maximum when repeatability matters. All batch members share a grid and integration schedule; they do not interact physically. GPU results can differ from separate runs by floating-point rounding.

## Save and resume a sweep

A saved sweep has a directory containing `sweep.json` and one directory per batch. Checkpointing stores a checkpoint in each batch directory. Call `sweep.run(resume=sweep.run_directory)` to continue after a signal stop; completed batches are skipped. Keep the sweep parameters and simulation settings the same on resume. See [Checkpoint and restart](checkpoint-and-restart.md).

For CLI sweeps, `polarism --sweep sweep.yaml` reads a file with `mode`, `parameters`, and optional `sweep` settings. Simulation settings come from the normal CLI configuration. Use `polarism --help` for the current CLI switches.
