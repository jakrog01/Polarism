# Parameter Sweeps

A parameter sweep runs a family of simulations from one base configuration. Each
member applies a set of parameter overrides. Polarism groups members that share
settings needed by the same grid and solver, then advances compatible members
together on the selected device.

## Python API

```python
import polarism as ps

cfg = ps.Config()
cfg.result.save_results = True
cfg.result.save_hdf5 = True
cfg.result.output_directory = "sweep_results"
parameters = {"laser.P0": [0.02, 0.03], "physics.g_C": [0.001, 0.002]}
sweep = ps.Sweep(cfg, parameters, mode="product")
print(sweep.members)
print(sweep.groups)
with sweep.run() as result:
    density = result.member(0).fields["density"][:]
```

`mode="product"` makes the Cartesian product in parameter insertion order;
the example has four members. `mode="zip"` pairs values at the same position,
so all parameter sequences must have equal length. `sweep.members` is the
ordered list of override dictionaries. `sweep.groups` lists dictionaries with
`overrides` for shared group settings and `indices` into `members`.

With result storage enabled, `run()` returns `SweepResults`. With storage
disabled, it returns an ordered list of dictionaries containing final `psi`
and a `reservoir` tuple of final density arrays. A signal stop returns `None`;
resume with `sweep.run(resume=sweep.run_directory)`.

Override paths include `physics.*`, `laser.*`, `laser.extra.*`, and
`potential.*`. In `laser.mode="multiple"`, use zero-based `lasers[i].*`, such
as `lasers[1].delay` or `lasers[1].extra.name`, for entries in the laser YAML
file. Extra keys must already exist in the configuration or laser entry.
Physics, laser, and potential changes can vary inside a batch. Grid, solver,
reservoir type, boundary condition, compute engine, result, checkpoint, and
sweep settings form separate groups. These settings control array shapes,
component construction, step counts, or output behavior, which must agree
within one batch.

Each batch has a leading member axis. Physics constants that differ are passed
per member. Pump envelopes and potentials are shared when identical, or stacked
when they differ. Initial fields are constructed separately with each member's
seed and initialization settings, as in single runs. A batch of one uses the
normal single-run path. Built-in CPU batches produce fields bitwise equal to
their corresponding single runs; GPU results can differ within floating-point
rounding.

## Batch size and performance

Set `cfg.sweep.batch_size` to choose a fixed maximum batch size. Otherwise,
GPU sweeps estimate memory per member, cap the batch using
`cfg.sweep.memory_fraction` (default `0.8`) and `cfg.sweep.max_batch` if set,
then measure throughput at candidate sizes. The choice is cached under
`$XDG_CACHE_HOME/polarism/sweep_batches.json`, or `~/.cache/polarism/` when
`XDG_CACHE_HOME` is unset. The first sweep for a configuration pays the tuning
cost, which can be a few seconds. CPU sweeps use all members of a group unless
`batch_size` or `max_batch` limits them.

The largest batch is not always fastest. A single run whose fields fit in the
GPU's L2 cache (48 MB on an RTX 4070 SUPER, so up to roughly 1024² in single
precision) reads most data from the cache, while a larger batch spills to device
memory and can be several times slower per member. Batching pays off where
single runs are limited by kernel launch latency, that is on small grids; on
large grids the autotuner usually picks a small batch or runs members one at a
time. Autotuning measures completed member steps per second rather than
choosing the largest batch that fits in memory.

Measured on an **NVIDIA RTX 4070 SUPER**, single precision, double reservoir,
1000 steps. Speedup is relative to sequential single runs; batch size is 64
unless shown otherwise.

| Solver | Grid | Batch size | Speedup |
| --- | ---: | ---: | ---: |
| `split-step-fft-cuda` | 64² | 64 | 23.5× |
| `split-step-fft-cuda` | 128² | 64 | 8.9× |
| `ifrk-fft-cuda` | 64² | 64 | 23.7× |
| `ifrk-fft-cuda` | 128² | 64 | 4.1× |
| `rk4-cuda` | 64² | 64 | 6.8× |
| `rk4-cuda` | 128² | 64 | 2.2× |
| `split-step-fft-cuda` | 256² | 8 | 2.3× |
| `ifrk-fft-cuda` | 256² | 4 | 2.0× |
| `rk4-cuda` | 256² | 8 | 1.1× |

To measure your GPU, run `polarism-bench --batch 8 --grid-sizes 256 --steps
1000 --reservoir double --precisions single`. Batched benchmark output includes
speedup and nanoseconds per member, grid element, and step.

## Output and restart

A saved sweep gets one timestamped directory under `result.output_directory`.
Its `sweep.json` records member parameters, group and batch assignments, and
configuration. Each `batch_NNN/` directory holds normal result files and, when
enabled, its own `checkpoint.h5`. Dynamic field datasets have shape
`(frames, members, *S)` and scalar datasets `(frames, members)`, where `S` is
`(nx,)` in 1D or `(ny, nx)` in 2D. Identical
static fields may be stored once without a member axis.

```python
with ps.load_results(sweep.run_directory) as result:
    print(result.parameters)
    print(result.parameter_table["laser.P0"])
    first = result.member(0)
    density = result.fields["density"][::10, 0]
    norm = result.scalars["norm"][:, :]
```

`ps.load_results` detects `sweep.json` and returns `SweepResults`.
`parameters` is the ordered list of member dictionaries; `parameter_table`
maps each parameter path to an array in member order. `member(i)` returns a
`Results`-compatible view. HDF5 `fields` and `scalars` are read lazily with
frame and member axes; slice them to avoid loading the entire sweep. Close the
result or use a context manager to release HDF5 handles.

For restart, set `cfg.checkpoint.every` or `cfg.checkpoint.path`, then pass the
sweep directory to `run(resume=...)`. Completed batches are skipped; an
incomplete batch resumes from its checkpoint. The manifest preserves the
original batch plan even if batch size settings change. Keep simulation
settings and sweep parameters the same for resume. On SLURM, arrange a signal
before the time limit and resubmit the job against the same sweep directory;
see [Checkpoint and Restart](checkpoint-and-restart.md#slurm-jobs).

`cfg.result.view_member` selects the global member index for live visualization
and animation. Only its batch creates the view or movie.

## CLI

Save this as `sweep.yaml`:

```yaml
mode: product
parameters:
  laser.P0: [0.02, 0.03]
  physics.g_C: [0.001, 0.002]
sweep:
  batch_size: 8
  max_batch: 16
  memory_fraction: 0.8
```

Run `polarism --sweep sweep.yaml` with the usual CLI configuration options,
including `--result.save-results` and a storage format. The YAML contains
overrides and sweep settings; other simulation settings come from CLI options.

## Plugins

A laser plugin that overrides `get_power` takes a slower per-member pump
fallback. Solver and reservoir plugins must accept `batch_physics` to vary
physics constants within a batch. Scalar observable plugins must return one
value per member in batched runs; field observables must keep the member axis.
