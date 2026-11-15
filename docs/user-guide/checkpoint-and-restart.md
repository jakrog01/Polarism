# Checkpoint and restart

Save a checkpoint at a fixed cadence so a long run can continue after interruption. The target `solver.total_time` stays measured from the original initial state, in ps.

## Save and resume

```python
from copy import deepcopy

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
cfg.checkpoint.every = 50.0
controller = ps.SimulationController(cfg)
controller.run()
checkpoint = controller.run_directory / "checkpoint.h5"
continued = deepcopy(cfg)
continued.solver.total_time = 300.0
continued.checkpoint.resume_from = str(checkpoint)
ps.SimulationController(continued).run()
```

`checkpoint.every` is in ps and must be a positive multiple of `solver.dt`. The controller writes at that cadence and at run end. It creates a run directory even when result storage is disabled. `checkpoint.path` selects a fixed checkpoint path; otherwise the file is `checkpoint.h5` in the run directory. A resumed run writes any result frames to a new run directory, with times continuing from the saved state.

A checkpoint contains condensate and reservoir state, time, step, configuration, and any active accumulator and adaptive step state. The write uses a temporary file and atomic replacement. A different grid shape, reservoir type, batch shape, or time step is rejected. Changing precision casts saved arrays and warns, so exact bitwise continuation requires the original precision. See [Results format](../reference/results-format.md) for the file layout.

`checkpoint.on_signal` is enabled by default. While `run()` is active in the main thread, SIGTERM and SIGUSR1 request a stop after the current solver chunk. The controller writes a final checkpoint, closes storage, and returns. On a scheduler, send the signal early enough for one chunk and a checkpoint write. For a sweep, pass its directory to `Sweep.run(resume=...)`; see [Parameter sweeps](parameter-sweeps.md).

For a prepared field, call `controller.set_initial_state(psi, reservoir_state=None, t=0.0)` before `run()`. `t` is in ps. Direct `save_checkpoint(path)` and `load_checkpoint(path)` methods are also available. Python event callbacks must be registered again after loading a checkpoint.
