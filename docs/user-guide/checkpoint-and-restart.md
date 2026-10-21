# Checkpoint and Restart

Set a checkpoint cadence in solver steps and keep `solver.total_time` as the
absolute target time. The controller writes `checkpoint.h5` in its timestamped
run directory at every selected step and at the run end. It creates that
directory even when result storage is disabled.

```python
from polarism import Config, SimulationController

cfg = Config()
cfg.solver.total_time = 10.0
cfg.checkpoint.every_steps = 1000
controller = SimulationController(cfg)
controller.run()
print(controller.run_directory / "checkpoint.h5")
```

To continue to a later time, point a new controller at the saved file. Its
result files go into a new run directory. Stored `time` values continue from
the checkpoint time. The initial checkpoint frame is not written again because
it already belongs to the earlier run; concatenate the two runs' frames in
directory order to reconstruct the uninterrupted output sequence.

```python
continued = Config()
continued.solver.total_time = 20.0
continued.checkpoint.resume_from = str(controller.run_directory / "checkpoint.h5")
SimulationController(continued).run()
```

The CLI accepts the same settings, including
`polarism --checkpoint.resume-from path/to/checkpoint.h5 --solver.total-time 20`.
The checkpoint records the absolute step and time, condensate field, spectral
field where needed, reservoir densities, timestep, config JSON, and format
version. The write uses a temporary file and atomic replacement. A different
grid shape, reservoir type, batch shape, or timestep is rejected. A different
precision casts the saved arrays and emits a warning, so bitwise continuation
is only expected at the original precision.

For parameter sweeps, each batch has its own checkpoint. Resume by passing the
sweep directory to `Sweep.run(resume=...)`; completed batches are skipped. See
[Parameter Sweeps](parameter-sweeps.md#output-and-restart) for details.

For a manually prepared field, call
`controller.set_initial_state(psi, reservoir_state=None, t=0.0)` before `run()`.
The optional density tuple must match the condensate shape, including a leading
batch axis. `save_checkpoint(path)` and `load_checkpoint(path)` are also
available directly.

With `checkpoint.on_signal=True`, SIGTERM and SIGUSR1 request a stop after the
current solver chunk. The controller writes a checkpoint, finalizes result
storage, and returns normally. Signal handlers are installed only while `run()`
is active in the main thread, then previous handlers are restored.

## SLURM jobs

Slurm delivers `--signal` to job steps started with `srun`, so launch Polarism
through `srun`. With a fixed checkpoint path, one job script both starts and
continues the run; submit it again until the target time is reached:

```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120

CHECKPOINT="$SLURM_SUBMIT_DIR/checkpoint.h5"
ARGS=(--solver.total-time 20 --checkpoint.every-steps 1000 --checkpoint.path "$CHECKPOINT")
if [ -f "$CHECKPOINT" ]; then
    ARGS+=(--checkpoint.resume-from "$CHECKPOINT")
fi
srun polarism "${ARGS[@]}"
```

`USR1@120` asks Slurm for the signal 120 seconds before the time limit; choose a
margin longer than one checkpoint write plus one solver chunk. Each submission
writes its results to a new run directory and advances only the remaining
steps. If the job script runs Polarism without `srun`, use
`#SBATCH --signal=B:USR1@120` and start it with `exec polarism ...` so that the
signal reaches the Python process instead of the batch shell.
