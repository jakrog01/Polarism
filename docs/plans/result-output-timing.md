# Plan: result output directory and save timing

## Problems

1. `StorageVisitor` names the output directory
   `<output_directory>/<YYYYmmdd-HHMMSS>`. Two runs started in the same second
   share the directory and `HDF5Storage` opens `results.h5` with mode `"w"`, so
   the second run silently overwrites the first (parameter sweeps of short runs).
2. `SimulationController.run` saves when `step % save_interval == 0` but labels
   the record with `t_after = (step + 1) * dt`. Records land at
   `dt, (k+1)*dt, (2k+1)*dt, ...` instead of multiples of `k*dt`, and the
   initial state at `t = 0` is never recorded. The same condition drives the
   animation.

## Changes

1. `polarism/results/visitors/storage_visitor.py`: create the run directory
   atomically. Try `<output_directory>/<timestamp>` with
   `mkdir(parents=True, exist_ok=False)`; on `FileExistsError` try
   `<timestamp>-1`, `<timestamp>-2`, ... until one is created. Store the result
   in `self.output_dir` as today. The storages keep creating files inside it
   (their own `mkdir(exist_ok=True)` stays as is).
2. `polarism/simulation_controller.py`, `run()`:
   - Before the loop, if any of storage, animation or live visualisation is
     active, emit one results step for the initial state at `t = state.t`
     (pump evaluated at that time, same arguments as the in-loop call), and
     advance `next_viz_time` if visualisation consumed it.
   - In the loop, save and animate when `(step + 1) % save_interval == 0`, so
     records land at `t = k * save_interval * dt`.
   - Keep everything else (visualisation stop check, finalize logic)
     unchanged. Factor the duplicated `results_manager.step(...)` call into
     one small private helper only if it avoids copying the block.
3. Tests:
   - `tests/integration/test_run_matrix.py`: besides the record count, assert
     the stored `time` dataset equals `[0, 7*dt, 14*dt, ...]` (use
     `np.testing.assert_allclose`).
   - New test (in `tests/unit/test_simulation_controller.py` or
     `tests/integration/test_cli_and_outputs.py`, whichever fits): two
     controllers created back to back with the same `output_directory` (in
     the same second) get different output directories and both `results.h5`
     files exist and are readable.
   - Fix any existing test that encodes the old timing.
4. Docs: if `docs/` describes when results are saved or how the output
   directory is named, update it (search for `save_interval` and
   `output_directory`).

## Out of scope

Storage aliasing, animation path handling, `animation_target_seconds`,
other refactors.

## Acceptance criteria

- New and updated tests pass and fail on the original code.
- `pytest -q` (default selection) passes.
- `pytest -q -m '' tests/integration tests/unit` passes.
- `mkdocs build --strict` passes.
