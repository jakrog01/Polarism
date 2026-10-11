# Your first plugin

This tutorial adds a continuous ring-shaped pump, runs a small CPU simulation,
checks the saved pump field, and then shows how to repeat the run on a GPU. The
working class is the same file shown on the [laser contract page](lasers.md).

## 1. Create the project layout

Install Polarism in your project's virtual environment and work from the
directory containing `polarism_plugins/`:

```text
my_project/
├── .venv/
├── polarism_plugins/
│   └── laser/
│       └── ring_laser.py
├── run.py
└── lasers.yaml              # optional, for multiple-laser mode
```

```bash
mkdir -p my_project/polarism_plugins/laser
cd my_project
python -m venv .venv
.venv/bin/pip install polarism
```

Copy [the example file](https://github.com/jakrog01/Polarism/blob/master/examples/plugins/polarism_plugins/laser/ring_laser.py)
into `polarism_plugins/laser/ring_laser.py`. The runtime imports it when a
controller is created from this directory. No `__init__.py` is required.

## 2. Read the registration and class

```python
--8<-- "examples/plugins/polarism_plugins/laser/ring_laser.py"
```

`@register_laser("example-ring")` is the selection name. The constructor calls
`AbstractLaser.__init__` for backend, dtype, power, center, and delay fields.
It reads `ring_radius` from `laser_config.extra`, then caches the spatial
envelope with `_finalize_spatial_envelope(X, Y)`. The three methods provide
amplitude, spatial shape, and time dependence; the base class combines them.

## 3. Select it from Python and run

Create `run.py`:

```python
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = cfg.grid.ny = 32
cfg.grid.lx = cfg.grid.ly = 40.0
cfg.laser.mode = "single"
cfg.laser.laser_type = "example-ring"
cfg.laser.extra = {"ring_radius": 8.0}
cfg.laser.P0 = 0.02
cfg.laser.sigma_space = 2.0
cfg.reservoir.reservoir_type = "single"
cfg.solver.method = "rk4-fdm"
cfg.solver.dt = 0.001
cfg.solver.total_time = 0.01
cfg.result.save_results = True
cfg.result.save_hdf5 = True
cfg.result.output_directory = "results"

ps.SimulationController(cfg).run()
```

```bash
.venv/bin/python run.py
```

The controller searches the **current working directory**, so run this from
`my_project/`. The configured ten steps are intentionally small for a first
check. The output writer creates a timestamped directory below `results/`.

## 4. Inspect the pump in HDF5 output

The controller exposes a `Pump` result node when
`cfg.laser.expose_results` is true (its default). HDF5 stores that field at
`fields/Pump`. Check the latest run:

```python
from pathlib import Path
import h5py

files = sorted(Path("results").glob("*/results.h5"))
assert files, "run.py did not write results.h5"
with h5py.File(files[-1], "r") as saved:
    pump = saved["fields/Pump"][:]
    print(pump.shape)           # (saved frames, 32, 32)
    assert pump.shape[1:] == (32, 32)
    assert pump.max() > 0
```

Save this as `inspect.py` and run `.venv/bin/python inspect.py`. A fresh
`results/` directory makes the last-file selection unambiguous.

## 5. Select it from YAML and the CLI

The CLI cannot supply arbitrary `extra` keys as individual flags. For a
custom radius in CLI use, create `lasers.yaml`:

```yaml
lasers:
  - laser_type: example-ring
    ring_radius: 8.0
    P0: 0.02
    sigma_space: 2.0
```

Unknown laser entry keys, including `ring_radius`, become that laser's
`LaserParameters.extra`. Run from `my_project/`:

```bash
.venv/bin/polarism --laser.mode multiple --laser.config-file lasers.yaml \
  --grid.nx 32 --grid.ny 32 --grid.lx 40 --grid.ly 40 \
  --reservoir.reservoir-type single --solver.method rk4-fdm \
  --solver.dt 0.001 --solver.total-time 0.01 \
  --result.save-results --result.save-hdf5 --result.output-directory results
```

For a single pump with the example's default radius, the direct CLI selection
is `--laser.laser-type example-ring`; set the remaining physics and output
flags as above. `--laser.mode multiple` instead reads the YAML entries.

## 6. Add a CPU test

The repository's [integration test](https://github.com/jakrog01/Polarism/blob/master/tests/integration/test_example_plugins.py)
copies example modules into `tmp_path`, loads them, runs a short controller
simulation, and restores the registries and `sys.modules` afterward. For your
project, begin with this focused check in `test_ring.py`:

```python
import numpy as np
import polarism as ps


def test_ring_runs(monkeypatch, tmp_path):
    # Copy your plugin tree into tmp_path before changing directories.
    import shutil
    shutil.copytree("polarism_plugins", tmp_path / "polarism_plugins")
    monkeypatch.chdir(tmp_path)
    cfg = ps.Config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.laser.laser_type = "example-ring"
    cfg.laser.extra = {"ring_radius": 4.0}
    cfg.reservoir.reservoir_type = "single"
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.002
    run = ps.SimulationController(cfg)
    run.run()
    assert np.isfinite(run.state.psi).all()
    assert run.pump_field(0).shape == (8, 8)
```

```bash
.venv/bin/pip install pytest
.venv/bin/pytest -q test_ring.py
```

!!! warning "Isolate registration across tests"
    Python caches `polarism_plugins.*` modules and Polarism registries are
    process-global. The short test above is suitable for a separate pytest
    process. For a larger suite, restore registry snapshots and remove the
    imported modules after each test, as the repository's integration fixture
    does.

## 7. Repeat on a GPU

Install the CUDA extra that matches your environment, for example
`.venv/bin/pip install "polarism[cuda12]"` for CUDA 12.x. In `run.py`, set:

```python
cfg.compute_engine.use_gpu = True
cfg.solver.method = "rk4-cuda"
cfg.solver.precision = "single"
```

Run `.venv/bin/python run.py` again on a machine with an available CUDA GPU.
The ring laser uses `self.xp`, so its spatial field stays on the selected
backend. Compare a small CPU and GPU result at a tolerance appropriate to
your model before scaling up. See [Running on GPUs](../user-guide/running-on-gpus.md)
for solver and precision details.
