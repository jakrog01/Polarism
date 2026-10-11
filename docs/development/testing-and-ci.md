# Testing and CI

The repository has multiple pytest layers, and they serve different purposes. This was previously undocumented; this page is the operational reference.

## Test suites

| Command | Purpose |
| --- | --- |
| `.venv/bin/pytest -q` | Default CPU-oriented regression suite |
| `.venv/bin/pytest -m compliance` | Solver-to-solver agreement and compatibility checks |
| `.venv/bin/pytest -m slow --use-gpu` | Long-running Phoenix benchmark workflow |
| `.venv/bin/pytest -q -m '' --use-gpu --junitxml=artifacts/reports/full_verification.xml` | Complete CPU and GPU verification, including Phoenix |

The default `pyproject.toml` configuration excludes `slow` and `compliance` markers from a plain `pytest` run.

## Recommended local workflow

Before a normal contribution:

```bash
pytest
mkdocs build
```

Before changing solver kernels, reservoir logic, or numerical defaults:

```bash
pytest
pytest -m compliance
mkdocs build
```

Before claiming GPU performance or benchmark parity:

```bash
pytest -m slow --use-gpu
```

Run the slow suite only on a machine where the runtime and available data make sense.

## Complete verification run

Use the following command for a complete verification pass:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

`-m ''` overrides the normal development selection and includes every marker:
unit, integration, reference, compliance, slow, and GPU. The command executes
the complete Phoenix matrix as well as the CPU/GPU agreement checks. Do not run
a second pytest process at the same time: Phoenix needs exclusive use of the
GPU for meaningful timing and reproducible comparisons.

The JUnit XML is the primary machine-readable execution record. All outputs
under `artifacts/` are local generated data and must not be committed.

## What the tests cover

- `test_solver_decay.py` checks exponential decay against an analytic expectation.
- `test_solver_uniformity.py` verifies that a uniform pump does not create artificial spatial structure.
- `test_reservoir_0d_stationary.py` checks the single-reservoir stationary limit.
- `test_solver_compliance.py` compares solver families and reservoir variants.
- `test_phoenix_benchmark.py` is a long benchmark-style validation path.

## Phoenix reference data

`test_phoenix_benchmark.py` reads its reference traces from
`tests/data/phoenix_benchmark/`, which is tracked in the repository. A fresh
clone therefore contains everything the eleven `test_phoenix_accuracy` comparisons need
(`rho_max.txt`, `psi_init.txt`, `pump.txt`, `potential.txt`,
`phoenix_lasers_setup.yaml`, `timing.json`, `frame_first.npz`,
`frame_last.npz`); there is no separate data download step. Unlike `artifacts/`,
this directory is verification *input*, not generated output.

The data was produced by `tests/data/phoenix_benchmark/example.ipynb` inside the
`robertschade/phoenix:latest` container (pyphoenix, fp64, GPU) and is marked
binary in `.gitattributes` so the byte-exact input checks stay valid regardless
of the local `core.autocrlf` setting.

## GitHub Actions workflows

- **Test** (`tests.yml`) runs for every push and pull request on Python 3.12
  and 3.13. It installs the `test` dependency extra and runs the CPU test
  selection, excluding slow, compliance, and GPU-only tests.
- **Docs** (`docs.yml`) builds the MkDocs site in strict mode for every pull
  request. It deploys to GitHub Pages only when called by the release workflow.
- **Release** (`release.yml`) runs for pull requests into `master` and for
  pushes to `master`; see [Branches and releases](contributing.md#branches-and-releases).

## Important limitation

The hosted workflows do not run the slow, compliance, or GPU suites. Run those
locally before opening a release pull request when solver behavior changed.
