# Testing and CI

The repository has multiple pytest layers, and they serve different purposes. This was previously undocumented; this page is the operational reference.

## Test suites

| Command | Purpose |
| --- | --- |
| `pytest` | Default CPU-oriented regression suite |
| `pytest -m compliance` | Solver-to-solver agreement and compatibility checks |
| `pytest -m slow --use-gpu` | Long-running Phoenix benchmark workflow |
| `pytest -q -m '' --use-gpu --junitxml=artifacts/reports/full_verification.xml` | Complete CPU and GPU verification, including Phoenix |

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

Use the following command for a full verification pass:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

`-m ''` overrides the normal development selection and includes every marker:
unit, integration, reference, compliance, slow, and GPU. The command executes
the complete Phoenix matrix as well as the CPU/GPU agreement checks. Do not run
a second pytest process at the same time: Phoenix needs exclusive use of the
GPU for meaningful timing and reproducible comparisons.

The JUnit XML is the primary machine-readable execution record. After a
successful run, generate the Polish-language tables and figures with:

```bash
.venv/bin/python -m scripts.generate_verification_evidence
.venv/bin/python -m scripts.generate_verification_artifacts
```

All outputs under `artifacts/` are local generated data and must not be
committed.

## What the tests cover

- `test_solver_decay.py` checks exponential decay against an analytic expectation.
- `test_solver_uniformity.py` verifies that a uniform pump does not create artificial spatial structure.
- `test_reservoir_0d_stationary.py` checks the single-reservoir stationary limit.
- `test_solver_compliance.py` compares solver families and reservoir variants.
- `test_phoenix_benchmark.py` is a long benchmark-style validation path.

## Phoenix reference data

`test_phoenix_benchmark.py` reads its reference traces from
`tests/data/phoenix_benchmark/`, which is tracked in the repository. A fresh
clone therefore contains everything the nine `test_accuracy` comparisons need
(`rho_max.txt`, `psi_init.txt`, `pump.txt`, `potential.txt`,
`phoenix_lasers_setup.yaml`, `timing.json`, `frame_first.npz`,
`frame_last.npz`); there is no separate data download step. Unlike `artifacts/`,
this directory is verification *input*, not generated output.

The data was produced by `tests/data/phoenix_benchmark/example.ipynb` inside the
`robertschade/phoenix:latest` container (pyphoenix, fp64, GPU) and is marked
binary in `.gitattributes` so the byte-exact input checks stay valid regardless
of the local `core.autocrlf` setting.

## GitHub Actions deployment flow

The repository now includes a documentation workflow that:

1. installs the project dependencies
2. builds the MkDocs site
3. deploys the built site to GitHub Pages on pushes to `master` or `main`

## Important limitation

The Pages workflow intentionally avoids running the test suites. This keeps the deployment job focused on documentation publishing and avoids spending GitHub-hosted runner time on validations that are better run locally or on hardware-aware infrastructure.

If you later add CI back, keep the heavier compliance and GPU-oriented validations separate from the documentation deployment path.
