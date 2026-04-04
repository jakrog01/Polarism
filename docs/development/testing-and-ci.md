# Testing and CI

The repository has multiple pytest layers, and they serve different purposes. This was previously undocumented; this page is the operational reference.

## Test suites

| Command | Purpose |
| --- | --- |
| `pytest` | Default CPU-oriented regression suite |
| `pytest -m compliance` | Solver-to-solver agreement and compatibility checks |
| `pytest -m slow --use-gpu` | Long-running Phoenix benchmark workflow |

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

## What the tests cover

- `test_solver_decay.py` checks exponential decay against an analytic expectation.
- `test_solver_uniformity.py` verifies that a uniform pump does not create artificial spatial structure.
- `test_reservoir_0d_stationary.py` checks the single-reservoir stationary limit.
- `test_solver_compliance.py` compares solver families and reservoir variants.
- `test_phoenix_benchmark.py` is a long benchmark-style validation path.

## GitHub Actions deployment flow

The repository now includes a documentation workflow that:

1. installs the project dependencies
2. builds the MkDocs site
3. deploys the built site to GitHub Pages on pushes to `master` or `main`

## Important limitation

The Pages workflow intentionally avoids running the test suites. This keeps the deployment job focused on documentation publishing and avoids spending GitHub-hosted runner time on validations that are better run locally or on hardware-aware infrastructure.

If you later add CI back, keep the heavier compliance and GPU-oriented validations separate from the documentation deployment path.
