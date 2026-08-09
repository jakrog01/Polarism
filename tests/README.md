# Verification suite

## How to run each tier

- `pytest -m "not slow and not compliance"` runs the default unit, integration, and requirements tiers.
- `pytest -m compliance` runs the reference and cross-check tiers.
- `pytest -m slow` runs convergence, memory, and GPU-speedup tiers.
- `pytest -m gpu` runs CPU/GPU agreement and speedup checks on CUDA hardware.

The default command deliberately deselects `slow` and `compliance` nodes to keep ordinary development feedback short. A full verification run must execute every tier above; the Phoenix reference matrix is run separately with `pytest tests/test_phoenix_benchmark.py -m slow` on CUDA hardware.

## What each tier proves

Unit tests verify single-file correctness. Integration verifies the end-to-end run matrix. Reference tests compare against analytic closed forms. Convergence tests support order-of-accuracy claims. Cross-checks compare solvers and CPU/GPU parity. Quality tests cover memory scaling, extensibility, reproducibility, and GPU speedup. The requirements matrix provides WF/WJ traceability to collected test nodes.

## Artefacts

Generated outputs are `artifacts/convergence/*.json`, `artifacts/benchmark/*.json`, and `artifacts/requirements_matrix.json` plus `artifacts/requirements_matrix.tex`.
Verification-ready PDF/PNG/JSON figures and LaTex fragments are generated with `.venv/bin/python -m scripts.generate_verification_artifacts` in `artifacts/reports/`.

## Adding a solver

Register the solver in `polarism/solver/solver_registry.py` and add its `_SOLVER_CAPABILITIES` entry. The run matrix and convergence tiers derive their coverage from that capability table.
