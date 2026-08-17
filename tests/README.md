# Verification suite

## How to run each tier

- `pytest -m "not slow and not compliance"` runs the default unit, integration, and requirements tiers.
- `pytest -m compliance` runs the reference and cross-check tiers.
- `pytest -m slow` runs convergence, memory, and GPU-speedup tiers.
- `pytest -m gpu` runs CPU/GPU agreement and speedup checks on CUDA hardware.
- Continuous integration runs `pytest -m 'not slow and not compliance and not gpu'` on `ubuntu-latest` for every push and PR.

The default command deliberately deselects `slow` and `compliance` nodes to keep ordinary development feedback short. A full verification run must execute every tier above; the Phoenix reference matrix is run separately with `pytest tests/test_phoenix_benchmark.py -m slow` on CUDA hardware.

## Dane referencyjne Phoenix

Dziewięć porównań `test_phoenix_accuracy` (trzy przypadki fizyczne × trzy solwery FDM)
czyta dane referencyjne z `tests/data/phoenix_benchmark/<przypadek>/`. Katalog jest
śledzony w repozytorium, więc świeży klon wystarcza do odtworzenia porównań — nie ma
osobnego kroku pobierania danych. Na przypadek zapisane są: `rho_max.txt` (przebieg
`max|ψ|²` z PHOENIX), `psi_init.txt`, `pump.txt`, `potential.txt`,
`phoenix_lasers_setup.yaml`, `timing.json` oraz `frame_first.npz` i `frame_last.npz`.
Test najpierw sprawdza, czy siatka, pompa, potencjał i warunek początkowy Polarism
zgadzają się z plikami PHOENIX, a dopiero potem porównuje wynik.

Dane wygenerowano notatnikiem `tests/data/phoenix_benchmark/example.ipynb`
uruchomionym w kontenerze `robertschade/phoenix:latest` (pyphoenix, fp64, GPU).
Pliki są oznaczone w `.gitattributes` jako binarne, aby porównania pozostały
bajtowo identyczne niezależnie od ustawienia `core.autocrlf`.

## Pełna walidacja CPU i GPU

Do raportu końcowego uruchom całą baterię jednym procesem:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

Polecenie obejmuje wszystkie markery, pełną macierz Phoenix oraz test
zgodności CPU/GPU. Nie uruchamiaj równolegle drugiego pytest na tej samej
karcie GPU. Po powodzeniu można utworzyć lokalne dowody do pracy:

```bash
.venv/bin/python -m scripts.generate_verification_evidence
.venv/bin/python -m scripts.generate_verification_artifacts
```

Pliki `artifacts/` są wynikami wykonania i pozostają poza historią Git.

## What each tier proves

Unit tests verify single-file correctness. Integration verifies the end-to-end run matrix. Reference tests compare against analytic closed forms. Convergence tests support order-of-accuracy claims. Cross-checks compare solvers and CPU/GPU parity. Quality tests cover memory scaling, extensibility, reproducibility, and GPU speedup. The requirements matrix provides WF/WJ traceability to collected test nodes.

## Artefacts

Generated outputs are `artifacts/convergence/*.json`, including `artifacts/convergence/fitted_orders.json`; `artifacts/benchmark/gpu_speedup.json` with `environment` and `entries` keys; `artifacts/benchmark/hardware.tex`; and `artifacts/requirements_matrix.json` plus `artifacts/requirements_matrix.tex`.
Verification-ready PDF/PNG/JSON figures and LaTex fragments are generated with `.venv/bin/python -m scripts.generate_verification_artifacts` in `artifacts/reports/`.
Convergence analysis writes `figures/fig1_time_convergence.pdf` and `figures/fig2_space_convergence.pdf`.
Pipeline manifests and scenario metadata carry an `environment` object with the stable runtime fields; seeded threshold scans additionally write `threshold_ensemble.json`.

Regenerate convergence figures from artifacts:

    python -m scripts.analyse_convergence

This reads artifacts/convergence/{time,space}_*.json, writes fitted_orders.json,
and produces figures/fig1_time_convergence.pdf and figures/fig2_space_convergence.pdf.

## Convergence analysis

Run `.venv/bin/python -m scripts.analyse_convergence` after the convergence artefacts have been generated. The fitter drops the coarsest dt, then drops a non-finest point only when its log-log slope to the finest point is more than 0.5 below its adjacent-pair slope; it always retains the finest point and requires at least 3 points before attempting a fit.

## Adding a solver

Register the solver in `polarism/solver/solver_registry.py` and add its `_SOLVER_CAPABILITIES` entry. The run matrix and convergence tiers derive their coverage from that capability table.
