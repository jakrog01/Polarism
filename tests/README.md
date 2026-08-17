# Verification suite

## How to run each tier

- `pytest -m "not slow and not compliance"` runs the default unit, integration, and requirements tiers.
- `pytest -m compliance` runs the reference and cross-check tiers.
- `pytest -m slow` runs convergence, memory, and GPU-speedup tiers.
- `pytest -m gpu` runs CPU/GPU agreement and speedup checks on CUDA hardware.
- Continuous integration runs `pytest -m 'not slow and not compliance and not gpu'` on `ubuntu-latest` for every push and PR.

The default command deliberately deselects `slow` and `compliance` nodes to keep ordinary development feedback short. A full verification run must execute every tier above; the Phoenix reference matrix is run separately with `pytest tests/test_phoenix_benchmark.py -m slow --use-gpu` on CUDA hardware.

## Dane referencyjne Phoenix

Jedenaście porównań `test_phoenix_accuracy` obejmuje trzy przypadki fizyczne
dla trzech solverów FDM oraz dwa przypadki bez potencjału dla `ifrk4-fft-cuda`.
Porównania czytają dane referencyjne z `tests/data/phoenix_benchmark/<przypadek>/`. Katalog jest
śledzony w repozytorium, więc świeży klon wystarcza do odtworzenia porównań — nie ma
osobnego kroku pobierania danych. Na przypadek zapisane są: `rho_max.txt` (przebieg
`max|ψ|²` z PHOENIX), `psi_init.txt`, `pump.txt`, `potential.txt`,
`phoenix_lasers_setup.yaml`, `timing.json` oraz `frame_first.npz` i `frame_last.npz`.
Test najpierw sprawdza, czy siatka, pompa, potencjał i warunek początkowy Polarism
zgadzają się z plikami PHOENIX, a dopiero potem porównuje wynik.

Progi `ifrk4-fft-cuda` opisują zaakceptowaną granicę ograniczonej zgodności, a
nie zgodność z PHOENIX na poziomie solverów FDM. W przypadku 01 bezpośrednia
różnica maksimum gęstości FDM–IFRK przy 500 ps wyniosła 0.05305, czyli 97.1%
błędu końcowego IFRK–PHOENIX równego 0.05462. Różnica L2 całego pola gęstości
nie narastała: spadła z 0.02412 przy 10 ps do 0.01938 przy 500 ps. Pomiar
wskazuje operator przestrzenny jako dominujące źródło końcowej rozbieżności
skalarnej, ale nie potwierdza monotonicznego rozjazdu całej trajektorii ani
wpływu szumu. Pełne dane są w `artifacts/crosscheck/ifrk4_divergence.json`.
Każdy nowy raport zapisuje zakres interpretacji w polu `validation_scope`
pliku `metrics.json` oraz w `metrics.txt`. Metryka `frame_phase_rmse` przed
porównaniem usuwa stały globalny offset fazy U(1) na masce gęstości
referencyjnej.

Katalogi FDM w `tests/test_results/test_phoenix_benchmark/` bez `metrics.json`
pochodzą sprzed wprowadzenia bieżącego formatu raportu. Wartość zera przy
niezmierzonym użyciu pamięci w ich archiwalnych `metrics.txt` nie jest aktualnym
dowodem pomiarowym. Raporty w nowym formacie powstają dopiero po ponownym
uruchomieniu odpowiedniego przypadku; repozytorium nie traktuje starych plików
graficznych ani tekstowych jako wyniku obecnej walidacji.

Dane wygenerowano notatnikiem `tests/data/phoenix_benchmark/example.ipynb`
uruchomionym w kontenerze `robertschade/phoenix:latest` (pyphoenix, fp64, GPU).
Pliki są oznaczone w `.gitattributes` jako binarne, aby porównania pozostały
bajtowo identyczne niezależnie od ustawienia `core.autocrlf`.

Bez flagi `--use-gpu` pytest konfiguruje backend CPU, także dla macierzy Phoenix.
Flaga jest wymagana do odtworzenia raportów `ifrk4-fft-cuda` i wyników GPU.

## Pełna walidacja CPU i GPU

Do raportu końcowego uruchom całą baterię jednym procesem:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

Polecenie obejmuje wszystkie markery, pełną macierz Phoenix oraz test
zgodności CPU/GPU. Nie uruchamiaj równolegle drugiego pytest na tej samej
karcie GPU. Pliki `artifacts/` są wynikami wykonania i pozostają poza historią
Git.

## What each tier proves

Unit tests verify single-file correctness. Integration verifies the end-to-end run matrix. Reference tests compare against analytic closed forms. Convergence tests support order-of-accuracy claims. Cross-checks compare solvers and CPU/GPU parity. Quality tests cover memory scaling, extensibility, reproducibility, and GPU speedup. The requirements matrix provides WF/WJ traceability to collected test nodes.

## Artefacts

Generated outputs are `artifacts/convergence/*.json`, including `artifacts/convergence/fitted_orders.json`; `artifacts/benchmark/gpu_speedup.json` with `environment` and `entries` keys; `artifacts/benchmark/hardware.tex`; and `artifacts/requirements_matrix.json` plus `artifacts/requirements_matrix.tex`.
Pipeline manifests and scenario metadata carry an `environment` object with the stable runtime fields; seeded threshold scans additionally write `threshold_ensemble.json`.

Dla serii czasowej `rk4-fdm_single` najdrobniejszy punkt `dt=0.000125`
pozostaje w regresji: jego błąd `1.61e-6` jest ponad pięć rzędów wielkości
powyżej podłogi `1e-11`, a trzy końcowe nachylenia lokalne wynoszą 3.983,
3.986 i 3.973. Niższe `fit_r2=0.9968` powoduje punkt `dt=0.002` z lokalnym
nachyleniem 5.173, a nie kontaminacja najdrobniejszego punktu podłogą
Richardsona. Fit nadal spełnia pasmo rzędu i kryterium jakości.

## Adding a solver

Register the solver in `polarism/solver/solver_registry.py` and add its `_SOLVER_CAPABILITIES` entry. The run matrix and convergence tiers derive their coverage from that capability table.
