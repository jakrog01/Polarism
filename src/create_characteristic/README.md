# create_characteristic

Pipeline do generowania 2D map charakterystyk progowania polarytonów.

Skanuje przestrzeń parametrów `(pulse_energy, pulse_separation)` i produkuje mapę `max|ψ|²(E, sep)` wraz z wizualizacją progu kondensacji.

## Cel

Każdy punkt siatki to jedna pełna symulacja GPE + rezerwuar (ifrk4-fft-cuda) z domyślnymi parametrami GaAs. Wynikiem jest:

- Mapa 2D `max|ψ|²` jako heatmap
- Mapa progu kondensacji (binarna: powyżej/poniżej kryterium `psi_sq_max >= threshold_criterion`)
- Izolinia progu nałożona na heatmapę
- CSV i JSON ze wszystkimi wynikami per punkt

## Struktura konfiguracji

```yaml
global:
  grid:       # nx, ny, lx, ly — rozmiar siatki (domyślnie 1024x1024, 160x160 um)
  physics:    # stałe GaAs: hbar, m_eff, gamma_C, gamma_R, g_C, g_R, g_I, R, kappa
  boundary_condition:  # CAP sin², strength, mask_width_percent
  potential:  # zero
  reservoir:  # quadratic-double
  solver:     # dt, method (ifrk4-fft-cuda)

laser:
  laser_type: pulse-gaussian
  sigma_space, sigma_time, cutoff_sigma, n_pulses
  power_definition: pulse_energy

sweep:
  energy_min / energy_max / energy_step       # oś energii impulsu
  separation_min / separation_max / separation_step  # oś separacji [ps]
  post_pulse_time: 80.0       # czas po ostatnim impulsie [ps]
  adaptive_total_time: true   # total_time per punkt zależy od separacji
  scalar_check_every: 100     # sprawdzenie skalara co N kroków
  early_stop_on_divergence: true
  max_concurrent: 16          # Slurm: max równoległych tasków

output:
  save_per_point_trace: false   # domyślnie nie zapisuj przebiegów czasowych
  threshold_criterion: 5.0e-2   # kryterium kondensacji
```

### Adaptive total_time

Dla każdego punktu czas symulacji jest obliczany jako:

```
total_time = 2 * cutoff_sigma * sigma_time + (n_pulses - 1) * pulse_separation + post_pulse_time
```

Przy domyślnych parametrach (9 impulsów, sigma_time=1.7 ps, cutoff_sigma=3.0):
- sep=12 ps  → total_time ≈ 10.2 + 96 + 80 = 186.2 ps
- sep=122 ps → total_time ≈ 10.2 + 976 + 80 = 1066.2 ps

## Rozmiar domyślnej siatki

```
energy:     1000..3200 co 200 → 12 wartości
separation: 12..122    co 10  → 12 wartości
Łącznie:    144 punkty
```

## Uruchomienie na Rysy

```bash
# dry-run (brak sbatch)
bash src/create_characteristic/submit.sh --dry-run

# pełne zgłoszenie do Slurm
bash src/create_characteristic/submit.sh

# własny config
bash src/create_characteristic/submit.sh --config src/create_characteristic/scenarios/moj_sweep.yaml
```

Wymaga pliku `slurm.env` w katalogu głównym repozytorium z zmiennymi:
`SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_MEM`, `SLURM_GPUS`, `SLURM_CPUS`,
`SLURM_TIME`, `TETYDA_RUNS_BASE`, `FINALIZE_MEM`, `FINALIZE_CPUS`, `FINALIZE_TIME`.

## Artefakty wynikowe

```
runs/<run_id>/
  config.yaml              — kopia konfiguracji
  manifest.json            — metadane runu
  point_index.json         — lista wszystkich punktów siatki
  points/
    point_000000.json      — wynik per punkt (psi_sq_max, status, walltime, ...)
    ...
  characteristic_map.csv   — tabela wszystkich punktów
  characteristic_map.json  — to samo w formacie JSON
  threshold_summary.json   — próg kondensacji per separacja, statystyki
  results/
    psi_max_heatmap.png         — mapa max|ψ|² z izolinią progu
    psi_max_heatmap_log.png     — to samo w skali log₁₀
    threshold_map.png           — binarna mapa kondensacji (zielony/czerwony)
  logs/
    local_point_XXXXXX.log / sweep_JOBID_TASKID.out
    finalize.log
```

## Mapa i kryterium progu

**Oś X**: separacja impulsów [ps]
**Oś Y**: energia impulsu [j.u. lub pulse_energy]
**Kolor**: `max|ψ|²` — maksimum gęstości kondensatu w całej symulacji

Kryterium kondensacji: `psi_sq_max >= threshold_criterion` (domyślnie `5e-2`).

Punkt oznaczony czerwonym `×` na mapie oznacza dywergencję numeryczną (NaN/Inf).

`threshold_summary.json` zawiera `threshold_energy_per_separation` — minimalną energię, dla której nastąpiła kondensacja, dla każdej wartości separacji.

## Scenariusze

Własne konfiguracje umieszczaj w `scenarios/*.yaml`.
Przykłady i opisy: [scenarios/README.md](scenarios/README.md).
