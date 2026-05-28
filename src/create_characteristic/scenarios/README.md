# Scenariusze create_characteristic

Własne konfiguracje 2D sweepów. Pliki `*.yaml` w tym katalogu są ignorowane przez git.

## Konwencja nazewnictwa

```
gaas_<opis>_characteristic.yaml
```

Przykład: `gaas_9pulse_2um_characteristic.yaml`

## Minimalny przykład

```yaml
global:
  grid:
    nx: 512
    ny: 512
    lx: 80.0
    ly: 80.0
    grid_type: periodic
  physics:
    hbar: 0.6582119514
    m_eff: 0.32
    gamma_C: 0.1
    gamma_R: 0.15
    gamma_I: 0.001
    g_C: 0.00024
    g_R: 0.0006
    g_I: 0.0
    R: 0.023
    kappa: 0.05
    init_eps: 0.001
    init_mode: filtered_complex_gaussian
    init_k_cutoff_um: 3.0
    init_seed: 42
    kinetic_relaxation_eta: 0.0
    reservoir_diffusion_I: 0.0
    reservoir_diffusion_R: 0.0
  boundary_condition:
    absorption: cap
    profile_type: sin2
    strength: 5.0
    mask_width_percent: 0.2
  potential:
    potential_type: zero
  reservoir:
    reservoir_type: quadratic-double
  solver:
    dt: 0.01
    method: ifrk4-fft-cuda
    laplacian: five-point

laser:
  laser_type: pulse-gaussian
  x0: 0.0
  y0: 0.0
  sigma_space: 2.0
  sigma_time: 1.7
  cutoff_sigma: 3.0
  n_pulses: 9
  power_definition: pulse_energy

sweep:
  energy_min: 1500.0
  energy_max: 2500.0
  energy_step: 100.0
  separation_min: 20.0
  separation_max: 80.0
  separation_step: 10.0
  post_pulse_time: 80.0
  adaptive_total_time: true
  scalar_check_every: 100
  early_stop_on_divergence: true
  max_concurrent: 8

output:
  save_per_point_trace: false
  threshold_criterion: 5.0e-2
```

## Uruchomienie

```bash
bash src/create_characteristic/submit.sh --config src/create_characteristic/scenarios/moj_sweep.yaml
```
