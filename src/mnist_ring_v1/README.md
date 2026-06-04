# mnist_ring_v1 — Phase 1 MNIST Polariton Classifier

Ring + trigger TTFS encoding of MNIST features → far-field k-space readout → linear classifier.

## Architecture

```
MNIST 28x28 → PCA(16) → normalize [0,1] → TTFS delays → 16 ring spots + trigger
                                                          ↓
                                          GPE 2D simulation (ring+trigger condensation)
                                                          ↓
                                          psi_k^2 readout → PCA(128) + scalars → Ridge classifier
```

## Physical model

- Grid: 1024×1024, 160×160 μm, periodic, dx=dy=0.15625 μm
- Solver: `rk4-cuda` + `isotropic-9pt` + `kinetic_relaxation_eta=1e-5`
- Reservoir: `quadratic-double`
- Init: `filtered_complex_gaussian`, `k_cutoff=3.0 μm⁻¹`, fixed `seed=42`
- Physics: GaAs polaritons, hbar=0.6582, m_eff=0.32, γ_C=0.1, g_C=0.00024, R=0.023, κ=0.05

## TTFS encoding

- 16 ring spots at R=20 μm, rotated by π/16 off +x axis
- `sigma_space=0.85 μm` (FWHM ≈ 2 μm), 1 pulse per spot
- Feature → delay: `t_i = T_max × (1 - clip(f_i, 0, 1))`, T_max=100 ps
- `delay` = Polarism field; peak fires at `delay + cutoff*sigma_t ≈ delay + 4.5 ps`
- Trigger: central spot, delay=T_max, power=0.8×P_trigger_threshold
- All ring spots: power=0.8×P_ring_threshold (subthreshold alone)

## Readout

- Fixed window: t ∈ [T_max+5, T_max+35] ps
- Feature: `fftshift(fft2(psi))`, crop 64×64, `log10(P/sum(P) + 1e-10)`
- PCA(128) on train-only far-field features + 4 scalar diagnostics
- Ridge classifier

## Pipeline stages

### 1. Calibration (GPU)
```bash
python -m mnist_ring_v1.stages.gpu.calibrate --run-dir <run_dir> --mode all
```
Finds P_ring_threshold (16-spot ring, 1 pulse) and P_trigger_threshold.
Updates `calibration_result.json`; manually set values in config.yaml before submitting.

### 2. Prepare run (CPU)
```bash
python -m mnist_ring_v1.config.prepare_run --config config.yaml --run-dir <run_dir>
```
Writes: `manifest.json`, `dataset_index.json`, `batch_index.json`, `projection_model.npz`, `feature_normalizer.npz`

### 3. GPU batch array (Slurm)
```bash
python -m mnist_ring_v1.stages.gpu.run_batch --run-dir <run_dir> --batch-index <N>
```
One Slurm array task = one batch of N_images. Writes `features/batch_XXXX.npz`, `metadata/batch_XXXX.json`.

### 4. Finalize (CPU)
```bash
python -m mnist_ring_v1.stages.cpu.finalize --run-dir <run_dir>
```
Aggregates batches → PCA on far-field → Ridge classifier → `results_summary.json`

### Full submit
```bash
bash submit.sh [config.yaml] [--dry-run]
```

## Run directory layout
```
<run_dir>/
├── config.yaml
├── manifest.json
├── dataset_index.json       (2500 entries: image_id, label, split, ring_delays, trigger_delay)
├── batch_index.json         (250 batches × 10 images)
├── projection_model.npz     (PCA: mean, components, explained_variance)
├── feature_normalizer.npz   (f_min, f_max from train set)
├── calibration_result.json  (from calibrate.py)
├── features/
│   └── batch_0000.npz .. batch_NNNN.npz  (kspace_features, labels, splits, image_ids)
├── metadata/
│   └── batch_0000.json .. (scalars: psi_sq_max, k_peak_um, t_cond, condensed, ...)
└── results_summary.json     (accuracy_train/test, confusion_matrix, diagnostics)
```

## GPU-h estimate (2000 train + 500 test = 2500 images)

| Grid | Solver | dt | Total time | Per image | Total 2500 |
|------|--------|-----|-----------|-----------|-----------|
| 1024² | rk4-cuda/9pt | 0.01 | 200ps | ~160-180s | ~111-125 GPU-h |
| Reference | ifrk4-fft-cuda/5pt | 0.01 | 200ps (w/render) | 226s | 157 GPU-h |

Safe budget: **180 GPU-h** (add 20% for rk4-cuda on 1024² without render).

## Open calibration

Before the full MNIST run:
1. Download full MNIST: `python -m dot_response_fit.tools.download_mnist`
2. Run calibration pilot (32 images): `python -m mnist_ring_v1.stages.gpu.calibrate --run-dir <run_dir> --mode all`
3. Update `config.yaml` `calibration.P_ring_threshold` and `calibration.P_trigger_threshold`
4. Run `--dry-run` to verify batch count and timing
5. Submit pilot 32 images, verify `condensed_fraction > 0.9` and `high_k_frac_0p8_nyq < 1e-3`

## Phase 1 success criteria

- Polariton readout accuracy > baseline 16-PCA linear (target: 80-88% on 2k/500)
- `condensed_fraction > 0.9`
- `high_k_frac_0p8_nyq < 1e-3` (no Nyquist artifacts)
- `k_peak_um` distribution centered around expected ring-mode scale

## Questions resolved

| # | Question | Decision |
|---|----------|---------|
| Q1 | TTFS mapping | `t_i = delay` (Polarism field); peak at `delay + 4.5 ps` |
| Q2 | sigma_space | `0.85 μm` (FWHM 2 μm); new threshold calibration required |
| Q3 | Ring threshold | New calibration for 16 spots × 1 pulse |
| Q4 | Pulses per spot | 1 pulse (clean TTFS) |
| Q5 | Readout window | Fixed window: T_max+5..T_max+35 ps |
| Q6 | Ring rotation | π/16 ≈ 11.25° (reduces grid axis alignment) |
| Q7 | sklearn | Fallback to scipy/numpy; sklearn used when available |
