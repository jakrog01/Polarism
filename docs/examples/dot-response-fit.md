# `dot_response_fit` Pipeline

`src/dot_response_fit/` is an example HPC workflow for fitting the spatial
Gaussian pump spot size used to represent MNIST pixels in a two-dimensional
Polarism simulation.

The pipeline compares two responses for each selected image:

- a time-only quadratic-double-reservoir ODE reference trace, `nC(t)`
- a scalar observable from the two-dimensional GPE simulation, usually
  `psi_sq_max`

The fitted parameter is `sigma_space`. It is selected globally across all
prepared images, not by fitting a single image.

## What It Does

At a high level, the workflow is:

1. load a batch of MNIST images
2. encode each selected image as a temporal sequence of Gaussian pump pulses
3. compute one ODE reference trace per image
4. search candidate `sigma_space` values on GPU simulations
5. accept a candidate only when every image has a finite score
6. run one full simulation job per image using the best `sigma_space`
7. write per-image plots and a final summary

This is intentionally different from `polariton_hpc_pipeline`: the unit of work
is an image case, and the fit score is an aggregate over the image batch.

## Entry Point

Run from a Rysy login node:

```bash
bash src/dot_response_fit/submit.sh \
  --config src/dot_response_fit/config.yaml
```

Useful options:

```bash
bash src/dot_response_fit/submit.sh --dry-run
bash src/dot_response_fit/submit.sh --wait
bash src/dot_response_fit/submit.sh --runs-dir /path/to/runs
```

The wrapper validates `config.yaml` and `slurm.env`, creates a timestamped run
directory, freezes the config, and submits the staged Slurm jobs.

## Pipeline Structure

| Stage | Module | Resource | Responsibility |
| --- | --- | --- | --- |
| 1 | `stages/cpu/prepare_reference.py` | CPU | select MNIST images, encode pulses, run ODE references |
| 2 | `stages/gpu/fit_dot_size.py` | GPU | score candidate `sigma_space` values across all images |
| 3 | `stages/gpu/run_scenario.py` | GPU array | run one full two-dimensional simulation per image |
| 4 | `stages/cpu/finalize.py` | CPU | aggregate per-image metadata and write summary plots |

The Slurm dependency chain is:

```text
prepare_reference -> fit_dot_size -> run_scenario image array -> finalize
```

`run_scenario.py` is an array job over image indices. The array length is
derived from `mnist.sample_indices`, `mnist.sample_index`, or `mnist.n_images`
in that priority order.

## Image Selection

The `mnist` section controls the input batch:

```yaml
mnist:
  data_path: ~/data/mnist.npz
  digit_class: null
  sample_index: null
  sample_indices: null
  n_images: 10
  seed: 42
  max_pixels: 50
  spatial_width_um: 80.0
```

Selection priority:

1. `sample_indices`: explicit non-empty list of indices within the filtered
   pool
2. `sample_index`: one image, backward-compatible single-image mode
3. `n_images`: random draw from the filtered pool using `seed`

If `digit_class` is set, the pool contains only that MNIST digit. The
`sample_indices` values are indices within this filtered pool, not absolute
dataset indices.

For each image, the brightest `max_pixels` pixels are selected and then sorted
by flat image index so that the temporal pulse order follows raster order.

## Encoding and Physics

The encoding section maps pixel intensity to pump amplitude:

```yaml
encoding:
  min_amp: 10.0
  max_amp: 32.3
  pulse_width_fwhm: 5.0
  separation: 43.0
```

Each selected pixel becomes one pulsed Gaussian laser in the two-dimensional
simulation. The same pulse amplitudes and center times are used by the ODE
reference. The temporal Gaussian width is converted from FWHM to `sigma_time`.

The production ODE uses the canonical state order:

```text
[nR, nI, nC]
```

where `nR` is the active reservoir, `nI` is the inactive reservoir, and `nC` is
the scalar condensate analogue. In HDF5 output, the field named `nA`
corresponds to this active reservoir `nR`.

## Fit Configuration

The fit section controls candidate search:

```yaml
fit:
  sigma_space_values: [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]
  observable: psi_sq_max
  aggregate: mean_rmse
  max_runtime_minutes: 120
  dt_factor: 5
  n_fit_pixels: null
  scalar_stride: 50
  n_ref_points: 2000
```

`n_fit_pixels: null` means the fit simulation uses all encoded pixels for each
image. This is the preferred mode because the ODE reference and the
two-dimensional pump see the same full pulse sequence.

If `n_fit_pixels` is set to an integer, the fit stage uses only the first
`n_fit_pixels` encoded pixels per image and recomputes a subset ODE reference
for scoring. This is a speed tradeoff and should be treated as an approximation.

## Scoring

For each image and candidate `sigma_space`:

1. run a scalar-only two-dimensional simulation
2. load or recompute the matching ODE reference trace
3. resample ODE `nC(t)` and the simulation observable to a common time window
4. normalize each trace by its own maximum
5. compute RMSE between the normalized traces

The aggregate score is the mean RMSE across the image batch. A candidate is
eligible only if every image produces a finite score. Candidates with failed
images are retained in `fit_result.json` for diagnostics but cannot be selected
as the best fit.

## Output Layout

A run directory contains a frozen config, logs, references, fit results, and
per-image simulation outputs. The important paths are:

```text
reference/images/index.json
reference/images/image_000/input_image.npy
reference/images/image_000/input_image.png
reference/images/image_000/encoded_events.json
reference/images/image_000/target_trace.npz

fit_result.json

results/images/image_000/scalars.npz
results/images/image_000/trace_comparison.png
results/images/image_000/diagnostic_overlay.png
results/images/image_000/image_meta.json

results/summary.png
results_summary.json
```

For backward compatibility, the first image is also copied to:

```text
reference/input_image.npy
reference/encoded_events.json
reference/target_trace.npz
```

## Visual Outputs

Each image result directory contains:

- `input_image.png`: the exact normalized MNIST image used for encoding
- `trace_comparison.png`: ODE `nC(t)` and two-dimensional observable on one
  normalized comparison plot
- `diagnostic_overlay.png`: a final diagnostic frame with the last active pixel
  marked on the spatial field and the MNIST thumbnail
- optional field PNGs and animation, depending on the `output` section

The animation omits the active-pixel marker. The marker is intentionally
restricted to the final diagnostic overlay.

## Raw HDF5 Policy

Raw HDF5 output is controlled by:

```yaml
output:
  archive_raw_hdf5: false
```

When `archive_raw_hdf5` is false, raw HDF5 is used only as a temporary render
source and is not retained in the final run output. When true, the per-image
HDF5 file is archived under `results/images/<image_id>/`.

## When To Use It

Use this pipeline when you want to evaluate whether one fitted pump spot size
can reproduce the time-response shape across a batch of encoded MNIST images.

If you only need generic multi-scenario Slurm orchestration, use
`polariton_hpc_pipeline` instead.
