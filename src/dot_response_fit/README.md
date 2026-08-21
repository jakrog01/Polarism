# Dot-Response Fit

`dot_response_fit` fits the spatial Gaussian pump spot size (`sigma_space`) for
MNIST pixel pulses by comparing:

- a time-only quadratic-double-reservoir ODE reference trace, `nC(t)`
- a two-dimensional Polarism scalar observable, usually `psi_sq_max`

The fit is global across a batch of images. A candidate `sigma_space` can win
only if every selected image has a finite RMSE score.

## Run

From a Rysy login node:

```bash
bash src/dot_response_fit/submit.sh --config src/dot_response_fit/config.yaml
```

Useful modes:

```bash
bash src/dot_response_fit/submit.sh --dry-run
bash src/dot_response_fit/submit.sh --wait
```

The wrapper expects `slurm.env` at the repository root; create it with
`cp slurm.env.example slurm.env` and replace the example cluster values. It
uses the repository `.venv` when available.

## Stages

```text
prepare_reference -> fit_dot_size -> run_scenario image array -> finalize
```

- `prepare_reference`: selects MNIST images, encodes pulses, writes ODE traces
- `fit_dot_size`: evaluates `sigma_space_values` across all images
- `run_scenario`: runs one full simulation per image with the best spot size
- `finalize`: writes `results_summary.json` and `results/summary.png`

## Key Config Fields

```yaml
mnist:
  sample_indices: null
  sample_index: null
  n_images: 10
  max_pixels: 50

fit:
  sigma_space_values: [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]
  observable: psi_sq_max
  aggregate: mean_rmse
  n_fit_pixels: null
```

Image selection priority is:

1. non-empty `sample_indices`
2. `sample_index`
3. random draw of `n_images`

`sample_indices` are indices inside the filtered MNIST pool. If `digit_class`
is set, the pool contains only that digit.

`n_fit_pixels: null` uses all selected pixels in the fit. Setting an integer
uses only the first `n_fit_pixels` pixels per image and recomputes the matching
subset ODE reference.

## Output

Important files in each run directory:

```text
reference/images/index.json
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

Raw HDF5 is retained only when `output.archive_raw_hdf5: true`.
