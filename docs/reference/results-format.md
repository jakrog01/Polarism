# Results format

Polarism writes sampled results in HDF5, NumPy or JSON format version 3. Use this page to inspect files directly or to write a reader. Checkpoints are separate HDF5 state files with version 2.

The controller creates a timestamped run directory under `result.output_directory`. `result.formats` selects registered storage names, including `hdf5`, `npy` and `json`. Field names depend on the result nodes selected by the run. `density` and `norm` are standard; `psi` is saved when `save_psi` is true. Static, time-independent pump fields are stored once. See [Output and results](../user-guide/output-and-results.md) for selecting quantities.

## Axes and units

Let `T` be saved frames, `B` sweep members, `C` condensate components, `C_R` reservoir components and `S` the spatial shape. `S=(nx,)` in 1D and `S=(ny,nx)` in 2D. The HDF5 writer stores each dynamic field as `(T, component, *S)` or `(T, B, component, *S)` when a component axis applies. It inserts a singleton component axis for ordinary scalar fields, including scalar runs. A spinor field has `C=2`; reservoir fields can have `C_R=1` or `2`. `stokes` uses a four-value `stokes` axis instead of `component`. Static fields have no time axis and may omit a shared member axis. NPY and JSON preserve the node's public array shape: `(T,*S)` for a scalar field, `(T,C,*S)` for a spinor field, with `B` after `T` in batched runs.

| Quantity | Shape | Units |
| --- | --- | --- |
| `time` | `(T,)` | ps |
| `grid/x`, `grid/y` | `(nx,)`, `(ny,)` | µm |
| `fields/psi` | `(T,C,*S)` or `(T,B,C,*S)` in HDF5 | µm⁻ᵈᐟ² |
| `fields/density` | same component convention as `psi` | µm⁻ᵈ |
| `fields/stokes` | `(T,4,*S)` or `(T,B,4,*S)` | µm⁻ᵈ |
| `fields/<name>` | `(T,*S)`, possibly with member and component axes | node-defined |
| `scalars/<name>` | `(T,)` or `(T,B)` | node-defined |
| `scalar_groups/<group>/<label>` | `(T,)` or `(T,B)` | node-defined |
| `static/<name>` | `(*S)`, `(B,*S)` or a spinor field with a component axis | node-defined |

`d` is `grid.ndim`. `momentum_density` uses `kx` and `ky` axes in place of `x` and `y`. `psi` is complex; JSON encodes a complex array as `{"real": [...], "imag": [...]}`. A result reader squeezes the singleton HDF5 component axis for scalar fields, so its public shape matches NPY and JSON. For spinors, component order is `plus`, `minus`. Stokes order is `S0`, `S1`, `S2`, `S3`.

## HDF5 layout

`results.h5` has root datasets `time`, optional `member_index`, `grid/x`, `grid/y`, `fields/<name>`, `scalars/<name>`, `scalar_groups/<group>/<label>`, `static/<name>` and `accumulated/<name>/value`. `config` is a group whose `json` attribute contains the full configuration. The root attributes are `format_version=3`, `polarism_version`, `ndim`, `grid_type`, `components` (`["scalar"]` or `["plus","minus"]`) and `units`, a JSON mapping for x, y and time. Batched output also has root attribute `members`, dataset `member_index` with `dims=["member"]` and `units=""`, and `config/member_seeds` when available. When steady-state stopping is active, root `stop_time` is a member-length array, aligned with `member_index` in batches; adaptive runs add attributes `adaptive_accepted`, `adaptive_rejected`, `adaptive_dt_min`, `adaptive_dt_max` and `adaptive_dt_last`.

Every frame or static dataset has `label`, `dims` and `units` attributes. `time` has `label="time"`, `dims=["time"]`, `units="ps"`; coordinate datasets have `dims=["x"]` or `["y"]` and `units="µm"`. HDF5 compression is `none`, `lzf` or `gzip`; compressed datasets use shuffle, and `compression_level` controls gzip only. Frames append along the time axis.

## NumPy layout

The directory contains `metadata.json`, `batch_000000.npz`, subsequent numbered batches, optional `static_<name>.npy`, `grid_x.npy`, `grid_y.npy`, `member_index.npy`, and `accumulated_<name>.npy`. Each NPZ batch contains `time`, `field_<name>`, `scalar_<name>` and `group_<group>_<label>` keys. `metadata.json` records `format_version=3`, `polarism_version`, `ndim`, `config` and `datasets`, whose entries have `label`, `dims` and `units` where applicable. Batched metadata adds `members` and `member_seeds`; steady-state runs add `stop_time`. NPY output uses uncompressed NPZ unless `result.compression` is other than `none`, in which case it uses `np.savez_compressed`.

## JSON layout

The directory contains `metadata.json` and `batch_000000.json`, followed by numbered batches. Each batch has `time`, `fields`, `scalars` and `scalar_groups` mappings. The first batch also has `static` when static fields exist. Batched output adds `member_index.json` and, when coordinates are supplied, `grid_x.json` and `grid_y.json`. `metadata.json` carries version, configuration, dataset dimensions and units, and optional member information and steady-state stop times. In unbatched JSON output, coordinate files are not written.

## Accumulated results

Accumulated results are finalized at the end of a run, apart from their intermediate checkpoint state. HDF5 stores each value at `accumulated/<name>/value`, with dataset attributes `dims`, `units`, `label`, `wigner_corrected` and `linear`. The parent group stores sampling metadata such as `samples`, `t_first` and `t_last` in ps. NPY writes `accumulated_<name>.npy` and stores the same metadata in `metadata.json` under `accumulated/<name>`. JSON stores both metadata and `value` in that section. Built-in spectra expose energy axes as separate accumulated values. `linear` indicates whether sweep members may be averaged directly; see [Observables and spectra](../user-guide/observables-and-spectra.md).

| Accumulator | Saved value names | Dimensions after optional `member`, `component` | Units |
| --- | --- | --- | --- |
| `momentum-distribution` | `momentum_distribution`, `g1` | `(ky,kx)` or `(kx)` for distribution; `(y,x)` or `(x)` for `g1` | particles/mode; `g1` is dimensionless |
| `g2` | `density_mean`, `density_squared_mean`, `g2` | `S` | µm⁻ᵈ; µm⁻²ᵈ; `g2` is dimensionless |
| `spectrum-k` | `spectrum_k`, `spectrum_k_energy` | `(E,kx)`; `(E)` with optional member axis | particles/mode/meV; meV |
| `spectrum-x` | `spectrum_x`, `spectrum_x_energy` | `(E,x)`; `(E)` with optional member axis | µm⁻ᵈ meV⁻¹; meV |

Spectrum energy coordinates increase along `E` and include the coherent-drive energy shift when enabled. Only results marked `linear=true` can be directly combined across sweep members. Derived `g1` and `g2` are nonlinear; recompute them from averaged moments.

## Checkpoint HDF5 layout

`SimulationController.save_checkpoint(path)` writes to a temporary file and atomically replaces the destination. Checkpoint version 2 contains `field` in canonical `(B,C,*S)` shape, optional `field_k` of the same shape, and `reservoir/<state_name>` datasets of shape `(B,C_R,*S)`. Built-in reservoir state names are `nR`, `nA` and `nI` as applicable; a plugin without `state_names` uses numeric names starting at `0`.

Root attributes are `checkpoint_version=2`, `polarism_version`, `step`, `t` and `t0` in ps, `dt` in ps, `ndim`, spatial `shape`, `components`, `batched`, `reservoir_type`, `config_json`, `stop_time` and `steady_last_step`. Batched sweeps add `member_configs_json`. A steady-state reference adds `steady_norm` and dataset `steady_density`. Adaptive runs add `step_controller` with attributes `h_next`, `err_prev`, `accepted`, `rejected`, `dt_min`, `dt_max` and `dt_last`; time values are in ps. Configured accumulators add `accumulators/<name>`: attributes `samples`, `t_first`, `t_last` (times in ps, absent times stored as NaN) and datasets for that accumulator's `state_dict()` arrays, such as `sum_k`, `sum_density`, `psd_sum` or `ring`.

`load_checkpoint` validates version, grid, component and batch shapes, timestep, reservoir names, relevant noise and stop settings, adaptive mode and accumulator configuration. A precision change casts arrays and warns. See [Checkpoint and restart](../user-guide/checkpoint-and-restart.md) for continuation.

## Reader API

`ps.load_results(path)` accepts a run directory, `results.h5` or a sweep directory. It returns `Results` for one run or `SweepResults` for a sweep. A run directory prefers HDF5 when present, then NPY, then JSON. `Results.time`, `fields`, `scalars`, `scalar_groups`, `static`, `accumulated`, `accumulated_meta`, `config`, `x`, `y` and `attrs` expose the saved data. HDF5 fields and accumulated values are lazy datasets; use slicing, a context manager or `close()`. `kx` and `ky` provide FFT-shifted angular wave numbers in rad/µm. `SweepResults.member(i)` gives a member view, and `SweepResults.mean(name)` combines linear accumulated values.
