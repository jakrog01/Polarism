# Migrating to 0.4

Polarism 0.4 reworks result output. Simulation numerics are unchanged; a run
with the same configuration produces the same fields as in 0.3.

## Renamed result datasets

Saved result names are now ASCII. Plots and movie titles keep the old text as
display labels.

| 0.3 name | 0.4 name |
| --- | --- |
| `|ψ|²` | `density` |
| `N(t)` | `norm` |
| `Pump` | `pump` |

Code that opens HDF5 files directly must use the new paths, for example
`fields/density` instead of `fields/|ψ|²`, and NPY keys such as
`field_density`. The simplest migration is to read through
`polarism.load_results`, which also opens 0.3 files and maps their names:

```python
import polarism as ps

with ps.load_results(run_dir) as result:
    density = result.fields["density"][:]
```

`cfg.result.animation_fields` still accepts the 0.3 names with a
`DeprecationWarning`.

## Static pump

A pump that does not change in time is stored once as `static/pump` (HDF5) or
`static_pump.npy` (NPY) instead of one copy per frame. Pulsed pumps and pumps
with a positive `delay` are still stored per frame under `fields/pump`.
`result.static["pump"]` and `result.fields["pump"]` distinguish the two cases.

## Compression

HDF5 output is uncompressed by default (0.3 used gzip level 4), and NPY batches
use `numpy.savez` instead of `numpy.savez_compressed`. Files are larger, but
writing is several times faster. Set `cfg.result.compression = "lzf"` or
`"gzip"` to compress.

## File metadata

HDF5 files carry `format_version` and `polarism_version` root attributes,
`grid/x` and `grid/y` coordinates, and `label`/`units` dataset attributes. NPY
runs write `metadata.json`, `grid_x.npy` and `grid_y.npy`.

## Scalar result nodes

For scalar nodes (`is_field=False`), storage now records the value returned by
`compute_fn`; `reduce_dim_fn` is applied only for CPU visitors such as the live
view. Built-in nodes are unaffected. A custom node whose `reduce_dim_fn` does
more than convert the value should move that computation into `compute_fn`.

## Storage backends

Built-in backends are created through the storage registry with the constructor
`StorageClass(output_dir, batch_size, config, x, y)` and receive host arrays on
a background thread. Subclasses of `BaseStorage` must implement
`write_static(name, field)` in addition to `dump_batch` and `finalize`.

## New options

| Option | Purpose |
| --- | --- |
| `result.formats` | output formats, including plugin formats |
| `result.observables` | extra result nodes from observable plugins |
| `result.compression`, `result.compression_level` | HDF5 compression |
| `result.save_psi` | store the complex condensate field |
| `checkpoint.every`, `checkpoint.path`, `checkpoint.resume_from`, `checkpoint.on_signal` | [checkpoint and restart](../user-guide/checkpoint-and-restart.md) |
