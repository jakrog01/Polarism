# Storage format plugins

Register a `BaseStorage` subclass with `polarism.results.storage.register_storage`.
Select it with `cfg.result.formats = ("example-scalar-csv",)`. The tuple is
combined with the built-in `save_hdf5`, `save_npy`, and `save_json` flags;
each selected name is constructed once. Unknown names fail configuration
validation and list the registered formats. The built-in names are `hdf5`,
`npy`, and `json`.

The constructor contract is `StorageClass(output_dir, batch_size, config, x, y)`.
`output_dir` is the run's `Path`; `config` is the full `Config`; `x` and `y`
are host coordinate arrays. Subclass `BaseStorage` and implement `dump_batch`,
`write_static`, and `finalize`. Its `add_to_batch(t, nodes, **context)` buffers
saved fields and scalars and calls `dump_batch` at the configured batch size.
`node_metadata` holds each saved node's label and optional units. Override
`add_to_batch` if a format supports only some node types.

Per-frame storage methods run on the background writer thread; construction
and `finalize` run on the simulation thread. `cached` values and
`scalar_groups` passed to `add_to_batch` are host NumPy arrays, including
0-D arrays for scalars. Flush the final batch in `finalize`. Storage errors
propagate to the simulation thread.

```python
--8<-- "examples/plugins/polarism_plugins/storage/scalar_csv.py"
```

This small example writes only saved scalar nodes, including built-in `norm`,
to `scalars.csv`. The [integration test](../plugins/index.md#try-and-test-a-plugin)
loads it through plugin discovery and checks the output.
