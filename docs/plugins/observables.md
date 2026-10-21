# Observable and visitor plugins

Register a factory with `polarism.results.register_observable(name)` and select
it with `cfg.result.observables = (name,)`. Factories receive the initialized
`SimulationController` and return one `ResultNode` or a list of nodes. Unknown
names fail configuration validation and list registered observables. There
are no built-in names in this registry.

`ResultNode.compute_fn(**context)` receives `state`, `grid`, `P_total`, and
`scalar_groups`, the context keys passed by
`SimulationController._record_results`. Use `compute_engine.xp` for NumPy or
CuPy operations. For `is_field=False`, storage records the value returned by
`compute_fn`; `reduce_dim_fn` is applied only for CPU visitors such as live
view. Return a 0-D backend array for a scalar to avoid a GPU sync during
result computation. Set `save` and `expose` to control storage and live view.

```python
--8<-- "examples/plugins/polarism_plugins/observable/center_of_mass.py"
```

The example computes the condensate x centre of mass in µm. It reads the
current wavefunction and the grid's x coordinates without transferring
arrays to the host.

For a custom consumer, subclass `ResultVisitor` and call
`controller.add_visitor(visitor)` before `controller.run()`. Nodes are built
when the first visitor is added, even when no configured output is enabled.
`needs(nodes)` returns the set of node names to compute; `visit_all(nodes,
t=..., cached=..., **context)` receives each sampled frame. `cached` maps
node names to `(field, reduced)` pairs. Set `data_location` to `"cpu"` or
`"device"` to choose host or backend arrays. `fatal_on_error` controls whether
a visit error stops the run. An optional `finalize()` is called at the end of
the run, including after an exception. Sampling uses `result.save_interval`
and always includes the initial state.
