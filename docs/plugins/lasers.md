# Laser plugins

## When to write one

Write a laser when the built-in uniform, Gaussian, exponential, and pulsed
profiles do not describe your pump. A laser produces a real pump field; the
controller combines all selected lasers through `PumpField`.

## Contract

`LaserFactory` calls `LaserClass(laser_config, X, Y, precision)` and expects an
`AbstractLaser` subclass. `X` and `Y` are backend arrays of shape `grid.shape`.
In 1D, `Y` is a zero array, so a 2D spatial profile is sampled at y = 0.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `__init__(laser_config, X, Y, precision="double")` | Factory, once per laser | Call base init; cache a real `grid.shape` envelope | Required |
| `_P_space(X, Y)` | `_finalize_spatial_envelope`, during init | Real spatial array of shape `grid.shape` | Abstract |
| `_amplitude(t)` | `power_coefficient`, on each pump sample | Scalar amplitude | Abstract |
| `_P_time(t)` | `power_coefficient`, on each pump sample | Scalar time factor | Abstract |
| `_finalize_spatial_envelope(X, Y)` | Subclass init, after parameters | Sets dtype-aware `_spatial_envelope` | Required call |
| `power_coefficient(t)` | Base `get_power`, and CUDA separable path | Scalar; applies `delay` | Inherited; override only if needed |
| `get_power(X, Y, t)` | `PumpField` at solver stage and output times | Read-only pump array; updates `self.P` | Inherited; override only if needed |

!!! warning "Keep coefficients scalar"
    The inherited `get_power()` checks `if coeff == 0`. Returning an array
    from `_amplitude` or `_P_time` makes that condition ambiguous.

## Step-by-step

1. Create `polarism_plugins/laser/ring_laser.py` below your run directory.
2. Import `register_laser` and `AbstractLaser`; decorate a new class with a
   unique name.
3. Call `super().__init__(laser_config, X, Y, precision)`, read custom values
   from `laser_config.extra`, then call `_finalize_spatial_envelope(X, Y)`.
4. Implement the three abstract methods with `self.xp` array operations.
5. Set `cfg.laser.laser_type` and `cfg.laser.extra`, then run a small simulation.
   The [first-plugin tutorial](first-plugin.md) includes Python, YAML, CLI,
   output inspection, and GPU steps.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/laser/ring_laser.py"
```

The decorator registers `example-ring`. The `radius` comes from `extra`,
while `sigma_space` supplies ring width. `_P_space` computes a Gaussian in
radial distance from the ring; `_amplitude` and `_P_time` make it continuous.
The base class handles delayed start, dtype conversion, and the latest `P`.

## GPU and performance notes

`self.xp` is NumPy or CuPy according to the configured backend. The spatial
envelope is computed once, then reused. `PumpField.separable_terms()` returns
one envelope and coefficient pair per laser when every laser inherits
`AbstractLaser.get_power` unchanged and has a cached envelope. It returns `[]`
for no lasers and `None` if any laser does not qualify. This includes plugin
lasers that keep the inherited `get_power`, and permits CUDA graph replay in
eligible `rk4-cuda`, `split-step-fft-cuda`, or `ifrk-fft-cuda` chunks with
multiple lasers. `separable_term()` remains available for one laser. Avoid
converting the field to NumPy inside `_P_space` or a per-step hook.

## Supported by solvers

| Solver family | Plugin laser behavior |
| --- | --- |
| `rk4-fdm`, `rk4-fdm-fused`, `split-step-fft`, `etd-rk2`, `ip-rk4` | Use the callable total pump field |
| `rk4-cuda`, `split-step-fft-cuda`, `ifrk-fft-cuda`, `ifrk4-fft-cuda` | Use the pump field on an active GPU; eligible inherited laser envelopes allow CUDA graph chunks |

## Pitfalls

- Call `_finalize_spatial_envelope` only after setting radius, width, or other
  fields used by `_P_space`.
- Return `(nx,)` in 1D and `(ny, nx)` in 2D.
- A custom `get_power` must update `self.P` if you want per-laser scalar output.
- `extra` keys are Python fields or YAML entry keys, not individual CLI flags.

## Testing

```python
cfg.laser.laser_type = "example-ring"
cfg.laser.extra = {"ring_radius": 4.0}
run = SimulationController(cfg)
assert run.pump_field(0).shape == run.grid.shape
assert np.isfinite(run.pump_field(0)).all()
```

Run the complete CPU example test with
`.venv/bin/pytest -q tests/integration/test_example_plugins.py`.
