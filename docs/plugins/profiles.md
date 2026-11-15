# Profile and envelope plugins

Spatial profiles and temporal envelopes are separate registries. A profile builds one array during setup; an envelope supplies a cheap host scalar during time stepping. Register both from a module loaded by your [plugin setup](index.md).

## Spatial profile contract

Decorate a function with `register_profile(name, params=(...))`. It receives `(grid, params, xp)` and returns a real array with shape `grid.shape`. `params` is the profile mapping with `x0` and `y0` defaults filled in. The framework applies `amplitude` after the function returns, checks the shape and finite real values, and rejects spec keys absent from `params` and the shared keys. Use `grid.X` and, in 2D, `grid.Y`; `xp` is NumPy or CuPy.

```python
from polarism import register_profile


@register_profile("parabolic-trap", params=("curvature",))
def parabolic_trap(grid, params, xp):
    result = (grid.X - params["x0"]) ** 2
    if grid.ndim == 2:
        result = result + (grid.Y - params["y0"]) ** 2
    return params["curvature"] * result
```

```yaml
potential:
  potential_type: profile
  profile: {type: parabolic-trap, curvature: 0.01, amplitude: 0.002}
```

In this example, `curvature` is 0.01 µm⁻² and `amplitude` is 0.002 meV. The final potential has units meV.

A plugin can attach a `validate(spec, ndim)` callable to its function for parameter and dimensionality checks before array creation. Raise `ValueError` with a useful message for invalid values.

## Temporal envelope contract

Decorate a factory taking the envelope spec and returning a callable `t -> float`. Declare accepted keys with `params`. The factory is called once during setup. The returned callable runs on the host and should do no device work. A returned callable may set `time_independent = True` when its value is constant; otherwise the flag defaults to false.

```python
from polarism import register_envelope


@register_envelope("linear-ramp", params=("duration",))
def linear_ramp(spec):
    duration = float(spec["duration"])
    if duration <= 0:
        raise ValueError("duration must be positive")
    return lambda t: min(1.0, max(0.0, t / duration))
```

```yaml
laser:
  laser_type: profile
  P0: 0.4
  profile: {type: gaussian, sigma: 4.0}
  envelope: {type: linear-ramp, duration: 2.0}
```

The ramp duration in this example is 2 ps. `create_profile`, `available_profiles`, `create_envelope`, and `available_envelopes` are exported from `polarism`. Registry names must be unique for predictable configuration loading. Built-in profile and envelope names and their parameters are listed in the [user guide](../user-guide/profiles-and-envelopes.md).
