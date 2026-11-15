# Dispersion plugins

A dispersion plugin replaces the parabolic kinetic energy
`ħ² k² / (2 m_eff)` of the spectral solvers with another relation E(k),
for example a quartic correction or a measured polariton branch.

## Contract

```python
from polarism.dispersion import Dispersion, register_dispersion

POLARISM_PLUGIN_API = 2


@register_dispersion("quartic")
class Quartic(Dispersion):
    def energy(self, grid, physics):
        hbar, m_eff = physics.get("hbar"), physics.get("m_eff")
        return hbar**2 * grid.k_squared / (2 * m_eff) + 1e-4 * grid.k_squared**2
```

The quartic coefficient in this example is `1e-4` meV·µm⁴. `energy(grid, physics)` returns the real energy E(k) in meV on the FFT grid:
an array with the shape of `grid.k_squared` (`(nx,)` or `(ny, nx)`), or one
that broadcasts against `(B, 1, *S)` when a parameter varies between sweep
members. `physics` is `BatchPhysics`; use `physics.get(name)`. The registered
name is available as `kind`. The class is created without arguments and held
by `model.dispersion`.

Select it in the configuration:

```yaml
physics:
  dispersion: quartic
```

`physics.dispersion` defaults to `parabolic`, the built-in
`ParabolicDispersion`. It is a group setting in sweeps, so members with
different dispersions run in separate groups. The default `matrix()` method
builds the spinor Hamiltonian from `energy()` with Zeeman and TE–TM terms;
plugins can override it for a different matrix dispersion.

## Solver support

| Solver | Dispersion |
| --- | --- |
| `split-step-fft`, `ip-rk4`, `etd-rk2` | any registered; propagators use `exp(-i E(k) dt / hbar)` |
| `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda`, `split-step-fft-cuda`, `ifrk-fft-cuda`, `ifrk4-fft-cuda` | `parabolic` only; others rejected at validation |

For `parabolic` the spectral solvers keep their exact built-in expressions.
`ip-rk4` still evaluates `kinetic_relaxation_eta` with the plain Laplacian. A
solver declares what it accepts in `SolverCapabilities.dispersion`: a set of
names, or `{"*"}` when it uses `model.dispersion` for every registered kind.
