# Term plugins

A term adds one extra contribution to the condensate equation without
changing the built-in physics. Use it for a loss, a nonlinearity, or an
energy shift that the core model does not provide, for example two-photon loss
or a field-dependent detuning.

## Contract

A term has a category and one method. The category says how the returned real
coefficient enters `d psi / dt`:

| `category` | Contribution | Coefficient |
| --- | --- | --- |
| `"energy"` | `(-i / hbar) * E * psi` | `E` in meV |
| `"rate"` | `G * psi` | `G` in 1/ps |

```python
from polarism.model_terms import Term, register_term

POLARISM_PLUGIN_API = 2


@register_term("nonlinear-loss")
class NonlinearLoss(Term):
    category = "rate"

    def coefficient(self, field, reservoir, reservoir_fields, t):
        return -0.5 * self.params["alpha"] * abs(field) ** 2
```

This term adds `-(alpha / 2) |psi|^2 psi`, so the density of a homogeneous
field without other losses decays as `n0 / (1 + alpha * n0 * t)`.

`Term.__init__(params, grid, physics)` stores `self.params` (the dictionary
from the configuration), `self.grid`, and `self.physics` (`BatchPhysics`).
Override it only to precompute arrays. `coefficient(field, reservoir,
reservoir_fields, t)` is called at every solver stage with that stage's values:

- `field` is the canonical `(B, 1, *S)` condensate array.
- `reservoir` is the reservoir object and `reservoir_fields` the stage's
  reservoir tuple, so `reservoir.get_active_density(reservoir_fields)` gives
  the active density at that stage.
- `t` is the stage time.

Return a real array broadcastable to `field`, or a scalar. Keep the result in
the field's precision (for example multiply by Python floats or by
`self.physics.get(name)`), so single-precision runs stay single precision.
Several terms are summed in configuration order.

## Configure

Terms live in the top-level `terms` section, a mapping from registered name to
its parameters:

```yaml
terms:
  nonlinear-loss:
    alpha: 0.01
```

```python
cfg.terms = {"nonlinear-loss": {"alpha": 0.01}}
```

Unknown names are rejected at validation with the list of registered terms.
Sweep a parameter with `terms.<name>.<parameter>`, for example
`terms.nonlinear-loss.alpha`. The parameter must already be present in the
configuration. Term settings are group settings: members with different values run in
separate groups.

## Solver support

Terms are evaluated as part of the time integration, so they follow each
solver's own accuracy.

| Solver | Terms | How |
| --- | --- | --- |
| `rk4-fdm` | yes | added to `model.rhs` at every RK4 stage |
| `ip-rk4` | yes | added to the nonlinear right-hand side at every stage |
| `etd-rk2` | yes | added to the nonlinear right-hand side of predictor and corrector |
| `split-step-fft` | yes | local factor `exp((G - iE/hbar) dt)` at `t + dt/2`, first order in `dt` for field-dependent terms |
| `rk4-fdm-fused`, `rk4-cuda`, `split-step-fft-cuda`, `ifrk-fft-cuda`, `ifrk4-fft-cuda` | no | rejected at validation |

A plugin solver that integrates with `model.rhs` supports terms automatically.
A solver with its own kernels reads `model.terms` and declares
`SolverCapabilities(terms=False)` until it implements them. `model.rhs` and
`model.extra_rhs(field, reservoir_fields, t)` return the contribution of the
extra terms; `model.extra_coefficients(...)` returns the summed `(E, G)`.
