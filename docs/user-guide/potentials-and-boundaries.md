# Potentials and boundaries

Use `potential` for an external energy landscape `V` in meV. Boundary absorption can damp waves near the grid edge.

```yaml
potential:
  potential_type: profile
  profile: {type: gaussian, sigma: 4.0, amplitude: -0.002}
  envelope: {type: step, t0: 50.0}
```

This applies a Gaussian well of depth 0.002 meV from 50 ps onward. The potential is zero before that time.

## Set a potential

`zero` is the default. `double-well-supergaussian` adds two wells defined by `x1`, `y1`, `V1`, `w1` and the matching fields for well 2, plus `order`. Positions and widths are in µm; `V1` and `V2` are in meV. `profile` uses any registered [spatial profile](profiles-and-envelopes.md), including a file map or lattice; its `amplitude` is in meV.

`potential.envelope` multiplies only the configured external potential: `V(r,t) = V(r) E(t)`. It defaults to a constant envelope. A complex absorbing potential at the edge remains active when the external potential is off. Solver support for time-varying potentials is listed in the [solver reference](../reference/solvers.md).

## Absorb near an edge

```yaml
boundary_condition:
  absorption: cap
  profile_type: sin2
  mask_width_percent: 0.2
  strength: 1.0
```

Here the absorbing region covers 20% of each axis at each edge. `strength` is a density-loss rate in ps⁻¹. `profile_type` is `sin2` or `parabolic`.

| `absorption` | Effect |
| --- | --- |
| `no-absorption` | No edge damping |
| `cap` | Adds a negative imaginary potential to the evolution |
| `mask` | Multiplies ψ after each step by a decay mask |

`cap` preserves the time order of a compatible solver. `mask` is an after-step split and makes the edge treatment first order in time. A periodic grid still wraps opposite edges; damping reduces outgoing amplitude before it wraps. A closed-interval grid includes both endpoints and mirrors the adjacent interior sample for finite differences. See [Grid and physics](grid-and-physics.md) for spacing and [Configuration reference](../reference/configuration.md) for all boundary fields.
