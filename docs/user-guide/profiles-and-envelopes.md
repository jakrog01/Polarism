# Spatial profiles and temporal envelopes

Use a spatial profile for a static potential or for the shape of a `profile` laser. Add a temporal envelope when the pump or potential should change with time.

```yaml
laser:
  laser_type: profile
  P0: 0.02
  profile: {type: gaussian, sigma: 4.0, x0: 2.0}
  envelope: {type: step, t0: 50.0}
```

This pump turns on at 50 ps, centred at x = 2 µm with width 4 µm. Its peak strength is 0.02 µm⁻² ps⁻¹ in 2D. A profile laser uses `power_definition: peak_amplitude`.

## Spatial profiles

A profile mapping has a `type` and optional `amplitude`, `x0` and `y0`. The coordinates shift it in µm. The default amplitude is 1; it is dimensionless for a pump or coherent drive and in meV for a potential. Unknown profile keys raise an error.

| `type` | Shape and required parameters |
| --- | --- |
| `uniform` | Constant across 1D or 2D |
| `gaussian` | `exp[-r²/(2 sigma²)]`; `sigma` in µm |
| `supergaussian` | `exp[-(r²/sigma²)^order/2]`; `sigma` in µm, positive `order` |
| `ring` | Gaussian radial band; `radius` and `width` in µm, 2D only |
| `file` | Interpolated real data; `path`, and `dataset` for HDF5 |
| `lattice` | Sum of Gaussian or super-Gaussian wells; `lattice`, `constant` and `width`, 2D only |

`file` accepts `.npy`, `.h5`, `.hdf5` and `.png`. The array must match the grid dimension and contain at least two samples per axis. HDF5 needs a dataset name. PNG color channels are averaged and scaled to [0, 1]; alpha is ignored. `extent` sets the source bounds in µm as `[xmin, xmax]` in 1D or `[xmin, xmax, ymin, ymax]` in 2D. Outside the extent the profile is zero. Interpolation is linear in 1D and bilinear in 2D.

For a lattice, `constant` is the lattice constant `a` in µm and `width` is the well width in µm. Choose `square`, `triangular`, `honeycomb`, `kagome` or `lieb`. `well: supergaussian` also needs a positive `order`. Neighbouring wells add, so the result can exceed one. `rotation` is in degrees; `disorder` sets the relative Gaussian depth variation and `seed` fixes its realization. `defects` lists omitted sites as `[i, j, s]`, where `s` is the sublattice index.

| Lattice | Basis vectors | Sublattice offsets | Nearest-neighbour distance |
| --- | --- | --- | --- |
| `square` | `(a,0), (0,a)` | `(0,0)` | `a` |
| `triangular` | `(a,0), (a/2,√3a/2)` | `(0,0)` | `a` |
| `honeycomb` | `(a,0), (a/2,√3a/2)` | `(0,0), (a/2,a/(2√3))` | `a/√3` |
| `kagome` | `(2a,0), (a,√3a)` | `(0,0), (a,0), (a/2,√3a/2)` | `a` |
| `lieb` | `(2a,0), (0,2a)` | `(0,0), (a,0), (0,a)` | `a` |

For example, set `potential.potential_type: profile` and `potential.profile: {type: lattice, lattice: square, constant: 3.0, width: 0.5, amplitude: -0.002}` for square wells 3 µm apart with width 0.5 µm and amplitude −0.002 meV.

## Temporal envelopes

An envelope returns a scalar multiplier. `constant` is the default. For a profile laser it sees time after `laser.delay`; the pump is zero before the delay. For a potential it sees simulation time directly.

| `type` | Behavior and parameters |
| --- | --- |
| `constant` | One at all times |
| `step` | Zero before `t0`, one from `t0` onward; `t0` in ps |
| `gaussian-pulses` | Sum of cut-off Gaussian pulses; `center`, `sigma` and `separation` in ps, plus `count` and `cutoff_sigma` |
| `data` | Linear interpolation of `times` in ps and dimensionless `values`, or a two-column `.npy` or `.csv` file |

For `gaussian-pulses`, `count` defaults to one. A positive `separation` is required for multiple pulses. Each pulse is zero beyond `cutoff_sigma × sigma` from its centre. For `data`, times must increase strictly; values outside the table are zero unless `hold: true`.

Profiles and envelopes can also shape a [coherent drive](coherent-drive.md). See [Configuration reference](../reference/configuration.md) for the fields and [profile plugins](../plugins/profiles.md) for extensions.
