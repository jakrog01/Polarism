# Spatial profiles and temporal envelopes

A spatial profile is a real array built once on the simulation grid. Use it as a static potential in meV or as the spatial part of an incoherent pump. A temporal envelope is a scalar function evaluated at each pump time. With `laser_type: profile`, the pump is

\[
P(\mathbf r,t)=P_0\,S(\mathbf r)\,E(t-\texttt{delay}),
\]

and is zero for `t < delay`. `S` includes `profile.amplitude`. Set `power_definition: peak_amplitude`; pulse energy normalization is unavailable for this laser type. `x0` and `y0` belong inside the profile mapping and shift its centre or source extent. Spatial distances are in µm, times in ps, and potential amplitudes in meV.

## Shared profile keys

Every profile mapping requires `type` and accepts `amplitude` (default `1.0`), `x0` and `y0` (both default `0.0`). The latter shift the profile relative to the simulation box. Unknown keys raise an error. The `amplitude` is dimensionless for pumps and in meV for potentials.

## Analytic profiles

`uniform` gives one everywhere and works in 1D or 2D:

```yaml
potential:
  potential_type: profile
  profile: {type: uniform, amplitude: -0.001}
```

`gaussian` gives `exp(-r²/(2 sigma²))` with required positive `sigma`:

```yaml
laser:
  laser_type: profile
  P0: 0.4
  profile: {type: gaussian, sigma: 4.0, x0: 2.0}
```

`supergaussian` gives `exp[-(r²/sigma²)^order/2]` with positive `sigma` and `order`:

```yaml
potential:
  potential_type: profile
  profile: {type: supergaussian, sigma: 5.0, order: 3.0, amplitude: -0.002}
```

`ring` is 2D and gives `exp[-(r-radius)²/(2 width²)]`:

```yaml
laser:
  laser_type: profile
  profile: {type: ring, radius: 6.0, width: 0.8}
```

## Profile from a file

`file` reads a real 1D or 2D `.npy` array, an `.h5` or `.hdf5` dataset, or a `.png` image. PNG RGB channels are averaged; alpha is ignored and pixel values are scaled to `[0, 1]`. Supply `dataset` for HDF5. `extent` lists source bounds `[xmin, xmax]` in 1D or `[xmin, xmax, ymin, ymax]` in 2D. It defaults to the simulation box. The source array covers both endpoints of its extent; interpolation is linear in 1D and bilinear in 2D. Values outside the extent are zero. An `x0` or `y0` shift moves the source extent with the image.

```yaml
potential:
  potential_type: profile
  profile:
    type: file
    path: patterns/trap.npy
    extent: [-20.0, 20.0, -15.0, 15.0]
    amplitude: -0.003
```

```yaml
laser:
  laser_type: profile
  profile: {type: file, path: patterns/pump.h5, dataset: pump, amplitude: 0.8}
```

```yaml
laser:
  laser_type: profile
  profile: {type: file, path: patterns/pump.png}
```

Each source axis needs at least two samples. The array must have the same number of dimensions as the grid.

## Lattices

`lattice` is 2D. It places unit-depth wells, then multiplies their sum by `amplitude`. `constant` is the lattice constant `a` in µm; `width` is the positive well width. `well` is `gaussian` by default or `supergaussian`, which also requires positive `order`. The well formula is `exp[-(r²/width²)^order/2]` for supergaussian and `exp[-r²/(2 width²)]` for gaussian. Neighbouring wells add and may make the total higher than one. `rotation` rotates the lattice counterclockwise in degrees about `(x0, y0)`.

| Lattice | Basis vectors | Sublattice offsets | Nearest neighbours |
| --- | --- | --- | --- |
| `square` | `(a,0)`, `(0,a)` | `(0,0)` | `a` |
| `triangular` | `(a,0)`, `(a/2,√3 a/2)` | `(0,0)` | `a` |
| `honeycomb` | `(a,0)`, `(a/2,√3 a/2)` | `(0,0)`, `(a/2,a/(2√3))` | `a/√3` |
| `kagome` | `(2a,0)`, `(a,√3 a)` | `(0,0)`, `(a,0)`, `(a/2,√3 a/2)` | `a` |
| `lieb` | `(2a,0)`, `(0,2a)` | `(0,0)`, `(a,0)`, `(0,a)` | `a` |

For example, a square array of potential wells:

```yaml
potential:
  potential_type: profile
  profile: {type: lattice, lattice: square, constant: 3.0, width: 0.5, amplitude: -0.002}
```

A triangular pump array:

```yaml
laser:
  laser_type: profile
  profile: {type: lattice, lattice: triangular, constant: 3.0, width: 0.5}
```

A honeycomb array with one removed site:

```yaml
potential:
  potential_type: profile
  profile:
    type: lattice
    lattice: honeycomb
    constant: 3.0
    width: 0.4
    defects: [[0, 0, 1]]
```

A rotated kagome array with deterministic depth variation:

```yaml
laser:
  laser_type: profile
  profile: {type: lattice, lattice: kagome, constant: 3.0, width: 0.4, rotation: 15.0, disorder: 0.1, seed: 7}
```

A Lieb array with supergaussian wells:

```yaml
potential:
  potential_type: profile
  profile: {type: lattice, lattice: lieb, constant: 3.0, width: 0.5, well: supergaussian, order: 3.0}
```

`disorder` is the standard deviation of a Gaussian relative depth change (default `0`); `seed` defaults to `0` and makes the realization reproducible. `defects` is a list of `[i, j, s]` unit-cell and sublattice indices. Sublattice indices follow the order shown in the table, starting at zero. Sites cover the box plus one cell margin, so wells near the edge contribute smoothly.

## Temporal envelopes

`constant` equals one; it is the default and is time independent:

```yaml
laser:
  laser_type: profile
  profile: {type: gaussian, sigma: 4.0}
  envelope: {type: constant}
```

`step` is zero before `t0` and one from `t0` onward:

```yaml
laser:
  laser_type: profile
  profile: {type: uniform}
  envelope: {type: step, t0: 2.0}
```

`gaussian-pulses` sums pulses with required `center` and positive `sigma`; `count` defaults to `1`, `separation` sets centre spacing when `count > 1`, and `cutoff_sigma` defaults to `3`. Each pulse is exactly zero outside its cutoff:

```yaml
laser:
  laser_type: profile
  profile: {type: ring, radius: 6.0, width: 0.8}
  envelope: {type: gaussian-pulses, center: 1.0, sigma: 0.1, count: 3, separation: 0.5, cutoff_sigma: 3.0}
```

`data` linearly interpolates `times` and `values`, requiring at least two strictly increasing times. It returns zero outside the table unless `hold: true`, which returns the edge values. A `.npy` file of shape `(N, 2)` or a two-column `.csv` is an alternative to inline values:

```yaml
laser:
  laser_type: profile
  profile: {type: uniform}
  envelope: {type: data, times: [0.0, 1.0, 2.0], values: [0.0, 1.0, 0.0]}
```

```yaml
laser:
  laser_type: profile
  profile: {type: uniform}
  envelope: {type: data, path: schedules/pump.csv, hold: true}
```

The envelope sees time after subtracting `laser.delay`. Multiple-laser YAML entries may each carry their own `profile` and `envelope` mappings. Sweeps can vary existing keys with paths such as `laser.profile.sigma`, `laser.envelope.t0`, `potential.profile.amplitude`, or `lasers[0].profile.width`. Changing a mapping's `type` creates a separate sweep group.
