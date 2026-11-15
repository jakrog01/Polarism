# Coherent drive

Use `drive` to inject condensate amplitude directly with a resonant source. It has a spatial profile, a temporal envelope and a rotating-frame energy.

```yaml
reservoir:
  reservoir_type: none
laser:
  P0: 0.0
noise:
  enabled: false
drive:
  enabled: true
  amplitude: 0.01
  energy: 0.4
  profile: {type: gaussian, sigma: 4.0}
```

This setup has no incoherent reservoir or initial noise. The source has a 4 µm Gaussian width and rotates at energy 0.4 meV.

## Set the source

In the rotating frame, `drive.energy` subtracts an energy `E_p` from the condensate Hamiltonian while the drive is enabled. The source adds `F₀ p(r) f(t) exp[i(kx x + ky y)]`, where `F₀` is `drive.amplitude`, `p` is `drive.profile` and `f` is `drive.envelope`. The [physics page](../explanation/physics.md) gives the full equation.

`drive.amplitude` has units meV µm⁻¹ in 2D or meV µm⁻¹ᐟ² in 1D. `drive.energy` is in meV and `kx` and `ky` are in rad µm⁻¹. On a 1D grid, `ky` must be zero. A profile's `amplitude` multiplies `drive.amplitude`; the envelope is dimensionless. Use any registered [profile or envelope](profiles-and-envelopes.md).

In a spinor run, `drive.polarization` gives the two circular Jones amplitudes in `[ψ+, ψ−]` order. It must contain two finite complex values with a nonzero norm; the implementation normalizes it. YAML can represent complex values as strings, for example `["1", "1j"]`. See [Spin and polarization](spin-and-polarization.md).

The source is available only in solvers that declare coherent-drive support. Check [solver compatibility](../reference/solvers.md) before changing methods. Saved results can include the spatial drive field; see [Output and results](output-and-results.md).
