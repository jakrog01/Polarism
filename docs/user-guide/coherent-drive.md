# Coherent drive

A coherent drive injects condensate amplitude directly. In the frame rotating
at the drive energy, the field obeys

$$
i\hbar\frac{\partial\psi}{\partial t}
= [H-E_p]\psi + F_0\,p(\mathbf r)\,f(t)\,
  e^{i(k_xx+k_yy)}.
$$

Here `drive.amplitude` is $F_0$, `drive.energy` is $E_p$ in meV,
`drive.profile` supplies the dimensionless spatial profile $p$, and
`drive.envelope` supplies the scalar time function $f$. The wave vectors
`kx` and `ky` are in rad/µm. In 1D, `ky` must be zero. The source amplitude
has units meV·µm⁻¹ in 2D and meV·µm⁻¹ᐟ² in 1D. Profile `amplitude`, if
specified, multiplies `drive.amplitude`.

The energy shift $-E_p$ is applied only while the drive is enabled. For a
spatially uniform source with no kinetic energy, potential, interactions, or
reservoir, the equation becomes

$$
\frac{d\psi}{dt}
=\left(\frac{iE_p}{\hbar}-\frac{\gamma_C}{2}\right)\psi
-\frac{iF}{\hbar},\qquad
\psi_{\mathrm{ss}}=
\frac{-iF/\hbar}{\gamma_C/2-iE_p/\hbar}.
$$

For a resonantly driven run without an incoherent reservoir:

```yaml
grid:
  nx: 256
  ny: 256
  lx: 100.0
  ly: 100.0
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
  kx: 0.0
  ky: 0.0
  profile: {type: gaussian, sigma: 10.0}
  envelope: {type: constant}
solver:
  method: ip-rk4
  dt: 0.001
  total_time: 10.0
```

The profile and envelope use the same [registries](profiles-and-envelopes.md)
as spatial potentials and profile lasers. `drive` is a static result field
containing $|F_0p(\mathbf r)e^{i\mathbf k\cdot\mathbf r}|$ when enabled.
`rk4-fdm`, `ip-rk4`, and `etd-rk2` support coherent drive. Other built-in
solvers reject it during compatibility checking.
