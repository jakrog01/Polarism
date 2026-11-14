# Spin and polarization

Set `physics.spin: spinor` to evolve two circular components, ordered
`ψ = (ψ+, ψ−)`. Step 8a supports this model on CPU with `rk4-fdm` on 1D or 2D
grids. Scalar runs remain the default.

## Equation and conventions

For σ = +1 or −1, the model evolves

\[
i\hbar\partial_t\psi_\sigma = \left[-\frac{\hbar^2\nabla^2}{2m}
+V+\sigma\frac{\Delta_Z}{2}+g_C|\psi_\sigma|^2
+\alpha_2|\psi_{-\sigma}|^2+g_Rn_\sigma+g_In_{I,\sigma}
+\frac{i\hbar}{2}(Rn_\sigma-\gamma_C)\right]\psi_\sigma
-\beta(\partial_x-i\sigma\partial_y)^2\psi_{-\sigma}.
\]

The **TE–TM sign convention** is `−β(∂x − iσ∂y)²ψ−σ` in position space,
so its Fourier-space off-diagonal element is `+β(kx − iσ ky)²`. The kinetic
energy is `ε(k) = ħ²k²/(2m)`. For a plane wave without interactions, the two
energies are `ε(k) ± sqrt((Δ_Z/2)² + β²k⁴)`.

The finite-difference reference uses central `∂xx` and `∂yy`, plus the
nine-point cross derivative
`∂xy f[j,i] = (f[j+1,i+1] − f[j+1,i−1] − f[j−1,i+1] + f[j−1,i−1])/(4 dx dy)`.
Periodic edges wrap; closed-interval edges mirror the adjacent interior
value. In 1D, the TE–TM operator reduces to `−β∂xx`.

## Pump and reservoirs

With two reservoirs, each component has its own active and inactive density
and is depleted by its corresponding `|ψσ|²`. The incoherent pump is split as
`P± = P(1 ± ρ_c)/2`, where `ρ_c = laser.circular_polarization`. `ρ_c = +1`
pumps only σ+; `ρ_c = −1` pumps only σ−. The default `ρ_c = 0` splits the
pump equally. Set `reservoir.components: 1` for one shared reservoir, fed by
the full pump and depleted by `|ψ+|² + |ψ−|²`; its density feeds both fields.

The coherent drive uses `drive.polarization`, a normalized Jones vector in
the same circular order. A scalar deterministic initial condition is added
to σ+; a two-component field loaded through `initial_condition.type: file`
sets both components. Initial random fields are independent. Truncated-Wigner
increments use independent Philox streams 1 and 2 for σ+ and σ−.

## Example

```yaml
grid:
  ndim: 2
  nx: 64
  ny: 64
  lx: 40.0
  ly: 40.0
physics:
  spin: spinor
  alpha_2: -0.0001
  zeeman: 0.1
  te_tm: 0.02
reservoir:
  reservoir_type: single
  components: 2
laser:
  laser_type: uniform
  P0: 0.4
  circular_polarization: 0.5
drive:
  enabled: true
  amplitude: 0.01
  polarization: ["1", "1j"]
solver:
  method: rk4-fdm
  dt: 0.001
  total_time: 1.0
result:
  save_results: true
  save_hdf5: true
  save_psi: true
```

`state.psi` has shape `(2, *space)` or `(batch, 2, *space)`; `psi_plus` and
`psi_minus` select the corresponding component. Saved `psi` and `density`
retain that axis. `density_plus` and `density_minus` are separate fields.
`stokes` stores `(S0, S1, S2, S3)` in that order, with
`S0 = |ψ+|² + |ψ−|²`, `S1 = 2 Re(ψ+* ψ−)`,
`S2 = 2 Im(ψ+* ψ−)`, and `S3 = |ψ+|² − |ψ−|²`.
`degree_of_polarization = sqrt(S1² + S2² + S3²)/S0`, set to zero where
`S0 = 0`. `center-of-mass` provides `center_x_plus`, `center_x_minus` and
their y counterparts in 2D. `momentum-density` provides the full component
field and separate plus/minus fields.
