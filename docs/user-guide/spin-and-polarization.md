# Spin and polarization

Set `physics.spin: spinor` to evolve two circular condensate components, ψ+ and ψ−. Scalar evolution is the default.

```yaml
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
  P0: 0.02
  circular_polarization: 0.5
```

Here `alpha_2` is a cross-spin interaction in meV µm², `zeeman` is a splitting in meV, and `te_tm` is a TE-TM coefficient in meV µm². The total pump `P` is 0.02 µm⁻² ps⁻¹, split between the two components by its circular polarization.

## Coupling the components

`alpha_2` couples each component to the other component's density. `zeeman` shifts their energies in opposite directions. TE-TM splitting mixes the components through their spatial derivatives; its Fourier-space coupling depends on `(kx ∓ i ky)²`. In 1D the TE-TM operator reduces to an x derivative. The [physics page](../explanation/physics.md) states the sign convention and equations.

The pump split is `P± = P(1 ± ρc)/2`, where `ρc` is `laser.circular_polarization` from −1 to 1. At +1 only ψ+ receives pump; at −1 only ψ− does. Zero splits the pump equally.

By default, a spinor run gives each component its own active reservoir density and depletion. `reservoir.components: 1` instead feeds both fields from one shared density, filled by the full pump and depleted by the total condensate density. See [Pumps and reservoirs](pumps-and-reservoirs.md) for the reservoir models.

A coherent source uses a normalized two-entry [drive polarization](coherent-drive.md). A scalar deterministic [initial condition](initial-conditions-and-noise.md) starts ψ+ only; a two-component file can set both. Initial random fields and Wigner increments are independent between components.

Saved spinor fields retain the component axis. The Stokes fields use `S0, S1, S2, S3` in circular order: `S0 = |ψ+|² + |ψ−|²` and `S3 = |ψ+|² − |ψ−|²`. See [Output and results](output-and-results.md) for field names. [Solver compatibility](../reference/solvers.md) lists supported spinor methods.
