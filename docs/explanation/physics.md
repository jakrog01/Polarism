# Physics model

Polarism evolves a driven and lossy condensate field coupled to an optional nonresonant reservoir. This page defines the equations and units used by the built-in model.

## Scalar condensate

For a scalar field ψ, the finite difference reference right-hand side is equivalent to

$$
i\hbar\partial_t\psi = \left[-\frac{\hbar^2}{2m_{\mathrm{eff}}}\nabla^2 + V + g_C|\psi|^2 + g_R n_A + g_I n_I + \frac{i\hbar}{2}(R n_A-\gamma_C)\right]\psi + i\hbar\eta n_A\frac{\hbar}{2m_{\mathrm{eff}}}\nabla^2\psi + H_{\mathrm{extra}}\psi + i\hbar\Gamma_{\mathrm{extra}}\psi - E_p\psi + F.
$$

Here `n_A` means the active density for the chosen reservoir, and `n_I` is its inactive density or zero. For `single` and `quadratic-double`, the active density is named `nR`; for `double`, it is `nA`. `none` makes both zero. The coherent source `F` and rotating-frame energy `E_p` appear only when the drive is enabled. Extra energy and rate terms appear only when configured. A complex absorbing potential is included in `V` when selected. The `η` term is `physics.kinetic_relaxation_eta`; it damps a plane wave at a rate proportional to `η n_A k²`.

In 2D, position is in µm, time in ps, energy in meV, ψ in µm⁻¹, and reservoir densities in µm⁻². `ħ` is in meV ps and `m_eff` is in meV ps² µm⁻². `γ_C`, `γ_R`, `γ_A`, `γ_I`, and transfer rates are in ps⁻¹; `R` is in µm² ps⁻¹; `g_C`, `g_R`, and `g_I` are in meV µm²; `P` is in µm⁻² ps⁻¹; `η` is in µm². In 1D, ψ is in µm⁻¹ᐟ², densities in µm⁻¹, and interaction, scattering, pump, and η units change by the corresponding power of µm. See [Configuration reference](../reference/configuration.md) for parameter definitions.

## Reservoir rate equations

The built-in models use the following densities and rates. Each displayed equation may also include `D_j ∇²n_j`, where the corresponding `physics.reservoir_diffusion_*` coefficient is nonzero; `D_j` is in µm² ps⁻¹.

For `single`,

$$
\partial_t n_R=P-(\gamma_R+R|\psi|^2)n_R+D_R\nabla^2n_R.
$$

For `double`,

$$
\partial_t n_I=P-(\gamma_I+R_{IA})n_I+R_{AI}n_A+D_I\nabla^2n_I,
$$
$$
\partial_t n_A=R_{IA}n_I-(\gamma_A+R_{AI}+R|\psi|^2)n_A+D_A\nabla^2n_A.
$$

For `quadratic-double`,

$$
\partial_t n_I=P-\kappa n_I^2-\gamma_I n_I+D_I\nabla^2n_I,
$$
$$
\partial_t n_R=\kappa n_I^2-(\gamma_R+R|\psi|^2)n_R+D_R\nabla^2n_R.
$$

`κ` is in µm² ps⁻¹ in 2D. With `none`, there is no reservoir rate equation. Use `quadratic-double` as the production reservoir model; `double` is kept as a sanity-check and reference model. Reservoir diffusion uses periodic wrapping or mirrored closed-interval boundaries. For spinor fields, a shared reservoir responds to total condensate density; separate component reservoirs respond to their component density and weighted pump. See [Pumps and reservoirs](../user-guide/pumps-and-reservoirs.md).

## Spinor condensate and dispersion

For circular components ψ₊ and ψ₋, each component uses its own `g_C|ψ_±|²`, plus `alpha_2|ψ_∓|²`, and Zeeman shifts `±zeeman/2`. Its gain, loss, reservoir, drive, and relaxation follow the scalar equation with the appropriate component density. The TE-TM energy added to component ± is `-β(∂ₓ ∓ i∂ᵧ)²ψ_∓`, with `β = physics.te_tm` in meV µm². In Fourier space, the off-diagonal energies are `β(kₓ ∓ ikᵧ)²`. In 1D, `∂ᵧ = kᵧ = 0`.

The built-in `parabolic` dispersion is `E(k)=ħ²|k|²/(2m_eff)` in meV, with k in µm⁻¹. Spectral solvers use this Fourier-space energy or a dispersion plugin. Finite difference solvers approximate its Laplacian on the spatial grid. See [Spin and polarization](../user-guide/spin-and-polarization.md) and [Solver reference](../reference/solvers.md).

## Stochastic field and interpretation

With `noise.stochastic: wigner`, each cell and component receives a complex Gaussian increment per step:

$$
\Delta\psi=\sqrt{\frac{s(\gamma_C+Rn_A)\,dt}{4\Delta V}}(z_{\mathrm{re}}+iz_{\mathrm{im}}),\qquad z_{\mathrm{re}},z_{\mathrm{im}}\sim\mathcal N(0,1).
$$

Here `s` is `noise.stochastic_strength`, `dt` is in ps, and cell volume `ΔV` is `dx` in 1D or `dx dy` in 2D. The active density is sampled at the solver's injection point. The noise uses counter-based Philox draws. This implementation omits the `−g_C/ΔV` interaction energy drift and the `R n_A/(2ΔV)` reservoir correction of the full truncated-Wigner equations. [Observables and spectra](../user-guide/observables-and-spectra.md) explains the separate vacuum corrections applied to accumulated moments.

Initial noise seeds symmetry breaking but does not represent a measured density. The default initial seed is zero-mean complex Gaussian noise when `noise.enabled` is true. The optional filtered seed imposes an explicit momentum cutoff. For physical claims, test seed variation, spatial and temporal convergence, and solver agreement; high-k population near Nyquist can signal under-resolution or missing physical relaxation.

## Energy relaxation and model limits

The minimal open-dissipative model has no channel that moves energy from high-k fluctuations toward the condensate ground state. A filamentary or X-shaped density pattern is therefore not evidence of physics on its own. Check it against the seed cutoff and stochastic realization, the spatial resolution and domain size, finite difference stencil anisotropy, and the momentum distribution near the Nyquist edge.

The kinetic relaxation term `η n_A (ħ/2m_eff) ∇²ψ` in the scalar equation is the standard phenomenological fix. It is a physical model term, not a numerical filter: for a plane wave it adds a damping rate `η n_A (ħ/2m_eff) k²`, so it suppresses high-k modes most. The active density is `nR` for `single` and `quadratic-double`, `nA` for `double`, and zero for `none`. It is off by default (`physics.kinetic_relaxation_eta: 0.0`). Choose the smallest η that removes growth near the Nyquist edge in `n(k)` or the momentum density; values around `1e-5` to `3e-5` µm² are typical starting points for tight pump spots, and `1e-4` µm² is an aggressive diagnostic upper bound rather than a production value.

Reservoir diffusion smooths the reservoir densities, not ψ. It represents carrier or exciton transport and is not a cure for high-k growth in the condensate. Use it as a controlled model variant after the case without diffusion is understood.

## Validating results

Parameter values alone do not guarantee a correct result. For quantities you report:

- halve the time step until they stop changing;
- refine the grid and enlarge the domain;
- compare at least two solver families on a representative case;
- repeat noisy runs with several seeds.

Gain and loss make stability a question of resolving growth and decay, not only oscillation. The transfer rates `R_IA`, `R_AI` and the quadratic transfer `κ n_I²` add their own time scales, which can tighten the time step. See [Solvers and time stepping](../user-guide/solvers-and-time-stepping.md).

## References

- M. Wouters and I. Carusotto, *Excitations in a nonequilibrium Bose-Einstein condensate of exciton polaritons*, Phys. Rev. Lett. 99, 140402 (2007). <https://doi.org/10.1103/PhysRevLett.99.140402>
- M. Wouters and V. Savona, *Stochastic classical field model for polariton condensates*, Phys. Rev. B 79, 165302 (2009). <https://doi.org/10.1103/PhysRevB.79.165302>
- M. Wouters, T. C. H. Liew and V. Savona, *Energy relaxation in one-dimensional polariton condensates*, Phys. Rev. B 82, 245315 (2010). <https://arxiv.org/abs/1008.5320>
- I. Carusotto and C. Ciuti, *Quantum fluids of light*, Rev. Mod. Phys. 85, 299 (2013). <https://doi.org/10.1103/RevModPhys.85.299>
- E. Estrecho et al., *Single-shot condensation of exciton polaritons and the hole burning effect*, Nat. Commun. 9, 2944 (2018). <https://doi.org/10.1038/s41467-018-05349-4>
- J. D. Töpfer et al., *Engineering spatial coherence in lattices of polariton condensates*, Sci. Rep. 14, 13104 (2024). <https://www.nature.com/articles/s41598-024-63725-1>
