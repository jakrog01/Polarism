# Physics Background

Polarism evolves a driven-dissipative condensate field coupled to one or more reservoir variables. In the single-reservoir case used as the baseline model, the implemented system is:

$$
i \hbar \frac{\partial \psi}{\partial t}
= \left[
-\frac{\hbar^2}{2 m_{\mathrm{eff}}} \nabla^2
+ V
+ g_C |\psi|^2
+ g_R n_R
+ \frac{i}{2}\left(R n_R - \gamma_C\right)
\right] \psi,
$$

$$
\frac{\partial n_R}{\partial t}
= P(x, y, t) - \left(\gamma_R + R |\psi|^2\right) n_R.
$$

The exact reservoir structure depends on whether the simulation uses the single,
double, or quadratic-double reservoir model.

For the double-reservoir model, the condensate equation keeps the same form but replaces `n_R` with the active density `n_A`, while the reservoir sector becomes:

$$
\frac{\partial n_I}{\partial t}
= P(x, y, t) - \left(\gamma_I + R_{IA}\right) n_I + R_{AI} n_A,
$$

$$
\frac{\partial n_A}{\partial t}
= R_{IA} n_I - \left(\gamma_A + R_{AI} + R |\psi|^2\right) n_A.
$$

For pulsed nonresonant campaigns the `quadratic-double` reservoir is available.
It separates the directly pumped inactive population `nI` from the active
population `nR` that feeds the condensate:

$$
\frac{\partial n_I}{\partial t}
= P(x,y,t) - \kappa n_I^2 - \gamma_I n_I,
$$

$$
\frac{\partial n_R}{\partial t}
= \kappa n_I^2 - \gamma_R n_R - R n_R |\psi|^2.
$$

The condensate equation keeps the same form as above and uses `nR` as the active
reservoir density.  This is a phenomenological carrier-relaxation model: it is
not a microscopic semiconductor transport calculation, but it captures an
important delay between optical injection and stimulated feeding of the
condensate.

## Initial noise and stochastic seeds

The initial condensate field is not a measured density.  It is a seed for
spontaneous symmetry breaking and stochastic growth.  The code exposes this
choice explicitly:

- `legacy_positive_uniform`: the historical seed, `eps*(Uniform[0,1)+i Uniform[0,1))`
- `complex_gaussian_zero_mean`: unbiased complex Gaussian seed
- `filtered_complex_gaussian`: zero-mean complex Gaussian seed projected to a
  finite radial k band and RMS-normalized

The filtered seed should be interpreted as an explicit condensate-band cutoff,
not as an invisible smoothing of the evolving field.  This distinction matters:
the parabolic lower-polariton effective-mass model and finite-difference
Laplacian are not physically meaningful up to arbitrary grid/Nyquist momenta.
The cutoff is written to metadata and should be swept or justified when spatial
geometry is part of the claim.

## Energy relaxation and model limits

The minimal open-dissipative GPE/reservoir model does not automatically include
energy relaxation from high-k polariton states toward the low-k condensate.  In
single-shot and pulsed polariton literature, phenomenological relaxation terms
are often added when modelling ground-state condensation, filamentation, and
suppression of high-k fluctuations.

That means a diagonal or filamentary pattern in one simulation is not by itself
proof of physical filamentation.  It must be checked against:

- seed cutoff and stochastic realization
- spatial resolution and physical domain size
- finite-difference stencil anisotropy
- high-k spectral diagnostics
- optional physical relaxation terms if the high-k population remains dominant

## Numerical methods in the repository

- Finite-difference RK4 solvers provide the reference implementation and the most predictable behavior across boundary choices.
- Fused CUDA RK4 solvers trade portability for throughput and target large production runs.
- Spectral solvers use FFT or related transforms and are efficient when the problem matches their assumptions.
- `rk4-cuda` supports `five-point` and `isotropic-9pt` finite-difference
  Laplacian stencils.  The latter reduces grid-direction anisotropy but keeps the
  same physical kinetic operator.

## Practical interpretation

- The Laplacian term controls kinetic spreading and makes spatial resolution matter.
- The nonlinear interaction terms couple density back into the phase evolution.
- The gain-loss term means stability is not only about oscillation error but also about correctly resolving growth and decay.
- In the double-reservoir model, the inter-reservoir transfer rates \(R_{IA}\) and \(R_{AI}\) add extra timescales that can tighten the timestep requirements.
- In the quadratic-double model, the nonlinear transfer \(\kappa n_I^2\) and
  reservoir lifetimes can create memory across pulse trains.
- High-k growth near the grid/Nyquist edge is a numerical warning sign unless a
  physical high-k mechanism is explicitly part of the model.

## Implication for users

The repository does not guarantee correctness from parameter values alone. Reliable studies still require:

- time-step convergence checks
- grid refinement checks
- solver cross-comparison on representative cases

That validation philosophy is reflected in the bundled compliance and decay tests.

## Reference context

The implemented GPE/reservoir structure follows the common open-dissipative
polariton mean-field picture introduced and used in the polariton literature,
including gain/loss balance and reservoir-fed stimulated scattering.  For pulsed
single-shot experiments, the literature also emphasizes stochastic initial
conditions, reservoir depletion, and energy relaxation as important ingredients.
Useful starting points:

- Wouters and Carusotto, *Excitations in a Nonequilibrium Bose-Einstein Condensate of Exciton Polaritons*, PRL 99, 140402 (2007): <https://doi.org/10.1103/PhysRevLett.99.140402>
- Estrecho et al., *Single-shot condensation of exciton polaritons and the hole burning effect*, Nature Communications 9, 2944 (2018): <https://doi.org/10.1038/s41467-018-05349-4>
- studies of relaxation oscillations and expanding condensates that couple GPE dynamics to multi-step reservoir rate equations
