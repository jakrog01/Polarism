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

The exact reservoir structure depends on whether the simulation uses the single or double reservoir model.

For the double-reservoir model, the condensate equation keeps the same form but replaces `n_R` with the active density `n_A`, while the reservoir sector becomes:

$$
\frac{\partial n_I}{\partial t}
= P(x, y, t) - \left(\gamma_I + R_{IA}\right) n_I + R_{AI} n_A,
$$

$$
\frac{\partial n_A}{\partial t}
= R_{IA} n_I - \left(\gamma_A + R_{AI} + R |\psi|^2\right) n_A.
$$

## Numerical methods in the repository

- Finite-difference RK4 solvers provide the reference implementation and the most predictable behavior across boundary choices.
- Fused CUDA RK4 solvers trade portability for throughput and target large production runs.
- Spectral solvers use FFT or related transforms and are efficient when the problem matches their assumptions.

## Practical interpretation

- The Laplacian term controls kinetic spreading and makes spatial resolution matter.
- The nonlinear interaction terms couple density back into the phase evolution.
- The gain-loss term means stability is not only about oscillation error but also about correctly resolving growth and decay.
- In the double-reservoir model, the inter-reservoir transfer rates \(R_{IA}\) and \(R_{AI}\) add extra timescales that can tighten the timestep requirements.

## Implication for users

The repository does not guarantee correctness from parameter values alone. Reliable studies still require:

- time-step convergence checks
- grid refinement checks
- solver cross-comparison on representative cases

That validation philosophy is reflected in the bundled compliance and decay tests.
