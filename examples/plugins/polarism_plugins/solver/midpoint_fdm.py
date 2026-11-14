"""A small midpoint solver using the reference model right-hand side."""

from polarism.solver import AbstractSolver, ACCEPTED, SolverCapabilities, register_solver

POLARISM_PLUGIN_API = 2


@register_solver("example-midpoint-fdm")
class MidpointFDMSolver(AbstractSolver):
    """Integrate the model with the explicit midpoint method."""

    capabilities = SolverCapabilities(
        ndim={1, 2}, grid_types={"periodic", "closed-interval"},
        kinetic_relaxation=True,
        description="Reference finite-difference midpoint solver.",
    )

    def step(self, model, state, dt):
        psi = state.field
        fields = model.reservoir.get_state()
        rate, derivatives = model.rhs(psi, fields, state.t)
        midpoint = psi + 0.5 * dt * rate
        middle_fields = tuple(field + 0.5 * dt * derivative
                              for field, derivative in zip(fields, derivatives))
        middle_rate, middle_derivatives = model.rhs(
            midpoint, middle_fields, state.t + 0.5 * dt)
        state.field = model.boundary.after_step_action(psi + dt * middle_rate).astype(psi.dtype, copy=False)
        model.reservoir.set_state(tuple(field + dt * derivative
                                        for field, derivative in zip(fields, middle_derivatives)))
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
