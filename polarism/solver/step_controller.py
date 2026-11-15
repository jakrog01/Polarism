"""Adaptive trial size control and restartable statistics."""
from __future__ import annotations

import math
import sys


class StepController:
    """Choose trial sizes using an embedded error estimate and PI feedback."""

    def __init__(self, solver):
        from polarism.solver.tableaus import TABLEAUS

        self.dt_max = solver.config.solver.dt_max
        self.h_next = min(solver.config.solver.dt, self.dt_max or math.inf)
        tableau = getattr(solver, "tableau", TABLEAUS[solver.config.solver.tableau])
        self.k = min(tableau.order, tableau.embedded_order) + 1
        self.err_prev = 1.0
        self.no_growth = False
        self.accepted = 0
        self.rejected = 0
        self.dt_min = math.inf
        self.dt_max_used = 0.0
        self.dt_last = 0.0

    @property
    def statistics(self):
        """Return host-side counters and accepted trial sizes."""
        return dict(accepted=self.accepted, rejected=self.rejected,
                    dt_min=self.dt_min, dt_max=self.dt_max_used, dt_last=self.dt_last)

    def state_dict(self):
        """Return the data needed to continue the exact trial sequence."""
        return dict(h_next=self.h_next, err_prev=self.err_prev,
                    **self.statistics)

    def load_state_dict(self, values):
        """Restore the trial sequence and statistics."""
        self.h_next = float(values["h_next"])
        self.err_prev = float(values["err_prev"])
        self.no_growth = False
        self.accepted = int(values["accepted"])
        self.rejected = int(values["rejected"])
        self.dt_min = float(values["dt_min"])
        self.dt_max_used = float(values["dt_max"])
        self.dt_last = float(values["dt_last"])

    def trial(self, solver, model, state, target):
        """Run one trial, update feedback and return whether it was accepted."""
        proposal = self.h_next
        clamped = proposal >= target - state.t
        h = target - state.t if clamped else proposal
        if h < 16 * sys.float_info.epsilon * max(1, abs(state.t)):
            raise RuntimeError(f"adaptive step size underflow at t={state.t}")
        report = solver.step(model, state, h)
        if report.accepted and clamped:
            state.t = target
        error = report.error
        if error is None or not math.isfinite(error) or error < 0:
            error = math.inf
        if report.accepted:
            self.accepted += 1
            self.dt_min = min(self.dt_min, h)
            self.dt_max_used = max(self.dt_max_used, h)
            self.dt_last = h
            factor = 5.0 if error == 0 else 0.9 * error ** (-0.7 / self.k) * self.err_prev ** (0.4 / self.k)
            factor = min(5.0, max(0.2, factor))
            if self.no_growth:
                factor = min(1.0, factor)
            self.err_prev = max(error, 1e-4)
            self.no_growth = False
        else:
            self.rejected += 1
            factor = max(0.2, 0.9 * error ** (-1 / self.k))
            factor = min(1.0, factor)
            self.no_growth = True
        h_next = report.dt_next if report.dt_next is not None else h * factor
        if report.accepted and clamped:
            h_next = max(h_next, proposal)
        self.h_next = min(h_next, self.dt_max or math.inf)
        return report.accepted
