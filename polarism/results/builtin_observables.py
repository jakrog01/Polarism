"""Built-in frame observables selected with ``cfg.result.observables``.

Every node is computed on the active backend from ``state.psi``. In a batched
run scalars are device vectors with one value per member and fields keep the
member axis. The always-present ``norm`` node is not repeated here.
"""
from __future__ import annotations

import math

from polarism.compute_engine import compute_engine
from polarism.results.observable_registry import register_observable
from polarism.results.result_node import ResultNode

TWO_DIMENSIONAL_OBSERVABLES = frozenset({"angular-momentum", "vortices"})

_TINY = 1e-30
_VORTEX_DENSITY_FLOOR = 1e-3


def _node(name, label, units, compute_fn, *, field):
    """Build a saved node; fields are not exposed to the scalar live view."""
    return ResultNode(
        name=name, label=label, units=units, compute_fn=compute_fn,
        reduce_dim_fn=lambda value: value, cmap=None, scaling=None, clim=None,
        expose=not field, save=True, cut=None, is_field=field,
    )


def _axes(grid):
    """Return the spatial axes counted from the end."""
    return tuple(range(-grid.ndim, 0))


def _density(psi):
    """Return ``|psi|^2`` without a square root."""
    return psi.real ** 2 + psi.imag ** 2


def _derivative(xp, field, axis, spacing, periodic):
    """Differentiate along ``axis``: periodic central, or central with one-sided edges."""
    if periodic:
        return (xp.roll(field, -1, axis=axis) - xp.roll(field, 1, axis=axis)) / (2.0 * spacing)

    def at(index):
        key = [slice(None)] * field.ndim
        key[axis] = index
        return tuple(key)

    out = xp.empty_like(field)
    out[at(slice(1, -1))] = (field[at(slice(2, None))] - field[at(slice(None, -2))]) / (2.0 * spacing)
    out[at(0)] = (field[at(1)] - field[at(0)]) / spacing
    out[at(-1)] = (field[at(-1)] - field[at(-2)]) / spacing
    return out


def _hbar_over_m(controller):
    """Return ``hbar / m_eff`` as a float or, for per-member physics, a broadcastable array."""
    physics = controller.model.physics
    value = physics.get("hbar") / physics.get("m_eff")
    return value[:, 0] if getattr(value, "ndim", 0) else value


def _winding(xp, psi, periodic):
    """Return the int8 phase winding of every plaquette whose lower-left corner is the index.

    The plaquette ``(j, i)`` has corners ``(j, i)``, ``(j, i+1)``, ``(j+1, i+1)``,
    ``(j+1, i)`` traversed counter-clockwise. The wrapped phase steps of the four
    edges sum to ``2 pi`` times the enclosed charge. On closed-interval grids the
    last row and column have no plaquette and are zero.
    """
    conj = psi.conj()
    step_x = xp.angle(xp.roll(psi, -1, axis=-1) * conj)
    step_y = xp.angle(xp.roll(psi, -1, axis=-2) * conj)
    total = (step_x + xp.roll(step_y, -1, axis=-1)
             - xp.roll(step_x, -1, axis=-2) - step_y)
    charge = xp.rint(total / (2.0 * math.pi)).astype(xp.int8)
    if not periodic:
        charge[..., -1, :] = 0
        charge[..., :, -1] = 0
    return charge


@register_observable("angular-momentum")
def angular_momentum(controller):
    """Create the scalar ``angular_momentum`` = ``<L_z> / N`` in units of hbar, 2D only.

    ``L_z = x p_y - y p_x = -i hbar (x d_y - y d_x)``, so
    ``<L_z> / (N hbar) = Im sum(psi* (x d_y psi - y d_x psi)) / sum(|psi|^2)``
    about the grid origin, with the derivatives of ``current``. A vortex of
    charge m centred on the origin gives m.
    """
    periodic = controller.grid.boundary == "periodic"

    def compute(**context):
        xp = compute_engine.xp
        psi = context["state"].psi
        grid = context["grid"]
        axes = _axes(grid)
        d_x = _derivative(xp, psi, -1, grid.spacing[-1], periodic)
        d_y = _derivative(xp, psi, -2, grid.spacing[-2], periodic)
        real = psi.real.dtype
        rotated = grid.X.astype(real, copy=False) * d_y - grid.Y.astype(real, copy=False) * d_x
        return (psi.conj() * rotated).imag.sum(axis=axes) / xp.maximum(
            _density(psi).sum(axis=axes), _TINY)

    node = _node("angular_momentum", "Angular momentum <L_z>/N", "ħ", compute, field=False)
    if controller.state.components == 1:
        return node
    def selected(component):
        def value(**context):
            full = compute(**context)
            return full[:, component] if context["state"].batched else full[component]
        return value
    return [_node(f"angular_momentum_{suffix}", f"Angular momentum {suffix}", "ħ",
                  selected(component), field=False)
            for component, suffix in ((0, "plus"), (1, "minus"))]


@register_observable("center-of-mass")
def center_of_mass(controller):
    """Create ``center_x`` (and ``center_y`` in 2D) in µm.

    ``<r> = sum(|psi|^2 r) / sum(|psi|^2)`` over the grid; the cell area cancels.
    """
    def make(coordinate):
        def compute(**context):
            xp = compute_engine.xp
            psi = context["state"].psi
            grid = context["grid"]
            axes = _axes(grid)
            density = _density(psi)
            weight = getattr(grid, coordinate).astype(density.dtype, copy=False)
            return (density * weight).sum(axis=axes) / xp.maximum(density.sum(axis=axes), _TINY)
        return compute

    if controller.state.components == 2:
        def per_component(coordinate, component):
            def compute(**context):
                xp = compute_engine.xp
                psi = context["state"].psi_plus if component == 0 else context["state"].psi_minus
                density = _density(psi)
                axes = _axes(context["grid"])
                weight = getattr(context["grid"], coordinate).astype(density.dtype, copy=False)
                return (density * weight).sum(axis=axes) / xp.maximum(density.sum(axis=axes), _TINY)
            return compute
        nodes = []
        for coordinate, name in (("X", "center_x"), ("Y", "center_y")):
            if coordinate == "Y" and controller.grid.ndim == 1:
                continue
            for component, suffix in ((0, "plus"), (1, "minus")):
                nodes.append(_node(f"{name}_{suffix}", f"Centre of mass {coordinate.lower()} {suffix}",
                                   "µm", per_component(coordinate, component), field=False))
        return nodes
    nodes = [_node("center_x", "Centre of mass x", "µm", make("X"), field=False)]
    if controller.grid.ndim == 2:
        nodes.append(_node("center_y", "Centre of mass y", "µm", make("Y"), field=False))
    return nodes


@register_observable("current")
def current(controller):
    """Create the probability current fields ``current_x`` (and ``current_y`` in 2D).

    ``j = (hbar / m_eff) Im(psi* grad psi)`` with central differences: periodic
    on periodic grids, one-sided at the edges of closed-interval grids. The
    units are µm^-1 ps^-1 in 2D and ps^-1 in 1D.
    """
    periodic = controller.grid.boundary == "periodic"

    def make(axis):
        def compute(**context):
            xp = compute_engine.xp
            psi = context["state"].psi
            spacing = context["grid"].spacing[axis]
            gradient = _derivative(xp, psi, axis, spacing, periodic)
            if context["state"].components == 2 and context["state"].batched:
                physics = controller.model.physics
                return (physics.get("hbar") / physics.get("m_eff")) * (psi.conj() * gradient).imag
            return _hbar_over_m(controller) * (psi.conj() * gradient).imag
        return compute

    units = "µm^-1 ps^-1" if controller.grid.ndim == 2 else "ps^-1"
    nodes = [_node("current_x", "Current x", units, make(-1), field=True)]
    if controller.grid.ndim == 2:
        nodes.append(_node("current_y", "Current y", units, make(-2), field=True))
    return nodes


@register_observable("momentum-density")
def momentum_density(controller):
    """Create the field ``momentum_density`` = ``n(k)``, fftshifted, in µm^ndim.

    ``n(k) = |sum_r psi(r) exp(-i k r) prod(h)|^2 / (2 pi)^ndim`` evaluated with an
    FFT, so that ``sum(n) * prod(dk) = N`` with ``dk = 2 pi / (n_i h_i)``. The k axes
    are ``2 pi * fftshift(fftfreq(n_i, h_i))`` along ``kx`` (last axis) and ``ky``.
    """
    def compute(**context):
        xp = compute_engine.xp
        psi = context["state"].psi
        grid = context["grid"]
        axes = _axes(grid)
        spectrum = xp.fft.fftn(psi, axes=axes)
        scale = math.prod(grid.spacing) ** 2 / (2.0 * math.pi) ** grid.ndim
        return xp.fft.fftshift(_density(spectrum) * scale, axes=axes)

    node = _node("momentum_density", "Momentum density |ψ(k)|²",
                 f"µm^{controller.grid.ndim}", compute, field=True)
    if controller.state.components == 1:
        return node
    def component_compute(component):
        def selected(**context):
            value = compute(**context)
            return value[:, component] if context["state"].batched else value[component]
        return selected
    return [node, *(_node(f"momentum_density_{suffix}", f"Momentum density {suffix}",
                          f"µm^{controller.grid.ndim}", component_compute(component), field=True)
                    for component, suffix in ((0, "plus"), (1, "minus")))]


@register_observable("vortices")
def vortices(controller):
    """Create the int8 field ``vortex_charge`` and the scalar ``vortex_count``, 2D only.

    ``vortex_charge[j, i]`` is the phase winding, in units of 2 pi, around the
    plaquette with corners ``(j, i)`` to ``(j+1, i+1)``, positive for a
    counter-clockwise phase increase. ``vortex_count`` counts the non-zero
    plaquettes whose largest corner density exceeds ``1e-3 max |psi|^2`` of the
    member, which suppresses phase noise in empty regions.
    """
    periodic = controller.grid.boundary == "periodic"

    def charge(**context):
        return _winding(compute_engine.xp, context["state"].psi, periodic)

    def count(**context):
        xp = compute_engine.xp
        psi = context["state"].psi
        axes = _axes(context["grid"])
        density = _density(psi)
        right = xp.roll(density, -1, axis=-1)
        corner = xp.maximum(xp.maximum(density, right),
                            xp.maximum(xp.roll(density, -1, axis=-2), xp.roll(right, -1, axis=-2)))
        floor = _VORTEX_DENSITY_FLOOR * density.max(axis=axes, keepdims=True)
        found = (_winding(xp, psi, periodic) != 0) & (corner > floor)
        return found.sum(axis=axes)

    nodes = [
        _node("vortex_charge", "Vortex winding", "", charge, field=True),
        _node("vortex_count", "Vortex count", "", count, field=False),
    ]
    if controller.state.components == 1:
        return nodes
    def selected(component):
        def value(**context):
            full = count(**context)
            return full[:, component] if context["state"].batched else full[component]
        return value
    return [nodes[0], *(_node(f"vortex_count_{suffix}", f"Vortex count {suffix}", "",
                              selected(component), field=False)
                        for component, suffix in ((0, "plus"), (1, "minus")))]
