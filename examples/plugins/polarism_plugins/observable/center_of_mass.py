"""Condensate x centre of mass observable."""
from polarism.compute_engine import compute_engine
from polarism.results import ResultNode, register_observable


@register_observable("example-center-x")
def center_of_mass_x(controller):
    """Create a device-computed condensate centre of mass scalar."""
    def compute(**context):
        xp = compute_engine.xp
        density = xp.abs(context["state"].psi) ** 2
        return (density * context["grid"].X).sum() / xp.maximum(density.sum(), 1e-30)

    return ResultNode(
        name="center_x", label="Condensate centre x", units="µm",
        compute_fn=compute, reduce_dim_fn=lambda value: value,
        cmap=None, scaling=None, clim=None, expose=True, save=True,
        cut=None, is_field=False,
    )
