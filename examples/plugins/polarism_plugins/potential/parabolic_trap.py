from polarism.potential import register_potential


@register_potential("example-parabolic-trap")
def parabolic_trap(X, Y, potential_config):
    return potential_config.V1 * (X**2 + Y**2)
