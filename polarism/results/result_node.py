class ResultNode:
    def __init__(
        self, name, compute_fn, reduce_dim_fn, cmap, scaling, clim, expose, save, cut
    ):
        self.name = name
        self.compute_fn = compute_fn
        self.reduce_dim_fn = reduce_dim_fn
        self.cmap = cmap
        self.scaling = scaling
        self.clim = clim
        self.expose = expose
        self.save = save
        self.cut = cut

        self._field_history = []
        self._scalar_history = []

    def compute(self, **context):
        field = self.compute_fn(**context)
        reduced = self.reduce_dim_fn(field)

        self._field_history.append(field)
        self._scalar_history.append(reduced)

        return field, reduced
