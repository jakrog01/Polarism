class ResultsManager:
    def __init__(self):
        self.t = []
        self.nodes = []

    def step(self, t, **context):
        self.t.append(t)
        for node in self.nodes:
            node.compute(**context)
