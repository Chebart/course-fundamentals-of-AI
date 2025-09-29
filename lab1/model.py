from core.blocks import Linear, Sigmoid

class MLP:
    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        self.layers = [
            Linear(in_features, 256),
            Sigmoid(),
            Linear(256, out_features),
            Sigmoid()
        ]

    def __call__(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            for p, g in layer.parameters():
                params.append((p, g))
        return params

    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)