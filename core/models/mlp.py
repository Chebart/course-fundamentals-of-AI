from ..blocks import Linear, Sigmoid
from .abstract_model import AbstractModel

class MLP(AbstractModel):
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
        super().__init__()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)    