from ..blocks import Linear, Conv2D, MaxPool2D, AvgPool2D, ReLU, Softmax, Reshape
from .abstract_model import AbstractModel

class LeNet5(AbstractModel):  
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        self.layers = [
            Conv2D(in_channels, 6, kernel_size = 5),
            ReLU(),
            AvgPool2D(kernel_size = 2, stride = 2),
            Conv2D(6, 16, kernel_size = 5),
            ReLU(),
            AvgPool2D(kernel_size = 2, stride = 2),
            Reshape(input_shape=(-1, 16, 5, 5), output_shape=(-1, 400)),
            Linear(400, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, out_channels),
            Softmax()
        ]
        super().__init__()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)    