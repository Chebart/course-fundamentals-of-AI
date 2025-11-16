from .abstract_model import AbstractModel
from ..blocks import LSTMBlock, Linear
from core.data import Tensor

class LSTM(AbstractModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1
    ):
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMBlock(input_size, hidden_size))
            else:
                self.layers.append(LSTMBlock(hidden_size, hidden_size))

        super().__init__()

    def forward(self, x):
        # reset block cache
        for layer in self.layers:
            layer.reset_cache()

        # forward pass
        T, batch, _ = x.shape
        h_prev = [Tensor.zeros((batch, layer.hidden_size), dtype = x.dtype, device = x.device) for layer in self.layers]
        c_prev = [Tensor.zeros((batch, layer.hidden_size), dtype = x.dtype, device = x.device) for layer in self.layers]
        output = []

        for t in range(T):
            for l, layer in enumerate(self.layers):
                inp = x[t] if l == 0 else h_prev[l-1]
                h, c = layer.forward(inp, h_prev[l], c_prev[l])
                h_prev[l] = h
                c_prev[l] = c

            output.append(h_prev[-1].clone())

        return Tensor.stack(output, axis = 0, dtype = x.dtype, device = x.device), h_prev

    def backward(self, dLdy):
        T, batch, _ = dLdy.shape
        dLdh_next = [Tensor.zeros((batch, layer.hidden_size), dLdy.dtype, dLdy.device) for layer in self.layers]
        dLdc_next = [Tensor.zeros((batch, layer.hidden_size), dLdy.dtype, dLdy.device) for layer in self.layers]

        for t in reversed(range(T)):
            dLdh = dLdy[t]
            for l in reversed(range(len(self.layers))):
                dLdh_total = dLdh + dLdh_next[l]
                dLdc_total = dLdc_next[l]
                dx, dh_prev, dc_prev = self.layers[l].backward(dLdh_total, dLdc_total, t)
                dLdh_next[l] = dh_prev
                dLdc_next[l] = dc_prev
                dLdh = dx

class LSTMReg(AbstractModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1
    ):
        self.layers = [
            LSTM(input_size, hidden_size, num_layers),
            Linear(hidden_size, 1)
        ]
        super().__init__()

    def forward(self, x):
        self.x_shape = x.shape
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l == 0:
                x = x[0][-1]

        return x.reshape(-1)
    
    def backward(self, dLdy):
        for l, layer in enumerate(reversed(self.layers)):
            if l == 0:
                dLdy = dLdy.reshape((-1, 1))
            if l == 1:
                dLdy = Tensor.stack([dLdy for _ in range(self.x_shape[0])], 0, dLdy.dtype, dLdy.device)

            dLdy = layer.backward(dLdy)    