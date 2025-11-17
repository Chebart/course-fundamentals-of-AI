from .abstract_block import AbstractBlock
from .init_weights import xavier 
from core.data import Tensor

class Linear(AbstractBlock):
    """y = x * w.T + b"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: str = "fp32",
        bias: bool = True
    ):
        # Set bias flag
        self._bias = bias
        # Init trainable params and small increments
        self._w = xavier((out_features, in_features), dtype = dtype, uniform = True)
        self._b = Tensor.zeros((out_features), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
        
    def forward(self, x):
        self.x = x
        return self.x @ self._w.T + self._b
    
    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]
        
    def backward(self, dLdy):
        self._dw += dLdy.T @ self.x
        self._db += dLdy.sum(axis=0)
        dLdx = dLdy @ self._w
        return dLdx