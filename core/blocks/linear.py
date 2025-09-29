import numpy as np

from .init_weights import xavier 

class Linear():
    """y = x * w.T + b"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        # Set bias flag
        self._bias = bias
        # Init trainable params and small increments
        self._w = xavier((out_features, in_features), uniform = True)
        self._b = np.zeros((out_features), dtype = np.float64)
        self._dw = np.zeros_like(self._w, dtype = np.float64)
        self._db = np.zeros_like(self._b, dtype = np.float64)
        
    def __call__(
        self, 
        x: np.ndarray
    )-> np.ndarray:
        """Calculate forward pass"""
        self.x = x
        return self.x @ self._w.T + self._b
    
    def parameters(self):
        if self._bias:
            return [(self._w, self._dw), (self._b, self._db)]
        else:
            return [(self._w, self._dw)]
        
    def backward(
        self,
        dLdy: np.ndarray
    )-> np.ndarray:
        """Calculate backward pass"""
        self._dw = dLdy.T @ self.x
        self._db = dLdy.sum(axis=0)
        dLdx = dLdy @ self._w
        return dLdx