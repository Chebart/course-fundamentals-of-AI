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
        self._b = np.zeros((out_features), dtype = np.float32)
        self._dw = np.zeros_like(self._w)
        self._db = np.zeros_like(self._b)
        
    def __call__(
        self, 
        x: np.ndarray
    )-> np.ndarray:
        """Calculate forward pass"""
        self.x = x
        return self.x @ self._w.T + self._b
    
    @property
    def dw(self)-> np.ndarray:
        return self._dw
    
    @property
    def db(self)-> np.ndarray:
        return self._db
    
    def parameters(self):
        return [(self._w, self._dw), (self._b, self._db)]

    def update_params(
        self,
        opt_dw: np.ndarray,
        opt_db: np.ndarray = None,
    ):
        """Update trainable params using optimizer"""
        self._w -= opt_dw
        if self._bias and opt_db is not None:
            self._b -= opt_db

    def zero_grad(self):
        """Reset small increments"""
        self._dw = np.zeros_like(self._w)
        self._db = np.zeros_like(self._b)

    def backward(
        self,
        dLdy: np.ndarray
    )-> np.ndarray:
        """Calculate backward pass"""
        self._dw = dLdy.T @ self.x
        self._db = dLdy.sum(axis=0)
        dLdx = dLdy @ self._w
        return dLdx