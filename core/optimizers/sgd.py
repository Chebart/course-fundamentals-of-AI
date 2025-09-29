import numpy as np

class SGD():
    """Stochastic gradient descent optimizer"""
    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._velocity: float = 0.0
        
    def step(
        self, 
        model_params: list[tuple[np.ndarray, np.ndarray]]
    ):
        """Update model params"""
        for p, g in model_params:
            if self._weight_decay > 0:
                g += self._weight_decay * g
            p -= self._lr * g

    def zero_grad(
        self, 
        model_params: list[tuple[np.ndarray, np.ndarray]]
    ):
        """Reset grad of model params"""
        for _, g in model_params:
            g.fill(0)