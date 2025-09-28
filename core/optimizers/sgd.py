import numpy as np

class SGD():
    """Stochastic gradient descent optimizer"""
    def __init__(
        self,
        model_params: list[tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self.model_params = model_params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity: float = 0.0
        
    def step(self):
        for p, g in self.model_params:
            p -= self.lr * g

    def zero_grad(self):
        for _, g in self.model_params:
            g.fill(0)