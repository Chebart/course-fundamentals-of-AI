from abc import ABC, abstractmethod

import numpy as np

class AbstractModel(ABC):
    def __init__(self):
        if not hasattr(self, "layers"):
            raise NotImplementedError(
                "Models must have self.layers attribute!"
            )

    def __call__(self, x: np.ndarray)-> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray)-> np.ndarray:
        """Calculate forward pass"""
        pass

    def parameters(self)-> list[tuple[np.ndarray, np.ndarray]]:
        """Return parameters of the model"""
        params = []
        for layer in self.layers:
            for p_type, p, g in layer.parameters():
                params.append((p_type, p, g))
        return params

    @abstractmethod
    def backward(self, dLdy: np.ndarray)-> None:
        """Calculate backward pass"""
        pass