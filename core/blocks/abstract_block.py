from abc import ABC, abstractmethod

import numpy as np

class AbstractBlock(ABC):
    def __call__(self, x: np.ndarray)-> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray)-> np.ndarray:
        """Calculate forward pass"""
        pass

    @abstractmethod
    def parameters(self)-> list[tuple[np.ndarray, np.ndarray]]:
        """Return parameters of the block"""

    @abstractmethod
    def backward(self, dLdy: np.ndarray)-> np.ndarray:
        """Calculate backward pass"""
        pass