import numpy as np

class Sigmoid():
    """σ(x) = 1 / (1 + e⁻ˣ)"""
    def __call__(
        self, 
        x: np.ndarray,
    )-> np.ndarray:
        """Calculate forward pass"""
        self.y = 1 / (1 + np.exp(-x))
        return self.y
        
    def parameters(self):
        return []

    def backward(
        self,
        dLdy: np.ndarray,
    )-> np.ndarray:
        """Calculate backward pass"""
        return dLdy * self.y * (1 - self.y)