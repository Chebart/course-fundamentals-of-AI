from typing import Any

import numpy as np

class BCELoss():
    """L(y_true, y_pred) = -[y_true * log(y_pred) + (1 — y_true) * log(1 — y_pred)]"""
    def __init__(self):
        self.eps = 10e-8
        
    def __call__(
        self, 
        y_pred: np.ndarray,
        y_true: np.ndarray
    )-> np.ndarray:
        """Calculate forward pass"""
        return -(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log(1 - y_pred + self.eps)) / len(y_true)
        
    def backward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        model: Any
    )-> np.ndarray:
        """Calculate backward pass"""
        dLdy = (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)
        model.backward(dLdy)