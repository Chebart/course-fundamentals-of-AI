from typing import Any

import numpy as np

class HingeLoss():
    """L = max(0, 1 - y_true * preds)"""     
    def __call__(
        self, 
        y_pred: np.ndarray,
        y_true: np.ndarray
    )-> np.ndarray:
        """Calculate forward pass"""
        margins = y_pred * y_true
        return np.maximum(0, 1 - margins) / len(margins)
        
    def backward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        model: Any
    )-> np.ndarray:
        """Calculate backward pass"""
        margins = y_true * y_pred
        dLdy = np.where(1 - margins > 0, -y_true, 0) / len(margins)
        model.backward(dLdy)