import numpy as np

from .abstract_loss import AbstractLoss

class HingeLoss(AbstractLoss):
    """L = max(0, 1 - y_true * preds)"""     

    def forward(self, y_pred, y_true):
        margins = y_pred * y_true
        return np.maximum(0, 1 - margins) / len(margins)
        
    def backward(self, y_pred, y_true, model):
        """Calculate backward pass"""
        margins = y_true * y_pred
        dLdy = np.where(1 - margins > 0, -y_true, 0) / len(margins)
        model.backward(dLdy)