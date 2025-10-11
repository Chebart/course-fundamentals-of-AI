import numpy as np

from .abstract_loss import AbstractLoss

class BCELoss(AbstractLoss):
    """L(y_true, y_pred) = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]"""

    def __init__(self):
        self.eps = 10e-8
        
    def forward(self, y_pred, y_true):
        return -(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log(1 - y_pred + self.eps)) / len(y_true)
        
    def backward(self, y_pred, y_true, model):
        dLdy = (y_pred - y_true) / (y_pred * (1 - y_pred) + self.eps) / len(y_true)
        model.backward(dLdy)