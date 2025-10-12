import numpy as np

from .abstract_loss import AbstractLoss

class CrossEntropyLoss(AbstractLoss):
    """L(y_true, y_pred) = -sum(y_true_i * log(y_pred_i))"""

    def __init__(self):
        self.eps = 10e-8

    def _convert_labels_to_one_hot(self, labels):
        return np.eye(len(np.unique(labels)))[labels]

    def forward(self, y_pred, y_true):
        y_true = self._convert_labels_to_one_hot(y_true)
        return -np.sum(y_true * np.log(y_pred + self.eps)) / y_true.shape[0]
            
    def backward(self, y_pred, y_true, model):
        y_true = self._convert_labels_to_one_hot(y_true)
        dLdy = -y_true / (y_pred * y_true.shape[0] + self.eps)
        model.backward(dLdy)