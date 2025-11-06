import numpy as np

from .abstract_loss import AbstractLoss

class CrossEntropyLoss(AbstractLoss):
    """L(y_true, y_pred) = -sum(y_true_i * log(y_pred_i))"""

    def _convert_labels_to_one_hot(self, num_cls, labels):
        return np.eye(num_cls)[labels]

    def forward(self, y_pred, y_true):
        y_true = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
        return -np.sum(y_true * np.log(y_pred + self.eps)) / y_true.shape[0]
            
    def backward(self, y_pred, y_true):
        y_true = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
        dLdy = -y_true / (y_pred * y_true.shape[0] + self.eps)
        self.model.backward(dLdy)