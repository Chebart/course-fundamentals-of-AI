from .abstract_loss import AbstractLoss
from core.data import Tensor

class CrossEntropyLoss(AbstractLoss):
    """L(y_true, y_pred) = -sum(y_true_i * log(y_pred_i))"""

    def _convert_labels_to_one_hot(self, num_cls, labels):
        return Tensor.eye(num_cls, dtype=labels.dtype, device=labels.device)[labels]

    def forward(self, y_pred, y_true):
        y_true = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
        return -(y_true * (y_pred + self.eps).log()).sum() / y_true.shape[0]
            
    def backward(self, y_pred, y_true):
        y_true = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
        dLdy = -y_true / (y_pred * y_true.shape[0] + self.eps)
        dLdy = self.model.backward(dLdy)
        return dLdy