from .abstract_loss import AbstractLoss
from core.data import Tensor

class HingeLoss(AbstractLoss):
    """L = max(0, 1 - y_true * preds)"""     

    def forward(self, y_pred, y_true):
        margins = y_pred * y_true
        return Tensor.maximum(0, 1 - margins, dtype=margins.dtype, device=margins.device) / len(margins)
        
    def backward(self, y_pred, y_true):
        margins = y_true * y_pred
        dLdy = Tensor.where(1 - margins > 0, -y_true, 0, dtype=margins.dtype, device=margins.device) / len(margins)
        self.model.backward(dLdy)