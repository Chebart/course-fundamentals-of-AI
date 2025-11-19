from .abstract_loss import AbstractLoss

class MSELoss(AbstractLoss):
    """L(y_true, y_pred) = mean((y_true - y_pred)**2)"""
      
    def forward(self, y_pred, y_true):
        return ((y_pred - y_true)**2).sum() / y_true.shape[0]
        
    def backward(self, y_pred, y_true):
        dLdy = 2 * (y_pred - y_true) / y_true.shape[0]
        dLdy = self.model.backward(dLdy)
        return dLdy