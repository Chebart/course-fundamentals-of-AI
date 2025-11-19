from .abstract_loss import AbstractLoss

class BCELoss(AbstractLoss):
    """L(y_true, y_pred) = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]"""
      
    def forward(self, y_pred, y_true):
        return -(y_true * (y_pred + self.eps).log() + 
                (1 - y_true) * (1 - y_pred + self.eps).log()) / len(y_true)
        
    def backward(self, y_pred, y_true):
        dLdy = (y_pred - y_true) / (y_pred * (1 - y_pred) + self.eps) / len(y_true)
        dLdy = self.model.backward(dLdy)
        return dLdy