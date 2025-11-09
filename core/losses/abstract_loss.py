from abc import ABC, abstractmethod

from ..models import AbstractModel
from core.data import Tensor

class AbstractLoss(ABC):
    def __init__(self, model: AbstractModel):
        self.eps = 10e-8
        self.model = model

    def __call__(self, y_pred: Tensor, y_true: Tensor)-> Tensor:
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor)-> Tensor:
        """Calculate forward pass"""
        pass

    @abstractmethod
    def backward(self, y_pred: Tensor, y_true: Tensor)-> None:
        """Calculate backward pass"""
        pass