from abc import ABC, abstractmethod

import numpy as np

from ..models import AbstractModel

class AbstractLoss(ABC):
    def __init__(self, model: AbstractModel):
        self.eps = 10e-8
        self.model = model

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray)-> np.ndarray:
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray)-> np.ndarray:
        """Calculate forward pass"""
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray)-> None:
        """Calculate backward pass"""
        pass