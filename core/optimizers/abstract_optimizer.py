from typing import Literal, Optional
from abc import ABC, abstractmethod

import numpy as np

from ..models import AbstractModel

class AbstractOptimizer(ABC):
    def __init__(
        self, 
        model: AbstractModel,
        lr: float = 1e-3, 
        lmbda: float = 0.5,
        alpha: float = 0.5,
        reg_type: Optional[Literal["l1", "l2", "elasticnet"]] = None
    ):
        """
        Args:
            model (AbstractModel): Model with trainable parameters
            lr (float): Learning rate
            lmbda (float): Regularization coefficient
            alpha (float): ElasticNet mixing parameter
            reg_type (Optional[Literal["l1", "l2", "elasticnet"]]): Regularization type
        """
        self.model = model
        self.lr = lr
        self.reg_type = reg_type
        self.lmbda = lmbda
        self.alpha = alpha

    def get_grad_reg(self, w: np.ndarray)-> np.ndarray|float:
        """Get gradient of regularization term"""
        if self.reg_type == "l2":
            return self.lmbda * w
        elif self.reg_type == "l1":
            return self.lmbda * np.sign(w)
        elif self.reg_type == "elasticnet":
            return self.lmbda * (self.alpha * np.sign(w) + (1 - self.alpha) * w)
        else:
            return 0.0

    def zero_grad(self):
        """Reset grad of model params"""
        for _, _, g in self.model.parameters():
            g.fill(0)

    @abstractmethod
    def step(self):
        """Update model params"""
        pass