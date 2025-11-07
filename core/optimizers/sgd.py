from typing import Literal, Optional

from .abstract_optimizer import AbstractOptimizer 
from ..models import AbstractModel

class SGD(AbstractOptimizer):
    """Stochastic gradient descent optimizer"""
    def __init__(
        self,
        model: AbstractModel,
        lr: float = 1e-3, 
        lmbda: float = 0.5,
        alpha: float = 0.5,
        reg_type: Optional[Literal["l1", "l2", "elasticnet"]] = None,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(model, lr, lmbda, alpha, reg_type)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity: list = [0]*len(self.model.parameters())
        
    def step(self):
        for i, (p_type, p, g) in enumerate(self.model.parameters()):
            # get regularization grad
            if p_type == "w": 
                grad_reg = self.get_grad_reg(p)
            else:
                grad_reg = 0.0

            # add regularization term
            g += grad_reg

            # calculate new model params
            if not self.nesterov and self.momentum == 0:
                p -= self.lr * g
            else:
                self.velocity[i] = self.lmbda * self.velocity[i] + (1 - self.lmbda) * g
                g = g + self.lmbda * self.velocity[i] if self.nesterov else self.velocity[i]
                p -= self.lr * g