from typing import Literal, Optional

from .abstract_optimizer import AbstractOptimizer 
from ..models import AbstractModel

class Adam(AbstractOptimizer):
    """Adaptive moment estimation optimizer"""
    def __init__(
        self,
        model: AbstractModel,
        lr: float = 1e-3, 
        lmbda: float = 0.5,
        alpha: float = 0.5,
        reg_type: Optional[Literal["l1", "l2", "elasticnet"]] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        """
        Args:
            model (AbstractModel): Model with trainable parameters
            lr (float): Learning rate
            lmbda (float): Regularization coefficient
            alpha (float): ElasticNet mixing parameter
            reg_type (Optional[Literal["l1", "l2", "elasticnet"]]): Regularization type
            beta1 (float): Exponential decay rate for 1st moment
            beta2 (float): Exponential decay rate for 2nd moment
        """
        super().__init__(model, lr, lmbda, alpha, reg_type)
        self.m: list = [0]*len(self.model.parameters())
        self.v: list = [0]*len(self.model.parameters())
        self.beta1 = beta1
        self.beta2 = beta2
        self.t: int = 0
        
    def step(self):
        # Update iteration number
        self.t += 1
        for i, (p_type, p, g) in enumerate(self.model.parameters()):
            # get regularization grad
            if p_type == "w": 
                grad_reg = self.get_grad_reg(p)
            else:
                grad_reg = 0.0

            # add regularization term
            g += grad_reg

            # Update 1st and 2nd moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * (g)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # Do bias correction
            m_corrected = self.m[i] / (1 - self.beta1**self.t)
            v_corrected = self.v[i] / (1 - self.beta2**self.t)

            # update the parameters
            p -= self.lr * (m_corrected / (v_corrected.sqrt() + self.eps))