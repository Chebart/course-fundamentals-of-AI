from .abstract_block import AbstractBlock
from core.data import Tensor

class LeakyReLU(AbstractBlock):
    """LeakyReLU(x) = max(0,x) + negative_slope ∗ min(0,x)"""

    def __init__(self, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope
    
    def forward(self, x):
        self.x = x
        x_max = Tensor.maximum(0, self.x, dtype=self.x.dtype, device=self.x.device)
        x_min = Tensor.minimum(0, self.x, dtype=self.x.dtype, device=self.x.device)
        return x_max + self.negative_slope * x_min
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * ((self.x > 0) + self.negative_slope * (self.x <= 0))