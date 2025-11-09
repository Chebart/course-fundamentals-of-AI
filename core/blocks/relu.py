from .abstract_block import AbstractBlock
from core.data import Tensor

class ReLU(AbstractBlock):
    """ReLU(x)= max(0,x)"""
    
    def forward(self, x):
        self.x = x
        return Tensor.maximum(0, self.x, dtype=self.x.dtype, device=self.x.device)
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * (self.x > 0)