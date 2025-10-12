import numpy as np

from .abstract_block import AbstractBlock

class ReLU(AbstractBlock):
    """ReLU(x)= max(0,x)"""
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, self.x)
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * (self.x > 0)