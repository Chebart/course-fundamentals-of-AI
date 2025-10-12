import numpy as np

from .abstract_block import AbstractBlock

class Softmax(AbstractBlock):
    """ softmax(x) = exp(x) / sum(exp(x)) """
    
    def forward(self, x):
        self.y = np.exp(x - np.max(x))
        return self.y / np.sum(self.y)
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        dot = np.sum(dLdy * self.y, axis=-1, keepdims=True)
        return self.y * (dLdy - dot)