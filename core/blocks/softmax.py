import numpy as np

from .abstract_block import AbstractBlock

class Softmax(AbstractBlock):
    """ softmax(x) = exp(x) / sum(exp(x)) """
    
    def forward(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        self.y = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.y
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        dot = np.sum(dLdy * self.y, axis=-1, keepdims=True)
        return self.y * (dLdy - dot)