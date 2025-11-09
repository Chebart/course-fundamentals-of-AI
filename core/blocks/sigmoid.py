from .abstract_block import AbstractBlock

class Sigmoid(AbstractBlock):
    """σ(x) = 1 / (1 + e⁻ˣ)"""
    
    def forward(self, x):
        self.y = 1 / (1 + (-x).exp())
        return self.y
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * self.y * (1 - self.y)