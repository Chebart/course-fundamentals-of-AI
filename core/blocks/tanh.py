from .abstract_block import AbstractBlock

class Tanh(AbstractBlock):
    """tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)"""
    
    def forward(self, x):
        self.y = x.tanh()
        return self.y
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * (1 - self.y**2)