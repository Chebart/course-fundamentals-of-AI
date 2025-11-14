from .abstract_block import AbstractBlock

class Tanh(AbstractBlock):
    """tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)"""
    
    def forward(self, x):
        exp_x_plus = x.exp()
        exp_x_minus = (-x).exp()
        self.y = (exp_x_plus - exp_x_minus) / (exp_x_plus + exp_x_minus)
        return self.y
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy * (1 - self.y**2)