from .abstract_block import AbstractBlock

class Softmax(AbstractBlock):
    """ softmax(x) = exp(x) / sum(exp(x)) """
    
    def forward(self, x):
        x -= x.max(axis=-1, keepdims=True)
        exp_x = x.exp()
        self.y = exp_x / exp_x.sum(axis=-1, keepdims=True)
        return self.y
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        dot = (dLdy * self.y).sum(axis=-1, keepdims=True)
        return self.y * (dLdy - dot)