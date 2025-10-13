import numpy as np

from .abstract_block import AbstractBlock

class Reshape(AbstractBlock):
    """Reshape the input to match the specified shape"""
    
    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return x.reshape(self.output_shape)
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy.reshape(self.input_shape)