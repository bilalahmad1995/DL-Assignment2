import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Forward pass with ReLu as Activation Function
        self._input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        self._input_tensor = output_tensor
        return output_tensor
    
    def backward(self, error_tensor):
        

        gradient_tensor = np.where(self._input_tensor >0, 1, 1)
        return gradient_tensor