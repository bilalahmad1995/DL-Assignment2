import numpy as np
from Base import BaseLayer  # Import BaseLayer from your previous implementation

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self._last_input = None

    def forward(self, input_tensor):
        self._last_input = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Gradient of ReLU: 1 for input > 0, 0 otherwise
        return error_tensor * (self._last_input > 0)


