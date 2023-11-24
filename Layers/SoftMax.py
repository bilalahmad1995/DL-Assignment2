import numpy as np
from Base import BaseLayer  # Import BaseLayer from your previous implementation

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._last_input = None

    def forward(self, input_tensor):
        # Shift input for numerical stability
        shift = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exps = np.exp(shift)
        self._last_input = exps / np.sum(exps, axis=1, keepdims=True)
        return self._last_input

    def backward(self, error_tensor):
        # Compute gradient of SoftMax
        return self._last_input * (error_tensor - np.sum(error_tensor * self._last_input, axis=1, keepdims=True))

# Your other imports and code here
