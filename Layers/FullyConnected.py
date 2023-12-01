import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))  # +1 for bias
        self._optimizer = None
        self._gradient_weights = None
        self._last_input = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        self._last_input = np.append(input_tensor, np.ones([input_tensor.shape[0], 1]), axis=1)  # Adding bias term
        return np.dot(self._last_input, self.weights)

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self._last_input.T, error_tensor)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # Calculate and return the error tensor for the previous layer
        return np.dot(error_tensor, self.weights.T)[:, :-1]  # Removing the bias term

# Your other imports and code here
