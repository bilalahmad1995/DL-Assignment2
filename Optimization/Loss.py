import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Avoid division by zero and log(0)
        epsilon = 1e-15
        self.prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        self.label_tensor = label_tensor

        # Compute cross entropy loss
        cross_entropy = -np.sum(label_tensor * np.log(self.prediction_tensor))
        return cross_entropy / label_tensor.shape[0]  # Average over batch

    def backward(self, label_tensor):
        # Compute gradient of cross entropy loss
        batch_size = label_tensor.shape[0]
        self.label_tensor = label_tensor
        return (self.prediction_tensor - label_tensor) / batch_size

# Your other imports and code here
