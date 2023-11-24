class Sgd:
    def __init__(self, learning_rate: float):
        """
        Initializes the SGD optimizer with the given learning rate.

        :param learning_rate: The learning rate (a float).
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Updates the weights based on the gradient tensor.

        :param weight_tensor: The current weights (a tensor).
        :param gradient_tensor: The gradient of the loss with respect to the weights (a tensor).
        :return: Updated weights (a tensor).
        """
        # Update the weights based on the gradient
        # weight_new = weight_old - learning_rate * gradient
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
