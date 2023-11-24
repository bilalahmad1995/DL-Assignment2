class BaseLayer:
    def __init__(self):
        """
        Initializes the BaseLayer with basic properties.
        """
        self.trainable = False
        self.weights = None

    # You might also want to define placeholder methods for forward and backward operations.
    # These methods can be overridden by subclasses.

    def forward(self, input_tensor):
        """
        Forward pass for the layer.
        This method should be overridden by subclasses.

        :param input_tensor: The input tensor for the layer.
        :return: Output tensor after applying the layer operation.
        """
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, error_tensor):
        """
        Backward pass for the layer.
        This method should be overridden by subclasses.

        :param error_tensor: The error tensor from the previous layer in the network.
        :return: The gradient tensor to pass to the previous layer.
        """
        raise NotImplementedError("Backward method not implemented.")
