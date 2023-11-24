# Setting up foundation for all the layers

class BaseLayer:
    def __init__(self):
        # initialising boolean member 'Trainable' with false
        self.trainable = False
    def forward(self, input_tensor):
        ''' The forward pass of the layer
        Parameters:
        -input_tensor: The input data to the layer

        Output:
        -output_tensor: The output data FROM the layer
        '''
        # To ensure every inherited-layer must impliment its own forward method 
        raise NotImplementedError("implement forward pass in the derived classes")

    def backward(self, error_tensor):
        ''' The backward pass of the layer.

        Parameters:
        -error_tensor:  The gradient of the loss w.r.t the layer's output

        output:
        -gradient_tensor: The gradient of the Loss w.r.t the layer's input
        '''
        # To ensure every inherited-layer must impliment its own forward method 

        raise NotImplementedError('Implement the Backward pass in the derived classes ')