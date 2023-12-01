# defining the class Sgd for Stochastic Gradient Descent
class Sgd:
    def __init__(self,learning_rate ):
        self.learning_rate = learning_rate
# Ti implement basic gradient descent update scheme
    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight