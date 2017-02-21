import numpy as np
from scipy.stats import expon

class NodeFunction:

    def __init__(self, coefficients, transform=None):

        # Make sure coefficients are np arrays
        # Try to coerce them if they aren't already
        if not isinstance(coefficients, np.ndarray):
            coefficients = np.array(coefficients)
        self.coefficients = coefficients

        # Default transform is the identity
        if not transform:
            transform = lambda x: x
        self.transform = transform

    def lin_combo_with(self, _prev_layer):
        
        # Transform prev_layer
        prev = self.transform(_prev_layer)

        # Make sure prev is an np array
        if not isinstance(prev, np.ndarray):
            prev = np.array(prev)

        # Linearly combine the coefficients with the transformed inputs
        return self.coefficients.dot(prev)

    def sample(self, prev_layer):
        pass


class SimpleLinearModel(NodeFunction):

    def __init__(self, coefficients, transform=None):
        NodeFunction.__init__(self, coefficients, transform)

    def sample(self, prev_layer):
        """Simple linear model with n+1 coefficients.

        A 1 will be concatenated to the beginning of prev_layer for the bias
        coefficient beta_0
        """
        return self.lin_combo_with(prev_layer)

class ExponentialModel(NodeFunction):
    
    def __init__(self, coefficients, f_param=None, transform=None):
        NodeFunction.__init__(self, coefficients, transform)

        # Default f_param is nothing
        if not f_param:
            f_param = lambda *args: 0
        self.f_param = f_param
        self.exp_mod = f_param(self.coefficients)

    def sample(self, prev_layer):
        return expon.rvs(
                loc=self.lin_combo_with(prev_layer) - self.exp_mod)
   

_random_functions= [
        [
            np.log,
            np.sqrt,
            np.square,
            np.exp
            ],
        [
            lambda x,y: max(x, y),
            lambda x,y: min(x, y),
            lambda x,y: x + y,
            lambda x,y: x - y,
            lambda x,y: x * y,
            lambda x,y: x / y,
            lambda x,y: x ** y
            ]
        ]

def random_function(n_params):
    return None

    # Pick a random integer up to (not including) h
    rint = lambda h: scipy.stats.randint.rvs(0, h)

    # Build up a function to eat all arguments
    cur_function = lambda x: x

    while n_params > 1:

        # Pick random functions until all variables are used
        return
