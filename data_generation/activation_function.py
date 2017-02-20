import numpy as np

class ActivationFunction:

    def __init__(self, _coefficients, _transform=None):
        self.coefficients = _coefficients

        # Default transform is the identity
        if not _transform:
            _transform = lambda x: x
        self.transform = _transform

    def linear_combination(self):
        return self.coefficients.dot(np.concatenate(
                        ([1], self.transform(prev_layer))))

    def sample(self, prev_layer):
        pass


class SimpleLinearModel(ActivationFunction):

    def __init__(self, _coefficients, _transform=None):
        ActivationFunction.__init__(_coefficients, _transform)

    def sample(self, prev_layer):
        """Simple linear model with n+1 coefficients.

        A 1 will be concatenated to the beginning of prev_layer for the bias
        coefficient beta_0
        """

class ExponentialModel(ActivationFunction):
    
    def __init__(self, _coefficients, f_param, _transform=None):
        ActivationFunction.__init__(_coefficients, _transform)
        self.f_param = f_param
        self.exp_mod = f.param(self.coefficients)

    def sample(self, prev_layer):
        return scipy.stats.exp.rvs(loc=self.linear_combination(), scale=self.exp_mod)
        
