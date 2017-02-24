"""
This module creates single layer Bayesian networks of the following form:

          w_1_1
        / 
    z_1
  /     \ 
  |       w_1_M1
  /  .
y    .
  \  .
  |       w_k_1
  \     / 
    z_k
        \ 
          w_k_Mk


y       The true response. The true class for classification tasks and the true
        value for regression tasks. This could potentially be a vector, but the
        current implementation assumes that it will be a scalar.

k       The number of sources. The z layer is used to control the relative
        importance of every source, which is measured as the relative variance
        of each z_i. Thus each z_i needs at minimum a variance.


"""
import numpy as np

class Node:

    def __init__(self, f_sample, f_noise):
        self.f_sample = f_sample
        self.f_noise = f_noise

        # Last sample will be stored as a tuple(sample, noise)
        self.last_sample = None 

    def sample(self, prev_layer):
        self.last_sample = (self.f_sample(prev_layer), self.f_noise())
        return sum(self.last_sample)

    def __str__(self):
        return "<Node(" + str(self.f_sample) + ", " + str(self.f_noise) + ")>"


class NodeLayer:
    """Basically just an array of Node objects"""

    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, prev_layer):
        return np.array([node.sample(prev_layer) for node in self.nodes])
 
    def __getitem__(self, index):
        return self.nodes[index]

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return str(self.nodes)


class BayesNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self.out_dimension = len(layers[-1])

    def sample(self):

        # First layer has no inputs and is assumed to be the desired output
        y = self.layers[0]()
        result = np.array(y)

        # Sample rest of the inputs
        for layer in range(1, len(self.layers)):
            result = np.array([self.layers[layer][node](result)
                                for node in range(len(self.layers[layer]))])

        return (y, result)

    def bulk_sample(self, n_samples):
        
        # Allocate memory
        y = np.zeros(shape=(n_samples, 1))
        X = np.zeros(shape=(n_samples, self.out_dimension))

        # Sample
        for i in range(n_samples):
            y[i], X[i,:] = self.sample()

        return X, y

    def __str__(self):
        """Bulky multi-line string representation, layer by layer"""
        s = ""
        for i, layer in enumerate(self.layers):
            s += "Layer " + str(i) + ": " + str(layer) + "\n"
        return s


