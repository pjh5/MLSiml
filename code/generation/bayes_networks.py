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
    """A Node in a Bayesian Network.

    The most important method is sample_with(prev_layer), which takes in the
    previous layer (a vector as a numpy array) and outputs a single value OR a
    vector. Note that this may output a vector.
    """

    def __init__(self, f_sample):
        self.f_sample = f_sample

        # Save the most recent sample generated
        self.last_sample = None 

    def sample_with(self, prev_layer):
        self.last_sample = self.f_sample(prev_layer)
        return self.last_sample

    def __str__(self):
        return "<Node " + str(self.f_sample) + ">"


class VectorNode(Node):
    """A Node that outputs a vector instead of a scalar.

    Only changes __str__. This does not validate that the output is actually a
    vector nor does it change any functionality.
    """

    def __str__(self):
        return ("<VectorNode " + str(self.f_sample) + ">")


class NodeLayer:
    """Basically just an array of Node objects"""

    def __init__(self, nodes):
        self.nodes = nodes

    @classmethod
    def from_function_array(cls, functions):
        return cls([Node(function) for function in functions])

    @classmethod
    def from_vector_function(cls, vector_function):
        return cls([VectorNode(vector_function)])

    @classmethod
    def for_y(cls, y_function):
        return y_function

    def sample_with(self, prev_layer):
        return np.array([node.sample_with(prev_layer)
                                            for node in self.nodes]).flatten()
 
    def __getitem__(self, index):
        return self.nodes[index]

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return str(self.nodes)


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.dims = []

        # Validate dimensions
        result = layers[0]()
        self.dims = [1]
        for layer in layers[1:]:
            result = layer.sample_with(result)
            self.dims.append(len(result))

    def sample(self):

        # First layer has no inputs and is assumed to be the desired output
        y = self.layers[0]()
        result = np.array(y)

        # Sample rest of the inputs
        for layer in self.layers[1:]:
            result = layer.sample_with(result)

        return (y, result)

    def bulk_sample(self, n_samples):
        
        # Allocate memory
        y = np.zeros(shape=(n_samples))
        X = np.zeros(shape=(n_samples, self.dims[-1]))

        # Sample
        for i in range(n_samples):
            y[i], X[i,:] = self.sample()

        return X, y

    def __len__(self):
        return len(self.layers)

    def __str__(self):
        """Bulky multi-line string representation, layer by layer"""

        # Header for the network
        s = "<Network " + "-".join([str(d) for d in self.dims]) + "\n"

        # Each layer on another line
        for i, layer in enumerate(self.layers):
            s += "\tLayer " + str(i) + ": " + str(layer) + "\n"

        # Extra newline at the end
        s += "\t>\n"

        return s


