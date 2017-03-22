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
from mlsiml.utils import flatten
from mlsiml.utils import make_iterable

class Node:
    """A Node in a Bayesian Network.

    The most important method is sample_with(prev_layer), which takes in the
    previous layer (a vector as a numpy array) and outputs a single value OR a
    vector. Note that this may output a vector.
    """

    def __init__(self, f_sample, description):
        self.description = description
        self.f_sample = f_sample

        # Save the most recent sample generated
        self.last_sample = None

    def sample_with(self, prev_layer):
        self.last_sample = self.f_sample(prev_layer)
        return self.last_sample

    def short_string(self):
        return self.description

    def __str__(self):
        return "<" + self.description + " Node>"

class NodeLayer:
    """Basically just an array of Node objects"""

    def __init__(self, description, nodes):
        self.description = description
        self.nodes = make_iterable(nodes)

    @classmethod
    def from_function_array(cls, desc, funcs):
        return cls(desc, [Node(f, "Lambda") for f in make_iterable(funcs)])

    @classmethod
    def from_repeated(cls, description, node):
        return RepeatedNodeLayer(description, node)

    def sample_with(self, prev_layer):
        return np.array([node.sample_with(prev_layer)
                                            for node in self.nodes]).flatten()

    def transform(self, y, z):
        return (y, self.sample_with(z))

    def short_string(self):
        return "<" + self.description + " Layer>"

    def __str__(self):
        return ("<" + self.description + " Layer: [" +
                    '-'.join([n.short_string() for n in self.nodes]) + "]>")

class RepeatedNodeLayer(NodeLayer):

    def __init__(self, description, node):
        self.description = description
        self.node = node

    def sample_with(self, z):
        return np.array([self.node.sample_with(_z) for _z in z]).flatten()

    def __str__(self):
        return "{} Layer: [{} (repeated)]".format(self.description,
                                                    self.node.short_string())

class Network:

    def __init__(self, description, class_generator, layers):
        self.description = description
        self.class_generator = class_generator
        self.layers = flatten(layers)
        self.dims = []

        # Validate dimensions
        y = self.class_generator()
        result = (y, y)
        self.dims = [1]
        for layer in layers:
            result = layer.transform(*result)
            self.dims.append(len(result[1]))


    def sample(self):

        # First layer has no inputs and is assumed to be the desired output
        y = self.class_generator.sample()
        result = (y, y)

        # Sample rest of the inputs
        for layer in self.layers:
            result = layer.transform(*result)

        return result


    def bulk_sample(self, n_samples):
        """Samples the network n_samples times.

        Returns
        -------
        X   :   A n_samples x K numpy matrix of n_samples samples arranged into
                rows, where K is the output dimension of the last layer of the
                network (Note that the last output of the network will always
                be flattened into a 1 dimensional numpy array). These are the
                outputs of the last layer of the network.

        y   :   A 1 dimensional numpy vector of length n_samples. These are the
                corresponding outputs of the first layer of the network (which
                is assumed to be the class label).
        """

        # Allocate memory
        y = np.zeros(shape=(n_samples))
        X = np.zeros(shape=(n_samples, self.dims[-1]))

        # Sample
        for i in range(n_samples):
            y[i], X[i,:] = self.sample()

        return X, y


    def __len__(self):
        return len(self.layers)


    def pretty_string(self):
        """Bulky multi-line string representation, layer by layer"""

        # Header for the network
        s = "<Network " + "-".join([str(d) for d in self.dims]) + "\n"

        # Each layer on another line
        s += "\tLayer 0: " + str(self.class_generator) + "\n"
        for i, layer in enumerate(self.layers):
            s += "\tLayer " + str(i + 1) + ": " + str(layer) + "\n"

        # Extra newline at the end
        s += "\t>\n"

        return s


    def __str__(self):
        return ("<" + self.description + " Network [" +
                "-".join([str(d) for d in self.dims]) + "]>")

