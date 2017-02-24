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

from .node_functions import ExponentialModel
from .stats_functions import make_normal
from .stats_functions import normal
from .stats_functions import bernoulli
from .structured_generators import make_xor_class_generator


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



def example():
    no_noise = lambda: 0

    # y will be 65% 0 and 35% 1
    y_layer = NodeLayer([Node(lambda x: bernoulli(0.65), no_noise)])

    # z, sources
    # Two normal sources, first with 80% of variance
    # First will be  N(y*10 + (1-y)*18, 8)
    # Second will be N(y*0  + (1-y)*2, 2)
    z_dists = (
            (lambda f: lambda y: f(y*10 + (1-y)*18, 8))(normal),
            (lambda f: lambda y: f((1-y)*2        , 2))(normal))
    z_layer = NodeLayer((Node(f, make_normal(0, 1)) for f in z_dists))

    # x, outputs
    # 4 total outputs, two for each source
    # Modeling x in the exponential family
    # still unsure of how to do this well and randomly
    # right now, coefficients are ignored
    x_dists = (
            ExponentialModel([ 1, 0]),
            ExponentialModel([-1, 1], transform=lambda x: [x[0], x[0]**2]),
            ExponentialModel([0, 1], transform=lambda x: [x[1], x[1]**2]),
            ExponentialModel([ 0, -1])
            )
    x_layer = NodeLayer((Node(f.sample, make_normal(0, 1)) for f in x_dists))

    return BayesNetwork((y_layer, z_layer, x_layer))

def xor_example():
    percent_positive = 0.65
    num_z = 10
    num_x_per_z = 1
    base_var = 0.02
    max_beta = 1

    # Calculate total number of X
    num_x = num_z * num_x_per_z

    # adjust the variance for more x? TODO: ?
    var = np.sqrt(num_x) * base_var

    

    # y will be 65% positive (1) and 35% negative (0)
    y_layer = lambda: bernoulli(percent_positive)

    # z is a k-dimensional XOR
    z_layer = [make_xor_class_generator(num_z)]

    # x are normals on the z
    x_layer = []
    for source in range(num_z):
        zeros = np.ones(num_z) / num_z
        zeros[source] = 1
        x_layer += [(lambda _zeros: lambda z: normal(z.dot(_zeros), var))(zeros)
                                                for x in range(num_x_per_z)]
    xx_layer = [
            lambda z: normal((z).dot([.8, .2])**3, var),
            lambda z: normal((z).dot([.6, .4])**3, var),
            lambda z: normal((z).dot([.4, .6])**3, var),
            lambda z: normal((z).dot([.2, .8])**3, var)
            ]

    net = BayesNetwork([y_layer, z_layer, x_layer])
    return net
