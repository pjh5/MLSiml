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
import scipy.stats

from activation_function import SimpleLinearModel

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
        y = result = self.layers[0].sample(None)

        # Sample rest of the inputs
        for i in range(1, len(self.layers)):
            result = self.layers[i].sample(result)

        return (y, result)

    def bulk_sample(self, n_samples):
        
        # Allocate memory
        y = np.zeros(shape=(n_samples, 1))
        X = np.zeros(shape=(n_samples, self.out_dimension))

        # Sample
        for i in range(n_samples):
            y[i], X[i,:] = self.sample()

        return y, X

    def __str__(self):
        """Bulky multi-line string representation, layer by layer"""
        s = ""
        for i, layer in enumerate(self.layers):
            s += "Layer " + str(i) + ": " + str(layer) + "\n"
        return s


def make_normal(mean, var):
    return (lambda f: lambda: f.rvs())(scipy.stats.norm(loc=mean, scale=var))


def normal(mean, var):
    return scipy.stats.norm.rvs(loc=mean, scale=var)

def make_bernoulli(prob):
    return (lambda p: lambda: float(scipy.stats.uniform.rvs() > p))(prob)

def bernoulli(p):
    return float(scipy.stats.uniform.rvs() > p)


def example_network():
    no_noise = lambda: 0

    # y will be 65% 0 and 35% 1
    y_layer = NodeLayer([Node(lambda x: bernoulli(0.65), no_noise)])

    # z, sources
    # Two normal sources, first with 80% of variance
    # First will be  N(y*10 + (1-y)*18, 8)
    # Second will be N(y*0  + (1-y)*2, 2)
    z_dists = (( lambda f: lambda y: f(y*10 + (1-y)*18, 8))(normal),
                (lambda f: lambda y: f((1-y)*2        , 2))(normal))
    z_layer = NodeLayer((Node(f, make_normal(0, 1)) for f in z_dists))

    return GaussianBayesNetwork((y_layer, z_layer))


def summarize(X, y, decimals=4):

    # Assumes y is either 1 or 0
    pos_idxs = np.where(y == 1)[0]
    neg_idxs = np.where(y == 0)[0]

    # Divide dataset into positive and negatives
    Xs = (X[neg_idxs, :], X[pos_idxs, :])
    Ys = (y[neg_idxs], y[pos_idxs])

    # Make format string
    numstr = ", ".join(["{" + str(i) + ":10." + str(decimals) + "f}" for i
                                                        in range(X.shape[1])])

    # Output results
    print("Total number of samples: " + str(len(y)))
    print()
    print(str(len(Ys[1])) + " Positive Samples:")
    print("\tMin   : " + numstr.format( *np.min(Xs[1], axis=0)))
    print("\tMean  : " + numstr.format(*np.mean(Xs[1], axis=0)))
    print("\tMax   : " + numstr.format( *np.max(Xs[1], axis=0)))
    print()
    print("\tStdev : " + numstr.format(*np.sqrt(np.var(Xs[1], axis=0))))
    print("\tVar   : " + numstr.format( *np.var(Xs[1], axis=0)))
    print()

    print(str(len(Ys[0])) + " Negative Samples:")
    print("\tMin   : " + numstr.format( *np.min(Xs[0], axis=0)))
    print("\tMean  : " + numstr.format(*np.mean(Xs[0], axis=0)))
    print("\tMax   : " + numstr.format( *np.max(Xs[0], axis=0)))
    print()
    print("\tStdev : " + numstr.format(*np.sqrt(np.var(Xs[0], axis=0))))
    print("\tVar   : " + numstr.format( *np.var(Xs[0], axis=0)))


