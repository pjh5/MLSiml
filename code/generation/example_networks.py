from .bayes_networks import Node
from .bayes_networks import NodeLayer
from .bayes_networks import BayesNetwork
 
from .node_functions import ExponentialModel
from .stats_functions import normal_generator
from .stats_functions import normal
from .stats_functions import bernoulli
from .structured_generators import make_xor_class_generator

import numpy as np


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
    z_layer = NodeLayer((Node(f, normal_generator(0, 1)) for f in z_dists))

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
    x_layer = NodeLayer((Node(f.sample, normal_generator(0, 1)) for f in x_dists))

    return BayesNetwork((y_layer, z_layer, x_layer))

def xor_example(
        p=0.65,
        num_z=3,
        num_x_per_z=1,
        base_var=0.25,
        max_beta=1,
        xor_scale=3,
        xor_base=-1):

    # Calculate total number of X
    num_x = num_z * num_x_per_z

    # adjust the variance for more x? TODO: ?
    var = np.sqrt(num_x) * base_var

    

    # y will be 65% positive (1) and 35% negative (0)
    y_layer = lambda: bernoulli(p)

    # z is a k-dimensional XOR
    z_layer = [make_xor_class_generator(num_z, scale=xor_scale, base=xor_base)]

    # x are normals on the z
    x_layer = []
    for source in range(num_z):

        # Each x is connected only to its z
        zeros = np.ones(num_z) / num_z
        zeros[source] = 1
        x_layer += [(lambda _zeros: lambda z: normal(z.dot(_zeros), var))(zeros)
                                                for x in range(num_x_per_z)]

    return BayesNetwork([y_layer, z_layer, x_layer])
