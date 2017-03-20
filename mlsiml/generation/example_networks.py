from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network

from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Exponential as Exp
from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.stats_functions import BinaryCorruption
from mlsiml.generation.structured_generators import XOR
from mlsiml.generation.structured_generators import Hypersphere

import numpy as np


def exponential(p=0.5):

    # z, sources
    # Two normal sources, first with 80% of variance
    # First will be  N(y*10 + (1-y)*18, 8)
    # Second will be N(y*0  + (1-y)*2, 2)
    z_layer = NodeLayer("Normal", [
                                Normal(loc=lambda y: y*50 + (1-y)*30, scale=3),
                                Normal(loc=lambda y: y*5  + (1-y)*7 , scale=1)
                                ])

    # Extra layer to make sure parameters are > 0 for the next layer
    # Note that this has to be np.maximum and not np.max
    abs_layer = NodeLayer.from_function_array("Absolute Value",
                                                [lambda z: np.maximum(z, 1.1)])

    # x, outputs
    # 4 total outputs, two for each source
    # Modeling x in the exponential family
    # 'scale' is the mean of the distribution, 'loc' is an additional shift
    x_layer = NodeLayer("Exponential", [
                        Exp(loc=lambda z: z[0], scale=lambda z: z[0]),
                        Exp(loc=lambda z: z[0], scale=lambda z: (z**2 - z)[0]),
                        Exp(loc=lambda z: z[1], scale=lambda z: (z**2)[1]),
                        Exp(loc=lambda z: z[1], scale=lambda z: (3*z**3)[1])
                        ])

    return Network(Bernoulli(p), [z_layer, abs_layer, x_layer],
                                                    description="Exponential")

def exp_norm(p=0.5, num_z=2, scale=5, var=0.3):

    # z, sources
    z_layer = NodeLayer("Exponential", [Exp(scale=lambda y: scale*y + 1)
                                        for _ in range(num_z)])

    # x, normal noise
    x_layer = NodeLayer.from_repeated("Normal Noise", Normal(loc=lambda z: z,
                                                             scale=var))

    return Network(Bernoulli(p), [z_layer, x_layer], description="Exp-Norm")


def xor(p=0.5,
        num_z=3,
        num_x_per_z=1,
        var=0.2,
        max_beta=1.0,
        xor_scale=1,
        xor_base=0):

    # Calculate total number of X
    num_x = num_z * num_x_per_z

    # z is a k-dimensional XOR
    z_layer = NodeLayer("XOR", [XOR(dim=num_z,
                                    make_even=lambda z: z > 0.5,
                                    scale=xor_scale,
                                    base=xor_base)])

    # x are normals on the z
    x_layer = []
    for source in range(num_z):

        # Each x is connected only to its z
        beta = np.ones(num_z) * (1 - max_beta) / float(num_x)
        beta[source] = max_beta

        x_layer += [Normal(loc=(lambda b: lambda z: z.dot(b))(beta), scale=var)
                    for x in range(num_x_per_z)]

    x_layer = NodeLayer("Normal", x_layer)

    return Network(Bernoulli(p), [z_layer, x_layer], description="XOR")


def corrupted_xor(
        p=0.5,
        source_corruptions=[0.2, 0.2],
        xor_dim=2,
        var=0.1
        ):

    # Calculate total number of X
    N_corruptions = len(source_corruptions)
    num_z = N_corruptions * xor_dim
    num_x = num_z

    # Corrupt y with a certain percentage corruption
    # 1 -> N_corruption
    corruption_layer = NodeLayer("Corruption", [BinaryCorruption(level)
                                            for level in source_corruptions])

    # z is a k-dimensional XOR
    # N_corruptions -> num_Z = N_corruptions*xor_dim
    z_layer = NodeLayer.from_repeated("XOR", XOR(dim=num_z,
                                                 make_even=lambda z: z > .5
                                                 ))

    # x are normals on linear combinations of the z
    # num_z -> num_x = num_z
    x_layer = NodeLayer.from_repeated("Normal Noise", Normal(loc=lambda z: z,
                                                             scale=var))


    return Network(Bernoulli(p), [corruption_layer, z_layer, x_layer],
                                                description="Jimmy's")

def spherical(p=0.5, dim=3, var=0.2):

    spheres = NodeLayer("Sphere", [Hypersphere(dim=dim, scale=lambda z: z+1)])
    noise = NodeLayer.from_repeated("Normal Noise", Normal(loc=lambda z: z, scale=var))

    return Network(Bernoulli(p), [spheres, noise])
