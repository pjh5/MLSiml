from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network
 
from mlsiml.generation.stats_functions import Distribution
from mlsiml.generation.stats_functions import bernoulli
from mlsiml.generation.structured_generators import make_xor
from mlsiml.generation.structured_generators import make_xor_class_generator

import numpy as np

# Distributions
Normal = Distribution.Normal()
Exp = Distribution.Exponential()


def exponential(
        p=0.5):

    # z, sources
    # Two normal sources, first with 80% of variance
    # First will be  N(y*10 + (1-y)*18, 8)
    # Second will be N(y*0  + (1-y)*2, 2)
    z_layer = NodeLayer.from_function_array([
            Normal.sampler_for(loc=lambda y: y*50 + (1-y)*30, scale=3),
            Normal.sampler_for(loc=lambda y: y*5  + (1-y)*7 , scale=1)
            ], description="Normal")

    # Extra layer to make sure parameters are > 0 for the next layer
    # Note that this has to be np.maximum and not np.max
    abs_layer = NodeLayer.from_vector_function(
                    lambda z: np.maximum(z, 1.1), description="Absolute Value")

    # x, outputs
    # 4 total outputs, two for each source
    # Modeling x in the exponential family
    # 'scale' is the mean of the distribution, 'loc' is an additional shift
    x_layer = NodeLayer.from_function_array([
            Exp.sampler_for(loc=lambda z: z[0], scale=lambda z: z[0]),
            Exp.sampler_for(loc=lambda z: z[0], scale=lambda z: (z**2 - z)[0]),
            Exp.sampler_for(loc=lambda z: z[1], scale=lambda z: (z**2)[1]),
            Exp.sampler_for(loc=lambda z: z[1], scale=lambda z: (3*z**3)[1])
            ], description="Exponential")

    return Network(bernoulli(p), [z_layer, abs_layer, x_layer],
                                                    description="Exponential")

def exp_norm(
        p=0.5,
        num_z=2,
        scale=5,
        var=0.3):

    # z, sources
    z_layer = NodeLayer.from_function_array([
            Exp.sampler_for(scale=lambda y: scale*y + 1)
                for _ in range(num_z)], description="Exponential")

    # x, normal noise
    x_layer = NodeLayer.from_function_array([
        Normal.sampler_for(loc=(lambda i: lambda z: z[i])(x), scale=var)
                                    for x in range(num_z)
                                            ], description="Normal Noise")

    return Network(bernoulli(p), [z_layer, x_layer], description="Exp-Norm")


def xor(
        p=0.5,
        num_z=3,
        num_x_per_z=1,
        var=0.02,
        max_beta=1.0,
        xor_scale=1,
        xor_base=0):

    # Calculate total number of X
    num_x = num_z * num_x_per_z

    # y is just a bernoulli
    y_layer = bernoulli(p)

    # z is a k-dimensional XOR
    z_layer = NodeLayer.from_vector_function(
            make_xor_class_generator(num_z, scale=xor_scale, base=xor_base),
            description="XOR")

    # x are normals on the z
    x_layer = []
    for source in range(num_z):

        # Each x is connected only to its z
        beta = np.ones(num_z) * (1 - max_beta) / float(num_x)
        beta[source] = max_beta

        x_layer += [Normal.sampler_for(
                                    loc=(lambda _b: lambda z: z.dot(_b))(beta),
                                    scale=var
                                    )
                    for x in range(num_x_per_z)]

    x_layer = NodeLayer.from_function_array(x_layer, description="Normal")
 
    return Network(bernoulli(p), [z_layer, x_layer], description="XOR")


def corrupted_xor(
        p=0.5,
        source_corruptions=[0.8],
        xor_dim=5,
        var=0.1
        ):

    # Calculate total number of X
    num_z = len(source_corruptions)
    num_x = num_z * xor_dim

    # Corrupt y with a certain percentage corruption
    corruption_layer = NodeLayer.from_function_array([
                (lambda f: lambda y: y * f())(bernoulli(1 - corruption_level))
                        for corruption_level in source_corruptions
                                                ], description="Corruption")

    # z is a k-dimensional XOR
    z_layer = NodeLayer.from_function_array([
                        (lambda f: lambda y: f(xor_dim, y[0] > 0.5))(make_xor)
                            for _ in range(num_z)
                                            ], description="XOR")

    # x are normals on linear combinations of the z
    x_layer = NodeLayer.from_function_array([
        Normal.sampler_for(loc=(lambda i: lambda z: z[i])(x), scale=var)
                                    for x in range(num_x)
                                            ], description="Normal Noise")


    return Network(bernoulli(p), [corruption_layer, z_layer, x_layer],
                                                description="Jimmy's")

