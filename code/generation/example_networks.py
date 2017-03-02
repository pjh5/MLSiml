from .bayes_networks import NodeLayer
from .bayes_networks import Network
 
from .stats_functions import Distribution
from .stats_functions import bernoulli
from .structured_generators import make_xor_class_generator

import numpy as np

# Distributions
Normal = Distribution.Normal()
Exponential = Distribution.Exponential()


def example():
    no_noise = lambda: 0

    # y will be 65% 0 and 35% 1
    y_layer = bernoulli(0.65)

    # z, sources
    # Two normal sources, first with 80% of variance
    # First will be  N(y*10 + (1-y)*18, 8)
    # Second will be N(y*0  + (1-y)*2, 2)
    z_dists = [
            Normal.sampler_for(loc=lambda y: y*50 + (1-y)*30, scale=3),
            Normal.sampler_for(loc=lambda y: y*5  + (1-y)*7 , scale=1)
            ]
    z_layer = NodeLayer.from_function_array(z_dists)

    # Extra layer to make sure parameters are > 0 for the next layer
    # Note that this has to be np.maximum and not np.max
    abs_layer = NodeLayer.from_vector_function(lambda z: np.maximum(z, 1.1))

    # x, outputs
    # 4 total outputs, two for each source
    # Modeling x in the exponential family
    # 'scale' is the mean of the distribution, 'loc' is an additional shift
    exp_args = ['scale', 'loc']
    x_dists = [
            Exponential.sampler_for(
                            loc=lambda z: z[0], scale=lambda z: z[0]),
            Exponential.sampler_for(
                            loc=lambda z: z[0], scale=lambda z: (z**2 - z)[0]),
            Exponential.sampler_for(
                            loc=lambda z: z[1], scale=lambda z: (z**2)[1]),
            Exponential.sampler_for(
                            loc=lambda z: z[1], scale=lambda z: (3*z**3)[1])
            ]
    x_layer = NodeLayer.from_function_array(x_dists)

    return Network([y_layer, z_layer, abs_layer, x_layer])

def xor_example(
        p=0.5,
        num_z=3,
        num_x_per_z=1,
        base_var=0.02,
        max_beta=1,
        xor_scale=1,
        xor_base=0):

    # Calculate total number of X
    num_x = num_z * num_x_per_z

    # adjust the variance for more x? TODO: ?
    var = np.sqrt(num_x) * base_var

    

    # y will be 65% positive (1) and 35% negative (0)
    y_layer = bernoulli(p)

    # z is a k-dimensional XOR
    z_layer = NodeLayer.from_vector_function(
            make_xor_class_generator(num_z, scale=xor_scale, base=xor_base))
    print("z_layer is " + str(z_layer))

    # x are normals on the z
    x_layer = []
    for source in range(num_z):

        # Each x is connected only to its z
        beta = np.zeros(num_z)
        beta[source] = 1

        x_layer += [Normal.sampler_for(
                                    loc=(lambda _b: lambda z: z.dot(_b))(beta),
                                    scale=var
                                    )
                    for x in range(num_x_per_z)]

    x_layer = NodeLayer.from_function_array(x_layer)
    print("x_layer is " + str(x_layer))

    return Network([y_layer, z_layer, x_layer])
