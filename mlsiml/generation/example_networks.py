from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network

from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Exponential as Exp
from mlsiml.generation.stats_functions import Bernoulli

from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import CorruptionLayer
from mlsiml.generation.noise_functions import ExtraNoiseNodes

from mlsiml.generation.geometric_functions import XOR
from mlsiml.generation.geometric_functions import Shells

from mlsiml.generation.transformations import PlaneFlip
from mlsiml.generation.transformations import Identity

import numpy as np


def exponential(p=0.5, extra_noise=0):
    """Two normal 2D clusters (one per class), then fed into Exp

    Difficulty of problem determined by distance between the normal clusters.
    This network is not very interesting. You probably shouldn't use it.
    """

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
    abs_layer = NodeLayer.from_function_array("AbsValue",
                                                lambda z: np.maximum(z, 1.1))

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

    return Network("Exponential",
            Bernoulli(p),
            [
                z_layer,
                abs_layer,
                x_layer,
                ExtraNoiseNodes(extra_noise)
            ])


def exp_norm(p=0.5, dim=2, scale=5, var=0.3, extra_noise=0):
    """Normal(scale*Exponential(Bernoulli()), var)

    Difficulty controlled by scale; smaller is harder. This is still pretty
    hard because the exponentials overlap so much.
    """

    z_layer = NodeLayer("Exponential", [Exp(scale=lambda y: scale*y + 1)
                                        for _ in range(dim)])

    return Network("Exp-Norm",
            Bernoulli(p),
            [
                z_layer,
                NormalNoise(var=var),
                ExtraNoiseNodes(extra_noise)
            ])


def xor(p=0.5, dim=3, var=0.2, xor_scale=1, xor_base=0, extra_noise=0):
    """XOR(dim) + NormalNoise(var)

    Very difficult for dimensions > 9ish, even for SVMs. The default variance
    is usually adequate, and corresponds to almost touching clusters. When
    plotted the clusters will be very clearly separated, but in a way that is
    hard to classify.
    """

    return Network("XOR",
            Bernoulli(p),
            [
                NodeLayer("XOR", XOR(dim, scale=xor_scale, base=xor_base)),
                NormalNoise(var=var),
                ExtraNoiseNodes(extra_noise)
            ])


def corrupted_xor(p=0.5,
                    corruptions=[0.0, 0.0],
                    xor_dim=2,
                    var=0.1,
                    extra_noise=0
                    ):

    return Network("Corrupted XOR",
            Bernoulli(p),
            [
                CorruptionLayer(corruptions),
                NodeLayer.from_repeated("XOR", XOR(len(corruptions) * xor_dim)),
                NormalNoise(var=var),
                ExtraNoiseNodes(extra_noise)
            ])


def shells(p=0.5, dim=3, var=0.2, flips=0, extra_noise=0):
    return Network("Simple Shells",
            Bernoulli(p),
            [
                NodeLayer("Shells", Shells(dim)),
                NormalNoise(var=var),
                [PlaneFlip(dim=dim) for _ in range(flips)],
                ExtraNoiseNodes(extra_noise)
            ])




def crosstalk(p=0.5, source1=None, source2=None, shared=None):
    """Makes a 2 source network (z1 and z2) with shared information

    Params
    ======
    n1, type1   - Number and type of dimensions that will be made from solely
        z1. If n1 is 3 and type2 is Shells, then z1 3 of the final features
        will be sampled form a 3D shell. n1 is the dimension, type1 should be a
        constructor of a Node.

    n1, type2   - Same as n1, type1 but for source 2

    nshared     - The number of
    """
    if not source1:
        source1 = {
                "var":0.2,
                "dim":2
                }
    if not source2:
        source2 = {
                "var":15,
                "dim":3
                }
    if not shared:
        shared = {
                "dim":4
                }


    # Normal gaussians for the sources
    z1 = Normal(loc=lambda y: 1 + y, scale=source1["var"])
    z2 = Normal(loc=lambda y: 30*(1 + y), scale=source2["var"])
    sources = NodeLayer("Sources", [z1, z2])

    return Network("Crosstalk",
            Bernoulli(p),
            [
                sources,
                NodeLayer("Absolute Value",  lambda z: np.abs(z)),
                NodeLayer("Shells",
                    [
                        Shells(source1["dim"], radii=lambda z: z[0]),
                        Shells(source2["dim"], radii=lambda z: z[1]),
                        Shells(shared["dim"], radii=lambda z: z[0] + z[1])
                    ])
            ])

