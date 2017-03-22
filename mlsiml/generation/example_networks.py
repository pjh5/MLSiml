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
    from mlsiml.utils import flatten
    return Network("Simple Shells",
            Bernoulli(p), flatten(
            [
                NodeLayer("Shells", Shells(dim, radii=lambda z: z+1)),
                NormalNoise(var=var),
                [PlaneFlip(dim=dim) for _ in range(flips)],
                ExtraNoiseNodes(extra_noise)
            ]))

