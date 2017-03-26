"""
Nodes and NodeLayers to easily add noise to a network.

Example Usage:
=============

"""
import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Bernoulli


def NormalNoise(var=1):
    """Layer that adds normal noise of (symmetrical) variance var"""
    return NodeLayer.from_repeated("N(var={!s}) Noise".format(var),
                                            Normal(loc=lambda z: z, scale=var))


class BinaryCorruption(Node):
    """N-dimensions -> N-dimensions, all flipped or none flipped"""

    def __init__(self, p):
        self.description = "{:.1%} Corruption".format(p)
        self.bern = Bernoulli(p)

    def sample_with(self, z):
        flip = self.bern()
        return (1 - flip) * z + flip * (1 - z)

def CorruptionLayer(p=None, corruption_levels=None):
    """Returns a NodeLayer of binary corruptions

    Either p or corruption_levels must be defined, but not both. If p is given,
    then every node will have the same chance of corruption. If
    corruption_levels is given, then there will be one node for every
    corruption percentage given in corruption_levels.
    """

    # Exactly 1 of p or corruption_levels must be defined
    if not p and not corruption_levels:
        Error("Either p or corruption_levels must be specified.")
    if p and corruption_levels:
        Error("p and corruption_levels cannot both be specified.")

    # Given a p, repeat the binary corruption node
    if p:
        return NodeLayer.from_repeated("{:.1%} Corruption".format(p),
                                                        BinaryCorruption(p))

    # Given an array of corruption percentages, make a separate node for each
    return NodeLayer(str(corruption_levels) + " Corruption",
                    [BinaryCorruption(level) for level in corruption_levels])


class ExtraNoiseNodes(NodeLayer):
    """Layer that adds additional dimensions of Normal noise

    Example Usage:
    ==============
    network = Network("Example network", [
                    some_layer,
                    ExtraNoiseNodes(4)
                    ])

    In the network above, some_layer will output some vector of dimension K.
    The ExtraNoiseNodes will not alter this vector, but will output another
    vector of length K + 4, where the extra 4 dimensions are samples from
    random Normal distributions.

    Currently, the Normal distributions are sampled from
    Normals(Normal(0,20), Uniform(0,20)).
    """


    def __init__(self, dim, noise_nodes=None):
        self.dim = dim
        self.description = str(dim) + " Extra Noise Dimensions"

        # Default noise is normals with random mean and variances
        if not noise_nodes:
            noise_nodes = [Normal(loc=20*np.random.randn(),
                                  scale=20*np.random.rand())
                            for i in range(dim)]
        self.nodes = noise_nodes

    def sample_with(self, z):
        return np.append(z, [x.sample() for x in self.nodes])

