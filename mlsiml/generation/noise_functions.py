"""
Nodes and NodeLayers to easily add noise to a network.

Example Usage:
=============

"""
import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import RepeatedNodeLayer
from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.transformations import Identity


class NormalNoise(RepeatedNodeLayer):
    """Layer of spherical Normal noise added to every dimension"""

    def __init__(self, var=1):
        super().__init__("Normal(0, {!s}) Noise".format(var),
                Normal(loc=Identity(), scale=var))
        self.var = var

    def __str__(self):
        return "Normal(0, {!s}) Noise".format(self.var)


class BinaryCorruption(Node):
    """N-dimensions -> N-dimensions, all flipped or none flipped"""

    def __init__(self, p):
        self.description = "{:.1%} Corruption".format(p)
        self.bern = Bernoulli(p)

    def sample_with(self, z):
        flip = self.bern()
        return (1 - flip) * z + flip * (1 - z)

class CorruptionLayer(NodeLayer):
    """Returns a NodeLayer of binary corruptions

    Either p or corruption_levels must be defined, but not both. If p is given,
    then every node will have the same chance of corruption. If
    corruption_levels is given, then there will be one node for every
    corruption percentage given in corruption_levels.
    """

    def __init__(self, corruption_levels):
        super().__init__("{!s} Corruption".format(corruption_levels),
                    [BinaryCorruption(level) for level in corruption_levels])

    @classmethod
    def from_constant(cls, p):
        return NodeLayer.from_repeated("{:.0%} Corruption".format(p),
                                                            BinaryCorruption(p))


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

    def __str__(self):
        return "{!s} Extra Noise Nodes {!s}".format(self.dim, self.nodes)

