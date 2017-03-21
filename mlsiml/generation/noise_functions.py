import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Bernoulli


def NormalNoise(var=1):
    return NodeLayer.from_repeated("N(var={!s}) Noise".format(var),
                                            Normal(loc=lambda z: z, scale=var))


class BinaryCorruption(Node):

    def __init__(self, p):
        self.description = str(p) + " Corruption"
        self.bern = Bernoulli(p)

    def sample_with(self, z):
        flip = self.bern()
        return (1 - flip) * z + flip * (1 - z)

def CorruptionLayer(corruption_levels):
    return NodeLayer(str(corruption_levels) + " Corruption",
                    [BinaryCorruption(level) for level in corruption_levels])


class ExtraNoiseNodes(NodeLayer):

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

