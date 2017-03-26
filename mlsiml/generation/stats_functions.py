"""
Nodes that sample from statistical distributions.

Example Usage:
=============
network_layer = [
        Normal(loc=lambda z: z+1, scale=4),
        Exponential(scale=lambda z: 2*z)
        ]

The layer defined above will take in the output from a previous 2-dimensional
(2 Nodes) layer and output another 2 dimensional numpy array w. The first
output w[0] will be sampled from a Normal(z[0]+1, 4), and the second output
will be sampled from an Exponential(2*z[1]).

The distributions available are all from sklearn.stats, so available keywords
(like loc and scale) are defined in that documentation. These functions accept
any parameter that the sklearn.stats distribution does, and any parameter can
be specified either as a value (e.g. 4) or as a lambda that will be fed the
output from the previous layer (e.g. lambda z: z+1)
"""
import numpy as np
from scipy import stats

from mlsiml.generation.bayes_networks import Node
from mlsiml.utils import make_callable


class Distribution(Node):
    """A Node that samples from a statistical distribution

    Example Usage:
    =============
    norm_node = Distribution(sklearn.stats.norm, "Normal(4,2)",
                                                                loc=4, scale=2)
    norm_node.sample()          # sample from Normal(4,2)
    norm_node.sample_with(3)    # sample from Normal(4,2). 3 is ignored


    norm_node2 = Distribution(sklearn.stats.norm, "Normal Noise,
                                            var=2", loc=lambda z: z, scale=2)
    norm_node2.sample()         # error
    norm_node2.sample_with(0)   # sample from Normal(0,2)
    norm_node2.sample_with(3)   # sample from Normal(3,2)
    """

    def __init__(self, base, description, **kwargs):
        self.base = base
        self.description = description
        self._params = kwargs

        self.callable_kwargs = {
                kw:make_callable(arg)
                for kw, arg in kwargs.items()
                }

    def sample(self):
        return self.base.rvs(**self._params)

    def sample_with(self, z):
        params = {kw:arg(z) for kw, arg in self.callable_kwargs.items()}
        return self.base.rvs(**params)

    def __call__(self):
        return self.sample()

    def short_string(self):
        return self.description

    def __str__(self):
        return self.description + str(self._params)



def Normal(**kwargs):
    desc = "Normal({!s}, {!s})".format(
                                kwargs.get("loc", 0), kwargs.get("scale", 1))
    return Distribution(stats.norm, desc, **kwargs)

def Exponential(**kwargs):
    desc = "Exp({!s})".format(kwargs.get('scale', 1))
    return Distribution(stats.expon, desc, **kwargs)

def Bernoulli(p):
    return Distribution(stats.bernoulli, "Bern(" + str(p) + ")", p=p)

