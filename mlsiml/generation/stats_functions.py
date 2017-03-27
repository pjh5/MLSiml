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
from mlsiml.generation.transformations import Identity
from mlsiml.utils import make_callable
from mlsiml.utils import replace_key


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
    # Allow more common terminology "mean" and "var"
    replace_key(kwargs, "mean", "loc")
    replace_key(kwargs, "var", "scale")
    replace_key(kwargs, "variance", "scale")

    # Build a nice description string Normal(mean, var)
    loc = kwargs.get("loc", 0)
    scale = kwargs.get("scale", 1)
    desc = "Normal({!s}, {!s})".format(
            "lambda" if not _readable(loc) else loc,
            "lambda" if not _readable(scale) else scale)

    return Distribution(stats.norm, desc, **kwargs)

def Exponential(**kwargs):
    # Allow more common terminology lambda or beta
    if "lambda" in kwargs and "beta" in kwargs:
        raise Exception("Only one of lambda or beta can be specified")
    replace_key(kwargs, "lambda", "scale")
    if "beta" in kwargs:
        kwargs["scale"] = 1 / kwargs["beta"]
        kwargs.pop("beta")

    # Build a nice description string Exp(lambda)
    scale = kwargs.get("scale", 1)
    desc = "Exp({!s})".format("lambda" if _readable(scale) else scale)
    return Distribution(stats.expon, desc, **kwargs)

def Bernoulli(p):
    return Distribution(stats.bernoulli, "Bernoulli(" + str(p) + ")", p=p)

def _readable(thing):
    return isinstance(thing, Identity) or not callable(thing)
