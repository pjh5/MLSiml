"""
Functions to sample from distributions and to make generators that sample from
frozen distributions.
"""
import numpy as np
from scipy import stats

from mlsiml.generation.bayes_networks import Node
from mlsiml.utils import make_callable


class Distribution(Node):

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

