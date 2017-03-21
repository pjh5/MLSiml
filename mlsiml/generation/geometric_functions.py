"""
This module contains functions to create structure, high-dimensional, "barely"
separable data. By barely separable, we mean that this will create
N-dimensional data that is separable in N-dimensions but not separable in any
subset of N-1 dimensions. By separable we mean non-overlapping, but not
linearly separable.

This module makes relatively clean data. Default parameters will make regions
of data that have non-trivial margins between them.
"""
import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.utils import make_callable

class XOR(Node):

    def __init__(self, dim, make_even=None, base=None, scale=None):
        self.description = str(dim) + "D XOR"

        # Default values
        if not make_even:
            make_even = lambda z: z > 0.5
        if not base:
            base = lambda z: 0
        if not scale:
            scale = lambda z: 1

        # Save parameters
        self.dim        = dim
        self.make_even  = make_callable(make_even)
        self.base       = make_callable(base)
        self.scale      = make_callable(scale)

        # Save as _params to be consistent with stats_functions
        self._params = {"make_even":self.make_even, "base":self.base,
                        "scale":self.scale, "dim":self.dim}

    def sample_with(self, z):

        # Choose how many zeros
        if self.make_even(z):
            n_ones = 2 * np.random.randint(0, high=self.dim // 2 + 1)
        else:
            n_ones = 2 * np.random.randint(0, high=(self.dim + 1) // 2) + 1

        return self.scale(z) * np.random.permutation(
                np.concatenate((np.ones(n_ones), np.zeros(self.dim - n_ones)))
                ) + self.base(z)


class Shells(Node):

    def __init__(self, dim, scale=None):
        self.description = str(dim) + "D Sphere"

        # Default scale is 'z'
        if not scale:
            scale = lambda z: z

        self.dim = dim
        self.scale = make_callable(scale)

    def sample(self):
        x = np.random.normal(size=self.dim)
        return x / np.sqrt(np.square(x).sum())

    def sample_with(self, z):
        return self.sample() * self.scale(z)

