"""
This module contains functions to create structure, high-dimensional, "barely"
separable data. By barely separable, we mean that this will create
N-dimensional data that is separable in N-dimensions but not separable in any
subset of N-1 dimensions. By separable we mean non-overlapping, but not
linearly separable.
"""
import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.utils import make_callable


class XOR(Node):
    """Binary -> N-dimensional binary

    Usage
    ====
    xor = XOR(N)
    xor.sample_with(1)
        => returns N-dimensional binary vector with an even number of 1s
    xor.sample_with(0)
        => returns N-dimensional binary vector with an odd number of 1s
    """

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
    """Scalar Real -> N-dimensional Real

    Usage
    =====
    shell = Shells(N, radii=lambda z: z + y)    # N is an int, y is a scalar
    shell.sample_with(x)                        # x is a scalar
    # => returns N-dimensional point (x + y) distance (Euclidian) to origin
    """

    def __init__(self, dim, radii=None):
        self.description = str(dim) + "D Sphere"

        # Default radii is 'z'
        if not radii:
            radii = lambda z: z

        self.dim = dim
        self.radii = make_callable(radii)
        self._params = {"radii":self.radii, "dim":self.dim}

    def sample(self):
        x = np.random.normal(size=self.dim)
        return x / np.sqrt(np.square(x).sum())

    def sample_with(self, z):
        return self.sample() * self.radii(z)

