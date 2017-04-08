"""
Nodes that generate "structured" data, such as N-dimensional XORs or
N-dimensional hyperspheres.

"""
import numpy as np

from mlsiml.generation.bayes_networks import Node
from mlsiml.utils import Identity
from mlsiml.utils import make_callable
from mlsiml.utils import nice_str


class XorVector(Node):
    """Binary -> N-dimensional binary

    Sampling from an N-dimensional XOR is equivalent to sampling from the
    corners of a N-dimensional hypercube (of 0 and 1s), where every other
    corner is of a different class. There are two possible classes, "even"
    (with an even number of 1s) and "odd" (with an odd number of 1s).

    N-dimensional XORs are very very difficult for classification, as they have
    very complex decision boundaries.

    Usage
    ====
    xor = XorVector(N)
    xor.sample_with(1)      # odd xor
    xor.sample_with(0)      # even xor


    xor2 = XorVector(N, make_even=lambda z: z > 0.8)
    xor2.sample_with(0)     # odd xor
    xor2.sample_with(0.7)   # odd xor
    xor2.sample_with(0.8)   # even xor
    xor2.sample_with(1)     # even xor
    """

    def __init__(self, dim, make_even=None, base=None, scale=None):
        self.desc = str(dim) + "D XOR"

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


class ShellVector(Node):
    """Scalar Real -> N-dimensional Real

    Samples from shells, which are circles generalized to higher dimensions.
    When this says that it samples from a hypersphere of radius 4, it means
    that it samples from all points in 4 dimensions that have euclidian
    distance of 4 from the origin.

    Usage
    =====
    shell = ShellVector(N, radii=lambda z: z + 4)    # N is an int
    shell.sample_with(0)    # sample from N-dimensional hypersphere of radius 4
    shell.sample_with(3)    # sample from N-dimensional hypersphere of radius 7
    """

    def __init__(self, dim, radii=None):
        self.desc = str(dim) + "D Sphere"

        # Default radii is 'z'
        if not radii:
            radii = Identity()

        self.dim = dim
        self.radii = make_callable(radii)
        self._params = {"radii":self.radii, "dim":self.dim}

    def sample(self):
        x = np.random.normal(size=self.dim)
        return x / np.sqrt(np.square(x).sum())

    def sample_with(self, z):
        return self.sample() * self.radii(z)


class Trig(Node):
    """N -> N trigonometric transformation of every input

    Implements the function:
        amplitude * trig(frequency * (z_transform(z) - phase_shift)) + shift

    where "trig" is one of {np.sine, np.cosine, np.tangent, etc.}
    """

    def __init__(self, trig, z_transform=None,
                            amplitude=1, shift=0, frequency=1, phase_shift=0):
        """amplitude * trig(frequency * (z_transform(z) - phase_shift)) + shift

        Params
        ======
        trig    - numpy trigonometric function e.g. np.sin
        z_transform - function to apply to z before passing into the
            trigonometric function
        """
        # Default z_transform is the identity
        if not z_transform:
            z_transform = Identity()

        # Save raw params
        self._params = {"z_transform":z_transform,
                "amplitude":amplitude, "shift":shift,
                "frequency":frequency, "phase_shift":phase_shift}

        # Make a nice readable description
        str_params = [nice_str(t[1])
                for t in sorted(self._params.items(), key=lambda t: t[0])]
        self.desc = "{!s} * sin({!s}(z-{!s})) + {!s}".format(*str_params)

        # Store the sampling function
        # For slightest efficiency, store an uncallable _params if possible
        if not any(map(callable, self._params.values())):
            self.f_sample = (
                    lambda t,a,s,f,p: lambda z: a * t(f * (z - p)) + s
                    )(trig, amplitude, shift, frequency, phase_shift)
        else:
            self.f_sample = _make_sample_func(
                    trig,
                    z_transform,
                    make_callable(amplitude),
                    make_callable(shift),
                    make_callable(frequency),
                    make_callable(phase_shift)
                    )

    @classmethod
    def sine(cls, **kwargs):
        return cls(np.sin, **kwargs)

    @classmethod
    def cosine(cls, **kwargs):
        return cls(np.cos, **kwargs)

    @classmethod
    def tangent(cls, **kwargs):
        return cls(np.tan, **kwargs)


def _make_sample_func(trig, z_transform, amp, shift, freq, phase_shift):
    def trig_func(z):
        z = z_transform(z)
        return amp(z) * trig(freq(z) * (z - phase_shift(z))) + shift(z)
    return trig_func

