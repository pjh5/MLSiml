from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network

from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.stats_functions import Exponential as Exp
from mlsiml.generation.stats_functions import Normal
from mlsiml.generation.stats_functions import Uniform

from mlsiml.generation.noise_functions import BinaryCorruption
from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import CorruptionLayer
from mlsiml.generation.noise_functions import ExtraNoiseNodes
from mlsiml.generation.noise_functions import ExtraNoiseNodesDivide

from mlsiml.generation.geometric_functions import XorVector
from mlsiml.generation.geometric_functions import ShellVector
from mlsiml.generation.geometric_functions import Trig

from mlsiml.generation.transformations import PlaneFlip
from mlsiml.generation.transformations import Shuffle
from mlsiml.utils import Identity

import numpy as np


def exponential(p=0.5, extra_noise=0, **kwargs):
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
    abs_layer = NodeLayer("AbsValue", lambda z: np.maximum(z, 1.1))

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
            ], **kwargs)


def exp_norm(p=0.5, dim=2, scale=5, var=0.3, extra_noise=0, **kwargs):
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
            ], **kwargs)


def xor(p=0.5, dim=3, var=0.2, xor_scale=1, xor_base=0, extra_noise=0, **kwargs):
    """XorVector(dim) + NormalNoise(var)

    Very difficult for dimensions > 9ish, even for SVMs. The default variance
    is usually adequate, and corresponds to almost touching clusters. When
    plotted the clusters will be very clearly separated, but in a way that is
    hard to classify.
    """

    return Network("XOR",
            Bernoulli(p),
            [
                NodeLayer("XOR", XorVector(dim, scale=xor_scale, base=xor_base)),
                NormalNoise(var=var),
                ExtraNoiseNodes(extra_noise)
            ],
            split_indices=dim//2)


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
                NodeLayer.from_repeated("XOR", XorVector(len(corruptions) * xor_dim)),
                NormalNoise(var=var),
                ExtraNoiseNodesDivide(extra_noise, xor_dim)
            ],
            split_indices=(extra_noise + xor_dim*len(corruptions)) // 2
            )

def xor_sine(*, p=0.5,
        xor_corruption=0, xor_dim=2, xor_var=0.1,
        sine_corruption=0, sine_periods=2, sine_margin=1, sine_var=0,
        extra_noise=0
        ):

    # Map sine wave periods to real numbers
    sine_scale = sine_periods * np.pi

    # Corruption layer and x,y
    # 0: sine class
    # 1: x
    # 2: y
    # 3: xor class
    corruptions = NodeLayer("Corruptions", [
                    BinaryCorruption(sine_corruption),
                    Uniform(low=-sine_scale, high=sine_scale),
                    Uniform(low=-sine_scale, high=sine_scale),
                    BinaryCorruption(xor_corruption)
                    ])

    # XOR and sine
    #   ==> (source2_0,1, XOR(xor_dim), x, y, (x+y)*sine(x) + var)
    # 0: sine class
    # 1: sine x
    # 2: sine y
    # 3: (x+y) * sine(x)
    # 4 - 4+xor_dim: XOR
    sine_and_xor = NodeLayer("XOR and Sine", [
        lambda z: z[0],
        lambda z: z[1],
        lambda z: z[2],
        Trig.sine(z_transform=lambda z: z[1], amplitude=lambda z: z[1] + z[2]),
        XorVector(xor_dim, lambda z: z[3] > 0.5)
        ])

    # Normal noise
    # 0: sine x
    # 1: sine y
    # 2: (x+y) * sine(x)
    # 3 - 3+xor_dim: XOR
    normal_noise = NodeLayer("Normal Noise", [
        lambda z: z[1],
        lambda z: z[2],
        Normal(mean=lambda z: z[3] + sine_margin*z[0], var=sine_var),
        Normal(mean=lambda z: z[4], var=xor_var),
        Normal(mean=lambda z: z[5], var=xor_var),
        Normal(mean=lambda z: z[6], var=xor_var)
        ])

    return Network("Corrupted XOR",
            Bernoulli(p),
            [
                corruptions,
                sine_and_xor,
                normal_noise,
                ExtraNoiseNodesDivide(extra_noise, 3)
            ],
            split_indices=3 + extra_noise
            )

def shells(p=0.5, dim=3, var=0.2, flips=0, extra_noise=0, **kwargs):
    return Network("Simple Shells",
            Bernoulli(p),
            [
                NodeLayer("Shells", ShellVector(dim)),
                NormalNoise(var=var),
                [PlaneFlip(dim=dim) for _ in range(flips)],
                ExtraNoiseNodes(extra_noise)
            ],
            split_indices=dim//2
            )




def crosstalk(p=0.5,
        source1_var=0.2, source1_dim=3,
        source2_var=15, source2_dim=3,
        shared_var=0.1, shared_dim=5,
        extra_noise=0, **kwargs):
    """Makes a 2 source network (z1 and z2) with shared information

    Params
    ======
    n1, type1   - Number and type of dimensions that will be made from solely
        z1. If n1 is 3 and type2 is Shells, then z1 3 of the final features
        will be sampled form a 3D shell. n1 is the dimension, type1 should be a
        constructor of a Node.

    n1, type2   - Same as n1, type1 but for source 2

    nshared     - The number of
    """
    total_dim = source1_dim + source2_dim + shared_dim


    # Normal gaussians for the sources, almost touching, z2 var >> z1 var
    z1 = Normal(loc=lambda y: 1 + y, scale=source1_var)
    z2 = Normal(loc=lambda y: 30*(1 + y), scale=source2_var)
    sources = NodeLayer("Sources", [z1, z2])

    # Reposition "shared" and extra_noise dimensions among the 2 sources
    move_to_1 = list(
            range(
                source1_dim + source2_dim,
                int(source1_dim + source2_dim + shared_dim/2)
                ))
    move_to_1.extend(list(
        range(
            source1_dim + source2_dim + shared_dim,
            int(source1_dim + source2_dim + shared_dim + extra_noise/2))))

    return Network("Crosstalk",
            Bernoulli(p),
            [
                sources,
                NodeLayer("Absolute Value",  lambda z: np.abs(z)),
                NodeLayer("Stuff",
                    [
                        ShellVector(source1_dim, radii=lambda z: z[0]),
                        XorVector(source2_dim, make_even=lambda z: z[1] > 50),
                        ShellVector(shared_dim, radii=lambda z: z[0]*z[1])
                    ]),
                NormalNoise(var=shared_var),
                PlaneFlip(dim=total_dim),
                ExtraNoiseNodes(extra_noise),
                Shuffle(to_idx=0, from_indices=move_to_1)
            ], split_indices=int(source1_dim + shared_dim/2 + extra_noise/2))


def validate(**kwargs):
    return Network("Debug Network",
            Bernoulli(0.5),
            [
                NodeLayer("Normal", [
                    Normal(loc=Identity(), scale=0.2),
                    Uniform(),
                    Exp(beta=5)
                    ]),
                NodeLayer("Sine", Trig.sine())
            ], split_indices=1)

def sine(p=.5, periods=2, margin=1, var=0, extra_noise=0):
    """
    Samples from (x+y)*sine(x) + var for x,y in pi*[-periods, periods]

    Total height is 2 * pi * periods
    Soft margin between points is about margin - 2*var

    Params
    ======
    periods - How many periods to plot. Each additional period adds about one
        more big swell on each side. The higher the period, the more complex
        the decision boundary.
    margin  - The margin to have between the two classes. If var is 0, then
        this margin will be exact. Otherwise, the actual margin will be about
        "margin - 2*var". If "margin 2*var > 2 * pi * periods" then the
        decision boundary will become linear
    var - The amount of vertical noise. Note how this noise is added only in
        the direction of the margin (between the two classes).
    extra_noise - How many additional dimensions of pure noise to add. TODO
        right now all extra_noise is added to the second source

    For a wild time, change the 2nd output dimensions (lambda z: z[1]) to
    (lambda z: z[2] - z[1])
    """
    scale = periods * np.pi
    return Network("Debug Network",
            Bernoulli(p),
            [
                NodeLayer("Uniform", [
                    lambda z: z,
                    Uniform(low=-scale, high=scale),
                    Uniform(low=-scale, high=scale),
                    ]),
                NodeLayer("Sine", [
                    lambda z: z[0],
                    lambda z: z[1],
                    lambda z: z[2],
                    Trig.sine(
                        z_transform=lambda z: z[1],
                        amplitude=lambda z: z.sum()
                        )
                    ]),
                NodeLayer("Vertical Normal Noise", [
                    lambda z: z[1],
                    lambda z: z[2],
                    Normal(mean=lambda z: z[3] + margin*z[0], var=var)
                    ]),
                ExtraNoiseNodes(extra_noise)
            ],
            split_indices=(3 + extra_noise) // 2)

def cross_sine(
        p=.5, xmin=-5, xmax=10, ymin=-5, ymax=10, margin=5, var=0, extra_noise=0
        ):
    """
    Max height (max - 0) is the maximum of "(x*sin(x))^2 + margin". Due to the
        periodicity of sine, this may not be where x=xmax

    Soft margin between points is about "margin - 2*var"
        If var=0, then it will be "margin" exactly

    For a wild time, change the 2nd output dimensions (lambda z: z[2]) to
        (lambda z: z[4] - z[3] + z[2] - z[1]) for what looks like curved
        stacked mountains
    """
    # Until the last layer, dim[0] is kept as the class label
    return Network("Debug Network",
            Bernoulli(p),
            [
                NodeLayer("Uniform", [
                    lambda z: z,
                    Uniform(low=xmin, high=xmax),
                    Uniform(low=ymin, high=ymax),
                    ]),
                NodeLayer("Sine", [
                    lambda z: z[0],
                    lambda z: z[1],
                    lambda z: z[2],
                    Trig.sine(
                        z_transform=lambda z: z[1],
                        amplitude=lambda z: z[1]
                        ),
                    Trig.sine(
                        z_transform=lambda z: z[2],
                        amplitude=lambda z: z[2]
                        )
                    ]),
                NodeLayer("Vertical Normal Noise", [
                    lambda z: z[1],
                    lambda z: z[2],
                    Normal(mean=lambda z: z[3]*z[4] + margin*z[0], var=var)
                    ]),
                ExtraNoiseNodes(extra_noise)
            ],
            split_indices=1)

def trig(
        p=.5, xmin=-5, xmax=10, ymin=-5, ymax=10, margin=5, var=0, extra_noise=0
        ):
    """
    Max height (max - 0) is the maximum of "(x*sin(x))^2 + margin". Due to the
        periodicity of sine, this may not be where x=xmax

    Soft margin between points is about "margin - 2*var"
        If var=0, then it will be "margin" exactly

    For a wild time, change the 2nd output dimensions (lambda z: z[2]) to
        (lambda z: z[4] - z[3] + z[2] - z[1]) for what looks like curved
        stacked mountains
    """
    # Until the last layer, dim[0] is kept as the class label
    return Network("Debug Network",
            Bernoulli(p),
            [
                NodeLayer("Uniform", [
                    lambda z: z,
                    Uniform(low=xmin, high=xmax),
                    Uniform(low=ymin, high=ymax),
                    ]),
                NodeLayer("Sine", [
                    lambda z: z[0],
                    lambda z: z[1],
                    lambda z: z[2],
                    Trig.sine(
                        z_transform=lambda z: z[1],
                        amplitude=lambda z: z[1]
                        )
                    * Trig.cosine(
                        z_transform=lambda z: z[2],
                        amplitude=lambda z: 2*z[2] - z[1]
                        )
                    ]),
                NodeLayer("Vertical Normal Noise", [
                    lambda z: z[1],
                    lambda z: z[2],
                    Normal(mean=lambda z: z[3] + margin*z[0], var=var)
                    ]),
                ExtraNoiseNodes(extra_noise)
            ],
            split_indices=1)

