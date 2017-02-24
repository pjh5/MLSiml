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

def xor_generator(N, p=0.5):
    """Returns a function that returns N-dimensional XOR vectors

    Parmameters
    -----------
    N   : number of dimensions
    p   : proportion of positive (even) classes

    Returns
    -------
    TODO write this
    """
    # Sample a N-dimensional binary vector, with an even number of 1s with
    # probability p
    # TODO how do you actually do this?
    return None

