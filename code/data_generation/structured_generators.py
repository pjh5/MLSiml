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
import scipy


def make_xor(N, even):

    # Choose how many zeros
    if even:
        n_ones = 2*np.random.randint(0, high=N // 2 + 1)
    else:
        n_ones = 2*np.random.randint(0, high=(N + 1) // 2) + 1

    return 2 * np.random.permutation(
            np.concatenate(
                (np.ones(n_ones), np.zeros(N - n_ones))
                )
            ) + 1

def make_xor_class_generator(N):
    return (lambda f: lambda positive: f(N, positive > 0.5))(make_xor)


def make_xor_random_generator(N, p=0.5):
    return (lambda f: lambda: f(N, np.random.uniform() < p))(make_xor)
