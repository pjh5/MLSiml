import logging
import numpy as np

from mlsiml.utils import is_iterable, make_iterable


class Dataset():

    def __init__(self, X_train, X_test, Y_train, Y_test):
        if X_train is None or X_test is None or Y_train is None or Y_test is None:
            raise Exception("Dataset must have all of X_train, X_test, Y_train, and Y_test")

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def split(self, indices):
        indices = make_iterable(indices)
        splits = (
                np.split(self.X_train, indices, axis=1),
                np.split(self.X_test, indices, axis=1),
                )
        return [Dataset(splits[0][i], splits[1][i], self.Y_train, self.Y_test)
                for i in range(1 + len(indices))]

    def transform_with(self, func):
        return Dataset(func(self.X_train), func(self.X_test), self.Y_train, self.Y_test)


    def __str__(self):
        return "Dataset of dims({!s}, {!s}, {!s}, {!s})".format(
                self.X_train.shape,
                self.X_test.shape,
                self.Y_train.shape,
                self.Y_test.shape)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self


def concatenate(datasets):

    # TODO Make sure they are concatenating the same samples

    return Dataset(
            np.hstack((d.X_train for d in datasets)),
            np.hstack((d.X_test for d in datasets)),
            datasets[0].Y_train, datasets[0].Y_test)

def use(indices, sources):
    logging.debug("datset.use({!s}, {!s})".format(indices, sources))

    # Default is all sources
    if not indices:
        return sources

    # Array-like has to be indexed manually
    if is_iterable(sources):
        return [sources[i] for i in indices]

    # Integer or slice
    return sources[indices]
