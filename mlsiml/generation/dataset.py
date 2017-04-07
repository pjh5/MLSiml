
from functools import wraps
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from mlsiml.utils import make_iterable


def _validate_params(init):
    """Decorator to validate proper numpy matrices"""

    @wraps(init)
    def valid_init(self, X_train, X_test, Y_train, Y_test, *indices):

        # No matrix is allowed to be none
        if (X_train is None or X_test is None
                or Y_train is None or Y_test is None):
            raise Exception(
                    "Datasets need all of X_train, X_test, Y_train, and Y_test."
                    )

        # Number of rows must match across X and Y
        if (X_train.shape[0] != Y_train.shape[0]
                or X_test.shape[0] != Y_test.shape[0]):
            raise Exception(
                "X sets do not have the same number of rows as Ys. Shapes: "
                + "X_train:{!s}, X_test:{!s}, Y_train:{!s}, Y_test:{!s}".format(
                    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                )

        # Features must match for train and test sets
        if (X_train.shape[1] != X_test.shape[1]):
            raise Exception(
                "Train set does not have same # of columns as test set. Shapes: "
                + "X_train:{!s}, X_test:{!s}, Y_train:{!s}, Y_test:{!s}".format(
                    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                )

        # Ys must be scalars
        if (len(Y_train.shape) != 1 or len(Y_test.shape) != 1):
            raise Exception(
                "Y_train and Y_test must be vectors 1D arrays. Shapes:"
                + "X_train:{!s}, X_test:{!s}, Y_train:{!s}, Y_test:{!s}".format(
                    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                )

        return init(self, X_train, X_test, Y_train, Y_test, *indices)

    return valid_init


class MultiSourceDataset():
    """Representation of a multi-source dataset"""

    @_validate_params
    def __init__(self, X_train, X_test, Y_train, Y_test, split_index):

        # No split_index, then just make a single-source "Multisource"
        if split_index is None:
            self.X_trains = [X_train]
            self.X_tests = [X_test]

        # Otherwise split the data into so many sources
        else:
            split_index = make_iterable(split_index)
            self.X_trains = np.split(X_train, split_index, axis=1)
            self.X_tests = np.split(X_test, split_index, axis=1)

        # All sources have the same labels
        self.Y_train = Y_train
        self.Y_test = Y_test

    @classmethod
    def from_unsplit(cls, X, y, test_size, split_index):
        X_train, X_test, Y_train, Y_test = train_test_split(
                X, y, test_size=test_size
                )
        return cls(X_train, X_test, Y_train, Y_test, split_index)


    def transform_all_with(self, transform):
        """Transforms every source with the same transform, in place"""
        self.X_trains = map(transform, self.X_trains)
        self.X_tests = map(transform, self.X_tests)

        # Return self for pipelining
        return self

    def transform_with(self, transform_list):
        """Transforms every source with its own transform, in place"""

        # Transforms must exactly match up with sources
        if len(transform_list) != len(self):
            raise Exception(
                    ("There are {!s} sources but {!s} transformations. "
                    + "This:{!s}, transform_list:{!s}").format(
                        len(self), len(transform_list), self, transform_list
                        )
                    )

        for i, transform in enumerate(transform_list):
            self.X_trains[i] = transform(self.X_trains[i])
            self.X_tests[i] = transform(self.X_tests[i])

        # Return self for pipelining
        return self

    def combined(self):
        """Concatenates all sources together, still returns a
        MultiSourceDataset and not a Dataset
        """
        return MultiSourceDataset(
                np.hstack(self.X_trains),
                np.hstack(self.X_tests),
                self.Y_train,
                self.Y_test,
                None
                )

    def as_X_and_Y(self):
        """Concatenates everything back to just X and Y. Good for plotting"""
        return (
                np.vstack([np.hstack(self.X_trains), np.hstack(self.X_tests)]),
                np.concatenate([self.Y_train, self.Y_test])
                )

    def is_multisource(self):
        """Returns if this object still contains multi-source data"""
        return self.__len__() > 1

    def test_data(self):
        """Returns (X_test, Y_test) if there is only one source, and throws an
        exception otherwise
        """
        if self.is_multisource():
            raise Exception(
                    "Cannot get test data for multi-source data {!s}".format(
                        self)
                    )
        return self.X_test[0], self.Y_test

    def as_dataset(self):
        """Returns itself as a Dataset object, if there is only one source"""
        if self.is_multisource():
            raise Exception(
                    "Cannot get test data for multi-source data {!s}".format(
                        self)
                    )
        return self[0]


    def __len__(self):
        return len(self.X_trains)

    def __getitem__(self, idx):
        return Dataset(
                self.X_trains[idx],
                self.X_tests[idx],
                self.Y_train,
                self.Y_test
                )

    def __setitem__(self, index, dataset):

        # Can only set with instances of Dataset
        if not isinstance(dataset, Dataset):
            raise Exception(
                    "Cannot set MultiSource[{!s}] with type {!s}".format(
                        index, type(dataset)
                        )
                    )

        self.X_trains[index] = dataset.X_train
        self.X_tests[index] = dataset.X_test

    def __str__(self):
        return "{!s}-Source Dataset: X_trains:{!s}, X_tests:{!s}".format(
                len(self),
                list(map(lambda z: z.shape, self.X_trains)),
                list(map(lambda z: z.shape, self.X_tests))
            )


class Dataset():

    @_validate_params
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def split(self, indices):
        return MultiSourceDataset(
                self.X_train, self.X_test, self.Y_train, self.Y_test, indices
                )

    def transformed_with(self, transform):
        return Dataset(
                transform(self.X_train),
                transform(self.X_test),
                self.Y_train,
                self.Y_test
                )

    def is_multisource(self):
        """Returns if this object still contains multi-source data"""
        return False

    def test_data(self):
        """Returns (X_test, Y_test)"""
        return self.X_test, self.Y_test

    def __str__(self):
        return "Dataset of dims({!s}, {!s}, {!s}, {!s})".format(
                self.X_train.shape,
                self.X_test.shape,
                self.Y_train.shape,
                self.Y_test.shape)

    def __repr__(self):
        return self.__str__()


