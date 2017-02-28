import numpy as np
from sklearn import linear_model



# Simple container object
class DataSets:
    
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

class Classifier:

    def __init__(self, base_classifier, **kwargs):
        self.base_classifier = base_classifier(**kwargs)

    @classmethod
    def for_logistic_regression(cls, **kwargs):
        return cls(linear_model.LogisticRegression, **kwargs)

    def evaluate_on(self, datasets, verbose=False):
        self.base_classifier.fit(datasets.X_train, datasets.Y_train)
        y_hat = self.base_classifier.predict(datasets.X_test)
        
        # Print. Should we do this here?
        print("Error is "
                        + str(classification_accuracy(datasets.Y_test, y_hat)))
        return y_hat


def classification_accuracy(y_true, y_hat):
    return 1 - np.sum(np.abs(y_true - y_hat)) / float(y_true.shape[0])

def split_data(X, y, proportion_train=0.7):

    N = X.shape[0]
    n_train = np.floor(proportion_train * N).astype(int)

    # Training indices
    train_bin_idxs = np.random.permutation(np.concatenate(
                    [np.ones(n_train), np.zeros(N - n_train)]))
    train_idxs = train_bin_idxs.astype(bool)
    test_idxs = (-train_bin_idxs + 1).astype(bool)

    # Split
    X_train, Y_train = X[train_idxs, :], y[train_idxs]
    X_test, Y_test = X[test_idxs, :], y[test_idxs]

    return DataSets(X_train, X_test, Y_train, Y_test)
