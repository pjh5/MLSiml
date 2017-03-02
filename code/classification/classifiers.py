import numpy as np
from sklearn import linear_model
from sklearn import svm



# Simple container object
class DataSets:
    """Wrapper class for train/test splits of labelled X,Y data"""
    
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


class Classifier:
    """Wrapper class for sklearn classifiers to expose evaluate_on(datasplit)

    Written just so that a single evaluate_on(datasplit) method can be applied
    to all classification algorithms. This class unifies the interface for
    sklearn, although classifier specific parameters still need to be passed
    into the constructors for each algorithms.

    Supports constructor methods for several sklearn classification algorithms.

    Docstrings of all classifier constructs are copied from the corresponding
    sklearn docs and then edited to only be relevant to binary classification.
    """

    def __init__(self, base_classifier, **kwargs):
        """This probably shouldn't be called directly."""
        self.base_classifier = base_classifier(**kwargs)


    def evaluate_on(self, datasplit, verbose=False):
        self.base_classifier.fit(datasplit.X_train, datasplit.Y_train)
        y_hat = self.base_classifier.predict(datasplit.X_test)
        
        # Print. Should we do this here?
        print("Error is "
                        + str(classification_accuracy(datasplit.Y_test, y_hat)))
        return y_hat


    @classmethod
    def for_logistic_regression(cls, penalty='l2', solver='liblinear', **kwargs):
        """ Logistic Regression (aka logit, MaxEnt) classifier.

        This class implements regularized logistic regression using the
        'liblinear' library, 'newton-cg', 'sag' and 'lbfgs' solvers. It can
        handle both dense and sparse input.

        The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2
        regularization. The 'liblinear' solver supports both L1 and L2
        regularization.

        Parameters
        ----------
        penalty : str, 'l1' or 'l2', default: 'l2'
            Used to specify the norm used in the penalization. The 'newton-cg',
            'sag' and 'lbfgs' solvers support only l2 penalties.
        max_iter : int, default: 100
            Useful only for the newton-cg, sag and lbfgs solvers.
            Maximum number of iterations taken for the solvers to converge.
        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}, default: 'liblinear'
            Algorithm to use in the optimization problem.
            - For small datasets, 'liblinear' is a good choice, whereas 'sag' is
                faster for large ones.
            - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty.
            Note that 'sag' fast convergence is only guaranteed on features with
            approximately the same scale. You can preprocess the data with a
            scaler from sklearn.preprocessing.
            .. versionadded:: 0.17
               Stochastic Average Gradient descent solver.
        """
        return cls(linear_model.LogisticRegression, **kwargs)

    @classmethod
    def for_linear_svm(cls, dual=False, **kwargs):
        """Linear Support Vector Classification.

        Parameters
        ----------
        loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
            Specifies the loss function. 'hinge' is the standard SVM loss
            (used e.g. by the SVC class) while 'squared_hinge' is the
            square of the hinge loss.
        penalty : string, 'l1' or 'l2' (default='l2')
            Specifies the norm used in the penalization. The 'l2'
            penalty is the standard used in SVC. The 'l1' leads to ``coef_``
            vectors that are sparse.
        dual : bool, (default=False)
            Select the algorithm to either solve the dual or primal
            optimization problem. Prefer dual=False when n_samples > n_features.
        max_iter : int, (default=1000)
            The maximum number of iterations to be run.
        """
        return cls(svm.LinearSVC, **kwargs)

    @classmethod
    def for_svm(cls, kernel="rbf", **kwargs):
        """C-Support Vector Classification.
        The implementation is based on libsvm. The fit time complexity
        is more than quadratic with the number of samples which makes it hard
        to scale to dataset with more than a couple of 10000 samples.

        Parameters
        ----------
        C : float, optional (default=1.0)
            Penalty parameter C of the error term.
        kernel : string, optional (default='rbf')
             Specifies the kernel type to be used in the algorithm.
             It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
             a callable.
             If none is given, 'rbf' will be used. If a callable is given it is
             used to pre-compute the kernel matrix from data matrices; that matrix
             should be an array of shape ``(n_samples, n_samples)``.
        degree : int, optional (default=3)
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.
        gamma : float, optional (default='auto')
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            If gamma is 'auto' then 1/n_features will be used instead.
        coef0 : float, optional (default=0.0)
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
        probability : boolean, optional (default=False)
            Whether to enable probability estimates. This must be enabled prior
            to calling `fit`, and will slow down that method.
        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.
        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.
        """
        return cls(svm.SVC, **kwargs)


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
