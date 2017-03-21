import numpy as np

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from mlsiml.utils import flatten


##############################################################################
# Classifier Constructors                                                    #
##############################################################################

def for_logistic_regression(**kwargs):
    """ Logistic Regression (aka logit, MaxEnt) classifier.

    This class implements regularized logistic regression using the 'liblinear'
    library, 'newton-cg', 'sag' and 'lbfgs' solvers.

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.  Maximum number
        of iterations taken for the solvers to converge.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}, default: 'liblinear'
        Algorithm to use in the optimization problem. For small datasets,
        'liblinear' is a good choice, whereas 'sag' is faster for large ones.
        'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty.  Note that 'sag'
        fast convergence is only guaranteed on features with approximately the
        same scale. You can preprocess the data with a scaler from
        sklearn.preprocessing.
    """
    return _make_classifier(LogisticRegression, "Logistic Regression", **kwargs)

def for_knn(n_neighbors=5, **kwargs):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

    metric : string or DistanceMetric object (default = 'minkowski')
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        See the documentation of the DistanceMetric class for a list of
        available metrics.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    """
    kwargs["n_neighbors"] = n_neighbors
    return _make_classifier(KNeighborsClassifier, "KNN", **kwargs)

def for_linear_svm(**kwargs):
    """Linear Support Vector Classification.

    Parameters
    ----------
    loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
        Specifies the loss function. 'hinge' is the standard SVM loss (used
        e.g. by the SVC class) while 'squared_hinge' is the square of the hinge
        loss.

    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2' penalty is the
        standard used in SVC. The 'l1' leads to ``coef_`` vectors that are
        sparse.

    dual : bool, (default=False)
        Select the algorithm to either solve the dual or primal optimization
        problem. Prefer dual=False when n_samples > n_features

    max_iter : int, (default=1000)
        The maximum number of iterations to be run.
    """
    return _make_classifier(svm.LinearSVC, "Linear SVM", **kwargs)

def for_svm(kernel="rbf", **kwargs):
    """C-Support Vector Classification.
    The implementation is based on libsvm. The fit time complexity is more than
    quadratic with the number of samples which makes it hard to scale to
    dataset with more than a couple of 10000 samples.

    Parameters
    ----------
    C : float, optional (default=1.0) Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.  It must be one
         of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').  Ignored by all
        other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.  If gamma is 'auto'
        then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.  It is only significant in 'poly'
        and 'sigmoid'.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior to
        calling `fit`, and will slow down that method.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    """
    kwargs["kernel"] = kernel
    return _make_classifier(svm.SVC, "SVM", **kwargs)

def for_gaussian_nb(priors=None, **kwargs):
    """Gaussian Naive Bayes.

    Parameters
    ----------
    priors : Prior probabilities of the classes. If specified the priors are
    not adjusted according to the data.
    """
    kwargs['priors'] = priors
    return _make_classifier(GaussianNB, "Gaussian Naive Bayes", **kwargs)

def for_random_forest(n_estimators=10, **kwargs):
    """Random Forest Classifier.

    Parameters
    ----------
    n_estimators : Number of trees in forest
    criterion : Function to measure quality of tree. Default is the gini index.
    max_features : The number of features to consider when looking for the best
                    split.  default is sqrt(n_features)
    """
    kwargs['n_estimators'] = n_estimators
    return _make_classifier(RandomForestClassifier, "Random Forest", **kwargs)



##############################################################################
# Classifier Object Definitions                                              #
##############################################################################


class Classifier:
    """Wrapper class for sklearn classifiers to expose evaluate_on()

    Written just so that a single evaluate_on(X, X_test, Y, Y_test) method can
    be applied to all classification algorithms. This class unifies the
    interface for sklearn, although classifier specific parameters still need
    to be passed into the constructors for each algorithms.

    Docstrings of all classifier constructs are copied from the corresponding
    sklearn docs and then edited to only be relevant to binary classification.
    """

    def __init__(self, base_classifier, description, **kwargs):
        """This probably shouldn't be called directly."""
        self.base_classifier = base_classifier
        self.description = description
        self._params = kwargs
        self.last_evaluation_record = None
        self.preprocessors = []

    def with_preprocessing(self, transform):
        self.preprocessors.append(transform)
        return self # return self for pipelining

    def evaluate_on(self, X, X_test, Y, Y_test):

        # Preprocess the data
        for preprocess in self.preprocessors:
            preprocess.fit(X, Y)
            X, X_test, Y, Y_test = preprocess.process(X, X_test, Y, Y_test)

        # Fit and test the classifier on the datasplit
        self.base_classifier.fit(X, Y)
        y_hat = self.base_classifier.predict(X_test)
        accuracy = classification_accuracy(Y_test, y_hat)

        # Save record of this evaluation
        self.last_evaluation_record = self.get_params(deep=True).copy()
        self.last_evaluation_record['accuracy'] = accuracy

        # Return just the accuracy
        return accuracy

    def record_for_last_fit(self):
        self.last_evaluation_record

    def get_params(self, deep=False, original=False):
        params = self._params.copy()

        if not original:
            params['classifier_description'] = self.description

        # Include preprocesssing params if needed
        if deep:
            for preprocessor in self.preprocessors:
                params.update(preprocessor.get_params(mangled=True))

        return params

    def __str__(self):
        params = self.get_params(original=True)
        pres = self.preprocessors
        return "{} {!s} {!s}".format(self.description,
                                     params if params else "",
                                     pres if pres else "")

    def __repr__(self):
        return self.__str__()


class CVGridSearchClassifier(Classifier):

    def __init__(self, base_classifier, search_params, description, **kwargs):
        self._search_params = search_params
        self.full_record = None

        # Extract parameters of the cv function itself
        self.cv_kwargs = {}
        for kw in ['scoring', 'fit_params', 'n_jobs', 'pre_dispatch', 'cv']:
            if kw in kwargs:
                self.cv_kwargs[kw] = kwargs[kw]
                kwargs.pop(kw)

        # Base classifier is now a grid search over the params
        grid_classifier = GridSearchCV(
                base_classifier(**kwargs), search_params, **self.cv_kwargs)

        # Now init like normal with the GridSearchCV classifier
        super().__init__(grid_classifier, description, **kwargs)


    def evaluate_on(self, X, X_test, Y, Y_test):
        accuracy = super().evaluate_on(X, X_test, Y, Y_test)

        # Update the last record
        # Save the final parameters here since they were only determined after
        # "fit" has been called on this classifier
        for kw, val in self.base_classifier.best_params_.items():
            self.last_evaluation_record[kw] = val

        # Update the full record
        #######################################################################
        self.full_record = []
        cv_results = self.base_classifier.cv_results_

        # Extract parameters that were changed
        cv_params = [p[6:] for p in cv_results.keys() if p.startswith('param_')]

        # Loop over all cross validation runs
        current_params = self.get_params(deep=True)
        for cv_run in range(len(cv_results['mean_test_score'])):
            record = current_params.copy()

            # Add the cross validated parameters to the record
            for cv_param in cv_params:
                record[cv_param] = cv_results['param_' + cv_param][cv_run]

            # Add 'accuracy' to the record and save it
            record['CV_mean_accuracy'] = cv_results['mean_test_score'][cv_run]
            self.full_record.append(record)

        return accuracy

    def get_params(self, **kwargs):
        params = super().get_params(**kwargs)

        # Add CV search params, which aren't included in the base get_params()
        for kw, val in self._search_params.items():

            # If this classifier has been fit, then parameters have been chosen
            if self.full_record:
                params[kw] = self.base_classifier.best_params_[kw]

            # Otherwise, the values are still unspecified
            else:
                params[kw] = "UNSPECIFIED"

        return params

    def __str__(self):
        return "{} {!s}".format(super().__str__(),
                self.cv_kwargs if self.cv_kwargs else "")


##############################################################################
# Public Helper Functions                                                    #
##############################################################################

def classification_accuracy(y_true, y_hat):
    return 1 - np.sum(np.abs(y_true - y_hat)) / float(y_true.shape[0])


##############################################################################
# Helper Functions                                                           #
##############################################################################

def _make_classifier(base_classifier, description, search_params=None,
                                                                    **kwargs):
    if search_params:
        return CVGridSearchClassifier(
                        base_classifier, search_params, description, **kwargs)
    # No search params
    return Classifier(base_classifier(**kwargs), description)

