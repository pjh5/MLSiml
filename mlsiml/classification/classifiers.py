import logging
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from mlsiml.classification.workflow import WorkflowStep
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

def for_random_forest(n_estimators=10, criterion="gini", max_features="sqrt",
        **kwargs):
    """Random Forest Classifier.

    Parameters
    ----------
    n_estimators : Number of trees in forest
    criterion : Function to measure quality of tree. Default is the gini index.
    max_features : The number of features to consider when looking for the best
                    split.  default is sqrt(n_features)
    """
    kwargs['n_estimators'] = n_estimators
    kwargs['criterion'] = criterion
    kwargs['max_features'] = max_features
    return _make_classifier(RandomForestClassifier, "Random Forest", **kwargs)

def for_gaussian_nb(**kwargs):
    """Gaussian Naive Bayes.

    Parameters
    ----------
    priors : Prior probabilities of the classes. If specified the priors are
    not adjusted according to the data.
    """
    return _make_classifier(GaussianNB, "Gaussian Naive Bayes", **kwargs)



##############################################################################
# Classifier Object Definitions                                              #
##############################################################################

class Classifier(WorkflowStep):

    def fit(self, sources):
        """Default behavior is to fit a different classifier on every source"""

        # Only 1 source, use the 1 classifier
        if len(sources) == 1:
            self.workflow.fit(sources[0].X_train, sources[0].Y_train)

        # Multiple sources, train a different classifier for each source
        else:
            logging.info("Automatically converting classifier {!s} to {!s}"
                    + "different classifiers.".format(self, len(sources)))

            self.repeat_classifiers = [clone(self.workflow, safe=True)
                                                    for _ in range(len(sources))]

            # Fit a separate classifier on each source
            for classifier, source in zip(self.repeat_classifiers, sources):
                classifier.fit(source.X_train, source.Y_train)
        return self

    def transform(self, sources):

        # Only 1 source, just predict
        if len(sources) == 1:
            return sources[0].transform_with(self.workflow.transform)

        # Multiple sources, there should be a different classifier for each
        else:
            return [source.transform_with(classifier.transform)
                    for classifier, source in zip(self.repeat_classifiers, sources)]

    def predict(self, sources):

        # Only 1 source, just predict
        if len(sources) == 1:
            return self.workflow.predict(sources[0].X_test)

        # Multiple sources, there should be a different classifier for each
        else:
            return [classifier.predict(source.X_test)
                    for classifier, source in zip(self.repeat_classifiers, sources)]

class CVGridSearchClassifier(Classifier):

    def __init__(self, desc, base_classifier, search_params, **kwargs):

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
        super().__init__(desc, grid_classifier)


    def _evaluate_on(self, X, X_test, Y, Y_test):
        accuracy = super().evaluate_on(X, X_test, Y, Y_test)

        # Update the last record
        # Save the final parameters here since they were only determined after
        # "fit" has been called on this classifier
        for kw, val in self.workflow.best_params_.items():
            self.last_evaluation_record[kw] = val

        # Update the full record
        #######################################################################
        self.full_record = []
        cv_results = self.workflow.cv_results_

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

##############################################################################
# Helper Functions                                                           #
##############################################################################

def _make_classifier(base_classifier, desc, search_params=None, **kwargs):
    if search_params:
        return CVGridSearchClassifier(desc, base_classifier, search_params, **kwargs)

    return Classifier(desc, base_classifier(**kwargs))

