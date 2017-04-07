import logging

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from mlsiml.classification.workflow import SourceClassifier, CVSourceClassifier


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
    return _make_classifier(LogisticRegression, **kwargs)

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
    return _make_classifier(KNeighborsClassifier, **kwargs)

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
    return _make_classifier(svm.LinearSVC, **kwargs)

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
    return _make_classifier(svm.SVC, **kwargs)

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
    return _make_classifier(RandomForestClassifier, **kwargs)

def for_gaussian_nb(**kwargs):
    """Gaussian Naive Bayes.

    Parameters
    ----------
    priors : Prior probabilities of the classes. If specified the priors are
    not adjusted according to the data.
    """
    return _make_classifier(GaussianNB, **kwargs)



##############################################################################
# Classifier constructor
##############################################################################

def _make_classifier(base_classifier, *, search_params=None, **kwargs):
    """Wraps the base_classifier into a SourceClassifier, which can handle
    MultiSourceDataset objects
    """

    # If no search params, make the classifier with kwargs
    if not search_params:
        base_classifier = base_classifier(**kwargs)
        return SourceClassifier(base_classifier)

    # If there are search params, we have to wrap the base_classifier in a
    # GridSearchCV
    else:

        # Now there are search_params, so we must make a GridSearchCV for them
        # Extract parameters of the cv function itself
        cv_kwargs = {}
        for kw in ['scoring', 'fit_params', 'n_jobs', 'pre_dispatch', 'cv']:
            if kw in kwargs:
                cv_kwargs[kw] = kwargs[kw]
                kwargs.pop(kw)

        # Base classifier is now a grid search over the params
        base_classifier = GridSearchCV(
                base_classifier(**kwargs), search_params, **cv_kwargs
                )
        return CVSourceClassifier(base_classifier)

    return SourceClassifier(base_classifier)


