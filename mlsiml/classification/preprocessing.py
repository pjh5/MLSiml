
from sklearn.base import TransformerMixin
from sklearn import decomposition

# Distance metrics available from metric_learn package
# Docs can be found at http://all-umass.github.io/metric-learn/index.html
import metric_learn

from mlsiml.generation.workflow import Workflow



def Transform(TransformerMixin):

    def transform(X, Y):
        self.workflow.transform(X, Y)

    def fit_transform(X, Y):
        self.fit(X, Y)
        self.transform(X, Y)


def MetricLearnTransform(self, which, **kwargs):
    if not hasattr(metric_learn, which):
        raise Exception("Metric_learn method " + which + " not found.")
    return Transform(which, getattr(metric_learn, which)(**kwargs), **kwargs)


def PCA(self, n_components=None, **kwargs):
        kwargs["n_components"] = n_components
        return Transform("PCA", decomposition.PCA(**kwargs), **kwargs)


