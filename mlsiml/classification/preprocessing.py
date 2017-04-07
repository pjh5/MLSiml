
from sklearn.base import TransformerMixin
from sklearn import decomposition

# Distance metrics available from metric_learn package
# Docs can be found at http://all-umass.github.io/metric-learn/index.html
import metric_learn

from mlsiml.classification.workflow import SourceTransform

from sklearn.decomposition import PCA as sklearn_pca


class MetricLearn(SourceTransform):

    def __init__(self, which, train_ratio=0.1, **kwargs):
        if not hasattr(metric_learn, which):
            raise Exception("Metric_learn method " + which + " not found.")
        super().__init__(getattr(metric_learn, which)(**kwargs))
        self.train_ratio = train_ratio

    def fit(self, source):
        N = source.X_train.shape[0]
        sample_axies = np.random.randint(
                low=0, high=N, size=int(N * self.train_ratio)
                )
        self.transformer.fit(
                source.X_train[sample_axies,], source.Y_train[sample_axies,]
                )


def PCA(n_components=None, **kwargs):
    kwargs["n_components"] = n_components
    return SourceTransform(sklearn_pca(**kwargs))


