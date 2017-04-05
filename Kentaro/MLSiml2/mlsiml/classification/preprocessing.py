
# Distance metrics available from metric_learn package
# Docs can be found at http://all-umass.github.io/metric-learn/index.html
import metric_learn
import numpy as np

class Preprocessing:

    def fit(self, X_train, Y_train):
        raise NotImplemented()

    def process(self, X_train, X_test, Y_train, Y_test):
        raise NotImplemented()

    def get_params(self, mangled=False):
        raise NotImplemented()


class MetricLearnTransform(Preprocessing):

    def __init__(self, which, **kwargs):
        self.desc = which
        self._params = kwargs

        # Save which metric_learn submodule to use
        try:
            self.transformer = getattr(metric_learn, which)(**kwargs)
        except AttributeError:
            raise Error("Metric_learn method " + which + " not found.")

    def fit(self, X_train, Y_train, ratio = 0.1):
        size = X_train.shape[0]
        n_samples = int(round(size*ratio))
        sample_axies = np.random.randint(low=0, high=size, size=n_samples)
        self.transformer.fit(X_train[sample_axies,], Y_train[sample_axies,])

    def process(self, X_train, X_test, Y_train, Y_test):
        return (self.transformer.transform(X_train),
                self.transformer.transform(X_test),
                Y_train, Y_test)

    def get_params(self, mangled=False):
        prefix = "metriclearn_" if mangled else ""

        params = {prefix+kw : val for kw, val in self._params.items()}
        params[prefix + "type"] = self.desc

        return params

    def __str__(self):
        return "<MetricTransform {} {!s}>".format(self.desc, self._params)

    def __repr__(self):
        return self.__str__()      


def get_scale_function_for(data, low=-0.9, high=0.9, by_column=True,
        standardize=False):
    """TODO implement this when you add neural networks"""
    raise NotImplemented()
