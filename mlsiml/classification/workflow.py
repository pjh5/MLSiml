"""
Conceptually a sequence of steps in a ML workflow, this is composed of
preprocessing steps, transformation, other workflows, and classifiers (usually
just one classifier at the end).
"""
import re
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score

from mlsiml.generation import dataset
from mlsiml.utils import is_iterable



class Workflow(BaseEstimator, ClassifierMixin):

    def __init__(self, desc, steps, classifier):
        """Makes a new workflow (lowercase)

        Params
        steps   - Array-like of workflows (objects that also conform to this
            interface).
        """
        self.desc = desc
        self.steps = steps
        self.classifier = classifier

        # Build deep params and mangled, since we will always need them
        self.shallow_params = {True: {}, False: {}}
        self.deep_params = {True: {}, False: {}}
        for mangled in [True, False]:
            for step in self.steps:
                self.shallow_params[mangled].update(
                        step.get_params(deep=False, mangled=mangled)
                        )
                self.deep_params[mangled].update(
                        step.get_params(deep=True, mangled=mangled)
                        )
            self.shallow_params[mangled].update(
                    self.classifier.get_params(deep=False, mangled=mangled)
                    )
            self.deep_params[mangled].update(
                    self.classifier.get_params(deep=True, mangled=mangled)
                    )

        logging.debug(
            "Created workflow:\n{!s}\nwith deep parameters:\n{!s}\n\n".format(
            self, self.deep_params[True])
            )

    def get_params(self, deep=True, mangled=True):
        """Returns all params, where "all" is the sklearn definition"""
        if deep:
            return self.deep_params[mangled].copy()
        return self.shallow_params[mangled].copy()

    def fit(self, sources):
        """Propogates X and Y through all of the steps, fitting all of them

        Specifically, fits AND transforms X and Y through the all but the last
        step, and then fits the last step. The X and Y do have to be
        transformed for the fitting of the next layer to make sense.
        """
        for step in self.steps:
            sources = step.fit_transform(sources)
        self.classifier.fit(sources)
        return self

    def predict(self, X):
        """Requires that fit(X, Y) be called first"""
        for step in self.steps:
            X = step.transform(X)
        yhat = self.classifier.predict(X)

        # yhat should be just a single prediction at this point
        if not isinstance(yhat, np.ndarray):
            raise Exception("Workflow did not produce single prediction vector")

        # Unpack the prediction to just be a numpy array
        return yhat

    def evaluate_on(self, sources):

        # Fit and test the classifier on the datasplit
        self.fit(sources)
        y_hat = self.predict(sources)
        y_test = sources[0].Y_test  # all Y_test should be the same for all sources
        accuracy = accuracy_score(y_test, y_hat, normalize=True)

        # Return just the accuracy
        return accuracy

    def __str__(self):
        return "{} {!s} -> {!s}".format(
                self.desc,
                " -> ".join(["[{!s}]".format(step) for step in self.steps]),
                self.classifier
                )

    def __repr__(self):
        return self.__str__()



class WorkflowStep(BaseEstimator, TransformerMixin):
    """
    Conceptually a sequence of steps in a ML workflow, this is composed of
    preprocessing steps, transformation, other workflows, and classifiers
    (usually just one classifier at the end).

    This tries to conform to the sklearn classifier interface.
    """

    def __init__(self, desc, workflow, uses_sources=None):
        """Makes a new Workflow with the given steps and parameters

        Params
        ======
        desc    - A human-readable description of this workflow
        workflow    - An object that conforms to this interface, such as a
            sklearn.classifier
        """
        self.desc = desc
        self.uses_sources = uses_sources
        self.workflow = workflow

        logging.debug(
            "Created step in workflow:\n{!s}\nwith params:\n{!s}\n\n".format(
            self, self.get_params())
            )

    def get_params(self, deep=True, mangled=True):
        prefix = self.desc + "_" if mangled else ""
        return {prefix+k : v
                for k,v in self.workflow.get_params(deep=deep).items()}

    def set_params(self, *args, **kwargs):
        """Why do you want to change the parameters. Don't change the
        parameters. It just makes it harder to tell what parameters led to what
        results"""
        raise NotImplementedError()

    def fit(self, sources):
        sources = dataset.use(self.uses_sources, sources)

        # If there are multiple sources to read from then this does not know
        # what to do
        if len(sources) > 1:
            logging.error("{!s} sources for fit on workflow {!s}".format(
                len(sources), self))
            raise Exception("WorkflowStep can not fit {!s} sources".format(
                len(sources)))

        self.workflow.fit(sources[0].X_train, sources[0].Y_train)
        return self

    def transform(self, sources):
        sources = dataset.use(self.uses_sources, sources)
        return [d.transform_with(self.workflow.transform) for d in sources]

    def detail_str(self, deep=True):
        return "{}({})".format(self.desc,
                ", ".join(["{!s}={!s}".format(k, v)
                    for k, v in self.get_params(deep=deep).items()]))

    def __str__(self):
        return re.sub(r'\s+', ' ', self.workflow.__str__())

    def __repr__(self):
        return self.__str__()


class DoNothing(TransformerMixin):
    """Base class that obeys workflow interface but does nothing"""

    def __init__(self, desc, **kwargs):
        """Do nothing"""
        self.desc = desc
        self.params = kwargs

    def get_params(self, deep=True, mangled=True):
        return self.params.copy()

    def fit(self, sources):
        return self

    def transform(self, sources):
        return sources

    def __str__(self):
        return self.desc
