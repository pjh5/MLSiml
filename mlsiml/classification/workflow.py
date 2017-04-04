"""
Conceptually a sequence of steps in a ML workflow, this is composed of
preprocessing steps, transformation, other workflows, and classifiers (usually
just one classifier at the end).
"""
import re
import logging
logging.basicConfig(level=logging.DEBUG)


from sklearn.base import BaseEstimator

from mlsiml.utils import is_iterable


class Workflow(BaseEstimator):
    """
    Conceptually a sequence of steps in a ML workflow, this is composed of
    preprocessing steps, transformation, other workflows, and classifiers
    (usually just one classifier at the end).

    This tries to conform to the sklearn classifier interface.
    """

    def __init__(self, desc, workflow):
        """Makes a new Workflow with the given steps and parameters

        Params
        ======
        desc    - A human-readable description of this workflow
        workflow    - An object that conforms to this interface, such as a
            sklearn.classifier
        """
        self.desc = desc

        # Try to identify lists of workflows to convert to a sequence workflow
        if not hasattr(workflow, "fit") and is_iterable(workflow):
            logging.info("IS  iterable: {!s} is iterable".format(workflow))
            raise Exception("No")
            self.workflow = _WorkflowSequence(workflow)

        else:
            logging.info("NOT iterable: {!s}".format(workflow))
            self.workflow = workflow

    def get_params(self, deep=True, mangled=True):
        return self.workflow.get_params(deep=deep)

    def set_params(self, *args, **kwargs):
        """Why do you want to change the parameters. Don't change the
        parameters. It just makes it harder to tell what parameters led to what
        results"""
        raise NotImplementedError()

    def fit(self, X, Y):
        self.workflow.fit(X, Y)

    def detail_str(self, deep=True):
        return "{}({})".format(self.desc,
                ", ".join(["{!s}={!s}".format(k, v)
                    for k, v in self.get_params(deep=deep).items()]))

    def __str__(self):
        return re.sub(r'\s+', ' ', self.workflow.__str__())

    def __repr__(self):
        return self.__str__()


class _WorkflowSequence():

    def __init__(self, steps):
        """Makes a new workflow (lowercase)

        Params
        steps   - Array-like of workflows (objects that also conform to this
            interface).
        """
        self.steps = steps

        # Build deep params, since we will always need them
        self.params = {}
        self.deep_params = {}
        for step in self.steps:
            self.params.update(step.get_params(deep=False))
            self.deep_params.update(step.get_params(deep=True))

    def get_params(self, deep=True):
        return self.deep_params.copy() if deep else self.params.copy()

    def fit(self, X, Y):
        """Propogates X and Y through all of the steps, fitting all of them

        Specifically, fits AND transforms X and Y through the all but the last
        step, and then fits the last step. The X and Y do have to be
        transformed for the fitting of the next layer to make sense.
        """
        for step in self.steps[:-1]:
            X, Y = step.fit_transform(X, Y)
        self.steps[-1].fit(X, Y)
        return self

    def fit_transform(self, X, Y):
        for step in self.steps:
            X, Y = step.fit_transform(X, Y)
        return X, Y

    def transform(self, X, Y):
        """Requires that fit(X, Y) or fit_transform(X, Y) be called first"""
        for step in self.steps:
            X, Y = step.transform(X, Y)
        return X, Y


class UnsupervisedWrapper(Workflow):

    def fit(self, X, Y):
        self.workflow.fit(X)
        return self

    def transform(self, X, Y):
        return self.workflow.transform(X), Y

