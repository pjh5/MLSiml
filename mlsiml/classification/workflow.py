"""
Conceptually a sequence of steps in a ML workflow, this is composed of
preprocessing steps, transformation, other workflows, and classifiers (usually
just one classifier at the end).
"""
from abc import ABCMeta, abstractmethod
import logging
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score

from mlsiml.generation import dataset
from mlsiml.utils import dict_prefix, filter_truish, is_iterable


class WorkflowIterableBase():
    """Base class to Workflow and WorkflowStep to define get_params for them"""

    def __init__(self, iterable):
        self._iterable = iterable

        # Mangle all deep params with their index
        self.deep_params = {True:{}, False:{}}
        for deep in [True, False]:
            for idx, obj in enumerate(iterable):
                self.deep_params[deep].update(
                        dict_prefix(idx, obj.get_params(deep=deep))
                        )

    def get_params(self, deep=True):
        """Returns all params, where "all" is the sklearn definition"""
        return self.deep_params[deep]

    def get_cv_params(self):
        """Returns all cv_results_ (mangled) from any of the steps"""
        out = {}
        for obj in self._iterable:
            out.update(obj.get_cv_params())
        return filter_truish(out)



##############################################################################
# Workflow, sequence of preprocessing steps + final classifier
#   MultiSourceDataset -> y_hat and accuracy_score
##############################################################################

class Workflow(WorkflowIterableBase, BaseEstimator):

    def __init__(self, desc, num_sources, steps, classifier):
        """Makes a new workflow (lowercase)

        Params
        steps   - Array-like of workflows (objects that also conform to this
            interface).
        """

        # Validate that WorkflowSteps are valid
        ######################################################################
        # Keep track of the number of sources and make sure it goes to 1
        # Automatically cast all SourceTransforms to repeated WorkflowSteps
        newsteps = []
        cur_sources = num_sources
        for i, step in enumerate(steps):

            # Existing workflow steps are kept as steps
            if isinstance(step, RawWorkflowStep):
                newstep = step

            # Automatically wrap source transform in steps
            elif isinstance(step, RawSourceTransform):
                logging.info(
                        "Automatically wrapping {!s} in a WorkflowStep".format(
                            step
                            )
                        )
                newstep = WorkflowStep.repeated(step, cur_sources)

            # Don't know what to do otherwise
            else:
                raise Exception(
                        ("Workflow step # {!s} is not a known type of step. "
                        + "Step is {!s} of type {!s}").format(
                            i, step, type(step)
                            )
                        )

            # Now newstep must be a valid workflow step
            cur_sources = newstep.num_output_sources(cur_sources)
            newsteps.append(newstep)

        # There must be only 1 source left for the classifier
        if cur_sources > 1:
            raise Exception(
                    ("There are still {!s} sources left after all of the "
                    + "workflow steps. Workflow is \n{!s}").format(
                        cur_sources, self
                        )
                    )

        # Init succesfully
        ######################################################################

        super().__init__(newsteps + [classifier])
        self.desc = desc
        self.steps = newsteps
        self.classifier = classifier

        logging.debug(
            "Created workflow:\n{!s}\nwith deep parameters:\n{!s}\n\n".format(
            self, self.deep_params[True])
            )

    def fit(self, sources):
        """Propogates X and Y through all of the steps, fitting all of them

        Specifically, fits AND transforms X and Y through the all but the last
        step, and then fits the last step. The X and Y do have to be
        transformed for the fitting of the next layer to make sense.
        """
        for step in self.steps:
            sources = step.fit(sources).transform(sources)

        # By now we require that previous workflow steps have resulted in only
        # a single dataset
        self.classifier.fit(sources.as_dataset())
        return self

    def predict(self, sources):
        """Requires that fit(sources) be called first"""

        # Propogate sources through all the workflow steps
        for step in self.steps:
            sources = step.transform(sources)

        # Send the propogated source into the classifier
        yhat = self.classifier.predict(sources.as_dataset())

        # yhat should be just a single prediction at this point
        if not isinstance(yhat, np.ndarray) or len(yhat.shape) != 1:
            raise Exception(
                    "Workflow did not produce single prediction vector. "
                    + "Prediction is {!s}".format(yhat)
                    )
        return yhat

    def evaluate_on(self, sources):
        """Fits, then predicts, on the given sources, returning accuracy"""

        # Fit and test the classifier on the datasplit
        self.fit(sources)
        y_hat = self.predict(sources)
        accuracy = accuracy_score(sources.Y_test, y_hat, normalize=True)

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



##############################################################################
# WorkflowStep, one transformation step in a workflow
#   MultiSourceDataset -> MultiSourceDataset
##############################################################################

class RawWorkflowStep(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, sources):
        pass

    @abstractmethod
    def transform(self, sources):
        pass

    @abstractmethod
    def num_output_sources(self, num_input_sources):
        pass

    def get_cv_params(self):
        return {}

class WorkflowStep(WorkflowIterableBase, RawWorkflowStep):
    """A collection of transformations to transform a multisource-dataset with

    This is the default implementation. In this default (which should be used
    whenever possible), every source gets exactly one corresponding
    transformer. In other words, this default WorkflowStep is meant for
    wrapping transformation steps that operate in parallel on each source and
    which do not change the number of sources.
    """

    def __init__(self, transformers):
        super().__init__(transformers)
        self.transformers = transformers

    @classmethod
    def repeated(cls, transformer, repeats):
        return cls([transformer.clone() for _ in range(repeats)])

    def fit(self, sources):
        """Fits every transformer on its corresponding source

        Params:
        sources - MultiSourceDataset object
        """
        for transformer, source in zip(self.transformers, sources):
            transformer.fit(source)
        return self

    def transform(self, sources):
        """Transforms every source with its corresponding transformer

        Params:
        sources - MultiSourceDataset object
        """
        for i, transformer in enumerate(self.transformers):
            sources[i] = transformer.transform(sources[i])
        return sources

    def num_output_sources(self, num_input_sources):
        return num_input_sources


##############################################################################
# SourceTransform, wrapper around sklearn transformers to handle sources
#   .transform(X: np.ndarray) -> .transform(sources: Dataset)
##############################################################################

class RawSourceTransform(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, source):
        pass

    @abstractmethod
    def transform(self, source):
        pass

    def get_cv_params(self):
        return {}


class SourceTransform(RawSourceTransform):
    """Wrapper around a sklearn.transformations for MultiSourceDatasets"""

    def __init__(self, transform_base):
        """Wraps transform_base in a object that can handle MultiSourceDatasets

        Params
        ======
        transform_base  - The sklearn.transform object or an object that
            subclasses MatrixTransformation (the same interface).
        """
        self.transform_base = transform_base
        self.real_name = self.transform_base.__class__.__name__

    def fit(self, source):
        """Fits the wrapped transform to the given dataset source

        Params:
        source  - Dataset object
        """
        return self.transform_base.fit(source.X_train, source.Y_train)

    def transform(self, source):
        """Transforms the given source with the wrapped transformer

        Params:
        source  - Dataset object
        """
        return source.transformed_with(self.transform_base.transform)

    def clone(self):
        """Returns a safe deep copy of this transformer object"""
        return SourceTransform(clone(self.transform_base, safe=True))


    def get_params(self, deep=True):
        """Replace undescriptive "transformer_base" with actual name"""
        return dict_prefix(
                self.real_name, self.transform_base.get_params(deep=deep)
                )

    def get_cv_params(self):
        if hasattr(self.transform_base, "cv_results_"):
            return filter_truish(
                    {self.real_name + "_cv":self.transform_base.cv_results_}
                    )
        return {}

    def __str__(self):
        return "Wrapped({})".format(
                re.sub(r"\s+", " ", str(self.transform_base))
                )


##############################################################################
# SourceClassifier, wrapper around sklearn classifiers to handle sources
#   .predict(X: np.ndarray) -> .predict(sources: MultiSourceDataset)
##############################################################################

class SourceClassifier(SourceTransform):
    """A Classifier that acts on MultiSourceDataset objects

    This is just like a SourceTransform, except that it "transforms" data
    by classifying it with its predict method.

    A SourceTransform is created with a sklearn.classifier object (or another
    object that conforms to the same interface)
    """

    def predict(self, source):
        """Uses the fitted model to predict sources.X_test

        This throws an exception is sources has multiple X_tests, i.e. if
        sources is still a multi-source dataset.

        Params
        ======
        source - A Dataset object
        """
        # TODO re-using self.transform_base is hacky and confusing
        return self.transform_base.predict(source.X_test)

    def transform(self, source):
        return self.predict(source)

class CVSourceClassifier(SourceClassifier):
    """SourceClassifier for GridSearchCVs for better parameter naming"""

    def __init__(self, classifier):
        super().__init__(classifier)
        self.real_name = classifier.estimator.__class__.__name__

    def get_params(self, deep=True):
        params = self.transform_base.estimator.get_params()
        params = {
                re.sub(
                    "GridSearchCV_estimator", self.real_name + "_cv", k
                    ):v for k, v in params.items()
                }
        return params


class MatrixTransformation(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """sklearn's transformation interface"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    # fit_transform comes from TransformerMixin

