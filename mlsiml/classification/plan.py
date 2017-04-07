
from abc import ABCMeta, abstractmethod
from functools import wraps


class Workflow():

    def __init__(self, workflow_steps):
        pass

    def evaluate_on(self, sources):
        pass


class WorkflowStep(metaclass=ABCMeta):
    """Wrapper around transformations that have .transform(X, y) to use sources"""

    def __init__(self, transformers=None, repeat_transformer=None):

        # Exactly only one of transformers or repeat must be specified
        if not transformers and not repeat:
            raise Exception(
                    "Either transformers or repeat_transformer must be specified"
                    )
        if transformers and repeat:
            raise Exception(
                    "Only one of transformers or repeat_transformer must be specified"
                    )

        self.transformers = transformers
        self.repeat_transformer = repeat_transformer
        self.fit = False

    @classmethod
    def from_transformers(cls, transformers):
        return cls(transformers=transformers)

    @classmethod
    def from_repeated_transformer(cls, transformer):
        return cls(repeat_transformer=transformer)

    def fit(self, sources):

        # If there is only one transformer, clone it for every source
        if self.transformers is None:
            self.transformers = [self.repeat_transformer.clone(i)
                    for i in range(len(sources))]


        for transformer in self.transformers:
            transformer.fit(sources)
        return self

    def transform(self, sources):
        for transformer in self.transformers:
            transformer.transform(sources)


class SourceTransform():

    def __init__(self, transform_base, which_sources=None):
        self.transform_base = transform_base
        self.which_sources = which_sources

    def transform(self, sources):
        sources.transform_with(self.which_sources, self.transform_base)

    def clone(self, *, which_sources=None):
        return SourceTransform(
                sklearn.base.clone(self.transform_base, safe=True),
                which_sources=which_sources
                )


class Transformation(metaclass=ABCMeta, BaseEstimator, TransformerMixin):
    """sklearn's transformation interface"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    # fit_transform comes from TransformerMixin


def filter_sources(func):
    """Decorator to replace all sources with just the needed ones"""

    @wraps(func)
    def func_with_filtered_sources(self, sources, *args, **kwargs):
        if self.which_sources:
            func(self, sources[self.which_sources], *args, **kwargs)
        return func(self, sources, *args, **kwargs)
    return func_with_filtered_sources
