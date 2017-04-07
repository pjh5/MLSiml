

from mlsiml.generation import dataset
from mlsiml.classification.workflow import RawWorkflowStep


class Concatenate(RawWorkflowStep):
    """Concatenates all the sources together into 1"""

    def fit(self, sources):
        """Does nothing"""
        return self

    def transform(self, sources):
        """Concatenates all the sources together into 1"""
        return sources.combined()

    def num_output_sources(self, num_input_sources):
        """This will lead to only 1 source being left"""
        return 1

    def __str__(self):
        return "Concatenate"

