

from mlsiml.generation import dataset
from mlsiml.classification.workflow import DoNothing


class Concatenate(DoNothing):

    def __init__(self):
        super().__init__("Concatenate")

    def transform(self, sources):
        return [dataset.concatenate(sources)]

