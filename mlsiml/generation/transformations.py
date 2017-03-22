import numpy as np
from mlsiml.utils import make_callable


class Transformation():

    def transform(self, y, z):
        pass


class ClassFlipper(Transformation):

    def __init__(self, flip_predicate):
        self.flip_predicate = flip_predicate

    def transform(self, y, z):
        return int(not y) if self.flip_predicate(z) else y, z


class PlaneFlip(ClassFlipper):

    def __init__(self, dim=None, plane=None):

        # Either one of dim or plane must be specified
        if not dim and not plane:
            raise Exception("PlaneFlip needs either a plane or a dimension")

        # Default plane is random
        if not plane:
            plane = np.random.randn(dim)
        self.plane = plane

        super().__init__(lambda z: z.dot(plane) > 0)

    def __str__(self):
        return "Plane flip at {!s}".format(self.plane)
