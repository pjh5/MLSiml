import numpy as np

from mlsiml.classification.workflow import WorkflowStep
from mlsiml.utils import make_callable


class ClassFlipper(WorkflowStep):

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



class Identity():
    """A wrapper around lambda z: z with a nicer string representation"""

    def __call__(self, z):
        return z

    def __str__(self):
        return "z->z"

class Shuffle():

    def transform(self, array, to_idx, from_indices):
        """
        from_indices: list 
        to_idx: integer
        array: numpy array

        moves the elements in from indices 
        """
        from_list = []
        for i in from_indices:
            from_list.append(array[i])
        from_array = np.array(from_list)
    
        new_array = np.insert(array,to_idx,from_array, axis=0)

        from_indices.sort(reverse=True)
        for i in from_indices:
            if i < to_idx:
                new_array = np.delete(new_array, i, axis=0)
            else:
                new_array = np.delete(new_array, i+len(from_indices), axis=0)
        return new_array










