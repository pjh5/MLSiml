"""
This module creates layered networks of the following form:

A network is essentially a sequence of layers, each of which takes the output
of the previous layer (a vector), processes it in some waay, and then outputs a
different vector to the next layer.
"""
import logging
import numpy as np

from mlsiml.generation.dataset import MultiSourceDataset
from mlsiml.utils import flatten, make_iterable, to_flat_np_array


class Node:
    """A Node in a Bayesian Network.

    The most important method is sample_with(prev_layer), which takes in the
    previous layer (a vector as a numpy array) and outputs a single value OR a
    vector. Note that this may output a vector.
    """

    def __init__(self, f_sample, desc):
        self.desc = desc
        self.f_sample = f_sample

    def sample_with(self, prev_layer):
        return to_flat_np_array(self.f_sample(prev_layer))

    def short_string(self):
        return self.desc

    def _set_index(self, idx):
        self.idx = idx

    def __str__(self):
        return "<" + self.desc + " Node>"

    def __repr__(self):
        return self.__str__()


class NodeLayer:
    """Basically just an array of Node objects.

    NodeLayers are designed to encapsulate a transformation to a data
    generation process. NodeLayers take in vector output from earlier
    NodeLayers, provide a transformation / sample conditionally based on the
    previous output, and then output another vector (possibly of different
    dimension) to the next layer.

    A NodeLayer can be created from an array of callables, but it is better to
    define subclasses (with nice string representations) to encapsulate the
    logic.

    NodeLayers can also change or alter the class label, but this is a rarely
    needed feature. Subclasses that do not need this feature  need only
    implement .sample_with(z), which will automatically be called by
    .transform(y,z) when no .transform(y,z) is given.

    The output of the layer will always be coerced into a flat numpy array (a
    vector).
    """

    def __init__(self, desc, nodes):
        self.desc = desc
        nodes = flatten(make_iterable(nodes))

        # Convert functions into Nodes automatically
        self.nodes = [n if isinstance(n, Node) else Node(n, "Lambda")
                        for n in nodes]

        # Label all nodes with their indices
        for i, node in enumerate(self.nodes):
            node._set_index(i)

        logging.debug("Built layer '{}' with nodes {!s}".format(
                                                    self.desc, self.nodes))

    @classmethod
    def from_function_array(cls, desc, funcs):
        return cls(desc, [Node(f, "Lambda") for f in make_iterable(funcs)])

    @classmethod
    def from_repeated(cls, desc, node):
        return RepeatedNodeLayer(desc, node)

    def sample_with(self, z):
        return to_flat_np_array([n.sample_with(z) for n in self.nodes])

    def transform(self, y, z):
        return (y, self.sample_with(z))

    def short_string(self):
        return "<" + self.desc + " Layer>"

    def multiline_string(self):
        """Multi-line string representation, one row per node"""
        return "{} NodeLayer:\n{}".format(self.desc, "".join(
            ["\t{!s} {}\n".format(n.idx, n.desc) for n in self.nodes]))

    def __str__(self):
        return ("<" + self.desc + " Layer: [" +
                    '-'.join([n.short_string() for n in self.nodes]) + "]>")

    def __repr__(self):
        return self.__str__()


class RepeatedNodeLayer(NodeLayer):
    """A layer made from a single node, repeated as many times as needed

    Every node will be given the scalar from the corresponding node of the
    previous layer. Layers created with this method will not change the
    dimension of the sampled vector. Each Node will NOT be passed the entire
    vector output of the previous layer.
    """

    def __init__(self, desc, node):
        self.desc = desc
        self.node = node

    def sample_with(self, z):
        return to_flat_np_array([self.node.sample_with(_z) for _z in z])

    def __str__(self):
        return "{} Layer: [{} (repeated)]".format(self.desc,
                                                    self.node.short_string())

class Network:
    """
    A definition of a iterative process to generating data.

    Networks generate data backwards, first sampling the true class (the 'y',
    usually just from a Bernoulli), and then successively applying
    transformations to the class label to eventually produce a vector (the
    'x').

    A network is made up of a class_generator and several NodeLayers. When
    sampling from a network, the class label is sample from the
    class_generator. The class is saved, and also passed along to the first
    NodeLayer, which will then produce a vector which will be passed to the
    second NodeLayer, etc.
    """

    def __init__(self, desc, class_generator, layers, split_indices=None):
        """Creates a network

        Params
        ======
        desc     A human readable string that is used to describe the
            network. This is only used for debugging / printing purposes, but
            it really helps to have a descriptive string here.

        class_generator - Any object with a .sample() method that will generate
            a number. This should be either 1 or 0 for classification tasks.
            Technically, this can generate any numeric value, even numpy
            arrays, but all of the current networks assume that this will be
            either 0 or 1.

        layers  - An array of NodeLayer objects. This will be flattened, so it
            is okay if this is actually an array of arrays of NodeLayer
            objects, or if arrays are mixed into the array, as long as all
            objects have a .transform(y, z) method once flattened. Note that
            subclasses of NodeLayer need only implement .sample_with(z) if they
            won't ever change the class label, as NodeLayer will automatically
            call that method.
        """
        # Actually require split_indices
        if split_indices is None:
            raise Exception("Split_indices must be defined for all bayes_networks""")

        self.desc = desc
        self.class_generator = class_generator
        self.layers = flatten(layers)
        self.split_indices = split_indices
        self.dims = []

        # Make sure that all the dimensions of the network match up
        #######################################################################
        logging.debug("\nValidating Network Dimensions")

        # First check the class generator
        y = self.class_generator()
        result = (y, y)
        self.dims = [1]
        logging.debug("Layer 0: 1D ({!s} from {!s})".format(y, class_generator))

        # Check every other layer
        for i, layer in enumerate(self.layers):

            # Make sure every layer is actually a NodeLayer TODO fix this
            if not isinstance(layer, NodeLayer) and False:
                raise Exception(
                    "Layer {!s} '{!s}' is not a NodeLayer but a {!s}".format(
                        i+1, layer, type(layer))
                    )

            # Propogate the result and make the results dimensions agree
            result = layer.transform(*result)
            self.dims.append(len(result[1]))
            logging.debug(
                    "Layer {!s}: {!s}D ({!s} from {!s})".format(
                        i+1, len(result[1]), result[1], layer)
                    )
        logging.debug(
                "Final dims are {}\n".format("-".join(map(str, self.dims)))
                )


    def _sample_once(self):
        """Generates a (y,x) pair, where y is a scalar and x is a numpy array"""

        # First layer has no inputs and is assumed to be the desired output
        y = self.class_generator.sample()
        result = (y, y)

        # Sample rest of the inputs
        for layer in self.layers:
            result = layer.transform(*result)

        return result


    def sample(self, n_samples, test_size=0.3):
        """Samples the network n_samples times.

        Returns
        -------
        X   :   A n_samples x K numpy matrix of n_samples samples arranged into
                rows, where K is the output dimension of the last layer of the
                network (Note that the last output of the network will always
                be flattened into a 1 dimensional numpy array). These are the
                outputs of the last layer of the network.

        y   :   A 1 dimensional numpy vector of length n_samples. These are the
                corresponding outputs of the first layer of the network (which
                is assumed to be the class label).
        """

        # Allocate memory
        y = np.zeros(shape=(n_samples))
        X = np.zeros(shape=(n_samples, self.dims[-1]))

        # Sample all data
        for i in range(n_samples):
            try:
                y[i], X[i,:] = self._sample_once()
            except ValueError:
                raise Exception(
                        (
                            "Network output '{!s}' has to be a tuple of "
                            + "arrays of dimensions ({!s}, {!s})"
                        ).format(self.sample(), 1, self.dims[-1])
                        )

        # Split data into train and test sets, then into sources
        return MultiSourceDataset.from_unsplit(
                X, y, test_size, self.split_indices
                )


    def pretty_string(self):
        """Bulky multi-line string representation, layer by layer"""

        # Header for the network
        s = "<Network {}\n".format("-".join(map(str, self.dims)))

        # Each layer on another line
        s += "\tLayer 0: " + str(self.class_generator) + "\n"
        for i, layer in enumerate(self.layers):
            s += "\tLayer {!s}: {!s}\n".format(i + 1, layer)

        # Extra newline at the end
        s += "\t>\n"

        return s


    def __str__(self):
        return "<{} Network [{}]>".format(
                self.desc, "-".join(map(str, self.dims))
                )


