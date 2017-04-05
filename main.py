import logging
import sys

from sklearn.model_selection import train_test_split

from mlsiml.generation import example_networks, dataset
from mlsiml.integration.common import Concatenate
from mlsiml.classification import classifiers as Classifier
from mlsiml.classification.workflow import Workflow
from mlsiml.analysis import analysis
from mlsiml.utils import flatten, parse_to_args_and_kwargs


workflow_list = flatten([
    Workflow([Concatenate()], Classifier.for_logistic_regression()),
    Workflow([Concatenate()], Classifier.for_gaussian_nb()),
    ])


def main(which_example, sample_size=5000, plot=False, test=False, **kwargs):

    # If verbose, print out the args and kwargs
    logging.info("Creating network with: {}({})".format(which_example,
            ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])))

    # Build the network, sample from it, and split the data
    net = getattr(example_networks, which_example)(**kwargs)
    sources = net.sample(sample_size)

    # Display network info or data summary
    logging.info(net.pretty_string())

    # Test on classifier suites
    if test:
        print("Classifier performance:")
        for clsfr in workflow_list:
            print("{:8.3f}\t{!s}".format(clsfr.evaluate_on(sources), clsfr))
        print()

    # Plotting at the end
    if plot:
        logging.debug(sources)
        data = dataset.concatenate(sources)
        analysis.plot_data(data.X_train, data.Y_train)


def some_data(which="shells", sample_size=10000, **kwargs):
    """Return [X, X_test, Y, Y_test] for some sample data"""
    net = getattr(example_networks, which)(**kwargs)
    return  net.sample(sample_size)


# Allow running like "$> python main.py --xor=5"
if __name__ == "__main__":

    # Parse command line arguments into args and kwargs
    args, kwargs = parse_to_args_and_kwargs(sys.argv[1:])

    # Set log level based on passed in verbosity
    if "verbose" in kwargs:
        if kwargs["verbose"] >= 2:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug("Logging level set to DEBUG")
        else:
            logging.basicConfig(level=logging.INFO)
            logging.info("Logging level set to INFO")
        kwargs.pop("verbose")

    logging.info("Parsed arguments are args:{!s}, kwargs:{!s}".format(
                                                        args, kwargs))

    # Call main with arguments
    main(*args, **kwargs)

