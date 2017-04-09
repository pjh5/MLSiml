import logging
import sys


from mlsiml.classification import classifiers
from mlsiml.classification.preprocessing import PCA
from mlsiml.classification.workflow import Workflow
from mlsiml.integration.common import Concatenate

from mlsiml.generation import example_networks
from mlsiml.analysis import analysis
from mlsiml.utils import parse_to_args_and_kwargs

# Number of sources
NUMBER_OF_SOURCES = 2

# Default workflows to run
conc = Concatenate()
workflows = [
    Workflow("Random Forest", NUMBER_OF_SOURCES,
        [Concatenate()],
        classifiers.for_random_forest(search_params={
            'n_estimators':[10, 100]
            })
        ),
    Workflow("KNN", NUMBER_OF_SOURCES,
        [Concatenate()],
        classifiers.for_knn(search_params={'n_neighbors':[1, 10]})
        ),
    Workflow("RBF SVM", NUMBER_OF_SOURCES,
            [Concatenate()],
            classifiers.for_svm(search_params={
                'gamma':[0.01, 0.1, 1, 10],
                'C':[0.1, 1, 10, 100]
                })
            ),
    ]


def main(
        which_example, sample_size=5000, plot=False, test=False, summary=False, **kwargs
        ):

    # If verbose, print out the args and kwargs
    logging.info(
            "Creating network with: {}({})".format(
                which_example,
                ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])
                )
            )

    # Build the network, sample from it, and split the data
    net = getattr(example_networks, which_example)(**kwargs)
    sources = net.sample(sample_size)

    # Display network info or data summary
    logging.info(net.pretty_string())


    # Summarize data columns
    if summary:
        analysis.summarize(sources)

    # Test on classifier suites
    if test:
        print("Workflow performance:")
        for flow in workflows:
            print("{:8.3f}\t{!s}".format(flow.evaluate_on(sources), flow))
        print()

    # Plotting at the end
    if plot:
        logging.debug(sources)
        analysis.plot_data(*sources.as_X_and_Y())


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

    logging.info(
            "Parsed arguments are args:{!s}, kwargs:{!s}".format(args, kwargs)
            )

    # Call main with arguments
    main(*args, **kwargs)

