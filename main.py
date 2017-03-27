import logging
import sys

from sklearn.model_selection import train_test_split

from mlsiml.generation import example_networks
from mlsiml.analysis import analysis
from mlsiml.classification import classifiers as Classifier
from mlsiml.utils import parse_to_args_and_kwargs
from mlsiml.utils import flatten


classifier_list = flatten([
                Classifier.for_logistic_regression(),
                Classifier.for_knn(search_params={
                    'n_neighbors':[1, 2, 10]
                    }),
                Classifier.for_random_forest(search_params={
                    'n_estimators':[10, 100]
                    }),
                Classifier.for_svm(kernel='rbf', search_params={
                    'C':[0.1, 1, 10],
                    'gamma':[0.01, 0.1, 1],
                    }),
                Classifier.for_gaussian_nb()
                ])


def main(which_example, sample_size=5000, plot=False, test=False, **kwargs):

    # If verbose, print out the args and kwargs
    logging.info("Creating network with: {}({})".format(which_example,
            ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])))

    # Build the network, sample from it, and split the data
    net = getattr(example_networks, which_example)(**kwargs)
    X, y = net.bulk_sample(sample_size)
    datasplit = train_test_split(X, y, test_size=0.3)

    # Display network info or data summary
    logging.info(net.pretty_string())

    # Test on classifier suites
    if test:
        print("Classifier performance:")
        for clsfr in classifier_list:
            print("{:8.3f}\t{!s}".format(clsfr.evaluate_on(*datasplit), clsfr))
        print()

    # Plotting at the end
    if plot:
        analysis.plot_data(X, y)


def some_data(which="shells", sample_size=10000, **kwargs):
    net = getattr(example_networks, which)(**kwargs)
    X, y = net.bulk_sample(sample_size)
    datasplit = train_test_split(X, y, test_size=0.3)
    return datasplit


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

