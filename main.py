import sys

from sklearn.model_selection import train_test_split

from mlsiml.generation import example_networks
from mlsiml.analysis import analysis
from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.utils import parse_to_args_and_kwargs
from mlsiml.utils import flatten


def main(which_example, sample_size=1000, plot=False,
        summary=False, verbose=False, **kwargs):

    # If verbose, print out the args and kwargs
    if verbose:
        print("Creating network with: " + which_example + "("
            + ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])
            + ")")

    # Build the network, sample from it, and split the data
    net = getattr(example_networks, which_example)(**kwargs)
    X, y = net.bulk_sample(sample_size)
    datasplit = train_test_split(X, y, test_size=0.3)

    # Display network info or data summary
    if verbose:
        print(net.pretty_string())
        print()
    if summary:
        analysis.summarize(X, y)

    # Test on classifier suites
    print("Classifier performance:")
    for classifier in flatten([
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
            ]):

        print("{:8.3f}\t{!s}".format(classifier.evaluate_on(*datasplit),
                                                                    classifier))
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

    # If verbose, print out the args and kwargs
    if "verbose" in kwargs:
        print("Detected arguments:")
        print("\t  args: " + str(args))
        print("\tkwargs: " + str(kwargs))
        print()

    # Call main with arguments
    main(*args, **kwargs)

