import sys

from generation import example_networks
from analysis import analysis

from classification.classifiers import split_data
from classification.classifiers import Classifier

from utils import parse_to_args_and_kwargs
from utils import flatten


def main(which_example, sample_size=2500, plot=False,
        summary=False, verbose=False, **kwargs):

    # If verbose, print out the args and kwargs
    if verbose:
        print("Creating network with: " + which_example + "("
            + ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])
            + ")")

    # Build the network
    net = getattr(example_networks, which_example)(**kwargs)

    # Sample from the network
    X, y = net.bulk_sample(sample_size)

    # Show some summary of the data
    if summary:
        analysis.summarize(X, y)

    # Split to training and testing data
    datasets = split_data(X, y, proportion_train=0.7)

    # Display network info
    if verbose:
        print(net.pretty_string())
        print()


    # Test on classifier suites
    print("Classifier performance:")
    for classifier in flatten([
            Classifier.for_logistic_regression(),
            [Classifier.for_knn(n_neighbors=n) for n in [1, 2, 10]],
            Classifier.for_linear_svm(),
            [Classifier.for_svm(kernel=k) for k in ['rbf', 'sigmoid']],
            [Classifier.for_random_forest(n_estimators = k) for k in [10, 30]],
            Classifier.for_gaussian_nb()
            ]):

        print("{:8.3f}\t{!s}".format(classifier.evaluate_on(datasets), classifier))
    print()

    # Plotting at the end
    if plot:
        analysis.plot_data(X, y)



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

