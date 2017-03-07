import sys

from mlsiml.generation import example_networks
from mlsiml.analysis import analysis
from mlsiml.analysis import experiment
from mlsiml.classification.classifiers import split_data
from mlsiml.classification.classifiers import Classifier
from mlsiml.utils import parse_to_args_and_kwargs
from mlsiml.utils import flatten


def main(which_example, sample_size=2500, plot=False,
        summary=False, verbose=False, **kwargs):

    # If verbose, print out the args and kwargs
    if verbose:
        print("Creating network with: " + which_example + "("
            + ", ".join(["{!s}={!s}".format(k,v) for k,v in kwargs.items()])
            + ")")

    # Build the network, sample from it, and split the data
    net = getattr(example_networks, which_example)(**kwargs)
    X, y = net.bulk_sample(sample_size)
    datasets = split_data(X, y, proportion_train=0.7)

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
            [Classifier.for_knn(n_neighbors=n) for n in [1, 2, 10]],
            Classifier.for_linear_svm(),
            [Classifier.for_svm(kernel=k) for k in ['rbf', 'sigmoid']],
            [Classifier.for_random_forest(n_estimators = k) for k in [10, 30]],
            Classifier.for_gaussian_nb()
            ]):

        print("{:8.3f}\t{!s}".format(classifier.evaluate_on(datasets),
                                                                    classifier))
    print()

    # Plotting at the end
    if plot:
        analysis.plot_data(X, y)

def test(verbose=True):

    # Make experiment
    exp = experiment.Experiment(
            example_networks.xor,
            {
                "num_z":[3, 10],
                "max_beta":[1.0, 0.6],
                "xor_scale":[1, 3]
            },
            sample_sizes=[1000, 15000]
            )

    # Run experiment
    all_results = exp.run(verbose=verbose)

    return all_results


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

