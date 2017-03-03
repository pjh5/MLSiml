import sys

from generation.example_networks import example
from generation.example_networks import xor_example
from analysis import analysis

from classification.classifiers import split_data
from classification.classifiers import Classifier

from utils import parse_to_args_and_kwargs
from utils import flatten


def main(sample_size=2500, xor=0, show_plot=True, show_summary=False, **kwargs):

    # Build a network
    net = xor_example(num_z=xor, **kwargs) if xor > 0 else example(**kwargs)

    # Sample from the network
    X, y = net.bulk_sample(sample_size)

    # Show some summary of the data
    if show_summary:
        analysis.summarize(X, y)

    if show_plot:
        analysis.plot_data(X, y)

    # Split to training and testing data
    datasets = split_data(X, y, proportion_train=0.7)


    # Test on classifier suites
    for classifier in flatten([
            Classifier.for_logistic_regression(),
            [Classifier.for_knn(n_neighbors=n) for n in [1, 2, 3, 5, 10]],
            Classifier.for_linear_svm(),
            [Classifier.for_svm(kernel=k) for k in ['poly', 'rbf', 'sigmoid']],
            [Classifier.for_random_forest(n_estimators = k) for k in [10, 20, 30]],
            Classifier.for_gaussian_nb()
            ]):

        print(str(classifier.evaluate_on(datasets)) + " with " + str(classifier))
    print()



# Allow running like "$> python main.py --xor=5"
if __name__ == "__main__":

    args, kwargs = parse_to_args_and_kwargs(sys.argv[1:])
    if args:
        print("Discarding arguments: " + str(args))
    main(**kwargs)
