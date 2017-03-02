import sys

from generation.example_networks import example
from generation.example_networks import xor_example
from analysis import analysis

from classification.classifiers import split_data
from classification.classifiers import Classifier

from utils import parse_to_args_and_kwargs


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

    # Classify with logistic regression, with all default parameters
    print('\nTesting logistic regression')
    classifier = Classifier.for_logistic_regression()
    classifier.evaluate_on(datasets)

    # Classify with linear SVM, with all default parameters
    print('\nTesting a linear SVM')
    classifier = Classifier.for_linear_svm()
    classifier.evaluate_on(datasets)

    # Classify with a rbf SVM, with all default parameters
    print('\nTesting a SVM with a RBF kernel')
    classifier = Classifier.for_svm()
    classifier.evaluate_on(datasets)


    
# Allow running like "$> python main.py"
if __name__ == "__main__":

    args, kwargs = parse_to_args_and_kwargs(sys.argv[1:])
    main(**kwargs)
