import argparse

from generation.example_networks import example
from generation.example_networks import xor_example
from analysis import analysis

from classification.classifiers import split_data
from classification.classifiers import Classifier



def main(sample_size=2500, xor=0, show_plot=True, **kwargs):

    # Build a network
    net = xor_example(num_z=xor, **kwargs) if xor > 0 else example(**kwargs)

    # Sample from the network
    X, y = net.bulk_sample(sample_size)

    # Show some summary of the data
    analysis.summarize(X, y, show_plot)

    # Split to training and testing data
    datasets = split_data(X, y, proportion_train=0.7)

    # Classify with logistic regression, with all default parameters
    classifier = Classifier.for_logistic_regression()
    classifier.evaluate_on(datasets)


    
# Allow running like "$> python main.py"
if __name__ == "__main__":

    """
    # Arguments
    parser = argparse.ArgumentParser(
            description="Creates either an XOR-network or a " +
            "Normal-Exponential network, then samples from that network a " +
            "lot and shows summary statistics.")

    parser.add_argument("--xor", type=int, default=0,
            help="Uses a network with an XOR z-layer of the given dimension")

    parser.add_argument("-n", "--sample_size", type=int, default=2500,
            help="THe number of samples to take")

    parser.add_argument("-q", "--no-graph", action="store_true", default=False,
            help="Supresses the plot.")

    # Parse the arguments
    args = parser.parse_args()
    """

    main(**kwargs)
