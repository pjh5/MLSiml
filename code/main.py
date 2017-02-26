import argparse

from generation.example_networks import example
from generation.example_networks import xor_example
from analysis import analysis


def main(sample_size=2500, xor=True, show_plot=True, **kwargs):
    net = xor_example(**kwargs) if xor else example(**kwargs)
    X, y = net.bulk_sample(sample_size)
    analysis.summarize(X, y, show_plot)

    return X, y
    
# Allow running like "$> python main.py"
if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(
            description="Creates either an XOR-network or a " +
            "Normal-Exponential network, then samples from that network a " +
            "lot and shows summary statistics.")

    parser.add_argument("--xor", action="store_true", default=False,
            help="Uses a network with an XOR z-layer")

    parser.add_argument("-n", "--sample_size", type=int, default=2500,
            help="THe number of samples to take")

    parser.add_argument("-q", "--no-graph", action="store_true", default=False,
            help="Supresses the plot.")

    # Parse the arguments
    args = parser.parse_args()

    X, y = main(xor=args.xor, show_plot=not args.no_graph,
            sample_size=args.sample_size)
