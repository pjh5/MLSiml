from data_generation.example_networks import xor_example
from analysis import analysis


def main(**kwargs):
    net = xor_example(**kwargs)
    X, y = net.bulk_sample(5000)
    analysis.summarize(X, y)

    return X, y
    


if __name__ == "main":
    X, y = main()
