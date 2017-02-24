from data_generation.bayes_networks import xor_example
from analysis import analysis


def main():
    net = xor_example()
    X, y = net.bulk_sample(5000)
    analysis.summarize(X, y)

    return X, y
    


if __name__ == "main":
    X, y = main()
