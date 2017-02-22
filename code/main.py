from data_generation.bayes_networks import example
from analysis import analysis


def main():
    net = example()
    X, y = net.bulk_sample(1000)
    analysis.summarize(X, y)

    return X, y
    


if __name__ == "main":
    X, y = main()
