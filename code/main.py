from generation.example_networks import xor_example
from analysis import analysis


def main(sample_size=2500, pca=True, **kwargs):
    net = xor_example(**kwargs)
    X, y = net.bulk_sample(sample_size)
    analysis.summarize(X, y, pca)

    return X, y
    


if __name__ == "main":
    X, y = main()
