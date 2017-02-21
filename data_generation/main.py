
import bayes_networks
import analysis


def main():
    net = bayes_networks.example()
    X, y = net.bulk_sample(1000)
    analysis.summarize(X, y)

    return X, y
    


if __name__ == "main":
    X, y = main()
