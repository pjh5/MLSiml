from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier

from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network
from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import ExtraNoiseNodes
from mlsiml.generation.geometric_functions import ShellVector


# Classifiers
# Include every classifier, also with search_params too
classifiers = [
        Classifier.for_logistic_regression(),
        Classifier.for_knn(n_neighbors=1),
        Classifier.for_knn(search_params={
            'n_neighbors':[2, 10]
            }),
        Classifier.for_linear_svm(),
        Classifier.for_svm(kernel='rbf', search_params={
            'C':[0.1, 1, 10],
            'gamma':[0.01, 0.1, 1],
            }),
        Classifier.for_random_forest(n_estimators=100),
        Classifier.for_random_forest(search_params={
            "n_estimators":[200, 500]
            }),
        Classifier.for_gaussian_nb()
        ]


# Network Parameters
def shell_network(real_dims=0, noise_dims=0):
    return Network("Shells",
            Bernoulli(0.5),
            [
                NodeLayer("Shells", ShellVector(real_dims)),
                NormalNoise(var=0.2),
                ExtraNoiseNodes(noise_dims)
            ])
network = shell_network
network_params = {
         "real_dims":[3, 5, 7, 9],
        "noise_dims":[0, 2, 4, 8]
        }


# Experiment Parameters
sample_sizes = [2000]
test_size = 0.3


# Log file
logfile = "shell_experiment_small"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        classifiers=classifiers)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
