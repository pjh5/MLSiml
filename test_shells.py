from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.classification.workflow import Workflow

from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network
from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import ExtraNoiseNodes
from mlsiml.generation.geometric_functions import ShellVector

from mlsiml.integration.common import Concatenate

from sklearn.decomposition import PCA


# Workflows
workflows = [
    Workflow([Concatenate()], Classifier.for_logistic_regression()),
    Workflow([Concatenate()], Classifier.for_gaussian_nb()),
    Workflow([Concatenate()], Classifier.for_knn(search_params={
        'n_neighbors':[1, 10]
        })),
    Workflow([Concatenate(), Classifier.Classifier("PCA", PCA())], Classifier.for_logistic_regression()),
    Workflow([Concatenate(), Classifier.Classifier("PCA", PCA())], Classifier.for_gaussian_nb()),
    Workflow([Concatenate(), Classifier.Classifier("PCA", PCA())], Classifier.for_knn(search_params={
        'n_neighbors':[1, 10]
        })),
    Workflow([Classifier.Classifier("PCA", PCA()), Concatenate()], Classifier.for_logistic_regression()),
    Workflow([Classifier.Classifier("PCA", PCA()), Concatenate()], Classifier.for_gaussian_nb()),
    Workflow([Classifier.Classifier("PCA", PCA()), Concatenate()], Classifier.for_knn(search_params={
        'n_neighbors':[1, 10]
        }))
    ]


# Network Parameters
def shell_network(real_dims=0, noise_dims=0):
    return Network("Shells",
            Bernoulli(0.5),
            [
                NodeLayer("Shells", ShellVector(real_dims)),
                NormalNoise(var=0.2),
                ExtraNoiseNodes(noise_dims)
            ], split_indices=real_dims)

network = shell_network
network_params = {
         "real_dims":[3, 5, 7, 9],
        "noise_dims":[2, 4, 8]
        }


# Experiment Parameters
sample_sizes = [2000]
test_size = 0.3


# Log file
logfile = "shell_experiment_small"


# Make experiment
_exp = experiment.Experiment(
        network_class = network,
        network_params_dict = network_params,
        workflows = workflows,
        sample_sizes = sample_sizes,
        test_size = test_size
        )

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
