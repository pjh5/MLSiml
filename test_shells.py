
import logging

from mlsiml.analysis import experiment
from mlsiml.classification import classifiers
from mlsiml.classification import classifiers
from mlsiml.classification.workflow import Workflow
from mlsiml.classification.preprocessing import PCA

from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network
from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import ExtraNoiseNodes
from mlsiml.generation.geometric_functions import ShellVector

from mlsiml.integration.common import Concatenate



# Turn on logging
logging.basicConfig(level=logging.INFO)

# Number of sources
NUMBER_OF_SOURCES = 2

# Default workflows to run
conc = Concatenate()
workflows = [
    Workflow("Logistic Regression", NUMBER_OF_SOURCES,
        [Concatenate()],
        classifiers.for_logistic_regression()
        ),
    Workflow("Naive Bayes", NUMBER_OF_SOURCES,
        [Concatenate()],
        classifiers.for_gaussian_nb()
        ),
    Workflow("KNN", NUMBER_OF_SOURCES,
        [Concatenate()],
        classifiers.for_knn(search_params={'n_neighbors':[1, 10]})
        ),

    Workflow("PCA + Logistic Regression", NUMBER_OF_SOURCES,
        [Concatenate(), PCA()],
        classifiers.for_logistic_regression()
        ),
    Workflow("PCA + Naive Bayes", NUMBER_OF_SOURCES,
        [Concatenate(), PCA()],
        classifiers.for_gaussian_nb()
        ),
    Workflow("PCA + KNN", NUMBER_OF_SOURCES,
        [Concatenate(), PCA()],
        classifiers.for_knn(search_params={
            'n_neighbors':[1, 10]
            })
        ),

    Workflow("Separate PCA + Logistic Regression", NUMBER_OF_SOURCES,
        [PCA(), Concatenate()],
        classifiers.for_logistic_regression()
        ),
    Workflow("Separate PCA + Naive Bayes", NUMBER_OF_SOURCES,
        [PCA(), Concatenate()],
        classifiers.for_gaussian_nb()
        ),
    Workflow("Separate PCA + KNN", NUMBER_OF_SOURCES,
        [PCA(), Concatenate()],
        classifiers.for_knn(search_params={
            'n_neighbors':[1, 10]
            })
        )
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
