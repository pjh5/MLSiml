
import logging

from mlsiml.generation.example_networks import sine as sine_network

from mlsiml.analysis import experiment
from mlsiml.classification import classifiers
from mlsiml.classification import classifiers
from mlsiml.classification.workflow import Workflow
from mlsiml.classification.preprocessing import PCA

from mlsiml.integration.common import Concatenate



# Turn on logging
logging.basicConfig(level=logging.INFO)

# Number of sources
NUM_SOURCES = 2

# Default workflows to run
conc = Concatenate()
workflows = [
    Workflow("PCA -> Naive Bayes", NUM_SOURCES,
        [Concatenate(), PCA(n_components=3)],
        classifiers.for_gaussian_nb()
        ),
    Workflow("KNN", NUM_SOURCES,
        [Concatenate()],
        classifiers.for_knn(search_params={'n_neighbors':[1, 10]})
        ),
    Workflow("PCA -> RBF SVM", NUM_SOURCES,
        [Concatenate(), PCA(n_components=3)],
        classifiers.for_svm(kernel='rbf', search_params={
            'C':[1],
            'gamma':[0.1]
            })
        ),
    Workflow("Random Forest", NUM_SOURCES,
        [Concatenate()],
        classifiers.for_random_forest(search_params={
            'n_estimators':[10, 100]
            })
        )
    ]


# Network Parameters
network = sine_network
network_params = {
        "periods":[1, 2, 3, 4, 5],
        "extra_noise":[0, 1, 2, 3, 4]
        }


# Experiment Parameters
sample_sizes = [1000]
test_size = 0.3


# Log file
logfile = "sine_experiment"


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
