

from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.classification.workflow import Workflow
from mlsiml.generation.example_networks import xor as xor_network

from mlsiml.integration.common import Concatenate


# How many sources the workflows will expect
NUM_SOURCES = 2

# Workflows
workflows = [
        Workflow("Logistic Regression", NUM_SOURCES,
            [Concatenate()],
            Classifier.for_logistic_regression()
            ),
        Workflow("KNN", NUM_SOURCES,
            [Concatenate()],
            Classifier.for_knn(search_params={
                'n_neighbors':[1, 10]
                })
            ),
        Workflow("Random Forest", NUM_SOURCES,
            [Concatenate()],
            Classifier.for_random_forest(search_params={
                'n_estimators':[100]
                })
            ),
        Workflow("RBF SVM", NUM_SOURCES,
            [Concatenate()],
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1],
                'gamma':[0.003, 0.01],
                })
            )
        ]


# Network Parameters
network = xor_network
network_params = {
        'p':0.5,
        'num_z':[3, 7],
        'num_x_per_z':1,
        'var':[0.2, 0.3]
        }


# Experiment Parameters
sample_sizes = [2000]
test_size = 0.3


# Log file
logfile = "small_simple_xor"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        workflows=workflows)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
