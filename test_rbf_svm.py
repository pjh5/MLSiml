
from mlsiml.analysis import experiment

from mlsiml.classification import classifiers as Classifier
from mlsiml.classification.workflow import Workflow
from mlsiml.generation.example_networks import corrupted_xor

from mlsiml.integration.common import Concatenate
from mlsiml.classification.preprocessing import PCA


# How many sources the workflows should expect
NUM_SOURCES = 2

# Workflows
workflows = [
        Workflow("Raw RBF SVM", NUM_SOURCES,
            [Concatenate()],
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10, 100],
                'gamma':[0.001, 0.01, 0.1, 1]
                })
            ),
        Workflow("PCA -> RBF SVM", NUM_SOURCES,
            [Concatenate(), PCA()],
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10, 100],
                'gamma':[0.001, 0.01, 0.1, 1]
                })
            )
        ]

# Network Parameters
network = corrupted_xor
network_params = {
        'p':0.5,
        'corruptions':[[.0, .0],
                              [.0, .2],
                              [.0, .4],
                              [.2, .2],
                              [.2, .4],
                              [.4, .4]],
        'xor_dim':2,
        'var':0.15
        }


# Experiment Parameters
sample_sizes = [2500]
test_size = 0.3


# Log file
logfile = "small_rbf_svm"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        workflows=workflows)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
