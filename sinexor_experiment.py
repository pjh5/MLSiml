
import logging

from mlsiml.generation.example_networks import xor_sine

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
    Workflow("KNN", NUM_SOURCES,
        [Concatenate()],
        classifiers.for_knn(search_params={'n_neighbors':[1, 10]})
        ),
    Workflow("Random Forest", NUM_SOURCES,
        [Concatenate()],
        classifiers.for_random_forest(n_estimators=100)
        ),
    Workflow("RBF SVM", NUM_SOURCES,
        [Concatenate()],
        classifiers.for_svm(kernel='rbf', search_params={
            'C':[0.1, 1, 10],
            'gamma':[0.01, 0.1, 1]
            })
        )
    ]


# Network Parameters
network = xor_sine
network_params = {
        "xor_corruption":[0, 0.15, 0.3, 0.45],
        "xor_dim":[3, 5, 7],
        "xor_var":0.2,
        "sine_corruption":[0, 0.15, 0.3, 0.45],
        "sine_periods":[1, 2, 3],
        "extra_noise":0
        }


# Experiment Parameters
sample_sizes = [5000]
test_size = 0.3


# Files
logfile = "sinexor/results"
mat_dir = "sinexor"
mat_fileformat = (
        "sinexor"
        "_dim_{xor_dim}"
        "_periods_{sine_periods}"
        "_corruption_{xor_corruption}_{sine_corruption}"
        )


# Make experiment
_exp = experiment.Experiment(
        network_class = network,
        network_params_dict = network_params,
        workflows = workflows,
        sample_sizes = sample_sizes,
        test_size = test_size,
        save_data=True,
        mat_dir=mat_dir,
        mat_fileformat=mat_fileformat
        )

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
