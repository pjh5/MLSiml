
import logging


from mlsiml.analysis import experiment
from mlsiml.classification import classifiers
from mlsiml.classification.workflow import Workflow
from mlsiml.generation.example_networks import crosstalk as crosstalk_network
from mlsiml.classification.preprocessing import MetricLearn

from mlsiml.integration.common import Concatenate


# Turn on logging
logging.basicConfig(level=logging.INFO)

# Number of sources
NUMBER_OF_SOURCES = 2

# Metric_Learn Classifiers
# ITML needs constraints
# SDML needs a connectivity graph
# RCA needs a chunks array
# NCA takes forever?
# LFDA needs some other parameter too
workflows = [
        Workflow("Raw SVM", NUMBER_OF_SOURCES,
            [Concatenate()],
            classifiers.for_svm(kernel="rbf", search_params={
                "C":[0.1, 10],
                "gamma":[0.01]
                })
            ),
        Workflow("LMNN SVM", NUMBER_OF_SOURCES,
            [Concatenate(), MetricLearn("LMNN")],
            classifiers.for_svm(kernel="rbf", search_params={
                "C":[0.1, 10],
                "gamma":[0.01]
                })
            ),
        Workflow("2 LMNN SVM", NUMBER_OF_SOURCES,
            [MetricLearn("LMNN"), Concatenate()],
            classifiers.for_svm(kernel="rbf", search_params={
                "C":[0.1, 10],
                "gamma":[0.01]
                })
            )
        ]


# Network Parameters
network = crosstalk_network
network_params = {
        'p':0.5,
        "source1_var":0.2,
        "source2_var":15,
        "shared_var":0.1,
        "source1_dim":[1, 3],
        "source2_dim":[1, 3],
        "shared_dim" :[1, 3],
        "extra_noise":[1, 3]
        }


# Experiment Parameters
sample_sizes = [10000]
test_size = 0.3


# Log file
logfile = "metric_learn"


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
