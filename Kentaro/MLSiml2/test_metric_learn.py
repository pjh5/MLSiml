
from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.generation.example_networks import xor as xor_network
from mlsiml.classification.preprocessing import MetricLearnTransform


# Classifiers
# ITML needs constraints
# SDML needs a connectivity graph
# RCA needs a chunks array
# NCA takes forever?
# LFDA needs some other parameter too
classifiers = [Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 10],
                'gamma':[0.01],
                }).with_preprocessing(MetricLearnTransform(metric))
                for metric in ["Covariance", "LMNN"]]


# Network Parameters
network = xor_network
network_params = {
        'p':0.5,
        'dim':[3, 7],
        'var':[0.2, 0.3]
        }


# Experiment Parameters
sample_sizes = [1000]
test_size = 0.3


# Log file
logfile = "metric_learn"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        classifiers=classifiers)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
