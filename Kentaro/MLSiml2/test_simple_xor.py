from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.generation.example_networks import xor as xor_network
from mlsiml.utils import flatten


# Classifiers
classifiers = flatten([
            Classifier.for_logistic_regression(),
            Classifier.for_knn(search_params={
                'n_neighbors':[1]
                }),
            Classifier.for_random_forest(search_params={
                'n_estimators':[100]
                }),
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1],
                'gamma':[0.003, 0.01],
                })
            ])


# Network Parameters
network = xor_network
network_params = {
        'p':0.5,
        'dim':[3, 7]
        }


# Experiment Parameters
sample_sizes = [10000]
test_size = 0.3


# Log file
logfile = "simple_xor"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        classifiers=classifiers)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
