from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.generation.example_networks import xor as xor_network
from mlsiml.utils import flatten


# Classifiers
classifiers = flatten([
            Classifier.for_knn(search_params={
                'n_neighbors':[1, 3, 10]
                }),
            Classifier.for_random_forest(search_params={
                'n_estimators':[10, 30, 100]
                }),
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10, 100],
                'gamma':[0.003, 0.01, 0.1, 1]
                })
            ])


# Network Parameters
network = xor_network
network_params = {
        'p':0.5,
        'num_z':[3, 4, 5, 6, 7],
        'num_x_per_z':1,
        'var':[0.2, 0.3]
        }


# Experiment Parameters
sample_sizes = [10000]
train_proportions = 0.7


# Log file
logfile = "simple_xor.csv"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        train_proportions=train_proportions,
        classifiers=classifiers,
        output_file=logfile)

results = _exp.run()
