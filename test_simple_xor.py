from mlsiml.analysis import experiment
from mlsiml.classification.classifiers import Classifier
from mlsiml.generation.example_networks import xor as xor_network
from mlsiml.utils import flatten


# Classifiers
classifiers = flatten([
            Classifier.for_logistic_regression(),
            [Classifier.for_knn(n_neighbors=n)
                for n in [1, 2, 10]],
            Classifier.for_linear_svm(),
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10, 100],
                'gamma':[0.003, 0.01, 0.1, 1]}),
            Classifier.for_svm(kernel='poly'),
            [Classifier.for_random_forest(n_estimators = k)
                for k in [10, 30, 100]],
            ])


# Network Parameters
network = xor_network
network_params = {
        'p':0.5,
        'num_z':[3, 4, 5, 6, 7],
        'num_x_per_z':1,
        'var':[0.1, 0.2, 0.3]
        }


# Experiment Parameters
sample_sizes = [1000, 20000]
train_proportions = 0.7


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        train_proportions=train_proportions,
        classifiers=classifiers)

results = _exp.run(verbose=True)
