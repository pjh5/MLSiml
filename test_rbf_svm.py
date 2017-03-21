
from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier
from mlsiml.generation.example_networks import corrupted_xor


# Classifiers
classifiers = [Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10, 100],
                'gamma':[0.001, 0.01, 0.1, 1],
                })]


# Network Parameters
network = corrupted_xor
network_params = {
        'p':0.5,
        'source_corruptions':[[.0, .0],
                              [.0, .2],
                              [.0, .4],
                              [.2, .2],
                              [.2, .4],
                              [.4, .4]],
        'xor_dim':2,
        'var':0.15
        }


# Experiment Parameters
sample_sizes = [7500]
train_proportions = 0.7


# Log file
logfile = "rbf_svm"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        train_proportions=train_proportions,
        classifiers=classifiers)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
