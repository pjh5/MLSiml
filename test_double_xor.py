from mlsiml.analysis import experiment
from mlsiml.classification import classifiers as Classifier

from mlsiml.generation.stats_functions import Bernoulli
from mlsiml.generation.bayes_networks import NodeLayer
from mlsiml.generation.bayes_networks import Network
from mlsiml.generation.noise_functions import NormalNoise
from mlsiml.generation.noise_functions import ExtraNoiseNodes
from mlsiml.generation.geometric_functions import XOR
from mlsiml.generation.noise_functions import BinaryCorruption
from mlsiml.generation.example_networks import *

# Classifiers

classifiers = [
            Classifier.for_knn(search_params={
                'n_neighbors':[1, 2, 10]
                }),
            Classifier.for_random_forest(n_estimators=100),
            Classifier.for_svm(kernel='rbf', search_params={
                'C':[0.1, 1, 10],
                'gamma':[0.01, 0.1, 1],
                }),
            ]


# Network Parameters
# def xor_network(first_dims=0, second_dims=0, corruption=0,noise_dims=0):
#     return Network("XOR",
#             Bernoulli(0.5),
#             [
#                 NodeLayer("XOR", XOR(first_dims)),
#                 NodeLayer.from_repeated("XOR", XOR(second_dims)),
#                 NodeLayer("Corruption", BinaryCorruption(corruption)),
#                 NormalNoise(var=0.2),
#                 ExtraNoiseNodes(noise_dims)
#             ])

# p=0.5,
# corruptions=[0.0, 0.0],
#                     xor_dim=2,
#                     var=0.1,
#                     extra_noise=0

network = corrupted_xor
# network_params = {
#         "first_dims":[2],
#         "second_dims": [3, 5, 7],
#         "corruption": [0.0],
#         "noise_dims":[2,4,6,8,10,12,14,20,28]
#         }
network_params = {
    "corruptions": [[0, 0]],
    "xor_dim": [3, 5, 7],
    "extra_noise" : [2,4,6,8,10,12,14,20,28],
    "var" : 0.2
}

# Experiment Parameters
sample_sizes = [5000]
test_size = 0.3333333333


# Log file
logfile = "double_xor_base_noisedim"


# Make experiment
_exp = experiment.Experiment(network, network_params,
        sample_sizes=sample_sizes,
        test_size=test_size,
        classifiers=classifiers)

results = _exp.run(logfile=logfile)
df = results.as_dataframe()
