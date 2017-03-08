
from itertools import product as iter_product

from mlsiml.classification.classifiers import Classifier
from mlsiml.classification.classifiers import split_data
from mlsiml.utils import flatten
from mlsiml.utils import iterable

DEFAULT_CLASSIFIERS = flatten([
            Classifier.for_logistic_regression(),
            [Classifier.for_knn(n_neighbors=n) for n in [1, 2, 10]],
            Classifier.for_linear_svm(),
            [Classifier.for_svm(kernel=k) for k in ['rbf', 'poly']],
            [Classifier.for_random_forest(n_estimators = k) for k in [10, 30, 50]],
            Classifier.for_gaussian_nb()
            ])

# Default run only uses one sample size
DEFAULT_EXPERIMENT_PARAMS = {
        "sample_sizes":5000,
        "train_proportions":0.7
        }


class Experiment:


    def __init__(self,
            network_class,
            network_params_dict,
            sample_sizes=5000,
            train_proportions=0.7,
            classifiers=None,
            output_file=None):

        self.network_class = network_class
        self.network_params_dict = network_params_dict
        self.sample_sizes = iterable(sample_sizes)
        self.train_proportions = iterable(train_proportions)

        # Default classifiers
        if not classifiers:
            classifiers = DEFAULT_CLASSIFIERS
        self.classifiers = classifiers

        # Default output file
        if not output_file:
            output_file = "experiment_output"
        self.output_file = output_file


    def run(self, verbose=False):

        # Save all results in a massive dictionary
        # Keys are network parameter dictionaries e.g. {num_z:4}
        # Values are result dictionaries {classifier:evalution_result}
        all_results = {}
        
        # Network parameters
        for setting in _all_possible_settings(self.network_params_dict):
            network = self.network_class(**setting)

            # Sample sizes
            for sample_size in self.sample_sizes:
                X, y = network.bulk_sample(sample_size)

                # Proportion to split to training and testing data
                for p_train in self.train_proportions:
                    datasplit = split_data(X, y, proportion_train=p_train)

                    # Add experiment params to settings for saving
                    setting["sample_size"] = sample_size
                    setting["training_proportion"] = p_train

                    # Verbose output
                    if verbose:
                        print("\n" + "="*79)
                        print("BUILDING NETWORK WITH PARAMETERS:")
                        for kw,val in setting.items():
                            print("\t{:>20} : {!s}".format(kw, val))
                        print()

                    # Classifiers
                    results = {}
                    for classifier in self.classifiers:
                        result = classifier.evaluate_on(datasplit)
                        results[classifier] = result

                        # Verbose output
                        if verbose:
                            print("\t{:8.3f}\t{!s}".format(result, classifier))

                    all_results[frozenset(setting.items())] = results

        return all_results


def _all_possible_settings(a_dict):

    # First map the dictionary into a list of lists of parameters
    listlist = [
            [(k, v) for v in iterable(val)]
            for k, val in a_dict.items()]

    # Iterate over all possible combinations of the above
    for setting in iter_product(*listlist): 
        
        # Create a new dictionary and return it
        yield dict(setting)

