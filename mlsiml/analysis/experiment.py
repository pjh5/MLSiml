
import csv
from datetime import datetime
from itertools import product as iter_product
import os
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from mlsiml.classification import classifiers as Classifier
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


##############################################################################
# Experiment Class Definition                                                #
##############################################################################

class Experiment:


    def __init__(self,
            network_class,
            network_params_dict,
            sample_sizes=5000,
            train_proportions=0.7,
            classifiers=None):

        self.network_class = network_class
        self.network_params_dict = network_params_dict
        self.sample_sizes = iterable(sample_sizes)
        self.train_proportions = iterable(train_proportions)

        # Default classifiers
        if not classifiers:
            classifiers = DEFAULT_CLASSIFIERS
        self.classifiers = classifiers

        # Collect all possible keywords for logging purposes
        self.all_possible_keywords = sorted(list(set(
                ['accuracy', 'CV_mean_accuracy'] +
                ['sample_size', 'training_proportions'] +
                list(self.network_params_dict) +
                flatten([list(c.get_params(deep=True)) for c in self.classifiers])
                )))


    def run(self, logfile=None, verbosity=0):

        # Save all results in a massive list of dictionaries
        # Keys are network parameter dictionaries e.g. {num_z:4}
        # Values are result dictionaries {classifier:evalution_result}
        all_results = ExperimentResults(self.all_possible_keywords,
                                        logfile=logfile, verbosity=verbosity)

        # For every network parameter
        for setting in _all_possible_settings(self.network_params_dict):
            network = self.network_class(**setting)

            # For every sample size
            for sample_size in self.sample_sizes:
                X, y = network.bulk_sample(sample_size)

                # For every proportion to split to training and testing data
                for p_train in self.train_proportions:
                    datasplit = train_test_split(X, y, test_size=1 - p_train)

                    # Add experiment params to settings for saving
                    setting["sample_size"] = sample_size
                    setting["training_proportion"] = p_train

                    # Verbose output
                    if verbosity >= 0:
                        print("\n" + "="*79)
                        print("BUILDING NETWORK WITH PARAMETERS:")
                        for kw,val in setting.items():
                            print("\t{:>20} : {!s}".format(kw, val))
                        print()

                    # For every classifiers
                    for classifier in self.classifiers:
                        accuracy = classifier.evaluate_on(*datasplit)
                        all_results.add_record_for(setting, classifier)

        return all_results


##############################################################################
# Experiment Results Object Class Definition                                 #
##############################################################################


class ExperimentResults:

    def __init__(self, column_names, logfile=None, verbosity=0):
        self.records = []
        self.verbosity = verbosity
        self.logfile = logfile

        # Default logfile is just experiment_date
        if not logfile:
            self.logfile = "experiment_" + str(datetime.now())[:10] + ".csv"

        # Validate logfile name
        ######################################################################
        # Find a place to put the log, making sure never to overwrite a
        # previous log by adding [1] or [2] etc to the file name
        if os.path.isfile(self.logfile + ".csv"):
            run = 1
            while os.path.isfile("{}({!s}).csv".format(self.logfile, run)):
                run += 1
            self.logfile = "{}({!s}).csv".format(self.logfile, run)
        else:
            self.logfile = self.logfile + ".csv"

        # Write a header to the csv
        #####################################################################

        # Collect keywords in alphabetical order (so their order is defined)
        self.column_names = sorted(column_names)

        # Write the header row
        with open(self.logfile, 'w') as csvfile:
            logfile = csv.writer(csvfile)
            logfile.writerow(self.column_names)

    def add_record_for(self, settings_dict, classifier):

        # Always append the classifier's last evaluation record
        record = _make_record(settings_dict, classifier.last_evaluation_record)
        self.records.append(record)
        self.log(record)

        # Verbose output
        if self.verbosity >= 0:
            print("\t{:8.3f}\t{!s}".format(
                    classifier.last_evaluation_record['accuracy'], classifier))

        # CV Grid Search full record
        # If the classifier went through a CV grid search, add those results
        # too
        if hasattr(classifier, 'full_record'):
            for cvr in classifier.full_record:
                record = _make_record(settings_dict, cvr)
                self.records.append(record)
                self.log(record)

    def as_dataframe(self):
        return DataFrame.from_records(self.records)

    def log(self, record):

        # Build up row, using "" when a parameter is not found
        row = [record[kw] if kw in record else "" for kw in self.column_names]

        # Assume that settings will be the same every time
        with open(self.logfile, 'a') as csvfile:
            logfile = csv.writer(csvfile)
            logfile.writerow(row)

##############################################################################
# Private Helper Functions                                                   #
##############################################################################


def _make_record(network_dict, classifier_dict):
    record = network_dict.copy()

    # Merge dictionaries
    for k,v in classifier_dict.items():

        # Make sure that we don't overwrite a setting
        # TODO better error checking? Maybe just prepend 'network' to all
        # network settings?
        if k in record:
            raise Error(
                'Classifier setting {} already in network settings {}'.format(
                                                                    k, record))
        record[k] = v

    return record


def _all_possible_settings(a_dict):
    """Iterator over dictionaries of single key:value pairs

    Given a dictionary of key:<array of values> pairs, this function iterates
    over all possible combinations of single key:value pairs.
    """

    # First map the dictionary into a list of lists of parameters
    listlist = [
            [(k, v) for v in iterable(val)]
            for k, val in a_dict.items()]

    # Iterate over all possible combinations of the above
    for setting in iter_product(*listlist):

        # Create a new dictionary and return it
        yield dict(setting)

