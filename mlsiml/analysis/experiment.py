
import csv
from datetime import datetime
from itertools import product as iter_product
import logging
import os
from pandas import DataFrame

from mlsiml.classification import classifiers as Classifier
from mlsiml.utils import flatten
from mlsiml.utils import make_iterable


##############################################################################
# Experiment Class Definition                                                #
##############################################################################

class Experiment():
    """An experiment is a lot of workflows tested on a lot of networks.

    An experiment is defined by:
        A network to make data with
        A set of parameters to make the network with
        A list of workflows to evaluate on the data
        A list of sample sizes to repeat the experiment with
    """


    def __init__(self,
            network_class=None,
            network_params_dict=None,
            workflows=None,
            sample_sizes=None,
            test_size=0.3
            ):
        """Prepares an experiment which can then be run with "experiment.run()"

        Params
        NOTE, none of these are actually optional. You must specify all of them
        ======
        network_class   - The class of a BayesNetwork. This is the CLASS not
            the instance, so calling network_class()
            should create a new instance of the network with the given
            parameters

        network_params_dict - A dictionary of parameter options to make
            networks with. This isn't just keyword:value arguments, but can
            (should) also include keyword:[values to try]. E.g. if "dimension"
            is a key in network_params_dict with value [3, 5, 7] then this
            experiment will run over networks with dimensions of 3, 5, and 7.
            Specifically, a new network will be created for every possible
            combination of parameters in this dictionary (the cartesian product
            of all values of this dictionary).

        workflows   - Array-like of Workflow instances. For every network that
            is created in this experiment, all of these workflows will be
            evaluated on it.

        sample_sizes    - Array-like of sample sizes to repeat this experiment
            over. This has to be array-like, so wrap single values in lists
            e.g. sample_sizes=[5000] instead of sample_sizes=5000

        test_size   - (Optional) The proportion of the data (given in
            sample_sizes) to use for testing data
        """

        # Parameters aren't actually optional
        if not network_class or not network_params_dict or not workflows or not sample_sizes:
            raise Exception("Parameters to Experiment are not optional. All must be specified")

        self.network_class = network_class
        self.network_params_dict = network_params_dict
        self.workflows = workflows
        self.sample_sizes = make_iterable(sample_sizes)
        self.test_size = test_size

        # Collect all possible keywords for logging purposes
        self.all_possible_keywords = sorted(list(set(
                ['accuracy', 'CV_mean_accuracy'] +
                ['sample_size', 'test_size'] +
                list(self.network_params_dict.keys()) +
                flatten([list(w.get_params(deep=True, mangled=True).keys())
                    for w in self.workflows])
                )))


    def run(self, logfile=None):
        """Runs this experiment, writing outputs to logfile.
        This may take a very long time and use lots of CPU
        """

        # Save all results in a massive list of dictionaries
        # Keys are network parameter dictionaries e.g. {num_z:4}
        # Values are result dictionaries {workflow:evalution_result}
        all_results = ExperimentResults(self.all_possible_keywords, logfile=logfile)

        # For every network parameter
        for network_setting in _all_possible_settings(self.network_params_dict):
            network = self.network_class(**network_setting)
            network_setting["test_size"] = self.test_size

            logging.info(
                "\n"
                + "==========================================================="
                + "\nBuilding Network With Parameters:\n"
                + "\n".join(["\t{:>20} : {!s}".format(k, v) for k, v in network_setting.items()])
                + "\n")

            # For every sample size
            for sample_size in self.sample_sizes:
                logging.info(
                        "Sample {!s} points with {!s}/{!s} train-test split".format(
                            sample_size, 1 - self.test_size, self.test_size))

                network_setting["sample_size"] = sample_size
                sources = network.sample(sample_size, self.test_size)

                # For every workflows
                for workflow in self.workflows:
                    accuracy = workflow.evaluate_on(sources)
                    all_results.add_record_for(accuracy, network_setting, workflow)

        return all_results


##############################################################################
# Experiment Results Object Class Definition                                 #
##############################################################################


class ExperimentResults:

    def __init__(self, column_names, logfile=None):
        """Prepares the logfile by writing a header row to it"""
        self.records = []
        self.logfile = logfile

        # Default logfile is just experiment_date
        if not logfile:
            self.logfile = "experiment_" + str(datetime.now())[:10] + ".csv"

        # Validate logfile name
        ######################################################################
        # Find a place to put the log, making sure never to overwrite a
        # previous log by adding [1] or [2] etc to the file name
        if os.path.isfile(self.logfile + ".csv"):
            logging.info("Experiment logfile {} already exists".format(logfile + ".csv"))
            run = 1
            while os.path.isfile("{}({!s}).csv".format(self.logfile, run)):
                run += 1
            self.logfile = "{}({!s}).csv".format(self.logfile, run)
            logging.info("Writing experiment results to {}.csv instead".format(self.logfile))
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

    def add_record_for(self, accuracy, network_settings_dict, workflow):

        # Always append the workflow's last evaluation record
        record = _make_record(accuracy, network_settings_dict, workflow.get_params())
        self.records.append(record)
        self.log(record)

        # Verbose output
        logging.debug("\t{:8.3f}\t{!s}".format(accuracy, workflow))

        # CV Grid Search full record
        # If the workflow went through a CV grid search, add those results
        # too
        if hasattr(workflow.classifier, 'full_record'):
            for cvr in workflow.full_record:
                record = _make_record(network_settings_dict, cvr)
                self.records.append(record)
                self.log(record)

    def as_dataframe(self):
        return DataFrame.from_records(self.records)

    def log(self, record):

        # Build up row, using "" when a parameter is not found
        row = [record[kw] if kw in record else "" for kw in self.column_names]

        # Assume that network_settings will be the same every time
        with open(self.logfile, 'a') as csvfile:
            logfile = csv.writer(csvfile)
            logfile.writerow(row)

##############################################################################
# Private Helper Functions                                                   #
##############################################################################


def _make_record(accuracy, network_dict, workflow_dict):
    record = {"network_"+k : v for k, v in network_dict.items()}

    # Merge dictionaries
    for k,v in workflow_dict.items():

        # Make sure that we don't overwrite a setting
        # TODO better error checking? Maybe just prepend 'network' to all
        # network settings?
        if k in record:
            raise Exception(
                'Workflow setting {} already in network settings {}'.format(
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
            [(k, v) for v in make_iterable(val)]
            for k, val in a_dict.items()]

    # Iterate over all possible combinations of the above
    for setting in iter_product(*listlist):

        # Create a new dictionary and return it
        yield dict(setting)

