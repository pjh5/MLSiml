
import csv
from datetime import datetime
from itertools import product as iter_product
import logging
import os
from pandas import DataFrame

from mlsiml.classification import classifiers as Classifier
from mlsiml.utils import dict_prefix, flatten, make_iterable, truish


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


    def run(self, logfile=None):
        """Runs this experiment, writing outputs to logfile.
        This may take a very long time and use lots of CPU
        """

        # Save all results in a massive list of dictionaries
        # Keys are network parameter dictionaries e.g. {num_z:4}
        # Values are result dictionaries {workflow:evalution_result}
        all_results = ExperimentResults(logfile=logfile)

        # For every network parameter
        for network_setting in _all_possible_settings(self.network_params_dict):
            network = self.network_class(**network_setting)
            network_setting["test_size"] = self.test_size

            logging.info(
                "\n"
                + "==========================================================="
                + "\nBuilding Network With Parameters:\n"
                + "\n".join(
                    ["\t{:>20} : {!s}".format(k, v)
                        for k, v in network_setting.items()]
                    )
                + "\n"
                )

            # For every sample size
            for sample_size in self.sample_sizes:
                logging.info(
                        "Sample {!s} points with {!s}/{!s} train-test split".format(
                            sample_size, 1 - self.test_size, self.test_size
                            )
                        )

                network_setting["sample_size"] = sample_size
                sources = network.sample(sample_size, self.test_size)

                # For every workflows
                for workflow in self.workflows:
                    accuracy = workflow.evaluate_on(sources)
                    all_results.add_record_for(accuracy, network_setting, workflow)

                # After every network, write out the log
                # This is done here because we only know all of the parameters
                # (all of the CV grid search parameters) after all of the
                # workflows have been evaluated
                all_results.write()

        return all_results


##############################################################################
# Experiment Results Object Class Definition                                 #
##############################################################################


class ExperimentResults:

    def __init__(self, logfile=None):
        """Prepares the logfile by writing a header row to it"""
        self.records = []
        self.new_records = []
        self.logfile = logfile
        self.header_file_written = False

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

    def add_record_for(self, accuracy, network_settings_dict, workflow):

        # Mangle network settings
        net_work_record = dict_prefix("network", network_settings_dict)
        net_work_record.update(workflow.get_params(deep=True))
        net_work_record = {k:v for k, v in net_work_record.items() if truish(v)}

        # Append the "simple" record of accuracy + network/workflow params
        rec_with_acc = net_work_record.copy()
        rec_with_acc["accuracy"] = accuracy
        self.new_records.append(rec_with_acc)

        # Verbose output
        logging.debug("\t{:8.3f}\t{!s}".format(accuracy, workflow))

        # Cross-Validation data
        # For GridSearchCV classifiers, there's a lot of data hidden in
        # cv_results_. Add all of that data here too.
        all_cv_params = workflow.get_cv_params()
        for wf, cv_results in all_cv_params.items():

            # Mangle with which workflow if more than one CV result
            # TODO in the future only mangle if CV names are the same too
            prefix = wf if len(all_cv_params) > 1 else None

            # Extract parameters that were changed
            logging.debug("CV_PARAMS are {!s}".format(cv_results))

            # TODO only keep interesting parameters from cv results
            # cv_params = [p[6:] for p in cv_results.keys() if p.startswith('param_')]

            # Loop over all cross validation runs, adding a record for each
            for i in range(len(cv_results['mean_test_score'])):

                # Copy the ith cv_results into a new record
                record = dict_prefix(
                        prefix, {k:v[i] for k, v in cv_results.items() if truish(v)}
                        )

                # Combine the ith cv_results with the net/flow settings
                record.update(net_work_record)

                # Append the record
                self.new_records.append(record)


    def as_dataframe(self):
        return DataFrame.from_records(self.records)

    def write(self):

        # Write a header file if we need one
        # This is done here instead of in __init__ because it is very hard to
        # know all of the parameters (column names) in __init__
        if not self.header_file_written:

            # Determine column names
            self.column_names = set()
            for record in self.new_records:
                self.column_names.update(set(record.keys()))
            self.column_names = list(sorted(self.column_names))
            logging.debug("Found column names: {!s}".format(self.column_names))

            # Write the header
            with open(self.logfile, 'w') as fobj:
                csvfile = csv.writer(fobj)
                csvfile.writerow(self.column_names)

            self.header_file_written = True

        # Write out all of the new records
        with open(self.logfile, 'a') as fobj:
            csvfile = csv.writer(fobj)

            for record in self.new_records:
                csvfile.writerow(
                        [record.get(col, "") for col in self.column_names]
                        )

        # Move the new_records to records
        self.records += self.new_records
        self.new_records = []



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

