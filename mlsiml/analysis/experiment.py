
import csv
from datetime import datetime
from itertools import product as iter_product
import logging
import os
from pandas import DataFrame
from scipy.io import savemat

from mlsiml.classification import classifiers as Classifier
from mlsiml.utils import dict_prefix, filter_truish_str, flatten, make_iterable


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
            test_size=0.3,
            save_data=False,
            mat_dir=None,
            mat_fileformat=None
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
        if (not network_class or not network_params_dict or not workflows
                or not sample_sizes):
            raise Exception(
                    "Parameters to Experiment are not optional. "
                    "All must be specified"
                    )

        # Mat dir and fileformat must be specified if saving data
        if save_data and (not mat_dir or not mat_fileformat):
            raise Exception(
                    "Mat_dir and fileformat must be specified if saving data"
                    )

        self.network_class = network_class
        self.network_params_dict = network_params_dict
        self.workflows = workflows
        self.sample_sizes = make_iterable(sample_sizes)
        self.test_size = test_size
        self.save_data= save_data
        self.mat_dir = mat_dir
        self.mat_fileformat = mat_fileformat


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

            print(
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
                print(
                    "Sample {!s} points with {!s}/{!s} train-test split".format(
                        sample_size, 1 - self.test_size, self.test_size
                        )
                    )

                network_setting["sample_size"] = sample_size
                sources = network.sample(sample_size, self.test_size)

                # If we just want data, export everything as matlab matrices
                if self.save_data:
                    X, y = sources.as_X_and_Y()
                    split_idx = sources[0].X_train.shape[1] # TODO fix this

                    print(network_setting)
                    print(self.mat_fileformat)
                    savemat(
                            os.path.join(
                                self.mat_dir,
                                self.mat_fileformat.format(**network_setting)
                                ),
                            {"X":X, "y":y, "separator":[split_idx, X.shape[1]]}
                            )


                # For every workflows
                final_workflow_records = []
                cv_records = []
                for workflow in self.workflows:
                    accuracy, record, cv_params = workflow.evaluate_on(sources)

                    # Output accuracy to the console
                    print("\t{:8.3f}\t{!s}".format(accuracy, workflow))

                    # Record all parameters for writing to a csv later
                    final_workflow_records.append(record)
                    cv_records += cv_params

                # After every network, write out the log
                # This is done here because we only know all of the parameters
                # (all of the CV grid search parameters) after all of the
                # workflows have been evaluated
                all_results.write_records(
                        network_setting, final_workflow_records, cv_records
                        )

        return all_results


##############################################################################
# Experiment Results Object Class Definition                                 #
##############################################################################
# Constant for accessing which logfile/records/etc.
FOR_CV = True
NOT_FOR_CV = False

class ExperimentResults:

    def __init__(self, logfile=None):
        """Prepares the logfile by writing a header row to it"""

        # Default logfile is just experiment_date
        if not logfile:
            logfile = "experiment_" + str(datetime.now())[:10] + ".csv"

        # Validate logfile name
        ######################################################################
        # Find a place to put the log, making sure never to overwrite a
        # previous log by adding [1] or [2] etc to the file name
        if os.path.isfile(logfile + ".csv"):
            logging.info(
                    "Experiment logfile {} already exists".format(
                        logfile + ".csv"
                        )
                    )
            run = 1
            while os.path.isfile("{}({}).csv".format(logfile, run)):
                run += 1
            logfile = "{}({}).csv".format(logfile, run)
            logging.info(
                    "Writing experiment results to {}.csv instead".format(
                        logfile
                        )
                    )

        self.logfile = {NOT_FOR_CV:logfile + ".csv", FOR_CV:logfile + "_cv.csv"}
        self.records = {NOT_FOR_CV:[], FOR_CV:[]}
        self.header_rows_written = False

    def write_records(
            self, network_settings_dict, final_records_arr, cv_records_arr
            ):

        # Make the unified row for this network setting
        final_record = dict_prefix("network", network_settings_dict)
        for record in final_records_arr:
            final_record.update(record)
        final_record = filter_truish_str(final_record)

        # Write a header file if we need one
        # This is done here instead of in __init__ because it is very hard to
        # know all of the parameters (column names) in __init__. This is also
        # done after making the unified row so that we have those headers too
        if not self.header_rows_written:
            self.write_header_rows(
                    final_record.keys(), map(lambda p: p.keys(), cv_records_arr)
                    )
            self.header_rows_written = True

        # Write out final record
        self.records[NOT_FOR_CV].append(final_record)
        with open(self.logfile[NOT_FOR_CV], 'a') as fobj:
            csvfile = csv.writer(fobj)
            csvfile.writerow([
                final_record.get(col, "")
                for col in self.column_names[NOT_FOR_CV]
                ])

        # Write out all of the CV records
        with open(self.logfile[FOR_CV], 'a') as fobj:
            csvfile = csv.writer(fobj)

            for cv_record in cv_records_arr:

                # Build CV record off of final record
                full_cv_record = final_record.copy()
                full_cv_record.update(cv_record)

                # Write the full cv record
                self.records[FOR_CV].append(full_cv_record)
                csvfile.writerow([
                    full_cv_record.get(col, "")
                    for col in self.column_names[FOR_CV]
                    ])

    def write_header_rows(self, final_keys, cv_keys):
        self.column_names = {}

        # Headers for non-cv are easy
        self.column_names[NOT_FOR_CV] = set(final_keys)

        # Headers for CV are built off of non-cv
        self.column_names[FOR_CV] = set(final_keys)
        for additional_keys in cv_keys:
            self.column_names[FOR_CV].update(set(additional_keys))

        # Write the headers
        for for_cv in [FOR_CV, NOT_FOR_CV]:
            self.column_names[for_cv] = list(sorted(self.column_names[for_cv]))
            with open(self.logfile[for_cv], 'w') as fobj:
                csvfile = csv.writer(fobj)
                csvfile.writerow(self.column_names[for_cv])

        self.header_row_written = True

    def as_dataframes(self):
        return (
                DataFrame.from_records(self.records[NOT_FOR_CV]),
                DataFrame.from_records(self.records[FOR_CV])
                )

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

