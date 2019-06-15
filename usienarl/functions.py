# Import packages

import os.path
import datetime
import sys
import logging
import numpy
import argparse


def _positive_int(value: str) -> int:
    """
    TODO: summary

    :param value:
    :return:
    """
    int_value: int = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def command_line_parse():
    """
    TODO: summary

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace_directory", type=str)
    parser.add_argument("experiment_iterations", type=_positive_int)
    parser.add_argument("CUDA_devices", type=str)
    parser.add_argument("-render_during_training", action="store_true")
    parser.add_argument("-render_during_validation", action="store_true")
    parser.add_argument("-render_during_testing", action="store_true")
    args: dict = vars(parser.parse_args())
    return args["workspace_directory"], args["experiment_iterations"], args["CUDA_devices"], args["render_during_training"], args["render_during_validation"], args["render_during_testing"]


def run_experiments(experiments: [], experiment_iterations_number: int,
                    training_episodes: int, max_training_episodes: int,
                    validation_episodes: int,
                    testing_episodes: int, test_cycles: int,
                    print_trainable_variables: bool,
                    render_during_training: bool, render_during_validation: bool, render_during_test: bool,
                    workspace_path: str, file_name: str, logger: logging.Logger):
    """
    TODO: summary

    :param experiments:
    :param experiment_iterations_number:
    :param training_episodes:
    :param max_training_episodes:
    :param validation_episodes:
    :param testing_episodes:
    :param test_cycles:
    :param print_trainable_variables:
    :param render_during_training:
    :param render_during_validation:
    :param render_during_test:
    :param workspace_path:
    :param file_name:
    :param logger:
    :return:
    """
    # Generate the workspace directory if not defined
    workspace_directory = os.path.dirname(workspace_path)
    if not os.path.isdir(workspace_directory):
        try:
            os.makedirs(workspace_directory)
        except FileExistsError:
            pass
    # Generate the file directory with date if not defined
    file_path: str = workspace_path + "/" + os.path.splitext(os.path.basename(file_name))[
        0] + " " + datetime.date.today().strftime("%Y-%m-%d")
    if not os.path.isdir(file_path):
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass
    for experiment in experiments:
        # Generate a directory for the current experiment if not defined
        experiment_path: str = file_path + "/" + experiment.name
        if not os.path.isdir(experiment_path):
            try:
                os.makedirs(experiment_path)
            except FileExistsError:
                pass
        # Generate a directory for each iteration of the experiment if not defined
        # Prepare a table in which to store all results
        experiment_results_table: dict = dict()
        for iteration in range(experiment_iterations_number):
            iteration_path: str = experiment_path + "/" + str(iteration)
            if not os.path.isdir(iteration_path):
                try:
                    os.makedirs(iteration_path)
                except FileExistsError:
                    pass
            # Generate a metagraph and a summary directory for each iteration of experiment if not defined
            metagraph_path: str = iteration_path + "/metagraph"
            if not os.path.isdir(metagraph_path):
                try:
                    os.makedirs(metagraph_path)
                except FileExistsError:
                    pass
            summary_path: str = iteration_path + "/summary"
            if not os.path.isdir(summary_path):
                try:
                    os.makedirs(summary_path)
                except FileExistsError:
                    pass
            # Setup of the logger for the current experiment
            # Reset logger handlers for current experiment
            logger.handlers = []
            # Generate a console and a file handler for the logger
            console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            file_handler: logging.FileHandler = logging.FileHandler(iteration_path + "/info.log", "w+")
            # Set handlers properties
            console_handler.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
            formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            # Add the handlers to the logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            # Conduct the experiment
            if experiment.setup(iteration, logger):
                average_score, best_score, trained_episodes = experiment.conduct(training_episodes, validation_episodes,
                                                                                 max_training_episodes,
                                                                                 testing_episodes, test_cycles,
                                                                                 summary_path, metagraph_path,
                                                                                 logger,
                                                                                 print_trainable_variables,
                                                                                 render_during_training,
                                                                                 render_during_validation,
                                                                                 render_during_test,
                                                                                 iteration)
                # Save the result of the iteration in the table
                experiment_results_table[iteration] = (average_score, best_score, trained_episodes)
        # Print the results in both a results log file and in the console
        # Reset logger handlers for current experiment
        logger.handlers = []
        results_console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        results_file_handler: logging.FileHandler = logging.FileHandler(experiment_path + "/results.log", "w+")
        results_console_handler.setLevel(logging.INFO)
        results_file_handler.setLevel(logging.INFO)
        results_console_formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        results_file_formatter: logging.Formatter = logging.Formatter("%(message)s")
        results_console_handler.setFormatter(results_console_formatter)
        results_file_handler.setFormatter(results_file_formatter)
        # Add the file handler to the logger and print table of results
        logger.addHandler(results_file_handler)
        logger.info(experiment.name + " results:\n")
        logger.info("experiment iteration \t\t average score \t\t\t best score \t\t\t trained_episodes")
        for experiment_iteration, result in experiment_results_table.items():
            logger.info(str(experiment_iteration) + "\t\t\t\t" + str("%.3f" % result[0]) + "\t\t\t\t" + str(
                "%.3f" % result[1]) + "\t\t\t" + str(result[2]))
        logger.info("\n")
        # Add also the console handler and print verbose results
        logger.addHandler(results_console_handler)
        # Group all the final scores, best scores and training episode counts
        final_score_list: [] = []
        best_score_list: [] = []
        trained_episodes_list: [] = []
        for experiment_iteration, result in experiment_results_table.items():
            final_score_list.append(result[0])
            best_score_list.append(result[1])
            trained_episodes_list.append(result[2])
        max_final_score_index: int = numpy.argmax(final_score_list)
        max_best_score_index: int = numpy.argmax(best_score_list)
        min_training_episodes_index: int = numpy.argmin(trained_episodes_list)
        logger.info(
            "Maximum final average score over all " + str(test_cycles * testing_episodes) + " test episodes: " + str(
                "%.3f" % numpy.max(final_score_list)) + " at experiment number " + str(max_final_score_index))
        logger.info("Maximum best average score over " + str(testing_episodes) + " test episodes: " + str(
            "%.3f" % numpy.max(best_score_list)) + " at experiment number " + str(max_best_score_index))
        logger.info("Minimum training episodes for validation: " + str(
            numpy.min(trained_episodes_list)) + " at experiment number " + str(min_training_episodes_index))
        # Print average results for each indicator
        average_final_score: float = numpy.sum(final_score_list) / len(final_score_list)
        logger.info("Average of final average score over all experiments " + str("%.3f" % average_final_score))
        average_best_score: float = numpy.sum(best_score_list) / len(best_score_list)
        logger.info("Average of best average score over all experiments: " + str("%.3f" % average_best_score))
        average_training_episodes: float = numpy.sum(trained_episodes_list) / len(trained_episodes_list)
        logger.info("Average of training episodes over all experiments: " + str("%.3f" % average_training_episodes))
        # Print standard deviation for each indicator
        # Compute a list of averages for array difference
        final_score_average_list = numpy.full(len(final_score_list), average_final_score)
        best_score_average_list = numpy.full(len(best_score_list), average_best_score)
        training_episodes_average_list = numpy.full(len(trained_episodes_list), average_training_episodes)
        # Compute the actual standard deviation for each indicator
        stdv_final_score: float = numpy.sqrt(
            numpy.sum(numpy.power(final_score_list - final_score_average_list, 2)) / len(final_score_list))
        logger.info("Standard deviation of final average score over all experiments: " + str(
            "%.3f" % stdv_final_score) + ", which is a " + str(
            "%.2f" % ((100 * stdv_final_score) / average_final_score)) + "% of the average")
        stdv_best_score: float = numpy.sqrt(
            numpy.sum(numpy.power(best_score_list - best_score_average_list, 2)) / len(best_score_list))
        logger.info("Standard deviation of best average score over all experiments " + str(
            "%.3f" % stdv_best_score) + ", which is a " + str(
            "%.2f" % ((100 * stdv_best_score) / average_best_score)) + "% of the average")
        stdv_training_episodes: float = numpy.sqrt(
            numpy.sum(numpy.power(trained_episodes_list - training_episodes_average_list, 2)) / len(
                trained_episodes_list))
        logger.info("Standard deviation of training episodes over all experiments: " + str(
            "%.3f" % stdv_training_episodes) + ", which is a " + str(
            "%.2f" % ((100 * stdv_training_episodes) / average_training_episodes)) + "% of the average")
