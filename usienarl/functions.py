#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import os.path
import datetime
import sys
import logging
import numpy
import argparse

# Import required src

from usienarl import Experiment


def _positive_int(value: str) -> int:
    """
    Positive integer check/conversion rule for arg-parse.

    :param value: the value to check if it's integer
    :return: the int value if positive
    """
    int_value: int = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def command_line_parse():
    """
    Parse command line using arg-parse and get the default experiment user data.

    :return: the parsed arguments: workspace directory, experiment iterations, CUDA devices, optional render during training flag, optional render during validation flag, optional render during testing flag
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


def run_experiment(experiment: Experiment,
                   training_episodes: int,
                   max_training_episodes: int, episode_length_max: int,
                   validation_episodes: int,
                   testing_episodes: int, test_cycles: int,
                   render_during_training: bool, render_during_validation: bool, render_during_test: bool,
                   workspace_path: str, file_name: str, logger: logging.Logger,
                   experiment_iterations_number: int = 1):
    """
    Run the given experiment with the given parameters. It automatically creates all the directories to store the
    experiment results, summaries and meta-graphs for each iteration.

    :param experiment: the experiment to run
    :param training_episodes: the number of training episodes per volley
    :param max_training_episodes: the maximum number of training episodes in total
    :param episode_length_max: the maximum number of steps for each episode
    :param validation_episodes: the number of validation episodes per volley
    :param testing_episodes: the number of test episodes per test cycle
    :param test_cycles: the number of test cycles
    :param render_during_training: flag to render or not the environment during training
    :param render_during_validation: flag to render or not the environment during validation
    :param render_during_test: flag to render or not the environment during test
    :param workspace_path: path to workspace directory
    :param file_name: the name of the experiment script
    :param logger: the logger used to record information, warnings and errors
    :param experiment_iterations_number: the number of iterations of the experiment to run, by default just one
    """
    # Generate the workspace directory if not defined
    workspace_directory = os.path.dirname(workspace_path)
    if not os.path.isdir(workspace_directory):
        try:
            os.makedirs(workspace_directory)
        except FileExistsError:
            pass
    # Generate the file directory with date if not defined
    file_path: str = workspace_path + "/" + os.path.splitext(os.path.basename(file_name))[0] + "_" + datetime.date.today().strftime("%Y-%m-%d")
    if not os.path.isdir(file_path):
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass
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
        iteration_path: str = experiment_path
        # Set the iteration number to -1 if there is only one iteration
        if experiment_iterations_number <= 1:
            iteration = -1
        if iteration > -1:
            iteration_path = experiment_path + "/" + str(iteration)
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
        if experiment.setup(summary_path, metagraph_path, logger, iteration):
            average_total_reward, max_total_reward, average_scaled_reward, max_scaled_reward, trained_episodes, success = experiment.conduct(training_episodes, validation_episodes,
                                                                                                                                             max_training_episodes, episode_length_max,
                                                                                                                                             testing_episodes, test_cycles,
                                                                                                                                             logger,
                                                                                                                                             render_during_training,
                                                                                                                                             render_during_validation,
                                                                                                                                             render_during_test)
            # Save the result of the iteration in the table
            experiment_results_table[iteration] = (average_total_reward, max_total_reward, average_scaled_reward, max_scaled_reward, trained_episodes, success)
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
    logger.info("experiment iteration \t\t average total reward \t\t\t average scaled reward \t\t\t max total reward \t\t\t max scaled reward \t\t\t trained_episodes \t\t\t success")
    for experiment_iteration, result in experiment_results_table.items():
        logger.info(str(experiment_iteration) + "\t\t\t\t" + str("%.3f" % result[0]) + "\t\t\t\t" + str("%.3f" % result[1]) + "\t\t\t\t" + str("%.3f" % result[2]) + "\t\t\t\t" + str("%.3f" % result[3]) + "\t\t\t" + str(result[4]) + "\t\t\t" + "YES" if result[5] else "NO")
    logger.info("\n")
    # Add also the console handler and print verbose results
    logger.addHandler(results_console_handler)
    # Group all the final scores, best scores and training episode counts
    average_total_reward_list: [] = []
    average_scaled_reward_list: [] = []
    max_total_reward_list: [] = []
    max_scaled_reward_list: [] = []
    trained_episodes_list: [] = []
    success_list: [] = []
    for experiment_iteration, result in experiment_results_table.items():
        average_total_reward_list.append(result[0])
        max_total_reward_list.append(result[1])
        average_scaled_reward_list.append(result[2])
        max_scaled_reward_list.append(result[3])
        trained_episodes_list.append(result[4])
        success_list.append(result[5])
    max_average_total_reward_index: int = numpy.argmax(average_total_reward_list)
    max_max_total_reward_index: int = numpy.argmax(max_total_reward_list)
    max_average_scaled_reward_index: int = numpy.argmax(average_scaled_reward_list)
    max_max_scaled_reward_index: int = numpy.argmax(max_scaled_reward_list)
    min_training_episodes_index: int = numpy.argmin(trained_episodes_list)
    # Print best results for each indicator
    logger.info("Maximum average total reward over all " + str(test_cycles * testing_episodes) + " test episodes: " + str("%.3f" % numpy.max(average_total_reward_list)) + " at experiment number " + str(max_average_total_reward_index))
    logger.info("Maximum max total score over " + str(testing_episodes) + " test episodes: " + str("%.3f" % numpy.max(max_total_reward_list)) + " at experiment number " + str(max_max_total_reward_index))
    logger.info("Maximum average scaled reward over all " + str(test_cycles * testing_episodes) + " test episodes: " + str("%.3f" % numpy.max(average_scaled_reward_list)) + " at experiment number " + str(max_average_scaled_reward_index))
    logger.info("Maximum max scaled score over " + str(testing_episodes) + " test episodes: " + str("%.3f" % numpy.max(max_scaled_reward_list)) + " at experiment number " + str(max_max_scaled_reward_index))
    logger.info("Minimum training episodes for validation: " + str(numpy.min(trained_episodes_list)) + " at experiment number " + str(min_training_episodes_index))
    # Print average results for each indicator
    average_average_total_reward: float = numpy.average(average_total_reward_list)
    logger.info("Average of average total reward over all experiments " + str("%.3f" % average_average_total_reward))
    average_max_total_reward: float = numpy.average(max_total_reward_list)
    logger.info("Average of max total reward over all experiments: " + str("%.3f" % average_max_total_reward))
    average_average_scaled_reward: float = numpy.average(average_scaled_reward_list)
    logger.info("Average of average scaled reward over all experiments " + str("%.3f" % average_average_scaled_reward))
    average_max_scaled_reward: float = numpy.average(max_scaled_reward_list)
    logger.info("Average of max scaled reward over all experiments: " + str("%.3f" % average_max_scaled_reward))
    average_training_episodes: float = numpy.average(trained_episodes_list)
    logger.info("Average of training episodes over all experiments: " + str("%.3f" % average_training_episodes))
    success_percentage: float = numpy.round(numpy.count_nonzero(success_list), 2)
    logger.info("Success percentage over all experiments: " + str("%.2f" % success_percentage))
    # Print standard deviation for each indicator
    # Compute a list of averages for array difference
    average_total_reward_average_list = numpy.full(len(average_total_reward_list), average_average_total_reward)
    max_total_reward_average_list = numpy.full(len(max_total_reward_list), average_max_total_reward)
    average_scaled_reward_average_list = numpy.full(len(average_scaled_reward_list), average_average_scaled_reward)
    max_scaled_reward_average_list = numpy.full(len(max_scaled_reward_list), average_max_scaled_reward)
    training_episodes_average_list = numpy.full(len(trained_episodes_list), average_training_episodes)
    # Compute the actual standard deviation for each indicator
    stdv_average_total_reward: float = numpy.sqrt(numpy.sum(numpy.power(average_total_reward_list - average_total_reward_average_list, 2)) / len(average_total_reward_list))
    logger.info("Standard deviation of average total reward over all experiments: " + str("%.3f" % stdv_average_total_reward) + ", which is a " + str("%.2f" % ((100 * stdv_average_total_reward) / average_average_total_reward)) + "% of the average")
    stdv_max_total_reward: float = numpy.sqrt(numpy.sum(numpy.power(max_total_reward_list - max_total_reward_average_list, 2)) / len(max_total_reward_list))
    logger.info("Standard deviation of max total reward over all experiments " + str("%.3f" % stdv_max_total_reward) + ", which is a " + str("%.2f" % ((100 * stdv_max_total_reward) / average_max_total_reward)) + "% of the average")
    stdv_average_scaled_reward: float = numpy.sqrt(numpy.sum(numpy.power(average_scaled_reward_list - average_scaled_reward_average_list, 2)) / len(average_scaled_reward_list))
    logger.info("Standard deviation of average scaled reward over all experiments: " + str("%.3f" % stdv_average_scaled_reward) + ", which is a " + str("%.2f" % ((100 * stdv_average_scaled_reward) / average_average_scaled_reward)) + "% of the average")
    stdv_max_scaled_reward: float = numpy.sqrt(numpy.sum(numpy.power(max_scaled_reward_list - max_scaled_reward_average_list, 2)) / len(max_scaled_reward_list))
    logger.info("Standard deviation of max scaled reward over all experiments " + str("%.3f" % stdv_max_scaled_reward) + ", which is a " + str("%.2f" % ((100 * stdv_max_scaled_reward) / average_max_scaled_reward)) + "% of the average")
    stdv_training_episodes: float = numpy.sqrt(numpy.sum(numpy.power(trained_episodes_list - training_episodes_average_list, 2)) / len(trained_episodes_list))
    logger.info("Standard deviation of training episodes over all experiments: " + str("%.3f" % stdv_training_episodes) + ", which is a " + str("%.2f" % ((100 * stdv_training_episodes) / average_training_episodes)) + "% of the average")
