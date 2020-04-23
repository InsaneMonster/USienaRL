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
import pandas

# Import required src

from usienarl import Experiment


def softmax(sequence):
    """
    Compute the softmax of a sequence of values, i.e. the resulting probability distribution over that sequence.

    :param sequence: the sequence of which to compute the softmax
    :return: the probability distribution over the sequence
    """
    # Make sure the sequence is a numpy array
    if not isinstance(sequence, numpy.ndarray):
        sequence = numpy.array(sequence)
    # Make sure the length of the shape of the given array is 2
    assert len(sequence.shape) == 2
    # Get the element with max value in the given array as the normalization factor
    normalization_factor = numpy.max(sequence, axis=1)
    # Increase the shape size of the normalization factor to allow broadcasting
    normalization_factor = normalization_factor[:, numpy.newaxis]
    # Apply the normalization
    numerator = numpy.exp(sequence - normalization_factor)
    # Compute the denominator by summing all the normalized numerator elements
    denominator = numpy.sum(numerator, axis=1)
    # Increase the shape size of the denominator to allow broadcasting
    denominator = denominator[:, numpy.newaxis]
    # Return the softmax result probability distribution
    return numerator / denominator


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
    Parse command line using arg-parse and get built-in run experiment user data.

    :return: the parsed arguments: workspace directory, experiment iterations, CUDA devices, optional render during training flag, optional render during validation flag, optional render during testing flag
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace_directory", type=str)
    parser.add_argument("experiment_iterations", type=_positive_int)
    parser.add_argument("CUDA_devices", type=str)
    parser.add_argument("-render_during_training", action="store_true")
    parser.add_argument("-render_during_validation", action="store_true")
    parser.add_argument("-render_during_testing", action="store_true")
    args: dict = vars(parser.parse_args())
    return args["workspace_directory"], args["experiment_iterations"], args["CUDA_devices"], args["render_during_training"], args["render_during_validation"], args["render_during_testing"]


def run_experiment(logger: logging.Logger,
                   experiment: Experiment,
                   file_name: str, workspace_path: str,
                   training_volleys_episodes: int,
                   validation_volleys_episodes: int,
                   training_validation_volleys: int,
                   test_volleys_episodes: int,
                   test_volleys: int,
                   episode_length: int,
                   parallel: int = 1,
                   plots_dpi: int = 150,
                   render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False,
                   saves_to_keep: int = 1,
                   restore_path: str = None,
                   iterations: int = 1):
    """
    Run the given experiment with the given parameters. It automatically creates all the directories to store the
    experiment results, summaries and meta-graphs for each iteration.

    :param logger: the logger used to record information, warnings and errors
    :param experiment: the experiment to run
    :param file_name: the name of the experiment script
    :param workspace_path: the path of workspace directory
    :param training_volleys_episodes: the number of episodes per training volley
    :param validation_volleys_episodes: the number of episodes per validation volley
    :param training_validation_volleys: the number of training and validation volleys
    :param test_volleys_episodes: the number of episodes per test volley
    :param test_volleys: the number of test volleys
    :param episode_length: the maximum number of steps allowed for each episode in any volley
    :param parallel: the number of parallel episodes to run in the experiment
    :param plots_dpi: the dpi (quality) of plots saved during experiment run
    :param render_during_training: flag to render or not the environment during training
    :param render_during_validation: flag to render or not the environment during validation
    :param render_during_test: flag to render or not the environment during test
    :param saves_to_keep: the number of checkpoint metagraphs to keep
    :param restore_path: the optional checkpoint path from which to restore the model
    :param iterations: the number of iterations of the exact same experiment to run
    :return: the list of saved metagraph paths of each iteration of the experiment
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
    # Prepare a table in which to store all results and a list of metagraph paths
    experiment_results_table: dict = dict()
    for iteration in range(iterations):
        iteration_path: str = experiment_path
        # Set the iteration number to -1 if there is only one iteration
        if iterations <= 1:
            iteration = -1
        if iteration > -1:
            iteration_path = experiment_path + "/" + str(iteration)
            if not os.path.isdir(iteration_path):
                try:
                    os.makedirs(iteration_path)
                except FileExistsError:
                    pass
        # Generate a metagraph, summary and plots directory for each iteration of experiment if not defined
        plots_path: str = iteration_path + "/plots"
        if not os.path.isdir(plots_path):
            try:
                os.makedirs(plots_path)
            except FileExistsError:
                pass
        save_path: str = iteration_path + "/metagraph"
        if not os.path.isdir(save_path):
            try:
                os.makedirs(save_path)
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
        # Setup the experiment
        if experiment.setup(logger, plots_path, plots_dpi, parallel, summary_path, save_path, saves_to_keep, iteration):
            # Train the agent
            experiment.train(logger, training_volleys_episodes, validation_volleys_episodes, training_validation_volleys, episode_length,
                             render_during_training, render_during_validation, restore_path)
            # Test the agent
            experiment.test(logger, test_volleys_episodes, test_volleys, episode_length, save_path, render_during_test)
            # Save the result of the iteration in the table
            experiment_results_table[iteration] = ("YES" if experiment.successful else "NO",
                                                   numpy.round(experiment.avg_test_avg_total_reward, 3),
                                                   numpy.round(experiment.max_test_avg_total_reward, 3),
                                                   numpy.round(experiment.avg_test_avg_scaled_reward, 3),
                                                   numpy.round(experiment.max_test_avg_scaled_reward, 3),
                                                   experiment.trained_episodes)
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
    # Write results
    logger.info(experiment.name + " results:\n")
    with pandas.option_context('display.max_rows', 500, 'display.max_columns', 500, 'display.width', 500):
        data_frame = pandas.DataFrame.from_dict(experiment_results_table, orient="index")
        data_frame.index.name = 'Experiment iteration'
        data_frame.columns = ['Success', 'Average total reward', 'Max total reward', 'Average scaled reward', 'Max scaled reward', 'Trained episodes']
        # Print data frame removing index if there is only one iteration
        if iterations <= 1:
            logger.info(data_frame.to_string(index=False))
        else:
            logger.info(data_frame.to_string())
    logger.info("\n\n")
    # Also write results to csv file
    data_frame.to_csv(experiment_path + "/results_table.csv")
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
        success_list.append(True if result[0] == "YES" else False)
        average_total_reward_list.append(result[1])
        max_total_reward_list.append(result[2])
        average_scaled_reward_list.append(result[3])
        max_scaled_reward_list.append(result[4])
        trained_episodes_list.append(result[5])
    max_average_total_reward_index: int = int(numpy.argmax(average_total_reward_list))
    max_max_total_reward_index: int = int(numpy.argmax(max_total_reward_list))
    max_average_scaled_reward_index: int = int(numpy.argmax(average_scaled_reward_list))
    max_max_scaled_reward_index: int = int(numpy.argmax(max_scaled_reward_list))
    min_training_episodes_index: int = int(numpy.argmin(trained_episodes_list))
    # Print best results for each indicator
    logger.info("Maximum average total reward over all " + str(test_volleys * test_volleys_episodes) + " test episodes: " + str("%.3f" % numpy.max(average_total_reward_list)) + " at experiment number " + str(max_average_total_reward_index))
    logger.info("Maximum max total score over " + str(test_volleys_episodes) + " test episodes: " + str("%.3f" % numpy.max(max_total_reward_list)) + " at experiment number " + str(max_max_total_reward_index))
    logger.info("Maximum average scaled reward over all " + str(test_volleys * test_volleys_episodes) + " test episodes: " + str("%.3f" % numpy.max(average_scaled_reward_list)) + " at experiment number " + str(max_average_scaled_reward_index))
    logger.info("Maximum max scaled score over " + str(test_volleys_episodes) + " test episodes: " + str("%.3f" % numpy.max(max_scaled_reward_list)) + " at experiment number " + str(max_max_scaled_reward_index))
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
    success_percentage: float = numpy.round(100.0 * numpy.count_nonzero(success_list) / len(success_list), 2)
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
