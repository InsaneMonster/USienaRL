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


def command_line_parse(watch: bool = False):
    """
    Parse command line using arg-parse and get the default experiment user data.

    :param watch: optional boolean flag defining if the command line parse is for a watch script or for a regular experiment script (the default one)
    :return: the parsed arguments: workspace directory, experiment iterations, CUDA devices, optional render during training flag, optional render during validation flag, optional render during testing flag
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    if watch:
        parser.add_argument("checkpoint_directory", type=str)
        parser.add_argument("iteration_number", type=int)
        parser.add_argument("CUDA_devices", type=str)
        parser.add_argument("-render", action="store_true")
        args: dict = vars(parser.parse_args())
        return args["checkpoint_directory"], args["iteration_number"], args["CUDA_devices"], args["render"]
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
                   checkpoint_path: str = None, scopes_to_restore: [] = None,
                   experiment_iterations_number: int = 1,
                   intro: str = None,
                   plot_sample_density_training: int = 1, plot_sample_density_validation: int = 1) -> []:
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
    :param checkpoint_path: the optional checkpoint path to gather the model from, useful to further train an already trained model
    :param scopes_to_restore: the list of scopes to restore with the checkpoint, if any
    :param experiment_iterations_number: the number of iterations of the experiment to run, by default just one
    :param intro: the optional string intro of the experiment, describing what the experiment is about
    :param plot_sample_density_training: the optional number represent after how many episodes a sample for the episode-related training plots is sampled (default to 1, i.e. each episode)
    :param plot_sample_density_validation: the optional number represent after how many episodes a sample for the episode-related validation plots is sampled (default to 1, i.e. each episode)
    :return the list of saved metagraph paths of each iteration of the experiment
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
    experiment_metagraph_paths: [] = []
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
        # Generate a metagraph, summary and plots directory for each iteration of experiment if not defined
        plots_path: str = iteration_path + "/plots"
        if not os.path.isdir(plots_path):
            try:
                os.makedirs(plots_path)
            except FileExistsError:
                pass
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
        # Write the intro string in both the info file and on console if any
        if intro is not None:
            logger.info(intro)
        # Setup the experiment
        if experiment.setup(summary_path, metagraph_path, logger,checkpoint_path, scopes_to_restore, iteration):
            # Initialize the experiment
            experiment.initialize()
            # Conduct the experiment
            average_total_reward, max_total_reward, average_scaled_reward, max_scaled_reward, trained_episodes, success, metagraph_save_path = experiment.conduct(training_episodes, validation_episodes,
                                                                                                                                                                  max_training_episodes, episode_length_max,
                                                                                                                                                                  testing_episodes, test_cycles,
                                                                                                                                                                  logger,
                                                                                                                                                                  render_during_training,
                                                                                                                                                                  render_during_validation,
                                                                                                                                                                  render_during_test,
                                                                                                                                                                  plot_sample_density_training, plot_sample_density_validation)
            # Save the result of the iteration in the table
            experiment_results_table[iteration] = ("YES" if success else "NO",
                                                   numpy.round(average_total_reward, 3),
                                                   numpy.round(max_total_reward, 3),
                                                   numpy.round(average_scaled_reward, 3),
                                                   numpy.round(max_scaled_reward, 3),
                                                   trained_episodes)
            # Save the metagraph path in the list if the experiment iteration is successful
            if success:
                experiment_metagraph_paths.append(metagraph_save_path)
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
    # Write the intro if any
    if intro is not None:
        logger.info(experiment.name + ":")
        logger.info(intro)
        logger.info("\nResults:\n")
    else:
        logger.info(experiment.name + " results:\n")
    # Write results
    with pandas.option_context('display.max_rows', 500, 'display.max_columns', 500, 'display.width', 500):
        data_frame = pandas.DataFrame.from_dict(experiment_results_table, orient="index")
        data_frame.index.name = 'Experiment iteration'
        data_frame.columns = ['Success', 'Average total reward', 'Max total reward', 'Average scaled reward', 'Max scaled reward', 'Trained episodes']
        # Print data frame removing index if there is only one iteration
        if experiment_iterations_number <= 1:
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
    # Return the list of saved metagraph of successful iterations of the experiment, if any, otherwise an empty list
    return experiment_metagraph_paths


def watch_experiment(experiment: Experiment,
                     testing_episodes: int, test_cycles: int,
                     episode_length_max: int,
                     render: bool,
                     logger: logging.Logger,
                     checkpoint_path: str, scopes_to_restore: [] = None):
    """
    Run the given experiment with the given parameters. It automatically creates all the directories to store the
    experiment results, summaries and meta-graphs for each iteration.

    :param experiment: the experiment to run
    :param testing_episodes: the number of test episodes per test cycle
    :param test_cycles: the number of test cycles
    :param episode_length_max: the maximum number of steps for each episode
    :param render: flag to render or not the environment
    :param logger: the logger used to record information, warnings and errors
    :param checkpoint_path: the checkpoint path to gather the model from
    :param scopes_to_restore: the list of scopes to restore with the checkpoint, if any
    """
    # Setup of the logger for the current experiment
    # Reset logger handlers for current experiment
    logger.handlers = []
    # Generate a console handler for the logger
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    # Set handler properties
    console_handler.setLevel(logging.DEBUG)
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(console_handler)
    # Setup the experiment
    if experiment.setup(None, None, logger, checkpoint_path, scopes_to_restore):
        # Initialize the experiment
        experiment.initialize()
        # Watch the experiment
        experiment.watch(episode_length_max,
                         testing_episodes, test_cycles,
                         logger, render)
