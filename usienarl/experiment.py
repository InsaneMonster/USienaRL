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

import tensorflow
import numpy
import logging
import matplotlib.pyplot as plot
import matplotlib.ticker as mticker

# Import required src

from usienarl import Environment, Agent, Interface
from usienarl.volleys import EpisodeVolley, StepVolley, EpisodeVolleyType
from usienarl.interfaces import PassThroughInterface


class Experiment:
    """
    Base experiment abstract class.
    An experiment is a set of commands which allows to train or test a certain agent on a certain environment using a
    certain interface to translate states to observations and actions between the two.
    If an interface is not supplied, a default simple pass-through interface is used.
    An experiment can be used to train or test an agent. Both training and testing require setup.

    An experiment training phase consists of the following:
        - a warm-up volley, where the agent acts in warm-up mode for a certain amount of steps (usually required to fill
        an internal buffer of the agent model). This is used only by some models and it is run whether the
        agent requires it or not.
        - multiple set of training/validation volleys, where the agent acts in train/inference mode for a certain amount
        of episodes. Training volleys episodes and validation volleys episodes can be set separately, but for each
        training volley a corresponding validation volley is always run. The number of training/validation volleys
        define the length of a training phase.
    During training plots of most important metrics (average of total and scaled reward, standard deviation of total and
    scaled reward, average episode length) are saved with respect to each episode (on a volley basis) and with respect
    to each volley. An experiment training phase terminates when the number of required training/validation volleys is
    reached or when the agent is validated.

    An experiment testing phase consists of the following:
        - a set of test volleys, where the agent acts in inference mode. No plots are saved during test, but additional
        metrics are computed. Test volleys also set an attribute defining whether or not the agent is successful in
        overcoming the test phase.

    To run an experiment you must define your own. To define your own experiment, implement the abstract class in a
    specific child class. A child class only requires to have implemented the methods used to validate and test the
    agent's model and conclude the training and test phase respectively.
    """

    def __init__(self,
                 name: str,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None):
        # Define internal attributes
        self._name: str = name
        self._environment: Environment = environment
        self._agent: Agent = agent
        self._interface: Interface = interface if interface is not None else PassThroughInterface(self._environment)
        # Define global empty attributes
        self._plots_path: str or None = None
        self._plots_dpi: int or None = None
        self._tensorflow_config = None
        self._id: str or None = None
        self._parallel: int or None = None
        # Define data empty attributes
        self._warmup_volley: StepVolley or None = None
        self._training_volley: EpisodeVolley or None = None
        self._validation_volley: EpisodeVolley or None = None
        self._training_rewards: [] or None = None
        self._training_total_rewards: [] or None = None
        self._training_avg_total_rewards: [] or None = None
        self._training_scaled_rewards: [] or None = None
        self._training_avg_scaled_rewards: [] or None = None
        self._training_std_total_rewards: [] or None = None
        self._training_std_scaled_rewards: [] or None = None
        self._training_episode_lengths: [] or None = None
        self._training_avg_episode_lengths: [] or None = None
        self._validation_rewards: [] or None = None
        self._validation_total_rewards: [] or None = None
        self._validation_avg_total_rewards: [] or None = None
        self._validation_scaled_rewards: [] or None = None
        self._validation_avg_scaled_rewards: [] or None = None
        self._validation_std_total_rewards: [] or None = None
        self._validation_std_scaled_rewards: [] or None = None
        self._validation_episode_lengths: [] or None = None
        self._validation_avg_episode_lengths: [] or None = None
        self._training_validation_volley_counter: int or None = None
        self._trained_steps: int or None = None
        self._trained_episodes: int or None = None
        self._validated: bool or None = None
        self._test_rewards: [] or None = None
        self._test_total_rewards: [] or None = None
        self._test_volley: EpisodeVolley or None = None
        self._test_avg_total_rewards: [] or None = None
        self._test_scaled_rewards: [] or None = None
        self._test_avg_scaled_rewards: [] or None = None
        self._test_std_total_rewards: [] or None = None
        self._test_std_scaled_rewards: [] or None = None
        self._test_episode_lengths: [] or None = None
        self._test_avg_episode_lengths: [] or None = None
        self._test_volley_counter: int or None = None
        self._avg_test_avg_total_reward: float or None = None
        self._max_test_avg_total_reward: float or None = None
        self._avg_test_avg_scaled_reward: float or None = None
        self._max_test_avg_scaled_reward: float or None = None
        self._avg_test_std_total_reward: float or None = None
        self._min_test_std_total_reward: float or None = None
        self._avg_test_std_scaled_reward: float or None = None
        self._min_test_std_scaled_reward: float or None = None
        self._avg_test_avg_episode_length: int or None = None
        self._successful: bool or None = None

    def setup(self,
              logger: logging.Logger,
              plots_path: str or None = None, plots_dpi: int = 150,
              parallel: int = 1,
              summary_path: str = None, save_path: str = None, saves_to_keep: int = 0,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before both train and
        test. Until the experiment is not setup, most of its properties will return None.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param plots_path: the optional string path of the plots to save (both for the entire experiment and each one of its volleys), it's required for validation and training volleys.
        :param plots_dpi: the dpi of each plot graph (lower it to reduce plots size and quality)
        :param parallel: the amount of parallel episodes run for the experiment, must be greater than zero
        :param summary_path: the optional string path of the tensorboard summary directory to save during model training
        :param save_path: the optional string path of the metagraph agent directory to save at the end of each train volley
        :param saves_to_keep: the optional number of metagraphs to keep when saved, it does nothing if there is no save path
        :param iteration: number to append to the experiment name in all scopes and print statements (if not less than zero)
        :return: True if the setup of the experiment is successful, False otherwise
        """
        # Make sure parameters are valid
        assert(parallel > 0)
        assert(plots_dpi > 0)
        # Set attributes
        self._id = self._name + "_" + str(iteration) if iteration > -1 else self._name
        self._parallel = parallel
        self._plots_path = plots_path
        self._plots_dpi = plots_dpi
        # Start setup process
        logger.info("Setup of experiment " + self._id + "...")
        # Reset the tensorflow graph to a blank new graph
        tensorflow.reset_default_graph()
        # Execute the setup on the environment and on the agent and return if not both of them setup correctly
        logger.info("Environment setup...")
        if not self._environment.setup(logger, self._parallel):
            logger.info("Environment setup failed. Cannot setup the experiment!")
            return False
        logger.info("Environment setup is successful")
        logger.info("Agent setup...")
        # Note: the agent save path could be None
        agent_save_path: str or None = save_path + "/" + self._id if save_path is not None else None
        if not self._agent.setup(logger,
                                 self._id, self._parallel,
                                 self._interface.observation_space_type, self._interface.observation_space_shape,
                                 self._interface.agent_action_space_type, self._interface.agent_action_space_shape,
                                 summary_path, agent_save_path, saves_to_keep):
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful")
        # Configure tensorflow GPU
        # Note: allow growth in memory
        self._tensorflow_config = tensorflow.ConfigProto()
        self._tensorflow_config.gpu_options.allow_growth = True
        logger.info("Tensorflow configuration is successful")
        # Configure matplotlib
        # Note: make the matplotlib use agg-backend to make it compatible with a server and increase the chunksize to avoid unwanted errors
        plot.switch_backend('agg')
        plot.rcParams['agg.path.chunksize'] = 1000
        logger.info("Matplotlib configuration is successful")
        # If setup of both environment and agent is successful, reset internal attributes
        # Note: from now on most attributes are accessible
        self._warmup_volley = None
        self._training_volley = None
        self._validation_volley = None
        self._training_rewards = []
        self._training_total_rewards = []
        self._training_avg_total_rewards = []
        self._training_scaled_rewards = []
        self._training_avg_scaled_rewards = []
        self._training_std_total_rewards = []
        self._training_std_scaled_rewards = []
        self._training_episode_lengths = []
        self._training_avg_episode_lengths = []
        self._validation_rewards = []
        self._validation_total_rewards = []
        self._validation_avg_total_rewards = []
        self._validation_scaled_rewards = []
        self._validation_avg_scaled_rewards = []
        self._validation_std_total_rewards = []
        self._validation_std_scaled_rewards = []
        self._validation_episode_lengths = []
        self._validation_avg_episode_lengths = []
        self._training_validation_volley_counter = 0
        self._trained_steps = 0
        self._trained_episodes = 0
        self._validated = False
        self._test_volley = None
        self._test_rewards = []
        self._test_total_rewards = []
        self._test_avg_total_rewards = []
        self._test_scaled_rewards = []
        self._test_avg_scaled_rewards = []
        self._test_std_total_rewards = []
        self._test_std_scaled_rewards = []
        self._test_episode_lengths = []
        self._test_avg_episode_lengths = []
        self._test_volley_counter = 0
        self._avg_test_avg_total_reward = None
        self._max_test_avg_total_reward = None
        self._avg_test_avg_scaled_reward = None
        self._max_test_avg_scaled_reward = None
        self._avg_test_std_total_reward = None
        self._min_test_std_total_reward = None
        self._avg_test_std_scaled_reward = None
        self._min_test_std_scaled_reward = None
        self._avg_test_avg_episode_length = None
        self._successful = False
        # Setup is successful
        logger.info("Experiment setup is successful")
        return True

    def _save_plots(self,
                    episode_volley_type: EpisodeVolleyType):
        """
        Save training or validation plots according to the given type. If type is test it does nothing.

        :param episode_volley_type: the type of episode volley to save the appropriate plots. Test is not valid and ignored
        """
        # If test volley avoid saving plots
        if episode_volley_type == EpisodeVolleyType.test:
            return
        # Make sure there is a plots path
        assert (self._plots_path is not None and self._plots_path)
        # Save plots according to the requested type
        if episode_volley_type == EpisodeVolleyType.training:
            plot.plot(list(range(len(self._training_avg_total_rewards))), self._training_avg_total_rewards, 'r-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel("Training volley")
            plot.ylabel("Average total reward")
            plot.savefig(self._plots_path + "/training_volleys_total_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._training_avg_scaled_rewards))), self._training_avg_scaled_rewards, 'r--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel("Training volley")
            plot.ylabel("Average scaled reward")
            plot.savefig(self._plots_path + "/training_volleys_scaled_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._training_std_total_rewards))), self._training_std_total_rewards, 'c-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel("Training volley")
            plot.ylabel("Standard deviation of total reward")
            plot.savefig(self._plots_path + "/training_volleys_std_total_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._training_std_scaled_rewards))), self._training_std_scaled_rewards, 'c--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Training volley')
            plot.ylabel('Standard deviation of scaled reward')
            plot.savefig(self._plots_path + "/training_volleys_std_scaled_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._training_avg_episode_lengths))), self._training_avg_episode_lengths, 'b-.')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Training volley')
            plot.ylabel('Average episode length (steps)')
            plot.savefig(self._plots_path + "/training_volleys_average_episode_length.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
        else:
            plot.plot(list(range(len(self._validation_avg_total_rewards))), self._validation_avg_total_rewards, 'm-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Validation volley')
            plot.ylabel('Average total reward')
            plot.savefig(self._plots_path + "/validation_volleys_total_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._validation_avg_scaled_rewards))), self._validation_avg_scaled_rewards, 'm--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Validation volley')
            plot.ylabel('Average scaled reward')
            plot.savefig(self._plots_path + "/validation_volleys_scaled_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._validation_std_total_rewards))), self._validation_std_total_rewards, 'y-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Validation volley')
            plot.ylabel('Standard deviation of total reward')
            plot.savefig(self._plots_path + "/validation_volleys_std_total_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._validation_std_scaled_rewards))), self._validation_std_scaled_rewards, 'y--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Validation volley')
            plot.ylabel('Standard deviation of scaled reward')
            plot.savefig(self._plots_path + "/validation_volleys_std_scaled_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(self._validation_avg_episode_lengths))), self._validation_avg_episode_lengths, 'b-.')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plot.xlabel('Validation volley')
            plot.ylabel('Average episode length (steps)')
            plot.savefig(self._plots_path + "/validation_volleys_average_episode_length.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()

    def train(self,
              logger: logging.Logger,
              training_volleys_episodes: int, validation_volleys_episodes: int, volleys: int,
              episode_length: int,
              render_training: bool = False, render_validation: bool = False,
              restore_path: str = None):
        """
        Train the experiment's agent for a certain number of episodes per training/validation volley and a certain
        number of volleys. Training and validation volleys are always run in sequence (training first then validation)
        and in the same amount.

        If a restore path is given the model is restored from that (according to the way the agent restores its inner
        model), otherwise a new model is used. Also if restoration fails a new model is used.

        This generates a tensorflow session and initialize both environment and agent associated with the experiment.
        After this is run all training/validation results are accessible as properties.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param training_volleys_episodes: the number of training episodes per training volley, must be greater than zero
        :param validation_volleys_episodes: the number of validation episodes per validation volley, must be greater than zero
        :param volleys: the number of training/validation volleys, must be greater than zero
        :param episode_length: the max length in steps of each episode, must be greater than zero
        :param render_training: boolean flag to render or not the environment during training volleys execution
        :param render_validation: boolean flag to render or not the environment during validation volleys execution
        :param restore_path: the path of the model to restore is stored, if None or empty a new model is built
        """
        # Make sure parameters are valid
        assert(training_volleys_episodes > 0 and validation_volleys_episodes > 0 and volleys > 0 and episode_length > 0)
        # Start experiment training process
        logger.info("Experiment " + self._id + ": training process started")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            logger.info("Initializing the environment...")
            self._environment.initialize(logger, session)
            logger.info("Environment initialized")
            logger.info("Initializing the agent..")
            self._agent.initialize(logger, session)
            logger.info("Agent initialized")
            # Restore the agent if possible
            if restore_path is not None and restore_path:
                logger.info("Restoring agent graph with checkpoint path: " + restore_path)
                if self._agent.restore(logger, session, restore_path):
                    logger.info("Agent restored")
            # Execute warm-up if the agents require so
            if self._agent.warmup_steps > 0:
                # Generate warm-up volley
                self._warmup_volley = StepVolley(self._environment, self._agent, self._interface, self._parallel,
                                                 self._agent.warmup_steps, episode_length)
                # Setup and run warm-up volley
                logger.info("Setup of warm-up volley...")
                if not self._warmup_volley.setup():
                    logger.error("Warm-up volley failed to setup. Exiting...")
                    return
                logger.info("Setup of warm-up volley completed")
                self._warmup_volley.run(logger, session)
                logger.info("Warm-up volley is completed")
            # Generate training and validation volleys
            self._training_volley = EpisodeVolley(self._environment, self._agent, self._interface, self._parallel,
                                                  EpisodeVolleyType.training, self._plots_path, self._plots_dpi,
                                                  training_volleys_episodes, episode_length)
            self._validation_volley = EpisodeVolley(self._environment, self._agent, self._interface, self._parallel,
                                                    EpisodeVolleyType.validation, self._plots_path, self._plots_dpi,
                                                    validation_volleys_episodes, episode_length)
            # Execute training/validation volleys
            start: int = self._training_validation_volley_counter
            for volley_index in range(start, volleys):
                # Setup and run training volley
                logger.info("_____________________________________________________________________________________")
                logger.info("Setup of training volley " + str(self._training_validation_volley_counter) + "...")
                if not self._training_volley.setup(self._training_validation_volley_counter,
                                                   self._trained_steps, self._trained_episodes):
                    logger.error("Training volley failed to setup. Exiting...")
                    return
                logger.info("Setup of training volley " + str(self._training_validation_volley_counter) + " completed")
                self._training_volley.run(logger, session, render_training)
                # Update counter of trained steps and episodes
                self._trained_steps += self._training_volley.steps
                self._trained_episodes += self._training_volley.episodes
                # Store training volley statistics
                self._training_rewards.append(self._training_volley.rewards.copy())
                self._training_total_rewards.append(self._training_volley.total_rewards.copy())
                self._training_avg_total_rewards.append(self._training_volley.avg_total_reward)
                self._training_scaled_rewards.append(self._training_volley.scaled_rewards.copy())
                self._training_avg_scaled_rewards.append(self._training_volley.avg_scaled_reward)
                self._training_std_total_rewards.append(self._training_volley.std_total_reward)
                self._training_std_scaled_rewards.append(self._training_volley.std_scaled_reward)
                self._training_episode_lengths.append(self._training_volley.episode_lengths.copy())
                self._training_avg_episode_lengths.append(self._training_volley.avg_episode_length)
                # Print training volley results
                logger.info("Training volley " + str(self._training_validation_volley_counter) + " is completed")
                logger.info("Training steps up to now: " + str(self._trained_steps))
                # Save/update plots for all training volleys up to now (except at first volley, just one volley is not enough to plot anything)
                if self._training_validation_volley_counter > 0:
                    if self._training_validation_volley_counter == 0:
                        logger.info("Save all training volleys plots...")
                    else:
                        logger.info("Update all training volleys plots...")
                    self._save_plots(EpisodeVolleyType.training)
                    if self._training_validation_volley_counter == 0:
                        logger.info("Plots saved successfully")
                    else:
                        logger.info("Plots updated successfully")
                logger.info("_____________________________________________________________________________________")
                # Setup and run validation volley
                logger.info("---------------------------------------------------------------------------------------")
                logger.info("Setup of validation volley " + str(self._training_validation_volley_counter) + "...")
                if not self._validation_volley.setup(self._training_validation_volley_counter):
                    logger.error("Validation volley failed to setup. Exiting...")
                    return
                logger.info("Setup of validation volley " + str(self._training_validation_volley_counter) + " completed")
                self._validation_volley.run(logger, session, render_validation)
                # Store validation volley statistics
                self._validation_rewards.append(self._validation_volley.rewards.copy())
                self._validation_total_rewards.append(self._validation_volley.total_rewards.copy())
                self._validation_avg_total_rewards.append(self._validation_volley.avg_total_reward)
                self._validation_scaled_rewards.append(self._validation_volley.scaled_rewards.copy())
                self._validation_avg_scaled_rewards.append(self._validation_volley.avg_scaled_reward)
                self._validation_std_total_rewards.append(self._validation_volley.std_total_reward)
                self._validation_std_scaled_rewards.append(self._validation_volley.std_scaled_reward)
                self._validation_episode_lengths.append(self._validation_volley.episode_lengths.copy())
                self._validation_avg_episode_lengths.append(self._validation_volley.avg_episode_length)
                # Print validation volley results
                logger.info("Validation volley " + str(self._training_validation_volley_counter) + " is completed")
                # Save/update plots for all training volleys up to now (except at first volley, just one volley is not enough to plot anything)
                if self._training_validation_volley_counter > 0:
                    if self._training_validation_volley_counter == 0:
                        logger.info("Save all validation volleys plots...")
                    else:
                        logger.info("Update all validation volleys plots...")
                    self._save_plots(EpisodeVolleyType.validation)
                    if self._training_validation_volley_counter == 1:
                        logger.info("Plots saved successfully")
                    else:
                        logger.info("Plots updated successfully")
                logger.info("---------------------------------------------------------------------------------------")
                # Save the agent metagraph
                self._agent.save(logger, session)
                # Update the training/validation volley counter
                self._training_validation_volley_counter += 1
                # Check for validation
                self._validated = self._is_validated(logger)
                if self._validated:
                    logger.info("Validation of the agent is successful")
                    break
            logger.info("End of training")

    def test(self,
             logger: logging.Logger,
             episodes: int, volleys: int,
             episode_length: int,
             restore_path: str,
             render: bool = True):
        """
        Test the experiment's agent whose model is defined with the given checkpoint metagraph for a certain number
        of episodes per volley and a certain number of volleys.

        This generates a tensorflow session and initialize both environment and agent associated with the experiment.
        After this is run all test results are accessible as properties.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes for each test volley, must be greater than zero
        :param volleys: the number of test volleys, must be greater than zero
        :param episode_length: the max length in steps of each episode, must be greater than zero
        :param restore_path: the path where the metagraph of the model to test is stored
        :param render: boolean flag to render or not the environment
        """
        # Make sure parameters are valid
        assert(episodes > 0 and volleys > 0 and episode_length > 0)
        # Start experiment testing process
        logger.info("Experiment " + self._id + ": testing process started")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            logger.info("Initializing the environment...")
            self._environment.initialize(logger, session)
            logger.info("Environment initialized")
            logger.info("Initializing the agent..")
            self._agent.initialize(logger, session)
            logger.info("Agent initialized")
            # Load the pre-trained model at given checkpoint path
            if restore_path is None or not restore_path:
                logger.error("A checkpoint path is required to test!")
                return
            # Restore the agent model
            logger.info("Restoring agent graph with checkpoint path: " + restore_path)
            if not self._agent.restore(logger, session, restore_path):
                return
            logger.info("Agent restored")
            # Generate test volley
            self._test_volley = EpisodeVolley(self._environment, self._agent, self._interface, self._parallel,
                                              EpisodeVolleyType.test, self._plots_path, self._plots_dpi,
                                              episodes, episode_length)
            # Execute test volleys
            start: int = self._test_volley_counter
            for volley_index in range(start, volleys):
                # Setup and run test volley
                logger.info("Setup of test volley " + str(self._test_volley_counter) + "...")
                if not self._test_volley.setup():
                    logger.error("Test volley failed to setup. Exiting...")
                    return
                self._test_volley.run(logger, session, render)
                # Store test volley statistics
                self._test_rewards.append(self._test_volley.rewards.copy())
                self._test_total_rewards.append(self._test_volley.total_rewards.copy())
                self._test_avg_total_rewards.append(self._test_volley.avg_total_reward)
                self._test_scaled_rewards.append(self._test_volley.scaled_rewards.copy())
                self._test_avg_scaled_rewards.append(self._test_volley.avg_scaled_reward)
                self._test_std_total_rewards.append(self._test_volley.std_total_reward)
                self._test_std_scaled_rewards.append(self._test_volley.std_scaled_reward)
                self._test_episode_lengths.append(self._test_volley.episode_lengths.copy())
                self._test_avg_episode_lengths.append(self._test_volley.avg_episode_length)
                # Increase test volley counter
                self._test_volley_counter += 1
            logger.info("End of test")
            # Store the average and the best total and scaled rewards over all test volleys
            self._avg_test_avg_total_reward = numpy.round(numpy.average(self._test_avg_total_rewards), 3)
            self._max_test_avg_total_reward = numpy.round(numpy.max(self._test_avg_total_rewards), 3)
            self._avg_test_avg_scaled_reward = numpy.round(numpy.average(self._test_avg_scaled_rewards), 3)
            self._max_test_avg_scaled_reward = numpy.round(numpy.max(self._test_avg_scaled_rewards), 3)
            # Store the average and the minimum standard deviation of total and scaled rewards over test volleys
            self._avg_test_std_total_reward = numpy.round(numpy.average(self._test_std_total_rewards), 3)
            self._min_test_std_total_reward = numpy.round(numpy.min(self._test_std_total_rewards), 3)
            self._avg_test_std_scaled_reward = numpy.round(numpy.average(self._test_std_scaled_rewards), 3)
            self._min_test_std_scaled_reward = numpy.round(numpy.min(self._test_std_scaled_rewards), 3)
            # Store the average episode length over all test volleys
            self._avg_test_avg_episode_length = numpy.rint(numpy.average(self._test_avg_episode_lengths))
            # Print test results
            logger.info("Results over " + str(volleys) + " test volleys of " + str(episodes) + " episodes each are:")
            logger.info("Average test total reward: " + str(self._avg_test_avg_total_reward))
            logger.info("Average test scaled reward: " + str(self._avg_test_avg_scaled_reward))
            logger.info("Max test total reward: " + str(self._max_test_avg_total_reward))
            logger.info("Max test scaled reward: " + str(self._max_test_avg_scaled_reward))
            logger.info("Average test standard deviation of total reward: " + str(self._avg_test_std_total_reward))
            logger.info("Average test standard deviation of scaled reward: " + str(self._avg_test_std_scaled_reward))
            logger.info("Min test standard deviation of total reward: " + str(self._min_test_std_total_reward))
            logger.info("Min test standard deviation of scaled reward: " + str(self._min_test_std_scaled_reward))
            logger.info("Average test episode length: " + str(self._avg_test_avg_episode_length))
            # Check if test is successful
            self._successful = self._is_successful(logger)
            if self._successful:
                logger.info("The experiment terminated successfully")
            else:
                logger.info("The experiment terminated without meeting the requirements")

    def _is_validated(self,
                      logger: logging.Logger) -> bool:
        """
        Check if the experiment training phase is to be considered validated.

        :param logger: the logger used to print the experiment information, warnings and errors
        :return: True if training is validated, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _is_successful(self,
                       logger: logging.Logger) -> bool:
        """
        Check if the experiment testing phase is to be considered successful.

        :param logger: the logger used to print the experiment information, warnings and errors
        :return: True if test is successful, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        The name of the experiment.
        """
        return self._name

    @property
    def agent(self) -> Agent:
        """
        The agent associated with the experiment.
        """
        return self._agent

    @property
    def environment(self) -> Environment:
        """
        The environment associated with the experiment.
        """
        return self._environment

    @property
    def interface(self) -> Interface:
        """
        The interface associated with the experiment.
        """
        return self._interface

    @property
    def id(self) -> str or None:
        """
        The id of the experiment, consisting of the name and the iteration number (if greater or equal than zero).
        It is None if experiment is not setup.
        """
        return self._id

    @property
    def plots_path(self) -> str or None:
        """
        The path of plots saved by the experiment and its volleys.
        It is None if experiment is not setup.
        """
        return self._plots_path

    def plots_dpi(self) -> int or None:
        """
        The dpi (quality) of plots saved by the experiment and its volleys.
        It is None if experiment is not setup.
        """
        return self._plots_dpi

    @property
    def parallel(self) -> int or None:
        """
        The amount of parallel episodes run by the environment and the agent of the experiment.
        It is None if experiment is not setup.
        """
        return self._parallel

    @property
    def warmup_volley(self) -> StepVolley or None:
        """
        The warm-up volley of the training phase.
        It is None if warm-up is not required by the agent or not started yet.
        """
        return self._warmup_volley

    @property
    def training_volley(self) -> EpisodeVolley or None:
        """
        The training volley of the training phase.
        It is None if training is not started yet.
        """
        return self._training_volley

    @property
    def validation_volley(self) -> EpisodeVolley or None:
        """
        The validation volley of the training phase.
        It is None if training is not started yet.
        """
        return self._validation_volley

    @property
    def test_volley(self) -> EpisodeVolley or None:
        """
        The test volley of the testing phase.
        It is None if testing is not started yet.
        """
        return self._test_volley

    @property
    def training_rewards(self) -> [] or None:
        """
        The list of all rewards (step by step) of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_rewards

    @property
    def training_total_rewards(self) -> [] or None:
        """
        The list of all total rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_total_rewards

    @property
    def training_avg_total_rewards(self) -> [] or None:
        """
        The list of average total rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_avg_total_rewards

    @property
    def training_scaled_rewards(self) -> [] or None:
        """
        The list of all scaled rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_scaled_rewards

    @property
    def training_avg_scaled_rewards(self) -> [] or None:
        """
        The list of average scaled rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_avg_scaled_rewards

    @property
    def training_std_total_rewards(self) -> [] or None:
        """
        The list of standard deviations of total rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_std_total_rewards

    @property
    def training_std_scaled_rewards(self) -> [] or None:
        """
        The list of standard deviations of scaled rewards of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_std_scaled_rewards

    @property
    def training_episode_lengths(self) -> [] or None:
        """
        The list of all episode lengths of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_episode_lengths

    @property
    def training_avg_episode_lengths(self) -> [] or None:
        """
        The list of average episode lengths of each volley of all the training volleys already executed.
        It is None if experiment is not setup.
        It is empty if no training volley has finished execution.
        """
        return self._training_avg_episode_lengths

    @property
    def validation_rewards(self) -> [] or None:
        """
        The list of all rewards (step by step) of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_rewards

    @property
    def validation_total_rewards(self) -> [] or None:
        """
        The list of all total rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_total_rewards

    @property
    def validation_avg_total_rewards(self) -> [] or None:
        """
        The list of average total rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_avg_total_rewards

    @property
    def validation_scaled_rewards(self) -> [] or None:
        """
        The list of all scaled rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_scaled_rewards

    @property
    def validation_avg_scaled_rewards(self) -> [] or None:
        """
        The list of average scaled rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_avg_scaled_rewards

    @property
    def validation_std_total_rewards(self) -> [] or None:
        """
        The list of standard deviations of total rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_std_total_rewards

    @property
    def validation_std_scaled_rewards(self) -> [] or None:
        """
        The list of standard deviations of scaled rewards of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_std_scaled_rewards

    @property
    def validation_episode_lengths(self) -> [] or None:
        """
        The list of all episode lengths of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_episode_lengths

    @property
    def validation_avg_episode_lengths(self) -> [] or None:
        """
        The list of average episode lengths of each volley of all the validation volleys already executed.
        It is None if experiment is not setup.
        It is empty if no validation volley has finished execution.
        """
        return self._validation_avg_episode_lengths

    @property
    def test_rewards(self) -> [] or None:
        """
        The list of all rewards (step by step) of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_rewards

    @property
    def test_total_rewards(self) -> [] or None:
        """
        The list of all total rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_total_rewards

    @property
    def test_avg_total_rewards(self) -> [] or None:
        """
        The list of average total rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_avg_total_rewards

    @property
    def test_scaled_rewards(self) -> [] or None:
        """
        The list of all scaled rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_scaled_rewards

    @property
    def test_avg_scaled_rewards(self) -> [] or None:
        """
        The list of average scaled rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_avg_scaled_rewards

    @property
    def test_std_total_rewards(self) -> [] or None:
        """
        The list of standard deviations of total rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_std_total_rewards

    @property
    def test_std_scaled_rewards(self) -> [] or None:
        """
        The list of standard deviations of scaled rewards of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_std_scaled_rewards

    @property
    def test_episode_lengths(self) -> [] or None:
        """
        The list of all episode lengths of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_episode_lengths

    @property
    def test_avg_episode_lengths(self) -> [] or None:
        """
        The list of average episode lengths of each volley of all the test volleys already executed.
        It is None if experiment is not setup.
        It is empty if no test volley has finished execution.
        """
        return self._test_avg_episode_lengths

    @property
    def trained_steps(self) -> int or None:
        """
        Trained steps, globally, up to now.
        It is None if experiment is not setup.
        """
        return self._trained_steps

    @property
    def trained_episodes(self) -> int or None:
        """
        Trained episodes, globally, up to now.
        It is None if experiment is not setup.
        """
        return self._trained_episodes

    @property
    def training_validation_volley_counter(self) -> int or None:
        """
        Current counter of training/validation volleys.
        It is None if experiment is not setup.
        """
        return self._training_validation_volley_counter

    @property
    def test_volley_counter(self) -> int or None:
        """
        Current counter of test volleys.
        It is None if experiment is not setup.
        """
        return self._test_volley_counter

    @property
    def avg_test_avg_total_reward(self) -> float or None:
        """
        The average total reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._avg_test_avg_total_reward

    @property
    def max_test_avg_total_reward(self) -> float or None:
        """
        The max average total reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._max_test_avg_total_reward

    @property
    def avg_test_avg_scaled_reward(self) -> float or None:
        """
        The average scaled reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._avg_test_avg_scaled_reward

    @property
    def max_test_avg_scaled_reward(self) -> float or None:
        """
        The max average scaled reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._max_test_avg_scaled_reward

    @property
    def avg_test_std_total_reward(self) -> float or None:
        """
        The average standard deviation total reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._avg_test_std_total_reward

    @property
    def min_test_std_total_reward(self) -> float or None:
        """
        The minimum standard deviation of total reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._min_test_std_total_reward

    @property
    def avg_test_std_scaled_reward(self) -> float or None:
        """
        The average standard deviation of scaled reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._avg_test_std_scaled_reward

    @property
    def min_test_std_scaled_reward(self) -> float or None:
        """
        The minimum standard deviation of scaled reward among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._min_test_std_scaled_reward

    @property
    def avg_test_avg_episode_length(self) -> int or None:
        """
        The average episode length among all test volleys.
        It is None if test phase has not finished execution.
        """
        return self._avg_test_avg_episode_length

    @property
    def validated(self) -> bool or None:
        """
        Flag defining if the experiment training is validated or not.
        It is None if experiment is not setup.
        """
        return self._validated

    @property
    def successful(self) -> bool or None:
        """
        Flag defining if the experiment testing is successful or not.
        It is None if experiment is not setup.
        """
        return self._successful
