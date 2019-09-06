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

# Import required src

from usienarl import Environment, Agent, Interface
from usienarl.interfaces import PassThroughInterface


class Experiment:
    """
    Base experiment abstract class.
    An experiment is a set of commands which allows to train or test or even both a certain agent on a certain
    environment using a certain interface to translate between the two.

    If an interface is not supplied, a default simple pass-through interface is used.

    To conduct an experiment, first make sure the setup phase is passed. Once an experiment is setup, the agent and
    the environment can be used.
    A default experiment consists of:
        - warm-up phase, where the agent acts in warm-up mode and which can be used or not depending on the agent
        inner model
        - volley phase, which consists in multiple sets of episodes dedicated to training after which the agent
        performance is validated
        - test phase, which consists in multiple cycles of test where the agent acts in inference mode and in which
        the agent performance is tested and which occurs only after validation is passed

    Each volley consist of two additional sub-phases:
        - train phase: where the agent acts in train mode and its internal model is updated
        - validation phase: where the agent acts in inference mode and its performance is tested

    To define your own experiment, implement the abstract class in a specific child class. A child class only requires
    to have implemented the methods used to validate and test the agent's model and conclude the experiment volley
    phase or the experiment as a whole.

    Two types of scores are produced, using the rewards given only by the environment:
        - total reward: the sum of all the rewards at each step in the episode, also called total episode reward
        - scaled reward: the average of all the rewards at each step in the episode, also called average step reward
    """

    def __init__(self,
                 name: str,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None):
        # Define experiment attributes
        self.name: str = name
        self._environment: Environment = environment
        self._agent: Agent = agent
        self._interface: Interface = interface if interface is not None else PassThroughInterface(self._environment)
        # Define empty experiment attributes
        self._agent_saver = None
        self._metagraph_path: str = None
        self._plots_path: str = None
        self._tensorflow_config = None
        self._current_id: str = None
        self._trained_steps: int = None
        self._trained_episodes: int = None
        self._scaled_rewards: [] = []

    def setup(self,
              summary_path: str, metagraph_path: str,
              logger: logging.Logger,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before conduct method.

        :param summary_path: the string path of the tensorboard summary directory to save during model training, set to None if the model is not being trained
        :param metagraph_path: the string path of the metagraph agent directory to save at the end of each train volley, set to None if the model is not being trained
        :param logger: the logger used to print the experiment information, warnings and errors
        :param iteration: number to append to the experiment name in all scopes and print statements (if not less than zero)
        :return: a boolean equals to True if the setup of the experiment is successful, False otherwise
        """
        # Save current experiment id using name and iteration
        self._current_id = self.name + "_" + str(iteration) if iteration > -1 else self.name
        # Start setup process
        logger.info("Setup of experiment " + self._current_id + "...")
        # Reset the tensorflow graph to a blank new graph
        tensorflow.reset_default_graph()
        # Execute the setup on the environment and on the agent and return if not both of them setup correctly
        logger.info("Environment setup...")
        if not self._environment.setup(logger):
            logger.info("Environment setup failed. Cannot setup the experiment!")
            return False
        logger.info("Environment setup is successful!")
        logger.info("Agent setup...")
        if not self._agent.setup(logger,
                                 self._interface.observation_space_type, self._interface.observation_space_shape,
                                 self._interface.agent_action_space_type, self._interface.agent_action_space_shape,
                                 self._current_id, summary_path):
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful!")
        # If setup of both environment and agent is successful, reset internal experiment variables
        self._metagraph_path = metagraph_path
        self._plots_path = self._metagraph_path.replace('/metagraph', '/plots')
        self._trained_steps: int = 0
        self._trained_episodes: int = 0
        # Initialize the agent internal model saver
        self._agent_saver = tensorflow.train.Saver(self._agent.trainable_variables)
        if self._metagraph_path is not None:
            logger.info("Agent internal model will be saved after each train volley")
            logger.info("Agent internal model metagraph save path: " + self._metagraph_path)
        # Initialize tensorflow gpu configuration
        self._tensorflow_config = tensorflow.ConfigProto()
        self._tensorflow_config.gpu_options.allow_growth = True
        logger.info("Tensorflow configuration is successful!")
        # Setup is successful
        logger.info("Experiment setup is successful. Now can conduct the experiment")
        return True

    def _warmup(self,
                logger: logging.Logger,
                episodes: int, episode_length: int,
                session,
                render: bool = False):
        """
        Execute a volley of warm-up of the agent on the environment.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in warm-up mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during warm-up
        """
        for episode in range(episodes):
            # Start the episode
            episode_rewards: [] = []
            state_current = self._environment.reset(logger, session)
            # Execute actions until the episode is completed or the maximum length is exceeded
            for step in range(episode_length):
                # Get the action decided by the agent with train policy
                observation_current = self._interface.environment_state_to_observation(logger, session, state_current)
                agent_action = self._agent.act_warmup(logger, session, self._interface, observation_current)
                # Get the next state with relative reward and completion flag
                environment_action = self._interface.agent_action_to_environment_action(logger, session, agent_action)
                state_next, reward, episode_done = self._environment.step(logger, environment_action, session)
                # Complete the step and send back information to the agent
                observation_next = self._interface.environment_state_to_observation(logger, session, state_next)
                # Set the observation next to None if final state is reached
                if episode_done:
                    observation_next = None
                self._agent.complete_step_warmup(logger, session, self._interface,
                                                 observation_current, agent_action, reward, observation_next,
                                                 step, episode, episodes)
                # Render if required
                if render:
                    self._environment.render(logger, session)
                # Add the reward to the list of rewards for this episode
                episode_rewards.append(reward)
                # Update the current state with the previously next state
                state_current = state_next
                # Check if the episode is completed
                if episode_done:
                    break
            # Compute total and average reward over the steps in the episode
            # episode_total_reward: float = numpy.sum(numpy.array(episode_rewards))
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_warmup(logger, session, self._interface,
                                                episode_rewards[-1], numpy.sum(numpy.array(episode_rewards)),
                                                episode, episodes)

    def _train(self,
               logger: logging.Logger,
               episodes: int, episode_length: int, episodes_max: int,
               session,
               render: bool = False):
        """
        Execute a volley of training of the agent on the environment.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during training
        :return: the list of rewards obtained in the volley, grouped by episode
        """
        # Define a list of rewards found in the volley
        volley_rewards: [] = []
        for episode in range(episodes):
            # Start the episode
            episode_rewards: [] = []
            state_current = self._environment.reset(logger, session)
            # Execute actions until the episode is completed or the maximum length is exceeded
            for step in range(episode_length):
                # Get the action decided by the agent with train policy
                observation_current = self._interface.environment_state_to_observation(logger, session, state_current)
                agent_action = self._agent.act_train(logger, session, self._interface, observation_current)
                # Get the next state with relative reward and completion flag
                environment_action = self._interface.agent_action_to_environment_action(logger, session, agent_action)
                state_next, reward, episode_done = self._environment.step(logger, environment_action, session)
                # Complete the step and send back information to the agent
                observation_next = self._interface.environment_state_to_observation(logger, session, state_next)
                # Set the observation next to None if final state is reached
                if episode_done:
                    observation_next = None
                self._agent.complete_step_train(logger, session, self._interface,
                                                observation_current, agent_action, reward, observation_next,
                                                step, self._trained_steps,
                                                episode, self._trained_episodes,
                                                episodes, episodes_max)
                # Render if required
                if render:
                    self._environment.render(logger, session)
                # Add the reward to the list of rewards for this episode
                episode_rewards.append(reward)
                # Update the current state with the previously next state
                state_current = state_next
                # Increase the number of trained steps
                self._trained_steps += 1
                # Check if the episode is completed
                if episode_done:
                    break
            # Add the list of episode rewards to the volley rewards
            volley_rewards.append(episode_rewards)
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_train(logger, session, self._interface,
                                               episode_rewards[-1], numpy.sum(numpy.array(episode_rewards)),
                                               self._trained_steps,
                                               episode, self._trained_episodes,
                                               episodes, episodes_max)
            # Increase the counter of trained episodes
            self._trained_episodes += 1
        # Return the list of all rewards seen in the volley
        return volley_rewards

    def _inference(self,
                   logger: logging.Logger,
                   episodes: int, episode_length: int,
                   session,
                   render: bool = False) -> []:
        """
        Execute a volley of inference of the agent on the environment.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in inference mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during inference
        :return: the list of rewards obtained in the volley, grouped by episode
        """
        # Define a list of rewards found in the volley
        volley_rewards: [] = []
        for episode in range(episodes):
            # Start the episode
            episode_rewards: [] = []
            state_current = self._environment.reset(logger, session)
            # Execute actions until the episode is completed or the maximum length is exceeded
            for step in range(episode_length):
                # Get the action decided by the agent with train policy
                observation_current = self._interface.environment_state_to_observation(logger, session, state_current)
                agent_action = self._agent.act_inference(logger, session, self._interface, observation_current)
                # Get the next state with relative reward and completion flag
                environment_action = self._interface.agent_action_to_environment_action(logger, session, agent_action)
                state_next, reward, episode_done = self._environment.step(logger, environment_action, session)
                # Complete the step and send back information to the agent
                observation_next = self._interface.environment_state_to_observation(logger, session, state_next)
                # Set the observation next to None if final state is reached
                if episode_done:
                    observation_next = None
                self._agent.complete_step_inference(logger, session, self._interface,
                                                    observation_current, agent_action, reward, observation_next,
                                                    step, episode, episodes)
                # Render if required
                if render:
                    self._environment.render(logger, session)
                # Add the reward to the list of rewards for this episode
                episode_rewards.append(reward)
                # Update the current state with the previously next state
                state_current = state_next
                # Check if the episode is completed
                if episode_done:
                    break
            # Add the list of episode rewards to the volley rewards
            volley_rewards.append(episode_rewards)
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_inference(logger, session, self._interface,
                                                   episode_rewards[-1], numpy.sum(numpy.array(episode_rewards)),
                                                   episode, episodes)
        # Return the list of all rewards seen in the volley
        return volley_rewards

    def conduct(self,
                training_episodes_per_volley: int, validation_episodes_per_volley: int,
                training_episodes_max: int, episode_length_max: int,
                test_episodes_per_cycle: int, test_cycles: int,
                logger: logging.Logger,
                render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False,
                checkpoint_path: str = None):
        """
        Conduct the experiment of the given number of training (up to the given maximum) and validation episodes
        per volley, with the associated given number of test episodes per cycle and the given number of test cycles after
        the validation has passed.

        A volley of training is always considered to stop the training exactly when the agent performs well enough
        to validate, by checking the appropriate condition. When validation is successful, appropriate condition is also
        checked on the test cycles to decide if the experiment is successful or not.

        Conducting an experiment will generate a tensorflow session for that experiment. It is required that the experiment
        is ready to be conducted by checking the result of the setup method before attempting at conducting it.

        If an already trained model checkpoint is given, the last checkpoint in the given path folder is used to further
        train the model. The scope of the experiment and of the agent has to be the same across both models.

        :param training_episodes_per_volley: the number of training episodes per volley before trying to validate the agent
        :param validation_episodes_per_volley: the number of validation episodes per volley after the training in such interval
        :param training_episodes_max: the maximum number of training episodes allowed at all
        :param episode_length_max: the max length in steps of each episode
        :param test_episodes_per_cycle: the number of episodes to play for each test cycle after validation has passed
        :param test_cycles: the number of test cycles to execute after validation has passed
        :param logger: the logger used to print the experiment information, warnings and errors
        :param render_during_training: boolean flag to render the environment during training, default to False
        :param render_during_validation: boolean flag to render the environment during validation, default to False
        :param render_during_test: boolean flag to render the environment during test, default to False
        :param checkpoint_path: the optional checkpoint path to load the pre-trained model from, it should be a valid model, default to None
        :return: the average and the max of the total average reward, the average and the max of the average average reward with respect to all test cycles, the training episodes required to validate, a success flag and the location in which the saved metagraph is stored
        """
        logger.info("Conducting experiment " + self._current_id + "...")
        # Define the list in which to store the evolution of total and scaled average reward to plot by default
        all_training_volleys_total_rewards: [] = []
        all_training_volleys_scaled_rewards: [] = []
        all_validation_volleys_total_rewards: [] = []
        all_validation_volleys_scaled_rewards: [] = []
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            self._environment.initialize(logger, session)
            self._agent.initialize(logger, session)
            # Load the pre-trained model if required
            if checkpoint_path is not None:
                checkpoint = tensorflow.train.get_checkpoint_state(checkpoint_path)
                # If checkpoint exists restore from checkpoint
                if checkpoint and checkpoint.model_checkpoint_path:
                    self._agent_saver.restore(session, tensorflow.train.latest_checkpoint(checkpoint_path))
                    logger.info("Model graph stored at " + checkpoint_path + " loaded successfully!")
                else:
                    logger.error("Checkpoint path specified is wrong: no model can be accessed at " + checkpoint_path)
            # Execute pre-training if the agent requires pre-training
            if self._agent.warmup_episodes > 0:
                logger.info("Warming-up for " + str(self._agent.warmup_episodes) + " episodes...")
                self._warmup(logger,
                             self._agent.warmup_episodes,
                             episode_length_max,
                             session)
            # Save last volley average training and validation total and scaled rewards
            last_training_volley_average_total_reward: float = 0.0
            last_training_volley_average_scaled_reward: float = 0.0
            last_validation_volley_average_total_reward: float = 0.0
            last_validation_volley_average_scaled_reward: float = 0.0
            # Save last volley training and validation rewards list
            last_training_volley_rewards: [] = []
            last_validation_volley_rewards: [] = []
            # Execute training until max training episodes number is reached or the validation score is above the threshold
            while self._trained_episodes < training_episodes_max:
                # Run train for training episodes per volley and get the average score
                logger.info("Training for " + str(training_episodes_per_volley) + " episodes...")
                last_training_volley_rewards = self._train(logger,
                                                           training_episodes_per_volley,
                                                           episode_length_max,
                                                           training_episodes_max,
                                                           session,
                                                           render_during_training)
                # Since rewards list are of different sizes, get their sum and average
                training_volley_total_rewards: numpy.ndarray = numpy.zeros(len(last_training_volley_rewards), dtype=float)
                training_volley_scaled_rewards: numpy.ndarray = numpy.zeros(len(last_training_volley_rewards), dtype=float)
                for index, episode_rewards in enumerate(last_training_volley_rewards):
                    training_volley_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                    training_volley_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                    # Update the global lists of all the average total and scale rewards for each training episode
                    all_training_volleys_total_rewards.append(numpy.sum(numpy.array(episode_rewards)))
                    all_training_volleys_scaled_rewards.append(numpy.average(numpy.array(episode_rewards)))
                # Compute average total reward and average scaled reward over training volley
                last_training_volley_average_total_reward: float = numpy.average(training_volley_total_rewards)
                last_training_volley_average_scaled_reward: float = numpy.average(training_volley_scaled_rewards)
                # Print training results
                logger.info("Training of " + str(training_episodes_per_volley) + " finished with following result:")
                logger.info("Average total reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_average_total_reward))
                logger.info("Average scaled reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_average_scaled_reward))
                logger.info("Total environmental steps in training mode up to now: " + str(self._trained_steps))
                # Save the agent internal model at the current step
                logger.info("Saving the agent...")
                self._agent_saver.save(session, self._metagraph_path + "/" + self._current_id)
                # Run inference for validation episodes per volley and get the average score
                logger.info("Validating for " + str(validation_episodes_per_volley) + " episodes...")
                last_validation_volley_rewards = self._inference(logger,
                                                                 validation_episodes_per_volley,
                                                                 episode_length_max,
                                                                 session,
                                                                 render_during_validation)
                # Since rewards list are of different sizes, get their sum and average
                validation_volley_total_rewards: numpy.ndarray = numpy.zeros(len(last_validation_volley_rewards), dtype=float)
                validation_volley_scaled_rewards: numpy.ndarray = numpy.zeros(len(last_validation_volley_rewards), dtype=float)
                for index, episode_rewards in enumerate(last_validation_volley_rewards):
                    validation_volley_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                    validation_volley_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                    # Update the global lists of all the average total and scale rewards for each validation episode
                    all_validation_volleys_total_rewards.append(numpy.sum(numpy.array(episode_rewards)))
                    all_validation_volleys_scaled_rewards.append(numpy.average(numpy.array(episode_rewards)))
                # Compute average total reward and average scaled reward over validation volley
                last_validation_volley_average_total_reward: float = numpy.average(validation_volley_total_rewards)
                last_validation_volley_average_scaled_reward: float = numpy.average(validation_volley_scaled_rewards)
                # Print validation results
                logger.info("Training of " + str(validation_episodes_per_volley) + " finished with following result:")
                logger.info("Average total reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_average_total_reward))
                logger.info("Average scaled reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_average_scaled_reward))
                # Check for validation
                if self._is_validated(logger,
                                      last_validation_volley_average_total_reward, last_validation_volley_average_scaled_reward,
                                      last_training_volley_average_total_reward, last_training_volley_average_scaled_reward,
                                      last_validation_volley_rewards, last_training_volley_rewards):
                    logger.info("Validation of the agent is successful")
                    break
            # Test the model and get all cycles total, scaled rewards and all rewards
            test_average_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_average_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_cycles_rewards: [] = []
            for test in range(test_cycles):
                # Run inference for test episodes per cycles and get the average score
                logger.info("Cycle " + str(test + 1) + " - Testing for " + str(test_episodes_per_cycle) + " episodes...")
                test_cycle_rewards: [] = self._inference(logger,
                                                         test_episodes_per_cycle,
                                                         episode_length_max,
                                                         session,
                                                         render_during_test)
                # Append the test cycle rewards to the full list
                test_cycles_rewards.append(test_cycle_rewards)
                # Since rewards list are of different sizes, get their sum and average
                test_cycle_total_rewards: numpy.ndarray = numpy.zeros(len(test_cycle_rewards), dtype=float)
                test_cycle_scaled_rewards: numpy.ndarray = numpy.zeros(len(test_cycle_rewards), dtype=float)
                for index, episode_rewards in enumerate(test_cycle_rewards):
                    test_cycle_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                    test_cycle_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                # Compute total and scaled reward over the test cycle
                test_cycle_average_total_reward: float = numpy.average(test_cycle_total_rewards)
                test_cycle_average_scaled_reward: float = numpy.average(test_cycle_scaled_rewards)
                # Save the rewards
                test_average_total_rewards[test] = test_cycle_average_total_reward
                test_average_scaled_rewards[test] = test_cycle_average_scaled_reward
                # Print test results
                logger.info("Testing of " + str(test_episodes_per_cycle) + " finished with following result:")
                logger.info("Average total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_total_reward))
                logger.info("Average scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_scaled_reward))
            # Get the average and the best total and scaled rewards over all cycles
            average_test_average_total_reward: float = numpy.round(numpy.average(test_average_total_rewards), 3)
            max_test_average_total_reward: float = numpy.round(numpy.max(test_average_total_rewards), 3)
            average_test_average_scaled_reward: float = numpy.round(numpy.average(test_average_scaled_rewards), 3)
            max_test_average_scaled_reward: float = numpy.round(numpy.max(test_average_scaled_rewards), 3)
            # Print final results and outcome of the experiment
            logger.info("Average test total reward is " + str(average_test_average_total_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Average test scaled reward is " + str(average_test_average_scaled_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Max test total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_total_reward))
            logger.info("Max test scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_scaled_reward))
            # Save plots
            logger.info("Saving plots...")
            plot.plot(list(range(len(all_training_volleys_total_rewards))), all_training_volleys_total_rewards)
            plot.xlabel('Training episode')
            plot.ylabel('Total reward')
            plot.savefig(self._plots_path + "/training_total_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_training_volleys_scaled_rewards))), all_training_volleys_scaled_rewards)
            plot.xlabel('Training episode')
            plot.ylabel('Scaled reward')
            plot.savefig(self._plots_path + "/training_scaled_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_validation_volleys_total_rewards))), all_validation_volleys_total_rewards)
            plot.xlabel('Validation episode')
            plot.ylabel('Total reward')
            plot.savefig(self._plots_path + "/validation_total_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_validation_volleys_scaled_rewards))), all_validation_volleys_scaled_rewards)
            plot.xlabel('Validation episode')
            plot.ylabel('Scaled reward')
            plot.savefig(self._plots_path + "/validation_scaled_rewards.png", dpi=300, transparent=True)
            plot.clf()
            logger.info("Plots saved successfully")
            # Check if the experiment is successful
            success: bool = self._is_successful(logger,
                                                average_test_average_total_reward, average_test_average_scaled_reward,
                                                max_test_average_total_reward, max_test_average_scaled_reward,
                                                last_validation_volley_average_total_reward,
                                                last_validation_volley_average_scaled_reward,
                                                last_training_volley_average_total_reward,
                                                last_training_volley_average_scaled_reward,
                                                test_cycles_rewards,
                                                last_validation_volley_rewards, last_training_volley_rewards)
            if success:
                logger.info("The experiment terminated successfully")
            else:
                logger.info("The experiment terminated without meeting the requirements")
            return average_test_average_total_reward, max_test_average_total_reward, average_test_average_scaled_reward, max_test_average_scaled_reward, self._trained_episodes, success, self._metagraph_path

    def watch(self,
              episode_length_max: int,
              test_episodes_per_cycle: int, test_cycles: int,
              logger: logging.Logger,
              checkpoint_path: str,
              render: bool = True):
        """
        Watch an experiment with an already trained agent given by the checkpoint path. The experiment needs to be
        setup and also trained somehow to access a proper checkpoint.

        Use this method to see how a saved agent model performs.

        :param episode_length_max: the max length in steps of each episode
        :param test_episodes_per_cycle: the number of episodes to play for each test cycle
        :param test_cycles: the number of test cycles to execute
        :param logger: the logger used to print the experiment information, warnings and errors
        :param checkpoint_path: the optional checkpoint path to load the pre-trained model from, it should be a valid model
        :param render: boolean flag to render the environment, default to True
        """
        logger.info("Watching experiment " + self._current_id + "...")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            self._environment.initialize(logger, session)
            self._agent.initialize(logger, session)
            # Load the pre-trained model if required
            if checkpoint_path is None:
                logger.error("Need to specify a checkpoint path to watch the experiment!")
                return
            else:
                checkpoint = tensorflow.train.get_checkpoint_state(checkpoint_path)
                # If checkpoint exists restore from checkpoint
                if checkpoint and checkpoint.model_checkpoint_path:
                    self._agent_saver.restore(session, tensorflow.train.latest_checkpoint(checkpoint_path))
                    logger.info("Model graph stored at " + checkpoint_path + " loaded successfully!")
                else:
                    logger.error("Checkpoint path specified is wrong: no model can be accessed at " + checkpoint_path)
                    return
                # Test the model and get all cycles total, scaled rewards and all rewards
                test_average_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_average_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_cycles_rewards: [] = []
                for test in range(test_cycles):
                    # Run inference for test episodes per cycles and get the average score
                    logger.info("Cycle " + str(test + 1) + " - Testing for " + str(test_episodes_per_cycle) + " episodes...")
                    test_cycle_rewards: [] = self._inference(logger,
                                                             test_episodes_per_cycle,
                                                             episode_length_max,
                                                             session,
                                                             render)
                    # Append the test cycle rewards to the full list
                    test_cycles_rewards.append(test_cycle_rewards)
                    # Since rewards list are of different sizes, get their sum and average
                    test_cycle_average_total_rewards: numpy.ndarray = numpy.zeros(len(test_cycle_rewards), dtype=float)
                    test_cycle_average_scaled_rewards: numpy.ndarray = numpy.zeros(len(test_cycle_rewards), dtype=float)
                    for index, episode_rewards in enumerate(test_cycle_rewards):
                        test_cycle_average_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                        test_cycle_average_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                    # Compute total and scaled reward over the test cycle
                    test_cycle_average_total_reward: float = numpy.average(test_cycle_average_total_rewards)
                    test_cycle_average_scaled_reward: float = numpy.average(test_cycle_average_scaled_rewards)
                    # Save the rewards
                    test_average_total_rewards[test] = test_cycle_average_total_reward
                    test_average_scaled_rewards[test] = test_cycle_average_scaled_reward
                    # Print test results
                    logger.info("Testing of " + str(test_episodes_per_cycle) + " finished with following result:")
                    logger.info("Average total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_total_reward))
                    logger.info("Average scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_scaled_reward))
                # Get the average and the best total and scaled rewards over all cycles
                average_test_total_reward: float = numpy.round(numpy.average(test_average_total_rewards), 3)
                max_test_total_reward: float = numpy.round(numpy.max(test_average_total_rewards), 3)
                average_test_scaled_reward: float = numpy.round(numpy.average(test_average_scaled_rewards), 3)
                max_test_scaled_reward: float = numpy.round(numpy.max(test_average_scaled_rewards), 3)
                # Print final results and outcome of the experiment
                logger.info("Average test total reward is " + str(average_test_total_reward))
                logger.info("Average test scaled reward is " + str(average_test_scaled_reward))
                logger.info("Max test total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_total_reward))
                logger.info("Max test scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_scaled_reward))

    def _is_validated(self,
                      logger: logging.Logger,
                      last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                      last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                      last_validation_volley_rewards: [], last_training_volley_rewards: []) -> bool:
        """
        Check if the experiment is to be considered validated using the given parameters.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param last_average_validation_total_reward: the average total reward during last volley validation phase
        :param last_average_validation_scaled_reward: the average scaled reward during last volley validation phase
        :param last_average_training_total_reward: the average total reward during last volley training phase
        :param last_average_training_scaled_reward: the average scaled reward during last volley training phase
        :param last_validation_volley_rewards: the list of rewards (grouped by episode) for each step during last volley validation phase
        :param last_training_volley_rewards: the list of rewards (grouped by episode) for each step during last volley training phase
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _is_successful(self,
                       logger: logging.Logger,
                       average_test_total_reward: float, average_test_scaled_reward: float,
                       max_test_total_reward: float, max_test_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       test_cycles_rewards: [],
                       last_validation_volley_rewards: [], last_training_volley_rewards: []) -> bool:
        """
        Check if the experiment is to be considered successful using the given parameters.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param average_test_total_reward: the average total reward during all the test cycles
        :param average_test_scaled_reward: the average scaled reward during all the test cycles
        :param max_test_total_reward: the maximum total reward during all the test cycles
        :param max_test_scaled_reward: the maximum scaled reward during all the test cycles
        :param last_average_validation_total_reward: the average total reward during last volley validation phase
        :param last_average_validation_scaled_reward: the average scaled reward during last volley validation phase
        :param last_average_training_total_reward: the average total reward during last volley training phase
        :param last_average_training_scaled_reward: the average scaled reward during last volley training phase
        :param test_cycles_rewards: the list of rewards (grouped by episode) for each step during all the test cycles
        :param last_validation_volley_rewards: the list of rewards (grouped by episode) for each step during last volley validation phase
        :param last_training_volley_rewards: the list of rewards (grouped by episode) for each step during last volley training phase
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()
