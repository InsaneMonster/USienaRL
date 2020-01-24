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
        self._checkpoint_path: str = None
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
              checkpoint_path: str = None, scopes_to_restore: [] = None,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before conduct method.

        If an already trained model checkpoint is given, the last checkpoint in the given path folder is used to further
        train the model. The scope of the experiment and of the agent has to be the same across both models or a list
        of scopes to restore has to be defined. Using that list, which is passed to the agent, it is possible to decide
        which variables to restore using the appropriate agent's property.

        :param summary_path: the string path of the tensorboard summary directory to save during model training, set to None if the model is not being trained
        :param metagraph_path: the string path of the metagraph agent directory to save at the end of each train volley, set to None if the model is not being trained
        :param logger: the logger used to print the experiment information, warnings and errors
        :param checkpoint_path: the optional checkpoint path to load the pre-trained model from, it should be a valid model, default to None
        :param scopes_to_restore: the scopes of the checkpoint path to restore into the model, it can be used to define what part of the model should be restored
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
                                 self._current_id, summary_path, checkpoint_path, scopes_to_restore):
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful!")
        # If setup of both environment and agent is successful, reset internal experiment variables
        self._metagraph_path = metagraph_path
        self._plots_path = self._metagraph_path.replace('/metagraph', '/plots')
        self._trained_steps: int = 0
        self._trained_episodes: int = 0
        # Initialize the agent internal model saver
        self._checkpoint_path = checkpoint_path
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
                steps: int, episode_length: int,
                session,
                render: bool = False):
        """
        Execute a volley of warm-up of the agent on the environment. This volley is measured in steps and it is usually
        run for how many steps are required by the model.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param steps: the number of steps for which to run the agent in warm-up mode
        :param episode_length: the max length in steps of each episode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during warm-up
        """
        episode: int = 0
        warmup_step: int = 0
        while True:
            # Start the episode
            episode = episode + 1
            episode_rewards: [] = []
            state_current = self._environment.reset(logger, session)
            # Execute actions until the episode is completed or the maximum length is exceeded
            for step in range(episode_length):
                # Increase counter and check if we can stop warming up
                warmup_step = warmup_step + 1
                if warmup_step > steps:
                    return
                # Print current warming-up progress every once in a while (if length is not too short)
                if steps > 100 and warmup_step % (steps // 10) == 0 and warmup_step > 0:
                    logger.info("Warmed-up for " + str(warmup_step) + "/" + str(steps) + " steps...")
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
                                                 step, episode, steps)
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
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_warmup(logger, session, self._interface,
                                                episode_rewards[-1], numpy.sum(numpy.array(episode_rewards)),
                                                episode, steps)

    def _train(self,
               logger: logging.Logger,
               episodes: int, episode_length: int, episodes_max: int,
               session,
               render: bool = False):
        """
        Execute a volley of training of the agent on the environment.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in train mode
        :param episode_length: the max length in steps of each episode
        :param episodes_max: the max number of allowed training episodes
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during training
        :return: the list of rewards obtained and episode steps performed in the volley, grouped by episode
        """
        # Define a list of rewards/steps found in the volley
        volley_rewards: [] = []
        volley_steps: [] = []
        for episode in range(episodes):
            # Print current training progress every once in a while (if length is not too short)
            if episodes > 100 and episode % (episodes // 10) == 0 and episode > 0:
                logger.info("Trained for " + str(episode) + "/" + str(episodes) + " episodes...")
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
                # Save volley steps up to termination point
                if step + 1 == episode_length or episode_done:
                    volley_steps.append(step)
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
        # Return the list of all rewards/steps seen in the volley
        return volley_rewards, volley_steps

    def _inference(self,
                   logger: logging.Logger,
                   episodes: int, episode_length: int,
                   session,
                   render: bool = False) -> []:
        """
        Execute a volley of inference of the agent on the environment.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in inference mode
        :param episode_length: the max length in steps of each episode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during inference
        :return: the list of rewards obtained and episode step performed in the volley, grouped by episode
        """
        # Define a list of rewards/steps found in the volley
        volley_rewards: [] = []
        volley_steps: [] = []
        for episode in range(episodes):
            # Print current validation progress every once in a while (if length is not too short)
            if episodes > 100 and episode % (episodes // 10) == 0 and episode > 0:
                logger.info("Validated for " + str(episode) + "/" + str(episodes) + " episodes...")
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
                # Save volley steps up to termination point
                if step + 1 == episode_length or episode_done:
                    volley_steps.append(step)
                # Check if the episode is completed
                if episode_done:
                    break
            # Add the list of episode rewards to the volley rewards
            volley_rewards.append(episode_rewards)
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_inference(logger, session, self._interface,
                                                   episode_rewards[-1], numpy.sum(numpy.array(episode_rewards)),
                                                   episode, episodes)
        # Return the list of all rewards/steps seen in the volley
        return volley_rewards, volley_steps

    def conduct(self,
                training_episodes_per_volley: int, validation_episodes_per_volley: int,
                training_episodes_max: int, episode_length_max: int,
                test_episodes_per_cycle: int, test_cycles: int,
                logger: logging.Logger,
                render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False,
                plot_sample_density_training: int = 1, plot_sample_density_validation: int = 1):
        """
        Conduct the experiment of the given number of training (up to the given maximum) and validation episodes
        per volley, with the associated given number of test episodes per cycle and the given number of test cycles after
        the validation has passed.

        A volley of training is always considered to stop the training exactly when the agent performs well enough
        to validate, by checking the appropriate condition. When validation is successful, appropriate condition is also
        checked on the test cycles to decide if the experiment is successful or not.

        Conducting an experiment will generate a tensorflow session for that experiment. It is required that the experiment
        is ready to be conducted by checking the result of the setup method before attempting at conducting it.

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
        :param plot_sample_density_training: the optional number represent after how many episodes a sample for the episode-related training plots is sampled (default to 1, i.e. each episode)
        :param plot_sample_density_validation: the optional number represent after how many episodes a sample for the episode-related validation plots is sampled (default to 1, i.e. each episode)
        :return: the average and the max of the total average reward, the average and the max of the average average reward with respect to all test cycles, the training episodes required to validate, a success flag and the location in which the saved metagraph is stored
        """
        # Make the matplotlib use agg-backend to make it compatible with a server and increase the chunksize to avoid unwanted errors
        plot.switch_backend('agg')
        plot.rcParams['agg.path.chunksize'] = 1000
        # Start experiment
        logger.info("Conducting experiment " + self._current_id + "...")
        # Define the list in which to store the evolution of total and scaled average reward to plot by default
        all_training_episodes_total_rewards: [] = []
        all_training_episodes_scaled_rewards: [] = []
        all_training_episodes_length: [] = []
        all_validation_episodes_total_rewards: [] = []
        all_validation_episodes_scaled_rewards: [] = []
        all_validation_episodes_length: [] = []
        all_training_volleys_total_rewards: [] = []
        all_training_volleys_scaled_rewards: [] = []
        all_training_volleys_std_total_rewards: [] = []
        all_training_volleys_std_scaled_rewards: [] = []
        all_training_volleys_average_episode_length: [] = []
        all_validation_volleys_total_rewards: [] = []
        all_validation_volleys_scaled_rewards: [] = []
        all_validation_volleys_std_total_rewards: [] = []
        all_validation_volleys_std_scaled_rewards: [] = []
        all_validation_volleys_average_episode_length: [] = []
        # Define a counter for training volleys and validation volleys
        training_volley_counter: int = 0
        validation_volley_counter: int = 0
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            self._environment.initialize(logger, session)
            self._agent.initialize(logger, session)
            self._environment.post_initialize(logger, session)
            # Load the pre-trained model if checkpoint path is given in setup
            if self._checkpoint_path is not None:
                checkpoint = tensorflow.train.get_checkpoint_state(self._checkpoint_path)
                # If checkpoint exists restore from checkpoint
                if checkpoint and checkpoint.model_checkpoint_path:
                    self._agent_saver.restore(session, tensorflow.train.latest_checkpoint(self._checkpoint_path))
                    logger.info("Model graph stored at " + self._checkpoint_path + " loaded successfully!")
                else:
                    logger.error("Checkpoint path specified is wrong: no model can be accessed at " + self._checkpoint_path)
            # Execute pre-training if the agent requires pre-training
            if self._agent.warmup_steps > 0:
                logger.info("Warming-up for " + str(self._agent.warmup_steps) + " steps...")
                self._warmup(logger,
                             self._agent.warmup_steps,
                             episode_length_max,
                             session)
            # Save last volley average training and validation total and scaled rewards
            last_training_volley_average_total_reward: float = 0.0
            last_training_volley_average_scaled_reward: float = 0.0
            last_training_volley_std_total_reward: float = 0.0
            last_training_volley_std_scaled_reward: float = 0.0
            last_validation_volley_average_total_reward: float = 0.0
            last_validation_volley_average_scaled_reward: float = 0.0
            last_validation_volley_std_total_reward: float = 0.0
            last_validation_volley_std_scaled_reward: float = 0.0
            # Save last volley training and validation rewards list
            last_training_volley_rewards: [] = []
            last_validation_volley_rewards: [] = []
            last_training_volley_steps: [] = []
            last_validation_volley_steps: [] = []
            # Execute training until max training episodes number is reached or the validation score is above the threshold
            while self._trained_episodes < training_episodes_max:
                # Run train for training episodes per volley and get the average score
                logger.info("Training for " + str(training_episodes_per_volley) + " episodes...")
                last_training_volley_rewards, last_training_volley_steps = self._train(logger,
                                                                                       training_episodes_per_volley,
                                                                                       episode_length_max,
                                                                                       training_episodes_max,
                                                                                       session,
                                                                                       render_during_training)
                training_volley_counter += 1
                # Since rewards list are of different sizes, get their sum and average
                training_volley_total_rewards: numpy.ndarray = numpy.zeros(len(last_training_volley_rewards), dtype=float)
                training_volley_scaled_rewards: numpy.ndarray = numpy.zeros(len(last_training_volley_rewards), dtype=float)
                for index, episode_rewards in enumerate(last_training_volley_rewards):
                    training_volley_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                    training_volley_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                    # Update the global lists of all the average total and scaled rewards for each training episode
                    all_training_episodes_total_rewards.append(numpy.sum(numpy.array(episode_rewards)))
                    all_training_episodes_scaled_rewards.append(numpy.average(numpy.array(episode_rewards)))
                # Push all the saved steps into the episode list
                all_training_episodes_length += last_training_volley_steps
                # Compute average/std total reward and average/std scaled reward and episode length (in steps) over training volley and append it to the volley list
                last_training_volley_average_total_reward: float = numpy.round(numpy.average(training_volley_total_rewards), 3)
                last_training_volley_average_scaled_reward: float = numpy.round(numpy.average(training_volley_scaled_rewards), 3)
                last_training_volley_std_total_reward: float = numpy.round(numpy.std(training_volley_total_rewards), 3)
                last_training_volley_std_scaled_reward: float = numpy.round(numpy.std(training_volley_scaled_rewards), 3)
                last_training_volley_average_episode_length: int = numpy.rint(numpy.average(numpy.array(last_training_volley_steps)))
                all_training_volleys_total_rewards.append(last_training_volley_average_total_reward)
                all_training_volleys_scaled_rewards.append(last_training_volley_average_scaled_reward)
                all_training_volleys_std_total_rewards.append(last_training_volley_std_total_reward)
                all_training_volleys_std_scaled_rewards.append(last_training_volley_std_scaled_reward)
                all_training_volleys_average_episode_length.append(last_training_volley_average_episode_length)
                # Print training results
                logger.info("Training of " + str(training_episodes_per_volley) + " episodes finished with following result:")
                logger.info("Average total reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_average_total_reward))
                logger.info("Standard deviation of total reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_std_total_reward))
                logger.info("Average scaled reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_average_scaled_reward))
                logger.info("Standard deviation of scaled reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_std_scaled_reward))
                logger.info("Average episode length over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_training_volley_average_episode_length) + " steps")
                logger.info("Total environmental steps in training mode up to now: " + str(self._trained_steps))
                # Save the agent internal model at the current step
                logger.info("Saving the agent...")
                self._agent_saver.save(session, self._metagraph_path + "/" + self._current_id)
                # Run inference for validation episodes per volley and get the average score
                logger.info("Validating for " + str(validation_episodes_per_volley) + " episodes...")
                last_validation_volley_rewards, last_validation_volley_steps = self._inference(logger,
                                                                                               validation_episodes_per_volley,
                                                                                               episode_length_max,
                                                                                               session,
                                                                                               render_during_validation)
                validation_volley_counter += 1
                # Since rewards list are of different sizes, get their sum and average
                validation_volley_total_rewards: numpy.ndarray = numpy.zeros(len(last_validation_volley_rewards), dtype=float)
                validation_volley_scaled_rewards: numpy.ndarray = numpy.zeros(len(last_validation_volley_rewards), dtype=float)
                for index, episode_rewards in enumerate(last_validation_volley_rewards):
                    validation_volley_total_rewards[index] = numpy.sum(numpy.array(episode_rewards))
                    validation_volley_scaled_rewards[index] = numpy.average(numpy.array(episode_rewards))
                    # Update the global lists of all the average total and scale rewards for each validation episode
                    all_validation_episodes_total_rewards.append(numpy.sum(numpy.array(episode_rewards)))
                    all_validation_episodes_scaled_rewards.append(numpy.average(numpy.array(episode_rewards)))
                # Push all the saved steps into the episode list
                all_validation_episodes_length += last_validation_volley_steps
                # Compute average/std total reward and average/std scaled reward over validation volley and append it to volley list
                last_validation_volley_average_total_reward: float = numpy.round(numpy.average(validation_volley_total_rewards), 3)
                last_validation_volley_average_scaled_reward: float = numpy.round(numpy.average(validation_volley_scaled_rewards), 3)
                last_validation_volley_std_total_reward: float = numpy.round(numpy.std(validation_volley_total_rewards), 3)
                last_validation_volley_std_scaled_reward: float = numpy.round(numpy.std(validation_volley_scaled_rewards), 3)
                last_validation_volley_average_episode_length: int = numpy.rint(numpy.average(numpy.array(last_validation_volley_steps)))
                all_validation_volleys_total_rewards.append(last_validation_volley_average_total_reward)
                all_validation_volleys_scaled_rewards.append(last_validation_volley_average_scaled_reward)
                all_validation_volleys_std_total_rewards.append(last_validation_volley_std_total_reward)
                all_validation_volleys_std_scaled_rewards.append(last_validation_volley_std_scaled_reward)
                all_validation_volleys_average_episode_length.append(last_validation_volley_average_episode_length)
                # Print validation results
                logger.info("Validation of " + str(validation_episodes_per_volley) + " episodes finished with following result:")
                logger.info("Average total reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_average_total_reward))
                logger.info("Standard deviation of total reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_std_total_reward))
                logger.info("Average scaled reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_average_scaled_reward))
                logger.info("Standard deviation of scaled reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_std_scaled_reward))
                logger.info("Average episode length over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(last_validation_volley_average_episode_length) + " steps")
                # Check for validation
                if self._is_validated(logger,
                                      last_validation_volley_average_total_reward, last_validation_volley_average_scaled_reward,
                                      last_training_volley_average_total_reward, last_training_volley_average_scaled_reward,
                                      last_validation_volley_std_total_reward, last_validation_volley_std_scaled_reward,
                                      last_training_volley_std_total_reward, last_training_volley_std_scaled_reward,
                                      last_validation_volley_rewards, last_training_volley_rewards,
                                      plot_sample_density_training, plot_sample_density_validation):
                    logger.info("Validation of the agent is successful")
                    break
            logger.info("End of training")
            # Test the model and get all cycles total, scaled rewards and all rewards and all episode lengths
            test_average_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_average_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_std_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_std_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_average_episode_length: numpy.ndarray = numpy.zeros(test_cycles, dtype=int)
            test_cycles_rewards: [] = []
            for test in range(test_cycles):
                # Run inference for test episodes per cycles and get the average score
                logger.info("Cycle " + str(test + 1) + " - Testing for " + str(test_episodes_per_cycle) + " episodes...")
                test_cycle_rewards, test_cycle_steps = self._inference(logger,
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
                test_cycle_average_total_reward: float = numpy.round(numpy.average(test_cycle_total_rewards), 3)
                test_cycle_average_scaled_reward: float = numpy.round(numpy.average(test_cycle_scaled_rewards), 3)
                test_cycle_std_total_reward: float = numpy.round(numpy.std(test_cycle_total_rewards), 3)
                test_cycle_std_scaled_reward: float = numpy.round(numpy.std(test_cycle_scaled_rewards), 3)
                # Compute average episode length over the test cycle
                test_cycle_average_episode_length: int = numpy.rint(numpy.average(numpy.array(test_cycle_steps)))
                # Display test additional optional statistics
                self._display_test_cycle_metrics(logger,
                                                 test_cycle_average_total_reward,
                                                 test_cycle_average_scaled_reward,
                                                 test_cycle_std_total_reward,
                                                 test_cycle_std_scaled_reward,
                                                 test_cycle_rewards,
                                                 plot_sample_density_training, plot_sample_density_validation)
                # Save the rewards and the episode average length
                test_average_total_rewards[test] = test_cycle_average_total_reward
                test_average_scaled_rewards[test] = test_cycle_average_scaled_reward
                test_std_total_rewards[test] = test_cycle_std_total_reward
                test_std_scaled_rewards[test] = test_cycle_std_scaled_reward
                test_average_episode_length[test] = test_cycle_average_episode_length
                # Print test results
                logger.info("Testing of " + str(test_episodes_per_cycle) + " episodes finished with following result:")
                logger.info("Average total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_total_reward))
                logger.info("Standard deviation of total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_std_total_reward))
                logger.info("Average scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_scaled_reward))
                logger.info("Standard deviation of scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_std_scaled_reward))
                logger.info("Average episode length over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_episode_length) + " steps")
            logger.info("End of test")
            # Get the average and the best total and scaled rewards over all cycles
            average_test_average_total_reward: float = numpy.round(numpy.average(test_average_total_rewards), 3)
            max_test_average_total_reward: float = numpy.round(numpy.max(test_average_total_rewards), 3)
            average_test_average_scaled_reward: float = numpy.round(numpy.average(test_average_scaled_rewards), 3)
            max_test_average_scaled_reward: float = numpy.round(numpy.max(test_average_scaled_rewards), 3)
            # Get the average and the minimum standard deviation of total and scaled rewards over all cycles
            average_test_std_total_reward: float = numpy.round(numpy.average(test_std_total_rewards), 3)
            min_test_std_total_reward: float = numpy.round(numpy.min(test_std_total_rewards), 3)
            average_test_std_scaled_reward: float = numpy.round(numpy.average(test_std_scaled_rewards), 3)
            min_test_std_scaled_reward: float = numpy.round(numpy.min(test_std_scaled_rewards), 3)
            # Get the average episode length over all cycles
            average_test_average_episode_length: int = numpy.rint(numpy.average(test_average_episode_length))
            # Print final results and outcome of the experiment
            logger.info("Average test total reward is " + str(average_test_average_total_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Average test scaled reward is " + str(average_test_average_scaled_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Max test total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_total_reward))
            logger.info("Max test scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_scaled_reward))
            logger.info("Average test standard deviation of total reward is " + str(average_test_std_total_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Average test standard deviation of scaled reward is " + str(average_test_std_scaled_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Min test standard deviation of total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(min_test_std_total_reward))
            logger.info("Min test standard deviation of scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(min_test_std_scaled_reward))
            logger.info("Average test episode length is " + str(average_test_average_episode_length) + " with " + str(self._trained_episodes) + " training episodes")
            # Save plots
            logger.info("Saving plots...")
            plot.plot(list(range(len(all_training_episodes_total_rewards)))[::plot_sample_density_training], all_training_episodes_total_rewards[::plot_sample_density_training], 'r-')
            plot.xlabel('Training episode')
            plot.ylabel('Total reward')
            plot.savefig(self._plots_path + "/training_episodes_total_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_training_episodes_scaled_rewards)))[::plot_sample_density_training], all_training_episodes_scaled_rewards[::plot_sample_density_training], 'r--')
            plot.xlabel('Training episode')
            plot.ylabel('Scaled reward')
            plot.savefig(self._plots_path + "/training_episodes_scaled_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_training_episodes_length)))[::plot_sample_density_training], all_training_episodes_length[::plot_sample_density_training], 'b-.')
            plot.xlabel('Training episode')
            plot.ylabel('Episode length (steps)')
            plot.savefig(self._plots_path + "/training_episodes_lengths.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_validation_episodes_total_rewards)))[::plot_sample_density_validation], all_validation_episodes_total_rewards[::plot_sample_density_validation], 'g-')
            plot.xlabel('Validation episode')
            plot.ylabel('Total reward')
            plot.savefig(self._plots_path + "/validation_episodes_total_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_validation_episodes_scaled_rewards)))[::plot_sample_density_validation], all_validation_episodes_scaled_rewards[::plot_sample_density_validation], 'g--')
            plot.xlabel('Validation episode')
            plot.ylabel('Scaled reward')
            plot.savefig(self._plots_path + "/validation_episodes_scaled_rewards.png", dpi=300, transparent=True)
            plot.clf()
            plot.plot(list(range(len(all_validation_episodes_length)))[::plot_sample_density_validation], all_validation_episodes_length[::plot_sample_density_validation], 'b-.')
            plot.xlabel('Validation episode')
            plot.ylabel('Episode length (steps)')
            plot.savefig(self._plots_path + "/validation_episodes_lengths.png", dpi=300, transparent=True)
            plot.clf()
            if training_volley_counter > 1:
                plot.plot(list(range(len(all_training_volleys_total_rewards))), all_training_volleys_total_rewards, 'r-')
                plot.xlabel('Training volley')
                plot.ylabel('Average total reward')
                plot.savefig(self._plots_path + "/training_volleys_total_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_training_volleys_scaled_rewards))), all_training_volleys_scaled_rewards, 'r--')
                plot.xlabel('Training volley')
                plot.ylabel('Average scaled reward')
                plot.savefig(self._plots_path + "/training_volleys_scaled_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_training_volleys_std_total_rewards))), all_training_volleys_std_total_rewards, 'c-')
                plot.xlabel('Training volley')
                plot.ylabel('Standard deviation of total reward')
                plot.savefig(self._plots_path + "/training_volleys_std_total_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_training_volleys_std_scaled_rewards))), all_training_volleys_std_scaled_rewards, 'c--')
                plot.xlabel('Training volley')
                plot.ylabel('Standard deviation of scaled reward')
                plot.savefig(self._plots_path + "/training_volleys_std_scaled_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_training_volleys_average_episode_length))), all_training_volleys_average_episode_length, 'b-.')
                plot.xlabel('Training volley')
                plot.ylabel('Average episode length (steps)')
                plot.savefig(self._plots_path + "/training_volleys_average_episode_length.png", dpi=300, transparent=True)
                plot.clf()
            if validation_volley_counter > 1:
                plot.plot(list(range(len(all_validation_volleys_total_rewards))), all_validation_volleys_total_rewards, 'm-')
                plot.xlabel('Validation volley')
                plot.ylabel('Average total reward')
                plot.savefig(self._plots_path + "/validation_volleys_total_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_validation_volleys_scaled_rewards))), all_validation_volleys_scaled_rewards, 'm--')
                plot.xlabel('Validation volley')
                plot.ylabel('Average scaled reward')
                plot.savefig(self._plots_path + "/validation_volleys_scaled_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_validation_volleys_std_total_rewards))), all_validation_volleys_std_total_rewards, 'y-')
                plot.xlabel('Validation volley')
                plot.ylabel('Standard deviation of total reward')
                plot.savefig(self._plots_path + "/validation_volleys_std_total_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_validation_volleys_std_scaled_rewards))), all_validation_volleys_std_scaled_rewards, 'y--')
                plot.xlabel('Validation volley')
                plot.ylabel('Standard deviation of scaled reward')
                plot.savefig(self._plots_path + "/validation_volleys_std_scaled_rewards.png", dpi=300, transparent=True)
                plot.clf()
                plot.plot(list(range(len(all_validation_volleys_average_episode_length))), all_validation_volleys_average_episode_length, 'b-.')
                plot.xlabel('Validation volley')
                plot.ylabel('Average episode length (steps)')
                plot.savefig(self._plots_path + "/validation_volleys_average_episode_length.png", dpi=300, transparent=True)
                plot.clf()
            logger.info("Plots saved successfully")
            # Check if the experiment is successful
            success: bool = self._is_successful(logger,
                                                average_test_average_total_reward, average_test_average_scaled_reward,
                                                max_test_average_total_reward, max_test_average_scaled_reward,
                                                average_test_std_total_reward, average_test_std_scaled_reward,
                                                min_test_std_total_reward, min_test_std_scaled_reward,
                                                last_validation_volley_average_total_reward,
                                                last_validation_volley_average_scaled_reward,
                                                last_training_volley_average_total_reward,
                                                last_training_volley_average_scaled_reward,
                                                last_validation_volley_std_total_reward,
                                                last_validation_volley_std_scaled_reward,
                                                last_training_volley_std_total_reward,
                                                last_training_volley_std_scaled_reward,
                                                test_cycles_rewards,
                                                last_validation_volley_rewards, last_training_volley_rewards,
                                                plot_sample_density_training, plot_sample_density_validation)
            if success:
                logger.info("The experiment terminated successfully")
            else:
                logger.info("The experiment terminated without meeting the requirements")
            return average_test_average_total_reward, max_test_average_total_reward, average_test_average_scaled_reward, max_test_average_scaled_reward, self._trained_episodes, success, self._metagraph_path

    def watch(self,
              episode_length_max: int,
              test_episodes_per_cycle: int, test_cycles: int,
              logger: logging.Logger,
              render: bool = True):
        """
        Watch an experiment with an already trained agent given by the checkpoint path defined during setup.
        Use this method to see how a saved agent model performs.

        :param episode_length_max: the max length in steps of each episode
        :param test_episodes_per_cycle: the number of episodes to play for each test cycle
        :param test_cycles: the number of test cycles to execute
        :param logger: the logger used to print the experiment information, warnings and errors
        :param render: boolean flag to render the environment, default to True
        """
        logger.info("Watching experiment " + self._current_id + "...")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_config) as session:
            # Initialize the environment and the agent
            self._environment.initialize(logger, session)
            self._agent.initialize(logger, session)
            # Load the pre-trained model at the checkpoint path given in setup
            if self._checkpoint_path is None:
                logger.error("Need to specify a checkpoint path to watch the experiment!")
                return
            else:
                checkpoint = tensorflow.train.get_checkpoint_state(self._checkpoint_path)
                # If checkpoint exists restore from checkpoint
                if checkpoint and checkpoint.model_checkpoint_path:
                    self._agent_saver.restore(session, tensorflow.train.latest_checkpoint(self._checkpoint_path))
                    logger.info("Model graph stored at " + self._checkpoint_path + " loaded successfully!")
                else:
                    logger.error("Checkpoint path specified is wrong: no model can be accessed at " + self._checkpoint_path)
                    return
                # Test the model and get all cycles total, scaled rewards and all rewards and all episode lengths
                test_average_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_average_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_std_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_std_scaled_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
                test_average_episode_length: numpy.ndarray = numpy.zeros(test_cycles, dtype=int)
                test_cycles_rewards: [] = []
                for test in range(test_cycles):
                    # Run inference for test episodes per cycles and get the average score
                    logger.info("Cycle " + str(test + 1) + " - Testing for " + str(test_episodes_per_cycle) + " episodes...")
                    test_cycle_rewards, test_cycle_steps = self._inference(logger,
                                                                           test_episodes_per_cycle,
                                                                           episode_length_max,
                                                                           session,
                                                                           render)
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
                    test_cycle_std_total_reward: float = numpy.round(numpy.std(test_cycle_total_rewards), 3)
                    test_cycle_std_scaled_reward: float = numpy.round(numpy.std(test_cycle_scaled_rewards), 3)
                    # Compute average episode length over the test cycle
                    test_cycle_average_episode_length: int = numpy.rint(numpy.average(numpy.array(test_cycle_steps)))
                    # Display test additional optional statistics
                    self._display_test_cycle_metrics(logger,
                                                     test_cycle_average_total_reward,
                                                     test_cycle_average_scaled_reward,
                                                     test_cycle_std_total_reward,
                                                     test_cycle_std_scaled_reward,
                                                     test_cycle_rewards)
                    # Save the rewards
                    test_average_total_rewards[test] = test_cycle_average_total_reward
                    test_average_scaled_rewards[test] = test_cycle_average_scaled_reward
                    test_std_total_rewards[test] = test_cycle_std_total_reward
                    test_std_scaled_rewards[test] = test_cycle_std_scaled_reward
                    test_average_episode_length[test] = test_cycle_average_episode_length
                    # Print test results
                    logger.info("Testing of " + str(test_episodes_per_cycle) + " episodes finished with following result:")
                    logger.info("Average total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_total_reward))
                    logger.info("Standard deviation of total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_std_total_reward))
                    logger.info("Average scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_scaled_reward))
                    logger.info("Standard deviation of scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_std_scaled_reward))
                    logger.info("Average episode length over " + str(test_episodes_per_cycle) + " test episodes: " + str(test_cycle_average_episode_length) + " steps")
                # Get the average and the best total and scaled rewards over all cycles
                average_test_average_total_reward: float = numpy.round(numpy.average(test_average_total_rewards), 3)
                max_test_average_total_reward: float = numpy.round(numpy.max(test_average_total_rewards), 3)
                average_test_average_scaled_reward: float = numpy.round(numpy.average(test_average_scaled_rewards), 3)
                max_test_average_scaled_reward: float = numpy.round(numpy.max(test_average_scaled_rewards), 3)
                # Get the average and the minimum standard deviation of total and scaled rewards over all cycles
                average_test_std_total_reward: float = numpy.round(numpy.average(test_std_total_rewards), 3)
                min_test_std_total_reward: float = numpy.round(numpy.min(test_std_total_rewards), 3)
                average_test_std_scaled_reward: float = numpy.round(numpy.average(test_std_scaled_rewards), 3)
                min_test_std_scaled_reward: float = numpy.round(numpy.min(test_std_scaled_rewards), 3)
                # Get the average episode length over all cycles
                average_test_average_episode_length: int = numpy.rint(numpy.average(test_average_episode_length))
                # Print final results
                logger.info("Average test total reward is " + str(average_test_average_total_reward))
                logger.info("Average test scaled reward is " + str(average_test_average_scaled_reward))
                logger.info("Max test total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_total_reward))
                logger.info("Max test scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_scaled_reward))
                logger.info("Average test standard deviation of total reward is " + str(average_test_std_total_reward))
                logger.info("Average test standard deviation of scaled reward is " + str(average_test_std_scaled_reward))
                logger.info("Min test standard deviation of total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(min_test_std_total_reward))
                logger.info("Min test standard deviation of scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(min_test_std_scaled_reward))
                logger.info("Average test episode length is " + str(average_test_average_episode_length))

    def initialize(self):
        """
        Initialize the experiment, resetting all class-specific variables and preparing for the next iteration of the same
        experiment setup.
        """
        raise NotImplementedError()

    def _is_validated(self,
                      logger: logging.Logger,
                      last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                      last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                      last_std_validation_total_reward: float, last_std_validation_scaled_reward: float,
                      last_std_training_total_reward: float, last_std_training_scaled_reward: float,
                      last_validation_volley_rewards: [], last_training_volley_rewards: [],
                      plot_sample_density_training: int = 1, plot_sample_density_validation: int = 1) -> bool:
        """
        Check if the experiment is to be considered validated using the given parameters.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param last_average_validation_total_reward: the average total reward during last volley validation phase
        :param last_average_validation_scaled_reward: the average scaled reward during last volley validation phase
        :param last_average_training_total_reward: the average total reward during last volley training phase
        :param last_average_training_scaled_reward: the average scaled reward during last volley training phase
        :param last_std_validation_total_reward: the standard deviation of total reward during last volley validation phase
        :param last_std_validation_scaled_reward: the standard deviation of scaled reward during last volley validation phase
        :param last_std_training_total_reward: the standard deviation of total reward during last volley training phase
        :param last_std_training_scaled_reward: the standard deviation of scaled reward during last volley training phase
        :param last_validation_volley_rewards: the list of rewards (grouped by episode) for each step during last volley validation phase
        :param last_training_volley_rewards: the list of rewards (grouped by episode) for each step during last volley training phase
        :param plot_sample_density_training: the optional number represent after how many episodes a sample for the episode-related training plots is sampled (default to 1, i.e. each episode)
        :param plot_sample_density_validation: the optional number represent after how many episodes a sample for the episode-related validation plots is sampled (default to 1, i.e. each episode)
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _display_test_cycle_metrics(self,
                                    logger: logging.Logger,
                                    last_test_cycle_average_total_reward: float,
                                    last_test_cycle_average_scaled_reward: float,
                                    last_test_cycle_std_total_reward: float,
                                    last_test_cycle_std_scaled_reward: float,
                                    last_test_cycle_rewards: [],
                                    plot_sample_density_training: int = 1, plot_sample_density_validation: int = 1):
        """
        Display additional optional metrics from the last test cycle.
        Note: average and total and scaled reward over all the cycle test episodes are already displayed.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param last_test_cycle_average_total_reward: the average total reward in the last test cycle
        :param last_test_cycle_average_scaled_reward: the average scaled reward in the last test cycle
        :param last_test_cycle_std_total_reward: the standard deviation of total reward in the last test cycle
        :param last_test_cycle_std_scaled_reward: the standard deviation of scaled reward in the last test cycle
        :param last_test_cycle_rewards: a list of all the rewards obtained in each episode over all the episodes in the last test cycles
        :param plot_sample_density_training: the optional number represent after how many episodes a sample for the episode-related training plots is sampled (default to 1, i.e. each episode)
        :param plot_sample_density_validation: the optional number represent after how many episodes a sample for the episode-related validation plots is sampled (default to 1, i.e. each episode)
        """
        raise NotImplementedError()

    def _is_successful(self,
                       logger: logging.Logger,
                       average_test_total_reward: float, average_test_scaled_reward: float,
                       max_test_total_reward: float, max_test_scaled_reward: float,
                       average_test_std_total_reward: float, average_test_std_scaled_reward: float,
                       min_test_std_total_reward: float, min_test_std_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       last_std_validation_total_reward: float, last_std_validation_scaled_reward: float,
                       last_std_training_total_reward: float, last_std_training_scaled_reward: float,
                       test_cycles_rewards: [],
                       last_validation_volley_rewards: [], last_training_volley_rewards: [],
                       plot_sample_density_training: int = 1, plot_sample_density_validation: int = 1) -> bool:
        """
        Check if the experiment is to be considered successful using the given parameters.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param average_test_total_reward: the average total reward during all the test cycles
        :param average_test_scaled_reward: the average scaled reward during all the test cycles
        :param max_test_total_reward: the maximum total reward during all the test cycles
        :param max_test_scaled_reward: the maximum scaled reward during all the test cycles
        :param average_test_std_total_reward: the average standard deviation of total reward during all the test cycles
        :param average_test_std_scaled_reward: the average standard deviation of scaled reward during all the test cycles
        :param min_test_std_total_reward: the minimum standard deviation of total reward during all the test cycles
        :param min_test_std_scaled_reward: the minimum standard deviation of scaled reward during all the test cycles
        :param last_average_validation_total_reward: the average total reward during last volley validation phase
        :param last_average_validation_scaled_reward: the average scaled reward during last volley validation phase
        :param last_average_training_total_reward: the average total reward during last volley training phase
        :param last_average_training_scaled_reward: the average scaled reward during last volley training phase
        :param last_std_validation_total_reward: the standard deviation of total reward during last volley validation phase
        :param last_std_validation_scaled_reward: the standard deviation of scaled reward during last volley validation phase
        :param last_std_training_total_reward: the standard deviation of total reward during last volley training phase
        :param last_std_training_scaled_reward: the standard deviation of scaled reward during last volley training phase
        :param test_cycles_rewards: the list of rewards (grouped by episode) for each step during all the test cycles
        :param last_validation_volley_rewards: the list of rewards (grouped by episode) for each step during last volley validation phase
        :param last_training_volley_rewards: the list of rewards (grouped by episode) for each step during last volley training phase
        ::param plot_sample_density_training: the optional number represent after how many episodes a sample for the episode-related training plots is sampled (default to 1, i.e. each episode)
        :param plot_sample_density_validation: the optional number represent after how many episodes a sample for the episode-related validation plots is sampled (default to 1, i.e. each episode)
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()
