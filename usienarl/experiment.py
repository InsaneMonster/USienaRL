# Import packages

import tensorflow
import numpy
import logging

# Import required src

from usienarl import Environment, Agent, Interface


class Experiment:
    """
    TODO: _summary
    """

    def __init__(self,
                 name: str,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None):
        # Define experiment attributes
        self._name: str = name
        self._environment: Environment = environment
        self._agent: Agent = agent
        self._interface: Interface = interface if interface is not None else Interface(self._environment)
        # Define empty experiment attributes
        self._agent_saver = None
        self._metagraph_path: str = None
        self._tensorflow_gpu_config = None
        self._current_id: str = None
        self._trained_steps: int = None
        self._trained_episodes: int = None

    def setup(self,
              summary_path: str, metagraph_path: str,
              logger: logging.Logger,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before conduct method.

        :param summary_path: the string path of the TensorBoard _summary directory to save during _model training
        :param metagraph_path: the string path of the saved _model directory to save at the end of each training interval
        :param logger: the logger used to print the experiment information, warnings and errors
        :param iteration: number to append to the experiment _name in all scopes and print statements (if not less than zero)
        :return: a boolean equals to True if the setup of the experiment is successful, False otherwise
        """
        # Save current experiment id using _name and iteration
        self._current_id = self._name + "_" + str(iteration) if iteration > -1 else self._name
        # Start setup process
        logger.info("Setup of experiment " + self._current_id + "...")
        # Reset the tensorflow graph to a blank new graph
        tensorflow.reset_default_graph()
        # Execute the setup on the environment and on the agent and return if not both of them setup correctly
        logger.info("Environment setup..")
        if not self._environment.setup(logger):
            logger.info("Environment setup failed. Cannot setup the experiment!")
            return False
        logger.info("Environment setup is successful!")
        logger.info("Agent setup...")
        if not self._agent.setup(logger,
                                 self._interface.observation_space_type, self._interface.observation_space_shape,
                                 self._interface.agent_action_space_shape, self._interface.agent_action_space_type,
                                 self._current_id, summary_path):
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful!")
        # If setup of both environment and agent is successful, reset internal experiment variables
        self._metagraph_path = metagraph_path
        self._trained_steps: int = 0
        self._trained_episodes: int = 0
        # Initialize the agent internal _model saver
        self._agent_saver = tensorflow.train.Saver(self._agent.trainable_variables)
        logger.info("Agent internal _model will be saved after each train volley")
        logger.info("Agent internal _model metagraph save path: " + self._metagraph_path)
        # Initialize tensorflow gpu configuration
        self._tensorflow_gpu_config = tensorflow.ConfigProto()
        self._tensorflow_gpu_config.gpu_options.allow_growth = True
        logger.info("CUDA GPU device configured for Tensorflow-gpu")
        # Setup is successful
        logger.info("Experiment setup is successful. Now can conduct the experiment")
        return True

    def _warmup(self,
                logger: logging.Logger,
                episodes: int, episode_length: int,
                session,
                render: bool = False):
        """
        Execute a volley of pre-training of the agent on the environment.
        How the pre-training is done depends on the experiment implementation.

        Note: to call a pre-train step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in pre-train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during pre-training
        :return: the float average of the score obtained in the played episodes
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
                self._agent.complete_step_warmup(logger, session,
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
            episode_total_reward: float = numpy.sum(numpy.array(episode_rewards))
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_warmup(logger, session,
                                                episode_rewards[-1], episode_total_reward,
                                                episode, episodes)

    def _train(self,
               logger: logging.Logger,
               episodes: int, episode_length: int, episodes_max: int,
               session,
               render: bool = False):
        """
        Execute a volley of training of the agent on the environment.
        How the training is done depends on the experiment implementation.

        Note: to call a train step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during training
        :return: the float average of the score obtained in the played episodes
        """
        # Define arrays of volley rewards
        volley_average_rewards: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        volley_total_rewards: numpy.ndarray = numpy.zeros(episodes, dtype=float)
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
                self._agent.complete_step_train(logger, session,
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
            # Compute total and average reward over the steps in the episode
            episode_total_reward: float = numpy.sum(numpy.array(episode_rewards))
            episode_average_reward: float = numpy.average(numpy.array(episode_rewards))
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_train(logger, session,
                                               episode_rewards[-1], episode_total_reward,
                                               self._trained_steps,
                                               episode, self._trained_episodes,
                                               episodes, episodes_max)
            # Increase the counter of trained episodes
            self._trained_episodes += 1
            # Add the episode rewards to the volley
            volley_total_rewards[episode] = episode_total_reward
            volley_average_rewards[episode] = episode_average_reward
        # Return the average of the total and of the averages over the episodes
        return numpy.average(volley_total_rewards), numpy.average(volley_average_rewards)

    def _inference(self,
                   logger: logging.Logger,
                   episodes: int, episode_length: int,
                   session,
                   render: bool = False):
        """
        Execute a volley of inference of the agent on the environment.
        How the inference is done depends on the experiment implementation.

        Note: to call an inference step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in inference mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during inference
        :return: the float average of the score obtained in the played episodes
        """
        # Define arrays of volley rewards
        volley_average_rewards: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        volley_total_rewards: numpy.ndarray = numpy.zeros(episodes, dtype=float)
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
                self._agent.complete_step_inference(logger, session,
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
            episode_total_reward: float = numpy.sum(numpy.array(episode_rewards))
            episode_average_reward: float = numpy.average(numpy.array(episode_rewards))
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_inference(logger, session,
                                                   episode_rewards[-1], episode_total_reward,
                                                   episode, episodes)
            # Add the episode rewards to the volley
            volley_total_rewards[episode] = episode_total_reward
            volley_average_rewards[episode] = episode_average_reward
        # Return the average of the total and of the averages over the episodes
        return numpy.average(volley_total_rewards), numpy.average(volley_average_rewards)

    def conduct(self,
                training_episodes_per_volley: int, validation_episodes_per_volley: int,
                training_episodes_max: int, episode_length_max: int,
                test_episodes_per_cycle: int, test_cycles: int,
                logger: logging.Logger,
                render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False):
        """
        Conduct the experiment of the given number of training (up to the given maximum) and validation episodes
        per volley, with the associated given number of test episodes per cycle and the given number of test cycles after
        the validation has passed.

        A volley of training is always considered to stop the training exactly when the _model performs well enough
        to validate, by checking the appropriate condition. When validation is successful, appropriate condition is also
        checked on the test cycles to decide if the experiment is successful or not.

        Conducting an experiment will generate a tensorflow session for that experiment. It is required that the experiment
        is ready to be conducted by checking the result of the setup method before running this one.

        :param training_episodes_per_volley: the number of training episodes per volley before trying to validate the _model
        :param validation_episodes_per_volley: the number of validation episodes per volley after the training in such interval
        :param training_episodes_max: the maximum number of training episodes allowed at all
        :param test_episodes_per_cycle: the number of episodes to play for each test cycle after validation has passed
        :param test_cycles: the number of test cycles to execute after validation has passed
        :param logger: the logger used to print the experiment information, warnings and errors
        :param render_during_training: boolean flag to render the environment during training
        :param render_during_validation: boolean flag to render the environment during validation
        :param render_during_test: boolean flag to render the environment during test
        :return: the average of averages score over all the test cycles, the best average score among all the test cycles and the training episodes required to validate the _model
        """
        logger.info("Conducting experiment " + self._current_id + "...")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_gpu_config) as session:
            # Initialize the environment and the agent
            self._environment.initialize(logger, session)
            self._agent.initialize(logger, session)
            # Execute pre-training if the agent requires pre-training
            if self._agent.warmup_episodes > 0:
                logger.info("Warming-up for " + str(self._agent.warmup_episodes) + " episodes...")
                self._warmup(logger,
                             self._agent.warmup_episodes,
                             episode_length_max,
                             session)
            # Execute training until max training episodes number is reached or the validation score is above the threshold
            while self._trained_episodes < training_episodes_max:
                # Run train for training episodes per volley and get the average score
                logger.info("Training for " + str(training_episodes_per_volley) + " episodes...")
                training_total_reward, training_average_reward = self._train(logger,
                                                                             training_episodes_per_volley,
                                                                             episode_length_max,
                                                                             training_episodes_max,
                                                                             session,
                                                                             render_during_training)
                # Print training results
                logger.info("Training of " + str(training_episodes_per_volley) + " finished with following result:")
                logger.info("Average total reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(training_total_reward))
                logger.info("Average scaled reward over " + str(training_episodes_per_volley) + " training episodes after " + str(self._trained_episodes) + " total training episodes: " + str(training_average_reward))
                logger.info("Total training steps: " + str(self._trained_steps))
                # Save the agent internal _model at the current step
                logger.info("Saving the _model...")
                self._agent_saver.save(session, self._metagraph_path + "/" + self._current_id)
                # Run inference for validation episodes per volley and get the average score
                logger.info("Validating for " + str(validation_episodes_per_volley) + " episodes...")
                validation_total_reward, validation_average_reward = self._inference(logger,
                                                                                     validation_episodes_per_volley,
                                                                                     episode_length_max,
                                                                                     session,
                                                                                     render_during_validation)
                # Print validation results
                logger.info("Training of " + str(validation_episodes_per_volley) + " finished with following result:")
                logger.info("Average total reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(validation_total_reward))
                logger.info("Average scaled reward over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._trained_episodes) + " total training episodes: " + str(validation_average_reward))
                # Check for validation
                if self._is_validated(validation_total_reward, validation_average_reward, training_total_reward, training_average_reward):
                    logger.info("Validation of the _model is successful")
                    break
            # Test the _model and get all cycles total and average rewards
            test_total_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            test_average_rewards: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            for test in range(test_cycles):
                # Run inference for test episodes per cycles and get the average score
                logger.info("Testing for " + str(test_episodes_per_cycle) + " episodes...")
                cycle_total_reward, cycle_average_reward = self._inference(logger,
                                                                           test_episodes_per_cycle,
                                                                           episode_length_max,
                                                                           session,
                                                                           render_during_test)
                # Print validation results
                logger.info("Testing of " + str(test_episodes_per_cycle) + " finished with following result:")
                logger.info("Average total reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(cycle_total_reward))
                logger.info("Average scaled reward over " + str(test_episodes_per_cycle) + " test episodes: " + str(cycle_average_reward))
                # Save the rewards
                test_total_rewards[test] = cycle_total_reward
                test_average_rewards[test] = cycle_average_reward
            # Get the average and the best total and average rewards over all cycles
            average_test_total_reward: float = numpy.average(test_total_rewards)
            max_test_total_reward: float = numpy.max(test_total_rewards)
            average_test_average_reward: float = numpy.average(test_average_rewards)
            max_test_average_reward: float = numpy.max(test_average_rewards)
            # Print final results and outcome of the experiment
            logger.info("Average test total reward is " + str(average_test_total_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Average test scaled reward is " + str(average_test_average_reward) + " with " + str(self._trained_episodes) + " training episodes")
            logger.info("Best test total reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_total_reward))
            logger.info("Best test scaled reward over " + str(test_episodes_per_cycle) + " and " + str(test_cycles) + " cycles is: " + str(max_test_average_reward))
            # Check if the experiment is successful
            if self._is_successful(average_test_total_reward, average_test_average_reward, max_test_total_reward, max_test_average_reward):
                logger.info("The experiment is successful")
            else:
                logger.info("The experiment is not successful")
            return average_test_total_reward, max_test_total_reward, average_test_average_reward, max_test_average_reward, self._trained_episodes

    def _is_validated(self,
                      average_validation_total_reward: float, average_validation_average_reward: float,
                      average_training_total_reward: float, average_training_average_reward: float) -> bool:
        """
        Check if the experiment is to be considered validated using the given parameters.

        :param total_training_episodes: the total number of training episodes up to the validation time
        :param total_training_steps: the total number of training steps up to the validation time
        :param average_validation_score: the average validation score in the last volley
        :param average_training_score: the average training score in the last volley
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _is_successful(self,
                       average_test_total_reward: float, average_test_average_reward: float,
                       max_test_total_reward: float, max_test_average_reward: float) -> bool:
        """
        Check if the experiment is to be considered successful using the given parameters.

        :param total_training_episodes: the total number of training episodes up to the test time
        :param total_training_steps: the total number of training steps up to the test time
        :param average_test_score: the average of averages of testing scores among all test cycles
        :param best_test_score: the best of averages of testing scores among all test cycles
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()
