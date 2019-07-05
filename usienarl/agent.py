#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# USienaRL is licensed under a MIT License.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/MIT>.

# Import packages

import numpy
import logging
import tensorflow

# Import required src

from usienarl import Environment, Interface


class Agent:
    """
    TODO: Summary

    """

    def __init__(self,
                 name: str):
        # Define agent name attribute
        self.name: str = name
        # Define environment attribute
        self.environment: Environment = None
        # Define environment-agent interface attribute
        self.interface: Interface = None
        # Define summary writer attributes
        self.summary_writer = None
        self._summary_path: str = None
        # Define train steps counter attribute
        self.train_steps_counter: int = 0

    def setup(self,
              logger: logging.Logger,
              experiment_scope: str, summary_path: str,
              environment: Environment, interface: Interface = None) -> bool:
        """
        Setup the agent for pre-training, training and inference.
        It is called before the tensorflow session generation.
        Note: this should generate the model and other components, if any.

        :param logger: the logger used to print the agent information, warnings and errors
        :param experiment_scope: the experiment scope encompassing the agent scope, if any
        :param summary_path:
        :param environment: the environment this agent is setup for
        :param interface: the interface used to interact with the environment. If not provided, a default pass-trough interface is used
        :return a boolean flag True if setup is successful, False otherwise
        """
        logger.info("Setup of agent " + self.name + "...")
        # Set the environment to the given one
        self.environment = environment
        # Reset the train steps counter
        self.train_steps_counter = 0
        # Reset the summary writer
        self._summary_path = summary_path
        self.summary_writer = tensorflow.summary.FileWriter(self._summary_path, graph=tensorflow.get_default_graph())
        logger.info("A Tensorboard summary for the agent be updated during training of its internal model")
        logger.info("Tensorboard summary path: " + self._summary_path)
        # Set the interface (using a default pass-through interface if not given a specific one)
        self.interface = interface if interface is not None else Interface(environment)
        # Try to generate the agent inner model
        return self._generate(logger, experiment_scope)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the agent before acting in the environment.
        It is called right after tensorflow session generation.
        Note: this should initialize the model and other components, if any.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def act_pre_train(self,
                      logger: logging.Logger,
                      session,
                      environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in pre-train mode (using pre-training policy,
        usually a random one).

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_current)
        # Decide which action to take by running the agent decision making process in pre-train mode
        agent_action = self._decide_pre_train(logger, session, observation_current)
        # Return the respective environment action
        return self.interface.agent_action_to_environment_action(logger, session, agent_action)

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in train mode (using train policy, usually
        an exploration heavy one).

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Increase the training steps counter of the agent (anytime it acts in the environment, the steps is increased by one)
        self.train_steps_counter += 1
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_current)
        # Decide which action to take by running the agent decision making process in train mode
        agent_action = self._decide_train(logger, session, observation_current)
        # Return the respective environment action
        return self.interface.agent_action_to_environment_action(logger, session, agent_action)

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in inference mode (using inference policy, usually
        the optimal one).

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_current)
        # Decide which action to take by running the agent decision making process in inference mode
        agent_action = self._decide_inference(logger, session, observation_current)
        # Return the respective environment action
        return self.interface.agent_action_to_environment_action(logger, session, agent_action)

    def store_pre_train(self,
                        logger: logging.Logger,
                        session,
                        environment_state_current: numpy.ndarray,
                        environment_action,
                        reward: float,
                        environment_state_next: numpy.ndarray,
                        episode_done: bool):
        """
        Let the agent store a step in the environment with the given parameters when executing pre-train.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param environment_action: the action taken in the environment leading from the current state to the next state
        :param reward: the reward obtained by the combination of current state, next state and action in-between
        :param environment_state_next: the next state of the environment wrapped in a numpy array (ndarray)
        :param episode_done: the completion flag of the episode
        """
        # Translate the environment states in agent observations
        observation_current: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_current)
        observation_next: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_next)
        # Translate the environment environment_action in an agent environment_action
        agent_action = self.interface.environment_action_to_agent_action(logger, session, environment_action)
        # Attempt to save the step using the agent observations
        self._save_step_pre_train(logger, session, observation_current, agent_action, reward, observation_next, episode_done)

    def store_train(self,
                    logger: logging.Logger,
                    session,
                    environment_state_current: numpy.ndarray,
                    environment_action,
                    reward: float,
                    environment_state_next: numpy.ndarray,
                    episode_done: bool):
        """
        Let the agent store a step in the environment with the given parameters when executing training.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param environment_action: the action taken in the environment leading from the current state to the next state
        :param reward: the reward obtained by the combination of current state, next state and action in-between
        :param environment_state_next: the next state of the environment wrapped in a numpy array (ndarray)
        :param episode_done: the completion flag of the episode
        """
        # Translate the environment states in agent observations
        observation_current: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_current)
        observation_next: numpy.ndarray = self.interface.environment_state_to_observation(logger, session, environment_state_next)
        # Translate the environment environment_action in an agent environment_action
        agent_action = self.interface.environment_action_to_agent_action(logger, session, environment_action)
        # Attempt to save the step using the agent observations
        self._save_step_train(logger, session, observation_current, agent_action, reward, observation_next, episode_done)

    def update(self,
               logger: logging.Logger,
               session,
               current_episode: int, total_episodes: int, current_step: int):
        """
        Let the agent update policy, usually updating the model behind it. The given parameters are usually used
        to decide if it's time to update or not.
        Note: usually at least the least step is required to perform an update. Each step is sent to the agent in the
        save step private method. Other data could be retrieved from the model directly in both decision and previous
        update phases.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param current_episode: the current episode of the experiment
        :param total_episodes: the total number of total_episodes in the experiment
        :param current_step: the current step number in the experiment
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _generate(self,
                  logger: logging.Logger,
                  experiment_scope: str) -> bool:
        """
        Generate the agent internal model.
        It is always called during setup.

        :param logger: the logger used to print the agent information, warnings and errors
        :param experiment_scope: the experiment scope encompassing the agent scope, if any
        :return a boolean flag True if setup is successful, False otherwise
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_pre_train(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in pre-train mode.
        Train mode usually implies a specific randomized policy, but it's not mandatory.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_train(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in train mode.
        Train mode usually implies a specific policy to better explore the environment, but it's not mandatory and
        could be the same of the inference decision.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_inference(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in inference mode.
        Inference mode usually implies the use of the main optimal policy to better exploit the environment, but it's
        not mandatory and could be the same of the train decision.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _save_step_pre_train(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current: numpy.ndarray,
                             agent_action,
                             reward: float,
                             agent_observation_next: numpy.ndarray,
                             episode_done: bool):
        """
        Let the agent save the given step in pre-train mode. It could employ a buffer or nothing at all but usually at
        least one step is required to perform an update.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent wrapped in a numpy array (ndarray)
        :param episode_done: the completion flag of the episode
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _save_step_train(self,
                         logger: logging.Logger,
                         session,
                         agent_observation_current: numpy.ndarray,
                         agent_action,
                         reward: float,
                         agent_observation_next: numpy.ndarray,
                         episode_done: bool):
        """
        Let the agent save the given step in train mode. It could employ a buffer or nothing at all but usually at
        least one step is required to perform an update.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent wrapped in a numpy array (ndarray)
        :param episode_done: the completion flag of the episode
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def get_trainable_variables(self,
                                experiment_scope: str):
        """
        Get the trainable variables of the agent (usually of the models of the agents).

        :param experiment_scope: the experiment scope encompassing the agent scope, if any
        :return: the trainable variables defined by this agent.
        """
        # Empty method, it should be implemented on a child class basis
        pass

    @property
    def pre_train_episodes(self) -> int:
        """
        Return the integer number of pre-training episodes required by the agent. By default it returns 0.

        :return: the integer number of pre-training episodes required by the agent internal model
        """
        # Empty property with default return value, it should be implemented on a child class basis if required
        return 0
