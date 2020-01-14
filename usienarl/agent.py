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

import logging
import tensorflow

# Import required src

from usienarl import Interface, SpaceType


class Agent:
    """
    Base agent abstract class.

    An agent defines who or what operates in a certain environment during a certain experiment.
    An agent can act and being updated, but the inner working (model, policies and so on) are left to be decided by the
    implementation.

    An agent needs to be generated before running, and generation is done when executing agent setup.

    Any agent can act in three different modes:
        - warmup mode, before training (not all agents required that, it depends on the agent inner model)
        - train mode, during training
        - inference mode, when exploiting

    One or more policies can be defined for each one of these modes.

    To define your own agent, implement the abstract class in a specific child class.
    """

    def __init__(self,
                 name: str):
        # Define agent attributes
        self._name: str = name
        # Define empty agent attributes
        self._scope: str = None
        self._summary_writer = None
        self._observation_space_type: SpaceType = None
        self._observation_space_shape = None
        self._agent_action_space_type: SpaceType = None
        self._agent_action_space_shape = None

    def setup(self,
              logger: logging.Logger,
              observation_space_type: SpaceType, observation_space_shape,
              agent_action_space_type: SpaceType, agent_action_space_shape,
              scope: str, summary_path: str) -> bool:
        """
        Setup the agent.
        It is called before the tensorflow session generation, if any.
        Note: this should generate the model and other components, if any.

        :param logger: the logger used to print the agent information, warnings and errors
        :param observation_space_type: the space type of the observation space
        :param observation_space_shape: the shape of the observation space
        :param agent_action_space_type: the space type of the agent action space
        :param agent_action_space_shape: the shape of the agent action space
        :param scope: the experiment scope encompassing the agent _scope, if any
        :param summary_path: the path of the summary writer of the agent
        :return a boolean flag True if setup is successful, False otherwise
        """
        logger.info("Setup of agent " + self._name + " with scope " + scope + "...")
        # Reset agent attributes
        self._scope = scope
        self._observation_space_type: SpaceType = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._agent_action_space_type: SpaceType = agent_action_space_type
        self._agent_action_space_shape = agent_action_space_shape
        # Reset the summary writer if required
        if summary_path is not None:
            self._summary_writer = tensorflow.summary.FileWriter(summary_path, graph=tensorflow.get_default_graph())
            logger.info("A Tensorboard summary for the agent will be updated during training of its internal model")
            logger.info("Tensorboard summary path: " + summary_path)
        # Try to generate the agent inner model
        return self._generate(logger, observation_space_type, observation_space_shape, agent_action_space_type, agent_action_space_shape)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  action_space_type: SpaceType, action_space_shape) -> bool:
        """
        Generate the agent internal model. Used to generate all custom components of the agent.
        It is always called during setup.

        :param logger: the logger used to print the agent information, warnings and errors
        :return a boolean flag True if setup is successful, False otherwise
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the agent before acting in the environment.
        It is called right after tensorflow session generation and after the environment is initialized.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current):
        """
        Take an action given the current agent observation in warmup mode. Usually it uses a random policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current):
        """
        Take an action given the current agent observation in train mode. Usually it uses an exploring policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current):
        """
        Take an action given the current agent observation in inference mode. Usually it uses the best possible policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current,
                             agent_action,
                             reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_steps_volley: int):
        """
        Complete a warmup step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        :param warmup_step_current: the current warmup step in the current episode
        :param warmup_episode_current: the current warmup episode
        :param warmup_steps_volley: the number of warmup steps in the volley
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        """
        Complete a train step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        :param train_step_current: the current train step in the current episode
        :param train_step_absolute: the current absolute number of train step (counting all volleys)
        :param train_episode_current: the current train episode
        :param train_episode_absolute: the current absolute number of train episode (counting all volleys)
        :param train_episode_volley: the number of train episodes in the volley
        :param train_episode_total: the total number of allowed train episodes
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current,
                                agent_action,
                                reward: float,
                                agent_observation_next,
                                inference_step_current: int,
                                inference_episode_current: int,
                                inference_episode_volley: int):
        """
        Complete an inference step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        :param inference_step_current: the current inference step in the current episode
        :param inference_episode_current: the current inference episode
        :param inference_episode_volley: the number of inference episodes in the volley
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_steps_volley: int):
        """
        Finish a _warmup episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param last_step_reward: the reward obtained in the last step in the passed episode
        :param episode_total_reward: the reward obtained in the passed episode
        :param warmup_episode_current: the current warmup episode
        :param warmup_steps_volley: the number of warmup steps in the volley
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        """
        Finish a train episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param last_step_reward: the reward obtained in the last step in the passed episode
        :param episode_total_reward: the reward obtained in the passed episode
        :param train_step_absolute: the current absolute number of train step (counting all volleys)
        :param train_episode_current: the current train episode
        :param train_episode_absolute: the current absolute number of train episode (counting all volleys)
        :param train_episode_volley: the number of train episodes in the volley
        :param train_episode_total: the total number of allowed train episodes
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: float,
                                   episode_total_reward: float,
                                   inference_episode_current: int,
                                   inference_episode_volley: int):
        """
        Finish an inference episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param interface: the interface between the agent and the environment
        :param last_step_reward: the reward obtained in the last step in the passed episode
        :param episode_total_reward: the reward obtained in the passed episode
        :param inference_episode_current: the current inference episode
        :param inference_episode_volley: the number of inference episodes in the volley
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def trainable_variables(self):
        """
        Get the trainable variables of the agent (usually of the internal model of the agent).
        Is is usually searched with the scope environment/agent.

        :return: the trainable variables defined by this agent.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def warmup_steps(self) -> int:
        """
        Return the integer number of warm-up steps required by the agent.

        :return: the integer number of warm-up steps required by the agent internal model
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

