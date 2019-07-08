#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# USienaRL is licensed under a MIT License.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/MIT>.

# Import packages

import logging
import tensorflow

# Import required src

from usienarl import SpaceType


class Agent:
    """
    Base abstract agent class. Use this class as base for any agent.
    To define an agent, implement all of the abstract methods in this class. Check each method docstring for further
    details.

    Usually, an agent is built for at least a _model class, often it also employs one or more classes of
    exploration policies. The final implementation, however, does not require nor limits what can be used and
    it all really depends in what is the end goal of the agent.

    Note: the agent always interact with an environment via an interface which translates environment actions and states
    to agent actions and observations. The agent always acts on the latter.

    In order to fully generate an agent, the setup method needs to be called. It is always called by the experiment.
    Within this method the agents generates the _model and connects to the environment by the interface.

    Each agent consist mainly of three different phases:
        - pre-training, if required set implement the pre_train_episodes method and return a value greater than zero
        - training, in which the inner _model of the agent is updated and usually an exploration policy is employed
        - inference, where the _model runs in full exploitation, usually used for validating and testing an agent

    The pre-train phase comprehends:
        - decide_pre_train: the agent is invoked to decide which action to execute in pre-training mode given the
        observation. Usually, this is a random action
        - save_step_pre_train: the agent is invoked to save the step in the environment after the decided action is
        taken, using all the environment data in the relative step. This can also be used to update some parameters
        at the end of the episode by checking the relative flag.

    The train phase comprehends:
        - decide_train: the agent is invoked to decide which action to execute in training mode given the observation.
        Usually this is an action involving an exploration policy.
        - save_step_train: the agent is invoked to save the step in the environment after the decided action is taken,
        using all the environment data in the relative step. This can also be used to update some parameters
        at the end of the episode by checking the relative flag.
        - update: the agent is invoked to update its inner _model. When this is called depends on the experiment
        implementation. Further customization is possible with the given parameters relative to the training process.

    The inference phase comprehends:
        - decide_inference: the agent is invoked to decide which action to execute in inference mode given the
        observation. Usually this is the time the agent tries to predict the best possible action according to its
        own inner _model.

    Attributes:
        - _name: _name of the agent, used as further _scope with respect to the experiment
        - environment: the environment in which the agent is called to act, set by the experiment during setup
        - summary_writer: the tensorboard _summary writer for the inner _model of the agent
        - train_steps_counter: the number of training steps in total executed by the agent in the current experiment
    """

    def __init__(self,
                 name: str):
        # Define agent attributes
        self._name: str = name
        # Define empty agent attributes
        self._scope: str = None
        self._summary_writer = None

    def setup(self,
              logger: logging.Logger,
              observation_space_type: SpaceType, observation_space_shape,
              agent_action_space_type: SpaceType, agent_action_space_shape,
              scope: str, summary_path: str) -> bool:
        """
        Setup the agent for pre-training, training and inference.
        It is called before the tensorflow session generation.
        Note: this should generate the _model and other components, if any.

        :param logger: the logger used to print the agent information, warnings and errors
        :param scope: the experiment _scope encompassing the agent _scope, if any
        :param summary_path: the path of the _summary writer of the agent
        :return a boolean flag True if setup is successful, False otherwise
        """
        logger.info("Setup of agent " + self._name + " with _scope " + scope + "...")
        # Reset agent attributes
        self._scope = scope
        # Reset the _summary writer
        self._summary_writer = tensorflow.summary.FileWriter(summary_path, graph=tensorflow.get_default_graph())
        logger.info("A Tensorboard _summary for the agent will be updated during training of its internal _model")
        logger.info("Tensorboard _summary path: " + summary_path)
        # Set the interface (using a default pass-through interface if not given a specific one)
        # self.interface = interface if interface is not None else Interface(environment)
        # Try to generate the agent inner _model
        return self._generate(logger, observation_space_type, observation_space_shape, agent_action_space_type, agent_action_space_shape)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  action_space_type: SpaceType, action_space_shape) -> bool:
        """
        Generate the agent internal _model. Used to generate all custom components of the agent.
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
        It is called right after tensorflow session generation.
        Note: this should initialize the _model and other components, if any. It can also reset all the agent internal
        attributes in order to prepare for the experiment.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   agent_observation_current):
        """
        Take an action given the current agent observation in _warmup mode. Usually it uses a random policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  agent_observation_current):
        """
        Take an action given the current agent observation in _warmup mode. Usually it uses an exploring policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current):
        """
        Take an action given the current agent observation in _warmup mode. Usually it uses the best possible policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :return: the action the agent decided to take
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current,
                             agent_action,
                             reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_episode_volley: int):
        """
        Finish a _warmup step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        """
        Finish a train step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                agent_observation_current,
                                agent_action,
                                reward: float,
                                agent_observation_next,
                                inference_step_current: int,
                                inference_episode_current: int,
                                inference_episode_volley: int):
        """
        Finish an inference step with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_episode_volley: int):
        """
        Finish a _warmup episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param episode_reward: the reward obtained in the passed episode
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        """
        Finish a train episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param episode_reward: the reward obtained in the passed episode
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   last_step_reward: float,
                                   episode_total_reward: float,
                                   inference_episode_current: int,
                                   inference_episode_volley: int):
        """
        Finish an inference episode with the given values.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param episode_reward: the sum of the rewards obtained in the passed episode
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def trainable_variables(self):
        """
        Get the trainable variables of the agent (usually of the internal _model of the agent).
        Is is usually searched with the _scope environment/agent.

        :return: the trainable variables defined by this agent.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def warmup_episodes(self) -> int:
        """
        Return the integer number of warm-up episodes required by the agent.

        :return: the integer number of warm-up episodes required by the agent internal _model
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

