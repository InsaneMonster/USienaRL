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
import numpy

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
        # Define internal attributes
        self._name: str = name
        # Define empty attributes
        self._saver = None
        self._scope: str or None = None
        self._summary_writer = None
        self._parallel: int or None = None
        self._summary_path: str or None = None
        self._save_path: str or None = None
        self._saves_to_keep: int or None = None
        self._observation_space_type: SpaceType or None = None
        self._observation_space_shape = None
        self._agent_action_space_type: SpaceType or None = None
        self._agent_action_space_shape = None
        self._save_counter: int or None = None

    def setup(self,
              logger: logging.Logger,
              scope: str,
              parallel: int,
              observation_space_type: SpaceType, observation_space_shape: (),
              agent_action_space_type: SpaceType, agent_action_space_shape: (),
              summary_path: str = None, save_path: str = None, saves_to_keep: int = 0) -> bool:
        """
        Setup the agent, preparing all its components for execution.

        :param logger: the logger used to print the agent information, warnings and errors
        :param scope: the experiment scope encompassing the agent scope
        :param parallel: the amount of parallel episodes run by the experiment, must be greater than zero
        :param observation_space_type: the space type of the observation space
        :param observation_space_shape: the shape of the observation space, as a tuple
        :param agent_action_space_type: the space type of the agent action space
        :param agent_action_space_shape: the shape of the agent action space, as a tuple
        :param summary_path: the optional path of the summary writer of the agent
        :param save_path: the optional path where to save metagraphs checkpoints of the agent's model
        :param saves_to_keep: the optional number of checkpoint saves to keep, it does nothing if there is no save path
        :return: True if setup is successful, False otherwise
        """
        # Make sure parameters are correct
        assert(parallel > 0 and saves_to_keep >= 0)
        logger.info("Setup of agent " + self._name + " with scope " + scope + "...")
        # Reset agent attributes
        self._scope = scope
        self._parallel = parallel
        self._summary_path = summary_path
        self._save_path = save_path
        self._saves_to_keep = saves_to_keep
        self._observation_space_type: SpaceType = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._agent_action_space_type: SpaceType = agent_action_space_type
        self._agent_action_space_shape = agent_action_space_shape
        # Try to generate the agent inner model
        if not self._generate(logger,
                              observation_space_type, observation_space_shape,
                              agent_action_space_type, agent_action_space_shape):
            return False
        # Define the summary writer if required
        if summary_path is not None:
            self._summary_writer = tensorflow.summary.FileWriter(summary_path, graph=tensorflow.get_default_graph())
            logger.info("A Tensorboard summary for the agent will be updated during training of its internal model")
            logger.info("Tensorboard summary path: " + summary_path)
        # Define the saver if required
        if self._save_path is not None and self._save_path and self._saves_to_keep > 0:
            self._saver = tensorflow.train.Saver(self.saved_variables, max_to_keep=self._saves_to_keep)
            if self._saves_to_keep > 1:
                logger.info("Agent model metagraph will be saved after each training/validation pair. A set of " + str(self._saves_to_keep) + " models will be stored.")
            else:
                logger.info("Agent model metagraph will be saved after each training/validation pair")
            logger.info("Agent model metagraphs are saved at " + self._save_path)
            self._save_counter: int = 0
        # Validate setup
        return True

    def restore(self,
                logger: logging.Logger,
                session,
                path: str) -> bool:
        """
        Restore the agent's model from the checkpoint at the given path.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param path: the path from which to restore, it is required
        :return: True if restore is successful, false otherwise
        """
        # Make sure parameters are correct
        assert(path is not None and path)
        # Get checkpoint from path
        checkpoint = tensorflow.train.get_checkpoint_state(path)
        # If no saver is defined, define one to restore from checkpoint
        if self._saver is None:
            self._saver = tensorflow.train.Saver(self.saved_variables)
        # If checkpoint exists restore from checkpoint
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(session, tensorflow.train.latest_checkpoint(path))
            logger.info("Model graph stored at " + path + " loaded successfully!")
            return True
        logger.error("Checkpoint path specified is wrong: no model can be accessed at " + path)
        return False

    def save(self,
             logger: logging.Logger,
             session):
        """
        Save the agent's model metagraph. It does nothing if a saver is not defined.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Check if the saver exists or something has to be saved
        if self._saver is None or self._save_path is None or not self._save_path or self._saves_to_keep <= 0:
            return
        logger.info("Saving the agent " + self._name + " metagraph at path " + self._save_path + "...")
        self._saver.save(session, self._save_path, self._save_counter)
        self._save_counter += 1
        logger.info("Agent " + self._name + " metagraph saved successfully")

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape: (),
                  agent_action_space_type: SpaceType, agent_action_space_shape: ()) -> bool:
        """
        Generate the agent's model. Used to generate all custom components of the agent.
        It is always called during setup.

        :param logger: the logger used to print the agent information, warnings and errors
        :param observation_space_type: the space type of the observation space
        :param observation_space_shape: the shape of the observation space, as a tuple
        :param agent_action_space_type: the space type of the agent action space
        :param agent_action_space_shape: the shape of the agent action space, as a tuple
        :return: True if setup is successful, False otherwise
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the agent before acting in the environment.
        The environment at this stage is already initialized.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current: numpy.ndarray,
                   warmup_step: int, warmup_episode: int) -> numpy.ndarray:
        """
        Take an action given the current agent observation in warmup mode.
        Usually it uses a random policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array
        :param warmup_step: the current absolute warm-up step of the experiment the agent is warming-up into
        :param warmup_episode: the current absolute warm-up episode of the experiment the agent is warming-up into
        :return: the action decided by the agent wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int) -> numpy.ndarray:
        """
        Take an action given the current agent observation in train mode.
        Usually it uses an exploring policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array
        :param train_step: the current absolute train step of the experiment the agent is training into
        :param train_episode: the current absolute train episode of the experiment the agent is training into
        :return: the action decided by the agent wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current: numpy.ndarray,
                      inference_step: int, inference_episode: int) -> numpy.ndarray:
        """
        Take an action given the current agent observation in validation/test mode.
        Usually it uses the best possible policy.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array
        :param inference_step: the current absolute validation/test step of the experiment the agent is executing into
        :param inference_episode: the current absolute validation/test episode of the experiment the agent is executing into
        :return: the action decided by the agent wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current: numpy.ndarray,
                             agent_action: numpy.ndarray,
                             reward: numpy.ndarray,
                             episode_done: numpy.ndarray,
                             agent_observation_next: numpy.ndarray,
                             warmup_step: int, warmup_episode: int):
        """
        Complete a warm-up step.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation wrapped in a numpy array
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between wrapped in a numpy array
        :param episode_done: the episode done flag raised by the environment wrapped in a numpy array
        :param agent_observation_next: the next observation of the agent wrapped in a numpy array
        :param warmup_step: the current absolute warm-up step of the experiment the agent is warming-up into
        :param warmup_episode: the current absolute warm-up episode of the experiment the agent is warming-up into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current: numpy.ndarray,
                            agent_action: numpy.ndarray,
                            reward: numpy.ndarray,
                            episode_done: numpy.ndarray,
                            agent_observation_next: numpy.ndarray,
                            train_step: int, train_episode: int):
        """
        Complete a train step.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation wrapped in a numpy array
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between wrapped in a numpy array
        :param episode_done: the episode done flag raised by the environment wrapped in a numpy array
        :param agent_observation_next: the next observation of the agent wrapped in a numpy array
        :param train_step: the current absolute train step of the experiment the agent is training into
        :param train_episode: the current absolute train episode of the experiment the agent is training into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current: numpy.ndarray,
                                agent_action: numpy.ndarray,
                                reward: numpy.ndarray,
                                episode_done: numpy.ndarray,
                                agent_observation_next: numpy.ndarray,
                                inference_step: int, inference_episode: int):
        """
        Complete an validation/test step..

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param agent_observation_current: the current observation of the agent
        :param agent_action: the action taken in the environment as seen by the agent leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param episode_done: the episode done flag raised by the environment wrapped in a numpy array
        :param agent_observation_next: the next observation of the agent
        :param inference_step: the current absolute validation/test step of the experiment the agent is executing into
        :param inference_episode: the current absolute validation/test episode of the experiment the agent is executing into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: numpy.ndarray,
                                episode_total_reward: numpy.ndarray,
                                warmup_step: int, warmup_episode: int):
        """
        Finish a warm-up episode.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment  wrapped in a numpy array
        :param last_step_reward: the reward obtained in the last step in the passed episode wrapped in a numpy array  wrapped in a numpy array
        :param episode_total_reward: the reward obtained in the passed episode wrapped in a numpy array
        :param warmup_step: the current absolute warm-up step of the experiment the agent is warming-up into
        :param warmup_episode: the current absolute warm-up episode of the experiment the agent is warming-up into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        """
        Finish a train episode.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param last_step_reward: the reward obtained in the last step in the passed episode  wrapped in a numpy array
        :param episode_total_reward: the reward obtained in the passed episode  wrapped in a numpy array
        :param train_step: the current absolute train step of the experiment the agent is training into
        :param train_episode: the current absolute train episode of the experiment the agent is training into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: numpy.ndarray,
                                   episode_total_reward: numpy.ndarray,
                                   inference_step: int, inference_episode: int):
        """
        Finish an validation/test episode.

        :param logger: the logger used to print the agent information, warnings and errors
        :param session: the session of tensorflow currently running
        :param interface: the interface between the agent and the environment
        :param last_step_reward: the reward obtained in the last step in the passed episode wrapped in a numpy array
        :param episode_total_reward: the reward obtained in the passed episode wrapped in a numpy array
        :param inference_step: the current absolute validation/test step of the experiment the agent is executing into
        :param inference_episode: the current absolute validation/test episode of the experiment the agent is executing into
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        The name of the agent.
        """
        return self._name

    @property
    def scope(self) -> str or None:
        """
        The scope of the agent.
        It is None if agent is not setup.
        """
        return self._scope

    @property
    def parallel(self) -> int or None:
        """
        The number of parallel episodes run by the agent.
        It is None if agent is not setup.
        """
        return self._parallel

    @property
    def observation_space_type(self) -> SpaceType or None:
        """
        The type of the observation space of the agent.
        It is None if agent is not setup.
        """
        return self._observation_space_type

    @property
    def observation_space_shape(self) -> () or None:
        """
        The shape of the observation space of the agent.
        Note: it may differ from the environment's state space shape.
        It is None if agent is not setup.
        """
        return self._observation_space_shape

    @property
    def action_space_type(self) -> SpaceType or None:
        """
        The type of the action space of the agent.
        It is None if agent is not setup.
        """
        return self._agent_action_space_type

    @property
    def action_space_shape(self) -> () or None:
        """
        The shape of the action space of the agent.
        Note: it may differ from the environment's action space shape.
        It is None if agent is not setup.
        """
        return self._agent_action_space_shape

    @property
    def summary_path(self) -> str or None:
        """
        The path of the Tensorboard summaries saved by the agent.
        It is None if agent is not setup.
        """
        return self._summary_path

    @property
    def save_path(self) -> str or None:
        """
        The path of the checkpoint metagraph saves of the agent.
        It is None if agent is not setup.
        """
        return self._save_path

    @property
    def saved_variables(self):
        """
        The trainable variables of the agent (usually of its model).
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def warmup_steps(self) -> int:
        """
        The integer number of warm-up steps required by the agent.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()
