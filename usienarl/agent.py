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

# Import require src

from usienarl import Environment, SpaceType


class Agent:
    """
    TODO: Summary

    """

    def __init__(self,
                 environment: Environment,
                 observation_space_type: SpaceType, observation_space_shape):
        # Define action space attribute
        # Note: action space type and shape are the same as the environment
        self.action_space_type: SpaceType = environment.action_space_type
        self.action_space_shape = environment.action_space_shape
        # Define observation space attributes
        # Note: they are defined by themselves since agent observation space type and shape could differ form
        # the environment state space type and shape
        self.observation_space_type: SpaceType = observation_space_type
        self.observation_space_shape = observation_space_shape

    def setup(self) -> bool:
        """
        Setup the agent for both training and inference.
        Note: this should generate the model and other components, if any.

        :return a boolean flag True if setup is successful, false otherwise
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def initialize(self,
                   session):
        """
        Initialize the agent before acting in the environment.
        Note: this should initialize the model and other components, if any.

        :param session: the session of tensorflow currently running, if any
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def act_inference(self,
                      session,
                      environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in inference mode (using inference policy, usually
        the optimal one).

        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self._observe(session, environment_state_current)
        # Decide which action to take by running the agent decision making process in inference mode
        return self._decide_inference(session, observation_current)

    def act_pre_train(self,
                      session,
                      environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in pre-train mode (using pre-training policy,
        usually a random one).

        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self._observe(session, environment_state_current)
        # Decide which action to take by running the agent decision making process in pre-train mode
        return self._decide_pre_train(session, observation_current)

    def act_train(self,
                  session,
                  environment_state_current: numpy.ndarray):
        """
        Let the agent take an action given the current environment state in train mode (using train policy, usually
        an exploration heavy one).

        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the action chosen by the agent
        """
        # Get the current agent observation on the environment given the current state
        observation_current: numpy.ndarray = self._observe(session, environment_state_current)
        # Decide which action to take by running the agent decision making process in train mode
        return self._decide_train(session, observation_current)

    def store(self,
              session,
              environment_state_current: numpy.ndarray,
              action,
              reward: float,
              environment_state_next: numpy.ndarray):
        """
        Let the agent store a step in the environment with the given parameters.

        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param action: the action taken in the environment leading from the current state to the next state
        :param reward: the reward obtained by the combination of current state, next state and action in-between
        :param environment_state_next: the next state of the environment wrapped in a numpy array (ndarray)
        """
        # Translate the environment states in agent observations
        observation_current: numpy.ndarray = self._observe(session, environment_state_current)
        observation_next: numpy.ndarray = self._observe(session, environment_state_next)
        # Attempt to save the step using the agent observations
        self._save_step(session, observation_current, action, reward, observation_next)

    def update(self,
               session,
               episode: int, episodes: int, step: int):
        """
        Let the agent update policy, usually updating the model behind it. The given parameters are usually used
        to decide if it's time to update or not.
        Note: usually at least the least step is required to perform an update. Each step is sent to the agent in the
        save step private method. Other data could be retrieved from the model directly in both decision and previous
        update phases.

        :param session: the session of tensorflow currently running, if any
        :param episode: the current episode of the experiment
        :param episodes: the total number of episodes in the experiment
        :param step: the current step number in the experiment
        :return: the tensorboard summary
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _observe(self,
                 session,
                 environment_state_current: numpy.ndarray) -> numpy.ndarray:
        """
        Let the agent translate the environment current state to its own current observation.
        Note: in its simplest form (e.g. a fully observable environment) the agent observation space can coincide with
        the environment state space.

        :param session: the session of tensorflow currently running, if any
        :param environment_state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :return: the agent current observation on the environment
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_pre_train(self,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in pre-train mode.
        Train mode usually implies a specific randomized policy, but it's not mandatory.

        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_train(self,
                      session,
                      agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in train mode.
        Train mode usually implies a specific policy to better explore the environment, but it's not mandatory and
        could be the same of the inference decision.

        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_inference(self,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Let the agent decide which action to take given the current agent observation in inference mode.
        Inference mode usually implies the use of the main optimal policy to better exploit the environment, but it's
        not mandatory and could be the same of the train decision.

        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :return: the action the agent decided to take
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _save_step(self,
                   session,
                   agent_observation_current: numpy.ndarray,
                   action,
                   reward: float,
                   agent_observation_next: numpy.ndarray):
        """
        Let the agent save the given step if required. It could employ a buffer or nothing at all but usually at least
        one step is required to perform an update.

        :param session: the session of tensorflow currently running, if any
        :param agent_observation_current: the current observation of the agent wrapped in a numpy array (ndarray)
        :param action: the action taken in the environment leading from the current observation to the next observation
        :param reward: the reward obtained by the combination of current observation, next observation and action in-between
        :param agent_observation_next: the next observation of the agent wrapped in a numpy array (ndarray)
        """
        # Empty method, it should be implemented on a child class basis
        pass

    @property
    def trainable_variables(self):
        """
        Return the trainable variables of the agent (usually of the models of the agents).

        :return: the trainable variables defined by this agent.
        """
        # Empty property, it should be implemented on a child class basis
        return None

    @property
    def require_pre_train(self):
        """
        Return a flag stating if pre-training is required or not. By default it returns False.

        :return: True if it requires pre-training, False otherwise
        """
        # Empty property with default return value, it should be implemented on a child class basis if required
        return False

