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

import enum
import logging
import numpy

# Define space types for observation and action spaces definition in the environment


class SpaceType(enum.Enum):
    """
    Space types for environment definition.

    If discrete, each value is defined by an integer.
    If continuous, each value is defined by a vector.
    """
    discrete = 1
    continuous = 2


class Environment:
    """
    Base environment abstract class. It is not ready to be used before setup.

    Actions, states, rewards and episode-done flags are all defined in a vectorized way to allow batch parallelization.
    The parallel amount if set during setup.

    You should always define your own environment, the base class cannot be used as-is.
    """

    def __init__(self,
                 name: str):
        # Make sure parameters are valid
        assert(name != "")
        # Define internal attributes
        self._name: str = name
        # Define empty attributes
        self._parallel: int or None = None

    def setup(self,
              logger: logging.Logger,
              parallel: int = 1) -> bool:
        """
        Setup the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param parallel: the amount of parallel episodes run by the environment
        :return: True if the setup is successful, false otherwise
        """
        # Make sure parameters are valid
        assert(parallel > 0)
        # Set parallel attribute
        self._parallel = parallel
        # Generate the environment and check if generation is successful
        return self._generate(logger)

    def _generate(self,
                  logger: logging.Logger) -> bool:
        """
        Generate the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :return: True if the environment generation is successful, false otherwise
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def close(self,
              logger: logging.Logger,
              session):
        """
        Close the environment (e.g. the window of a rendered OpenAI environment). Usually is not required.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def reset(self,
              logger: logging.Logger,
              session) -> numpy.ndarray:
        """
        Reset the environment to its default state. Requires the environment to be initialized.
        This is used when an episode is done to start a new episode.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: the current state of the environment on its default state, wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def step(self,
             logger: logging.Logger,
             session,
             action: numpy.ndarray) -> ():
        """
        Execute a step on the environment with the given action.
        Requires the environment to be initialized.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :param action: the action executed on the environment, wrapped in a numpy array
        :return: the next state, the reward obtained and a boolean flag indicating whether or not the episode is done, all wrapped in numpy arrays wrapped in a tuple
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def render(self,
               logger: logging.Logger,
               session):
        """
        Render the environment.
        Note: to render with text, it's advised to use print statement instead of the logger.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def sample_action(self,
                      logger: logging.Logger,
                      session) -> numpy.ndarray:
        """
        Sample a random action from the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: a random action of the environment wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def possible_actions(self,
                         logger: logging.Logger,
                         session) -> []:
        """
        Get a list of all actions' indexes possible at the current states of the environment if the action space
        is discrete.
        Get a list containing the lower and the upper bounds at the current states of the environment, each
        wrapped in numpy array with the shape of the  action space.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: a list of indices containing the possible actions or a list of upper and lower bounds arrays
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        The name of the environment.
        """
        return self._name

    @property
    def parallel(self) -> int:
        """
        The amount of parallelization of the environment.
        """
        return self._parallel

    @property
    def state_space_type(self) -> SpaceType:
        """
        The type of the state space of the environment.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def state_space_shape(self) -> ():
        """
        The shape of the state space of the environment, wrapped in a tuple.
        Note: it may differ from the agent's observation space shape.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def action_space_type(self) -> SpaceType:
        """
        The type of the action space of the environment.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def action_space_shape(self) -> ():
        """
        The shape of the action space of the environment, wrapped in a tuple.
        Note: it may differ from the agent's action space shape.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()
