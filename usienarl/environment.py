# Import packages

import enum
import logging

# Define space types for observation and action spaces definition in the environment


class SpaceType(enum.Enum):
    """
    Space types for environment definition.

    If discrete, each value is one-hot encoded.
    If continuous, each value can have infinite sub values.

    Usually, the discrete space has greater size than the continuous one (it requires more variables one-hot encoded
    to properly define each state, while in continuous it is defined by all the values in the state itself).

    Except for the properties and the setup method, all the methods can be used with tensorflow (which means, the
    environment can run on tensorflow). Usually, environments doesn't require tensorflow, in this case the child class
    will have a default value of None for the session in each method.
    """
    discrete = 1
    continuous = 2


class Environment:
    """
    Base class for each environment. It should not be used by itself, and should be extended instead.

    The environment is not created until the setup method is called.

    Attributes:
        - _name: the string _name of the environment
    """

    def __init__(self,
                 name: str):
        # Define the _name attribute
        self.name: str = name

    def setup(self,
              logger: logging.Logger) -> bool:
        """
        Setup the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :return a boolean flag True if the setup is successful, false otherwise
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the environment. Requires the environment to be setup.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def close(self,
              logger: logging.Logger,
              session):
        """
        Close the environment (e.g. the window of a rendered OpenAI environment).

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def reset(self,
              logger: logging.Logger,
              session):
        """
        Reset the environment to its default state. Requires the environment to be initialized.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: the current state of the environment on its default state
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def step(self,
             logger: logging.Logger,
             action,
             session):
        """
        Execute a step on the environment with the given action. It returns back all the data describing the transition.
        Requires the environment to be initialized.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :param action: the action executed on the environment, causing the step and the state change
        :return: the next state, the reward obtained and a boolean flag indicating whether or not the episode is completed
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

    def get_random_action(self,
                          logger: logging.Logger,
                          session):
        """
        Get a random action from the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: a random action of the environment in the shape supported the environment (discrete: an int, continuous: a tuple)
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def state_space_type(self):
        """
        Get the type (according to its SpaceType enum definition) of the state space of the environment.

        :return: the SpaceType describing the state space type of the environment
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def state_space_shape(self):
        """
        Get the shape of the state space of the environment.

        :return: the shape of the state space of the environment in the form of a tuple
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def action_space_type(self):
        """
        Get the type (according to its SpaceType enum definition) of the action space of the environment.

        :return: the SpaceType describing the action space type of the environment
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def action_space_shape(self):
        """
        Get the shape of the action space of the environment.

        :return: the shape of the action space of the environment in the form of a tuple
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()


