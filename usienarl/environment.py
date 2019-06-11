# Import packages

import enum

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
        - name: the string name of the environment
    """

    def __init__(self,
                 name: str):
        # Save the name and prepare the internal environment variable
        self.name: str = name
        self._environment = None

    def setup(self):
        """
        Setup the environment.
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def initialize(self,
                   session):
        """
        Initialize the environment (should be called after setup but before the first reset.

        :param session: the session of tensorflow currently running
        """
        pass

    def close(self,
              session):
        """
        Close the environment (e.g. the window of a rendered OpenAI environment).

        :param session: the session of tensorflow currently running
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def reset(self,
              session):
        """
        Reset the environment to its default state.

        :param session: the session of tensorflow currently running
        :return: the current state of the environment on its default state
        """
        # Empty method, it should be implemented on a child class basis
        return None

    def step(self,
             action: int,
             session):
        """
        Execute a step on the environment with the given action. It returns back all the data describing the transition.

        :param session: the session of tensorflow currently running
        :param action: the action executed on the environment, causing the step and the state change
        :return: an ordered sequence of next state, reward obtained and a boolean flag indicating whether or not the episode is completed
        """
        # Empty method, it should be implemented on a child class basis
        return None

    def render(self,
               session):
        """
        Render the environment.

        :param session: the session of tensorflow currently running
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def get_random_action(self,
                          session):
        """
        Get a random action from the environment.

        :param session: the session of tensorflow currently running
        :return: a random action of the environment in the shape supported the environment (discrete: an int, continuous: a tuple)
        """
        return None

    @property
    def observation_space_type(self):
        """
        Get the type (according to its SpaceType enum definition) of the observation space of the environment.

        :return: the SpaceType describing the observation space type of the environment
        """
        # Empty property, it should be implemented on a child class basis
        return None

    @property
    def observation_space_shape(self):
        """
        Get the shape of the observation space of the environment.

        :return: the shape of the observation space of the environment in the form of a tuple
        """
        # Empty property, it should be implemented on a child class basis
        return None

    @property
    def action_space_type(self):
        """
        Get the type (according to its SpaceType enum definition) of the action space of the environment.

        :return: the SpaceType describing the action space type of the environment
        """
        # Empty property, it should be implemented on a child class basis
        return None

    @property
    def action_space_shape(self):
        """
        Get the shape of the action space of the environment.

        :return: the shape of the action space of the environment in the form of a tuple
        """
        # Empty property, it should be implemented on a child class basis
        return None


