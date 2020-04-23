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

# Import required src

from usienarl import Environment, Agent, Interface


class Volley:
    """
    Base abstract class for a volley. A volley holds data concerning the episodes/steps executed by a certain
    agent in a certain environment.
    """
    def __init__(self,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface,
                 parallel: int):
        # Make sure parameters are correct
        assert (environment is not None and agent is not None and interface is not None and parallel > 0)
        # Define internal attributes
        self._environment: Environment = environment
        self._agent: Agent = agent
        self._interface: Interface = interface
        self._parallel: int = parallel
        # Define empty attributes
        self._number: int or None = None
        self._steps: int or None = None
        self._episodes: int or None = None
        self._start_steps: int or None = None
        self._start_episodes: int or None = None

    def setup(self,
              number: int = 0,
              start_steps: int = 0, start_episodes: int = 0) -> bool:
        """
        Setup the volley, preparing all required values to run. A volley cannot be run without setup first. This also
        initialize the volley.

        :param number: the current number of the volley (usually more volleys are run during an experiment), must be greater or equal than zero
        :param start_steps: the number of steps already executed (in previous volleys), must be greater or equal than zero
        :param start_episodes: the number of episodes already executed (in previous volleys), must be greater or equal than zero
        :return: True if setup is successful, false otherwise
        """
        # Make sure parameters are correct
        assert(number >= 0 and start_steps >= 0 and start_episodes >= 0)
        # Set current steps and episodes counters to their starting value
        self._number = number
        self._start_steps = start_steps
        self._start_episodes = start_episodes
        self._steps = 0
        self._episodes = 0
        # Initialize and return if it is successful
        return self._initialize()

    def _initialize(self) -> bool:
        """
        Initialize the volley attributes.

        :return: True if initialization is successful, false otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def run(self,
            logger: logging.Logger,
            session,
            render: bool = False):
        """
        Run the volley.

        :param logger: the logger used to print the volley information, warnings and errors
        :param session: the session of tensorflow currently running
        :param render: flag defining whether to render the environment or not at every step
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def parallel(self) -> int:
        """
        Number of parallel episodes run by the volley.
        """
        return self._parallel

    @property
    def number(self) -> int or None:
        """
        Number of the volley, i.e. 1st volley (number 0), 2nd volley (number 1), etc.
        It is None if volley is not setup.
        """
        return self._number

    @property
    def steps(self) -> int or None:
        """
        The number of steps run after current setup of the volley (considering as step zero the first one run after
        setup).
        It is None if volley is not setup.
        """
        return self._steps

    @property
    def episodes(self) -> int or None:
        """
        The number of episodes run after current setup of the volley (considering as step zero the first one run after
        setup).
        It is None if volley is not setup.
        """
        return self._episodes

    @property
    def start_steps(self) -> int or None:
        """
        The number of steps already executed when current setup of the volley is done.
        It is None if volley is not setup.
        """
        return self._start_steps

    @property
    def start_episodes(self) -> int or None:
        """
        The number of episodes already executed when current setup of the volley is done.
        It is None if volley is not setup.
        """
        return self._start_episodes
