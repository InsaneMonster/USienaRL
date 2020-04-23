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
import numpy

# Import required src

from usienarl import Environment, SpaceType


class Interface:
    """
    Base interface abstract class to define interaction between agent and environment.

    An interface allows to translate the environment state and actions to the agent observations and actions.
    The default interface is the pass-through interface, implementing a simple fully observable interface for the
    environment. The environment is specific for each interface.

    You should always define your own interface or use a pass-trough, the base class cannot be used as-is.
    """

    def __init__(self,
                 environment: Environment):
        # Make sure parameters are valid
        assert(environment is not None)
        # Define internal attributes
        self._environment: Environment = environment

    def sample_agent_action(self,
                            logger: logging.Logger,
                            session) -> numpy.ndarray:
        """
        Sample a random action as seen by the agent, i.e. a random action in the environment translated in agent action.
        The way the action is sampled depends on the environment.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: the random action as seen by the agent, wrapped in a numpy array
        """
        # Sample an action from the environment
        environment_action: numpy.ndarray = self._environment.sample_action(logger, session)
        # Translate it to agent action and return it
        return self.environment_action_to_agent_action(logger, session, environment_action)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action: numpy.ndarray) -> numpy.ndarray:
        """
        Translate the given agent action to the respective environment action, both wrapped in a numpy array to allow
        parallelization.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running
        :param agent_action: the action as seen by the agent, wrapped in a numpy array
        :return: the action as seen by the environment relative to the given agent action, wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action: numpy.ndarray) -> numpy.ndarray:
        """
        Translate the given environment action to the respective agent action, both wrapped in a numpy array to allow
        parallelization.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running
        :param environment_action: the action as seen by the environment, wrapped in a numpy array
        :return: the action as seen by the agent relative to the given environment action, wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state: numpy.ndarray) -> numpy.ndarray:
        """
        Translate the given environment state to the respective agent state, i.e. observation, both wrapped in a
        numpy array to allow parallelization.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running
        :param environment_state: the state as seen by the environment, wrapped in a numpy array
        :return: the state as seen (observed) by the agent relative to the given environment state, wrapped in a numpy array
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    def possible_agent_actions(self,
                               logger: logging.Logger,
                               session) -> numpy.ndarray:
        """
        Get a numpy array of all actions' indexes possible at the current states of the environment if the action space
        is discrete, translated into agent actions.
        Get a numpy array containing the lower and the upper bounds at the current states of the environment, each
        wrapped in numpy array with the shape of the  action space, translated into agent actions.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running
        :return: an array of indices containing the possible actions or an array of upper and lower bounds arrays, translated into agent actions
        """
        # Abstract method, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def environment(self) -> Environment:
        """
        The environment associated with the interface.
        """
        return self._environment

    @property
    def observation_space_type(self) -> SpaceType:
        """
        The type of the observation space of the agent.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def observation_space_shape(self) -> ():
        """
        The shape of the observation space of the agent.
        Note: it may differ from the environment's state space shape.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def agent_action_space_type(self) -> SpaceType:
        """
        The type of the action space of the agent.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def agent_action_space_shape(self) -> ():
        """
        The shape of the action space of the agent.
        Note: it may differ from the environment's action space shape.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()
