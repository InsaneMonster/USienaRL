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

# Import required src

from usienarl import Environment, SpaceType


class Interface:
    """
    TODO: _summary

    """

    def __init__(self,
                 environment: Environment):
        # Define interface attribute
        self.environment: Environment = environment

    def get_random_agent_action(self,
                                logger: logging.Logger,
                                session):
        """
        Get a random action as seen by the agent, i.e. a random action in the environment translated in agent action.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the random action as seen by the agent
        """
        # Get the random environment action
        environment_action = self.environment.get_random_action(logger, session)
        # Translate it to agent action and return it
        return self.environment_action_to_agent_action(logger, session, environment_action)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action):
        """
        Translate the given agent action to the respective environment action.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_action: the action as seen by the agent
        :return: the action as seen by the environment relative to the given agent action
        """
        raise NotImplementedError()

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action):
        """
        Translate the given environment action to the respective agent action.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_action: the action as seen by the environment
        :return: the action as seen by the agent relative to the given environment action
        """
        raise NotImplementedError()

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state):
        """
        Translate the given environment state to the respective agent state, i.e. observation.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state: the state as seen by the environment
        :return: the state as seen (observed) by the agent relative to the given environment state
        """
        raise NotImplementedError()

    @property
    def observation_space_type(self) -> SpaceType:
        """
        Get the type (according to its SpaceType enum definition) of the observation space.

        :return: the SpaceType describing the observation space type
        """
        raise NotImplementedError()

    @property
    def observation_space_shape(self):
        """
        Get the shape of the observation space. This may differ from the environment state space.

        :return: the shape of the observation space in the form of a tuple
        """
        raise NotImplementedError()

    @property
    def agent_action_space_type(self) -> SpaceType:
        """
        Get the type (according to its SpaceType enum definition) of the agent action space.

        :return: the SpaceType describing the agent action space type
        """
        raise NotImplementedError()

    @property
    def agent_action_space_shape(self):
        """
        Get the shape of the agent action space. This may differ from the environment action space.

        :return: the shape of the agent action space in the form of a tuple
        """
        raise NotImplementedError()
