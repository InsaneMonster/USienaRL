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
import logging

# Import required src

from usienarl import Environment, SpaceType


class Interface:
    """
    TODO: summary

    """

    def __init__(self,
                 environment: Environment):
        # Define environment attribute
        self.environment: Environment = environment

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action):
        """
        Translate the given agent action to the respective environment action.
        By default, it just returns the agent action.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param agent_action: the action as seen by the agent
        :return: the action as seen by the environment relative to the given agent action
        """
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action):
        """
        Translate the given environment action to the respective agent action.
        By default, it just returns the environment action.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_action: the action as seen by the environment
        :return: the action as seen by the agent relative to the given environment action
        """
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state: numpy.ndarray) -> numpy.ndarray:
        """
        Translate the given environment state to the respective agent state, i.e. observation.
        By default, it just returns the environment state.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :param environment_state: the state as seen by the environment
        :return: the state as seen (observed) by the agent relative to the given environment state
        """
        return environment_state

    @property
    def observation_space_type(self) -> SpaceType:
        """
        Get the type (according to its SpaceType enum definition) of the observation space.
        By definition, the observation space has the same type of the environment state space.

        :return: the SpaceType describing the observation space type
        """
        return self.environment.state_space_type

    @property
    def observation_space_shape(self):
        """
        Get the shape of the observation space. This may differ from the environment state space.
        By default, it returns the same of the environment state space shape.

        :return: the shape of the observation space in the form of a tuple
        """
        return self.environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        """
        Get the type (according to its SpaceType enum definition) of the agent action space.
        By definition, the agent action space has the same type of the environment action space.

        :return: the SpaceType describing the agent action space type
        """
        return self.environment.action_space_type

    @property
    def agent_action_space_shape(self):
        """
        Get the shape of the agent action space. This may differ from the environment action space.
        By default, it returns the same of the environment action space shape.

        :return: the shape of the agent action space in the form of a tuple
        """
        return self.environment.action_space_shape
