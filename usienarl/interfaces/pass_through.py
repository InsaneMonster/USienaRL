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

from usienarl import Interface, Environment, SpaceType


class PassThroughInterface(Interface):
    """
    Basic pass-through interface just connecting an environment to an agent. It does not modify nor the shape nor
    the type of the actions and of the states.

    It is used by default in any experiment where a specific interface is not supplied.
    """

    def __init__(self,
                 environment: Environment):
        # Generate the base interface
        super(PassThroughInterface, self).__init__(environment)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action: numpy.ndarray) -> numpy.ndarray:
        # Just return the agent action
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action: numpy.ndarray) -> numpy.ndarray:
        # Just return the environment action
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state: numpy.ndarray) -> numpy.ndarray:
        # Just return the environment state
        return environment_state

    def possible_agent_actions(self,
                               logger: logging.Logger,
                               session) -> numpy.ndarray:
        # Generate the vectorized version of the translation function
        vectorized_translation = numpy.vectorize(self.environment_action_to_agent_action)
        # Return the vectorized translation of the possible actions
        return vectorized_translation(logger, session, self._environment.possible_actions(logger, session))

    @property
    def observation_space_type(self) -> SpaceType:
        # Just return the environment state space type
        return self._environment.state_space_type

    @property
    def observation_space_shape(self):
        # Just return the environment state space shape
        return self._environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        # Just return the environment action space type
        return self._environment.action_space_type

    @property
    def agent_action_space_shape(self):
        # Just return the environment action space shape
        return self._environment.action_space_shape
