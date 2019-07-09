
# Import packages

import logging

# Import required src

from usienarl import Interface, Environment, SpaceType


class PassThroughInterface(Interface):
    """
    TODO: _summary

    """

    def __init__(self,
                 environment: Environment):
        # Generate the base interface
        super(PassThroughInterface, self).__init__(environment)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action):
        # Just return the agent action
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action):
        # Just return the environment action
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state):
        # Just return the environment state
        return environment_state

    @property
    def observation_space_type(self) -> SpaceType:
        # Just return the environment state space type
        return self.environment.state_space_type

    @property
    def observation_space_shape(self):
        # Just return the environment state space shape
        return self.environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        # Just return the environment action space type
        return self.environment.action_space_type

    @property
    def agent_action_space_shape(self):
        # Just return the environment action space shape
        return self.environment.action_space_shape
