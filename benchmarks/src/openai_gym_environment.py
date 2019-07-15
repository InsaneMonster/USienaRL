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

import gym
import logging

# Import required src

from usienarl.environment import Environment, SpaceType


class OpenAIGymEnvironment(Environment):
    """
    Wrapper environment for any OpenAI gym environment.
    """

    def __init__(self,
                 name: str):
        # Define environment specific attributes
        self._gym_environment = None
        # Generate the base environment
        super(OpenAIGymEnvironment, self).__init__(name)

    def setup(self,
              logger: logging.Logger) -> bool:
        # If the environment is already defined, close it first and then make it again (resetting it)
        if self._gym_environment is not None:
            self.close(logger, None)
        self._gym_environment = gym.make(self.name)
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        self._gym_environment.close()

    def reset(self,
              logger: logging.Logger,
              session):
        return self._gym_environment.reset()

    def step(self,
             logger: logging.Logger,
             action,
             session):
        state_next, reward, episode_done, _ = self._gym_environment.step(action)
        return state_next, reward, episode_done

    def render(self,
               logger: logging.Logger,
               session):
        self._gym_environment.render()

    def get_random_action(self,
                          logger: logging.Logger,
                          session):
        return self._gym_environment.action_space.sample()

    @property
    def state_space_type(self) -> SpaceType:
        # Get the state space type depending on the gym space type
        if isinstance(self._gym_environment.observation_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def state_space_shape(self):
        # Get the state space size depending on the gym space type
        if isinstance(self._gym_environment.observation_space, gym.spaces.Discrete):
            return (self._gym_environment.observation_space.n, )
        else:
            return self._gym_environment.observation_space.high.shape

    @property
    def action_space_type(self) -> SpaceType:
        # Get the action space type depending on the gym space type
        if isinstance(self._gym_environment.action_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def action_space_shape(self):
        # Get the action space size depending on the gym space type
        if isinstance(self._gym_environment.action_space, gym.spaces.Discrete):
            return (self._gym_environment.action_space.n, )
        else:
            return self._gym_environment.action_space.high.shape

