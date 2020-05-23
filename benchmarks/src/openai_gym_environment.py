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
import numpy

# Import usienarl

from usienarl.environment import Environment, SpaceType


class OpenAIGymEnvironment(Environment):
    """
    Wrapper environment for any OpenAI gym environment.
    It supports parallelization (by default it is 1, so no parallelization).
    """

    def __init__(self,
                 name: str):
        # Define environment attributes
        self._gym_environments: [] = []
        # Define environment empty attributes
        self._last_step_episode_done_flags: numpy.ndarray or None = None
        self._last_step_states: numpy.ndarray or None = None
        # Generate the base environment
        super(OpenAIGymEnvironment, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger) -> bool:
        # Close all previous environments, if any
        self.close(logger, None)
        # Generate all new parallel environments
        for i in range(self._parallel):
            self._gym_environments.append(gym.make(self._name))
        # Setup attributes
        self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
        if self.state_space_type == SpaceType.continuous:
            self._last_step_states: numpy.ndarray = numpy.zeros((self._parallel, *self.state_space_shape), dtype=float)
        else:
            self._last_step_states: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        # Close all the environments
        for i in range(len(self._gym_environments)):
            self._gym_environments[i].close()
        # Clear all the environments
        self._gym_environments = []

    def reset(self,
              logger: logging.Logger,
              session) -> numpy.ndarray:
        # Prepare list of return values
        start_states: [] = []
        # Reset all parallel environments
        self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
        if self.state_space_type == SpaceType.continuous:
            self._last_step_states: numpy.ndarray = numpy.zeros((self._parallel, *self.state_space_shape), dtype=float)
        else:
            self._last_step_states: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
        for i in range(len(self._gym_environments)):
            start_state = self._gym_environments[i].reset()
            start_states.append(start_state)
        # Return start states wrapped in a numpy array
        return numpy.array(start_states)

    def step(self,
             logger: logging.Logger,
             session,
             action: numpy.ndarray) -> ():
        # Make sure the action is properly sized
        assert (len(self._gym_environments) == action.shape[0])
        # Prepare list of return values
        states: [] = []
        rewards: [] = []
        episode_done_flags: [] = []
        # Make a step in all non completed environments
        for i in range(len(self._gym_environments)):
            # Add dummy values to return if this parallel environment episode is already done
            if self._last_step_episode_done_flags[i]:
                states.append(self._last_step_states[i])
                rewards.append(0.0)
                episode_done_flags.append(True)
                continue
            # Execute the step in this parallel environment
            state_next, reward, episode_done, _ = self._gym_environments[i].step(action[i])
            # Save results
            states.append(state_next)
            rewards.append(reward)
            episode_done_flags.append(episode_done)
            # Update last step flags and states
            self._last_step_episode_done_flags[i] = episode_done
            self._last_step_states[i] = state_next
        # Return the new states, rewards and episode done flags wrapped in numpy array
        return numpy.array(states), numpy.array(rewards), numpy.array(episode_done_flags)

    def render(self,
               logger: logging.Logger,
               session):
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Render all the environments
        for i in range(len(self._gym_environments)):
            self._gym_environments[i].render()

    def sample_action(self,
                      logger: logging.Logger,
                      session) -> numpy.ndarray:
        # Prepare list of return values
        actions: [] = []
        # Sample action from all parallel environment
        for i in range(len(self._gym_environments)):
            actions.append(self._gym_environments[i].action_space.sample())
        # Return sampled actions wrapped in numpy array
        return numpy.array(actions)

    def possible_actions(self,
                         logger: logging.Logger,
                         session) -> []:
        # Prepare list of return values
        possible_actions: [] = []
        # Compute the possible actions for all parallel environments
        for i in range(len(self._gym_environments)):
            # Get the array of possible actions depending on the type of the action space
            if self.action_space_type == SpaceType.discrete:
                possible_actions.append([action for action in range(self._gym_environments[i].action_space.n)])
            else:
                possible_actions.append([self._gym_environments[i].action_space.low, self._gym_environments[i].action_space.high])
        # Return the possible actions list
        return possible_actions

    @property
    def state_space_type(self) -> SpaceType:
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Get the state space type depending on the gym space type
        # Note: they are all equal so the first is used
        if isinstance(self._gym_environments[0].observation_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def state_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Get the state space size depending on the gym space type
        # Note: they are all equal so the first is used
        if isinstance(self._gym_environments[0].observation_space, gym.spaces.Discrete):
            return self._gym_environments[0].observation_space.n,
        else:
            return self._gym_environments[0].observation_space.high.shape

    @property
    def action_space_type(self) -> SpaceType:
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Get the action space type depending on the gym space type
        # Note: they are all equal so the first is used
        if isinstance(self._gym_environments[0].action_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def action_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Get the action space size depending on the gym space type
        # Note: they are all equal so the first is used
        if isinstance(self._gym_environments[0].action_space, gym.spaces.Discrete):
            return self._gym_environments[0].action_space.n,
        else:
            return self._gym_environments[0].action_space.high.shape

