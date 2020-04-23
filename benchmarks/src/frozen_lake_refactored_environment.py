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


class FrozenLakeRefactoredEnvironment(Environment):
    """
    Wrapper environment for any OpenAI gym FrozenLake environment with changed rewards.
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
        super(FrozenLakeRefactoredEnvironment, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger) -> bool:
        # Close all previous environments, if any
        self.close(logger, None)
        # Make sure the environment is the right one
        if self._name == "FrozenLake-v0" or self._name == "FrozenLake8x8-v0":
            # Generate all new parallel environments
            for i in range(self._parallel):
                self._gym_environments.append(gym.make(self._name))
            # Setup attributes
            self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
            self._last_step_states: numpy.ndarray = numpy.zeros(self._parallel, dtype=float)
            return True
        # Return an error if it is not the right one
        logger.error("FrozenLake Refactored environment requires a version of OpenAI gym FrozenLake to run!")
        return False

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
              session):
        # Prepare list of return values
        start_states: [] = []
        # Reset all parallel environments
        self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
        self._last_step_states: numpy.ndarray = numpy.zeros(self._parallel, dtype=float)
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
        assert (len(self._gym_environments) == action.size)
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
            # Change reward in order to make it easier to learn
            if reward == 0:
                reward = -1
            else:
                reward = 0
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
                         session) -> numpy.ndarray:
        # Prepare list of return values
        possible_actions: [] = []
        # Compute the possible actions for all parallel environments
        for i in range(len(self._gym_environments)):
            # Get the array of possible actions (always discrete on frozen lake)
            assert(self.action_space_type == SpaceType.discrete)
            possible_actions.append([action for action in range(self._gym_environments[i].action_space.n)])
        # Return the possible actions wrapped in a numpy array
        return numpy.array(possible_actions)

    @property
    def state_space_type(self) -> SpaceType:
        return SpaceType.discrete

    @property
    def state_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Just return the observation space size of the first parallel environment
        return self._gym_environments[0].observation_space.n,

    @property
    def action_space_type(self) -> SpaceType:
        return SpaceType.discrete

    @property
    def action_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._gym_environments) > 0)
        # Just return the action space size of the first parallel environment
        return self._gym_environments[0].action_space.n,

