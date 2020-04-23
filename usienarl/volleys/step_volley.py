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

import numpy
import logging

# Import required src

from usienarl import Environment, Volley, Agent, Interface


class StepVolley(Volley):
    """
    Step based volley. Used for warmup volleys.
    """
    def __init__(self,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface,
                 parallel: int,
                 steps_required: int, episode_length: int):
        # Generate base volley
        super(StepVolley, self).__init__(environment, agent, interface, parallel)
        # Make sure additional parameters are correct
        assert (steps_required > 0 and episode_length > 0)
        # Define internal attributes
        self._environment: Environment = environment
        self._agent: Agent = agent
        self._interface: Interface = interface
        self._parallel: int = parallel
        self._steps_required: int = steps_required
        self._episode_length: int = episode_length
        # Define empty attributes
        self._rewards: [] = []
        self._last_episode_done: numpy.ndarray or None = None
        self._last_reward: numpy.ndarray or None = None

    def _initialize(self) -> bool:
        # Reset empty attributes
        self._last_episode_done = None
        self._last_reward = None
        self._rewards = []
        # This initialization always succeed
        return True

    def run(self,
            logger: logging.Logger,
            session,
            render: bool = False):
        # Get the amount of parallel step batches
        parallel_step_batches: int = self._steps_required // self._parallel
        executed_parallel_step_batches: int = 0
        # Execute warm-up
        while True:
            # Initialize last reward and last episode done flags
            self._last_episode_done = numpy.zeros(self._environment.parallel, dtype=bool)
            self._last_reward = numpy.nan * numpy.ones(self._environment.parallel, dtype=float)
            # Execute actions until the all parallel step batches are completed or the maximum episode length is exceeded
            episode_rewards: [] = []
            state_current: numpy.ndarray = self._environment.reset(logger, session)
            for parallel_step_batch in range(self._episode_length):
                # Check if we can stop warming up
                if self._steps > self._steps_required:
                    return
                # Print current progress every once in a while (if length is not too short)
                if parallel_step_batches >= 1000 and executed_parallel_step_batches % (parallel_step_batches // 10) == 0 and executed_parallel_step_batches > 0:
                    logger.info("Warmed-up for " + str(self._steps) + "/" + str(self._steps_required) + " steps...")
                # Get the action decided by the agent
                observation_current: numpy.ndarray = self._interface.environment_state_to_observation(logger, session, state_current)
                agent_action: numpy.ndarray = self._agent.act_warmup(logger, session, self._interface, observation_current,
                                                                     self._start_steps + self._steps, self._start_episodes + self._episodes)
                # Get the next state with relative reward and completion flag
                environment_action: numpy.ndarray = self._interface.agent_action_to_environment_action(logger, session, agent_action)
                state_next, reward, episode_done = self._environment.step(logger, session, environment_action)
                # Send back information to the agent
                observation_next: numpy.ndarray = self._interface.environment_state_to_observation(logger, session, state_next)
                # Complete the step
                self._agent.complete_step_warmup(logger, session, self._interface,
                                                 observation_current, agent_action, reward, episode_done, observation_next,
                                                 self._start_steps + self._steps, self._start_episodes + self._episodes)
                # Render if required
                if render:
                    self._environment.render(logger, session)
                # Save the reward at the last step
                self._last_reward = numpy.where(episode_done * (1 - self._last_episode_done), reward, self._last_reward)
                if parallel_step_batch + 1 == self._episode_length:
                    self._last_reward = numpy.where(numpy.isnan(self._last_reward), reward, self._last_reward)
                # Add the reward to the list of rewards for this episode
                # Note: make sure the reward saved is NaN for all parallel episodes already completed
                episode_rewards.append(numpy.where(self._last_episode_done, numpy.nan, reward))
                # Update the current state with the previously next state
                state_current = state_next.copy()
                # Increase the number of warm-up steps
                # Note: the counter should be increased according to the completed episodes of the current parallel set
                self._steps += numpy.count_nonzero(episode_done == 0)
                # Increase the episode counter
                self._episodes += numpy.count_nonzero((episode_done * (1 - self._last_episode_done)) == 1)
                # Save the current episode done flags
                # Note: episode done flag is a numpy array so it should be copied
                self._last_episode_done = episode_done.copy()
                # Increase the counter of parallel warm-up steps batch already executed globally
                executed_parallel_step_batches += 1
                # Check if the episode is completed
                if all(episode_done):
                    break
            # Consider done also all parallel episodes truncated but not terminated
            self._episodes += numpy.count_nonzero(self._last_episode_done == 0)
            # Save rewards per episode
            self._rewards.append(episode_rewards.copy())
            # Complete the episode and send back information to the agent
            self._agent.complete_episode_warmup(logger, session, self._interface,
                                                self._last_reward, numpy.nansum(numpy.array(episode_rewards), axis=0),
                                                self._start_steps + self._steps, self._start_episodes + self._episodes)

    @property
    def last_episode_done(self) -> numpy.ndarray or None:
        """
        The episode done flag in the last step of the last episode of the volley.
        It is wrapped in a numpy array.
        It is None if volley is not setup.
        """
        return self._last_episode_done

    @property
    def last_reward(self) -> numpy.ndarray or None:
        """
        The reward in the last step of the last episode of the volley.
        It is wrapped in a numpy array.
        It is None if volley is not setup.
        """
        return self._last_reward

    @property
    def rewards(self) -> []:
        """
        The list of rewards per step grouped by episode of all episodes already executed.
        """
        return self._rewards
