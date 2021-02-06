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

import os
import numpy
import logging
import matplotlib.pyplot as plot
import matplotlib.ticker as mticker
import enum
import time

# Import required src

from usienarl import Environment, Volley, Agent, Interface


class EpisodeVolleyType(enum.Enum):
    """
    Enum type of the episode volley: training, validation or test.
    """
    training = 0
    validation = 1
    test = 2


class EpisodeVolley(Volley):
    """
    Episode based volley. Used for training, validation and test volleys. It is executed for a certain amount of
    episodes. When run in training and validation modes, plots of each episodes (averaged to always be 100 episodes)
    are saved.
    """
    def __init__(self,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface,
                 parallel: int,
                 episode_volley_type: EpisodeVolleyType,
                 plots_path: str or None, plots_dpi: int or None,
                 episodes_required: int, episode_length: int):
        # Generate base volley
        super(EpisodeVolley, self).__init__(environment, agent, interface, parallel)
        # Make sure additional parameters are correct
        assert(episode_volley_type is not None)
        # Note: plots path and DPI are required only if the volley is not a test one
        if episode_volley_type != EpisodeVolleyType.test:
            assert(plots_path is not None and plots_path)
            assert(plots_dpi is not None and plots_dpi > 0)
        assert(episodes_required > 0 and episode_length > 0)
        assert(parallel > 0)
        assert(episodes_required % parallel == 0)
        # Define internal attributes
        self._episode_volley_type: EpisodeVolleyType = episode_volley_type
        self._plots_path: str or None = plots_path
        self._plots_dpi: int = plots_dpi
        self._episodes_required: int = episodes_required
        self._episode_length: int = episode_length
        # Define empty attributes
        self._last_episode_done: numpy.ndarray or None = None
        self._last_reward: numpy.ndarray or None = None
        self._avg_total_reward: float or None = None
        self._avg_scaled_reward: float or None = None
        self._std_total_reward: float or None = None
        self._std_scaled_reward: float or None = None
        self._avg_episode_length: int or None = None
        self._avg_action_duration: float or None = None
        self._rewards: [] = []
        self._total_rewards: [] = []
        self._scaled_rewards: [] = []
        self._episode_lengths: [] = []
        self._actions_durations: [] = []

    def _initialize(self) -> bool:
        # Reset empty attributes
        self._last_episode_done = None
        self._last_reward = None
        self._avg_total_reward = None
        self._avg_scaled_reward = None
        self._std_total_reward = None
        self._std_scaled_reward = None
        self._avg_episode_length = None
        self._avg_action_duration = None
        self._rewards = []
        self._total_rewards = []
        self._scaled_rewards = []
        self._episode_lengths = []
        self._actions_durations = []
        # This initialization always succeed
        return True

    def run(self,
            logger: logging.Logger,
            session,
            render: bool = False):
        # Print info
        if self._episode_volley_type == EpisodeVolleyType.training:
            logger.info("Training for " + str(self._episodes_required) + " episodes...")
        elif self._episode_volley_type == EpisodeVolleyType.validation:
            logger.info("Validating for " + str(self._episodes_required) + " episodes...")
        else:
            logger.info("Testing for " + str(self._episodes_required) + " episodes...")
        # Get the amount of parallel batches required
        parallel_episode_batches: int = self._episodes_required // self._parallel
        # Execute the parallel episode batches
        for parallel_episode_batch in range(parallel_episode_batches):
            # Print current progress every once in a while (if length is not too short)
            if parallel_episode_batches >= 100 and (parallel_episode_batch + 1) % (parallel_episode_batches // 10) == 0 and parallel_episode_batch > 0:
                if self._episode_volley_type == EpisodeVolleyType.training:
                    logger.info("Trained for " + str((parallel_episode_batch + 1) * self._parallel) + "/" + str(self._episodes_required) + " episodes...")
                elif self._episode_volley_type == EpisodeVolleyType.validation:
                    logger.info("Validated for " + str((parallel_episode_batch + 1) * self._parallel) + "/" + str(self._episodes_required) + " episodes...")
                else:
                    logger.info("Tested for " + str((parallel_episode_batch + 1) * self._parallel) + "/" + str(self._episodes_required) + " episodes...")
            # Initialize last reward and last episode done flags
            self._last_reward = numpy.nan * numpy.ones(self._environment.parallel, dtype=float)
            self._last_episode_done = numpy.zeros(self._environment.parallel, dtype=bool)
            # Execute actions until the all parallel step batches are completed or the maximum episode length is exceeded
            episode_rewards: [] = []
            episode_actions_durations: [] = []
            state_current: numpy.ndarray = self._environment.reset(logger, session)
            for parallel_step_batch in range(self._episode_length):
                # Get the action decided by the agent
                observation_current: numpy.ndarray = self._interface.environment_state_to_observation(logger, session, state_current)
                time_before_action = time.clock()
                if self._episode_volley_type == EpisodeVolleyType.training:
                    agent_action: numpy.ndarray = self._agent.act_train(logger, session, self._interface, observation_current,
                                                                        self._start_steps + self._steps, self._start_episodes + self._episodes)
                else:
                    agent_action: numpy.ndarray = self._agent.act_inference(logger, session, self._interface, observation_current,
                                                                            self._start_steps + self._steps, self._start_episodes + self._episodes)
                time_after_action = time.clock()
                # Save the time, converted to milliseconds
                episode_actions_durations.append((time_after_action - time_before_action) * 1000)
                # Get the next state with relative reward and episode done flag
                environment_action: numpy.ndarray = self._interface.agent_action_to_environment_action(logger, session, agent_action)
                state_next, reward, episode_done = self._environment.step(logger, session, environment_action)
                # Send back information to the agent
                observation_next: numpy.ndarray = self._interface.environment_state_to_observation(logger, session, state_next)
                # Complete the step
                if self._episode_volley_type == EpisodeVolleyType.training:
                    self._agent.complete_step_train(logger, session, self._interface,
                                                    observation_current, agent_action, reward, episode_done, observation_next,
                                                    self._start_steps + self._steps, self._start_episodes + self._episodes)
                else:
                    self._agent.complete_step_inference(logger, session, self._interface,
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
                # Note: make sure the reward saved is NaN for all already completed episodes in the parallel batch
                episode_rewards.append(numpy.where(self._last_episode_done, numpy.nan, reward))
                # Update the current state with the previously next state
                state_current = state_next.copy()
                # Increase the number of trained steps
                # Note: the counter should be increased according to the completed episodes of the current parallel batch
                self._steps += numpy.count_nonzero(episode_done == 0)
                # Save volley steps at termination time
                # Note: saving the step of each final step of each volley is required to compute the average episode length
                step_array: numpy.ndarray = numpy.ones(self._environment.parallel, dtype=int) * (parallel_step_batch + 1)
                self._episode_lengths += step_array[episode_done * numpy.logical_not(self._last_episode_done)].tolist()
                if parallel_step_batch + 1 == self._episode_length:
                    self._episode_lengths += step_array[numpy.logical_not(episode_done)].tolist()
                # Increase the episode counter
                self._episodes += numpy.count_nonzero((episode_done * (1 - self._last_episode_done)) == 1)
                # Save the current episode done flags
                # Note: episode done flag is a numpy array so it should be copied
                self._last_episode_done = episode_done.copy()
                # Check if the episode is already completed
                if all(episode_done):
                    break
            # Consider done also all parallel episodes truncated but not terminated
            self._episodes += numpy.count_nonzero(self._last_episode_done == 0)
            # Save rewards per episode and total/scaled rewards
            self._rewards.append(episode_rewards.copy())
            self._total_rewards += numpy.nansum(numpy.array(episode_rewards), axis=0).tolist()
            self._scaled_rewards += numpy.nanmean(numpy.array(episode_rewards), axis=0).tolist()
            # Save the average actions duration for the episode
            self._actions_durations.append(round(numpy.average(numpy.array(episode_actions_durations)), 3))
            # Complete the episode and send back information to the agent
            if self._episode_volley_type == EpisodeVolleyType.training:
                self._agent.complete_episode_train(logger, session, self._interface,
                                                   self._last_reward, self._total_rewards[-1],
                                                   self._start_steps + self._steps, self._start_episodes + self._episodes)
            else:
                self._agent.complete_episode_inference(logger, session, self._interface,
                                                       self._last_reward, self._total_rewards[-1],
                                                       self._start_steps + self._steps, self._start_episodes + self._episodes)
        # Compute statistics
        self._avg_total_reward = numpy.round(numpy.average(numpy.array(self._total_rewards)), 3)
        self._avg_scaled_reward = numpy.round(numpy.average(numpy.array(self._scaled_rewards)), 3)
        self._std_total_reward = numpy.round(numpy.std(numpy.array(self._total_rewards)), 3)
        self._std_scaled_reward = numpy.round(numpy.std(numpy.array(self._scaled_rewards)), 3)
        self._avg_episode_length = numpy.rint(numpy.average(numpy.array(self._episode_lengths)))
        self._avg_action_duration = numpy.round(numpy.average(numpy.array(self._actions_durations)), 3)
        # Print results
        if self._episode_volley_type == EpisodeVolleyType.training:
            logger.info("Training for " + str(self._episodes_required) + " episodes finished with following result:")
        elif self._episode_volley_type == EpisodeVolleyType.validation:
            logger.info("Validating for " + str(self._episodes_required) + " episodes finished with following result:")
        else:
            logger.info("Testing for " + str(self._episodes_required) + " episodes finished with following result:")
        logger.info("Average total reward: " + str(self._avg_total_reward))
        logger.info("Standard deviation of total reward: " + str(self._std_total_reward))
        logger.info("Average scaled reward: " + str(self._avg_scaled_reward))
        logger.info("Standard deviation of scaled reward: " + str(self._std_scaled_reward))
        logger.info("Average episode length: " + str(self._avg_episode_length) + " steps")
        logger.info("Average action duration: " + str(self._avg_action_duration) + " msec")
        # Save the episodes plots
        self._save_plots(logger)

    def _save_plots(self,
                    logger: logging.Logger):
        """
        Save plots for each episode of the volley. If more than 100 episodes have been executed, an average of some
        interval of them is used to normalize to 100. This does nothing if volley type is test.
        """
        # If this is a test volley return
        if self._episode_volley_type == EpisodeVolleyType.test:
            return
        # Make sure there is a plots path
        assert (self._plots_path is not None and self._plots_path)
        # Make sure all lengths are the same
        assert (len(self._total_rewards) == len(self._scaled_rewards) == len(self._episode_lengths))
        # Print info
        if self._episode_volley_type == EpisodeVolleyType.training:
            logger.info("Save training volley " + str(self._number) + " episodes plots...")
        else:
            logger.info("Save validation volley " + str(self._number) + " episodes plots...")
        # Compute statistic on the volley episodes
        # Note: if more than 100 episodes normalize to 100 episodes using interval averages (in order to avoid cluttering the plot)
        amount: int = len(self._total_rewards)
        interval: int = max(1, amount // 100)
        avg_total_rewards: [] = [sum(self._total_rewards[i:i + interval]) / interval for i in range(0, amount, interval)]
        avg_scaled_rewards: [] = [sum(self._scaled_rewards[i:i + interval]) / interval for i in range(0, amount, interval)]
        avg_episode_lengths: [] = [sum(self._episode_lengths[i:i + interval]) / interval for i in range(0, amount, interval)]
        # Generate volley dir:
        plot_directory = os.path.dirname(self._plots_path + "/volley_" + str(self._number) + "/")
        if not os.path.isdir(plot_directory):
            try:
                os.makedirs(plot_directory)
            except FileExistsError:
                pass
        # Save plots according to the requested type
        if self._episode_volley_type == EpisodeVolleyType.training:
            plot.plot(list(range(len(avg_total_rewards))), avg_total_rewards, 'r-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Training episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Training episode")
            plot.ylabel("Total reward")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_training_episodes_total_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(avg_scaled_rewards))), avg_scaled_rewards, 'r--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Training episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Training episode")
            plot.ylabel("Scaled reward")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_training_episodes_scaled_rewards.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(avg_episode_lengths))), avg_episode_lengths, 'b-.')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Training episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Training episode")
            plot.ylabel("Episode length (steps)")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_training_episodes_lengths_v.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
        else:
            plot.plot(list(range(len(avg_total_rewards))), avg_total_rewards, 'r-')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Validation episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Validation episode")
            plot.ylabel("Total reward")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_validation_episodes_total_rewards_v.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(avg_scaled_rewards))), avg_scaled_rewards, 'r--')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Validation episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Validation episode")
            plot.ylabel("Scaled reward")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_validation_episodes_scaled_rewards_v.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
            plot.plot(list(range(len(avg_episode_lengths))), avg_episode_lengths, 'b-.')
            plot.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if interval > 1:
                plot.xlabel("Validation episode (averaged every " + str(interval) + " episodes)")
            else:
                plot.xlabel("Validation episode")
            plot.ylabel("Episode length (steps)")
            plot.savefig(plot_directory + "/v" + str(self._number) + "_validation_episodes_lengths_v.png", dpi=self._plots_dpi, transparent=True)
            plot.clf()
        # Print info
        if self._episode_volley_type == EpisodeVolleyType.training:
            logger.info("Plots of training volley " + str(self._number) + " saved successfully")
        else:
            logger.info("Plots of validation volley " + str(self._number) + " saved successfully")

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
    def avg_total_reward(self) -> float or None:
        """
        The average total reward of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._avg_total_reward

    @property
    def avg_scaled_reward(self) -> float or None:
        """
        The average scaled reward of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._avg_scaled_reward

    @property
    def std_total_reward(self) -> float or None:
        """
        The standard deviation of total reward of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._std_total_reward

    @property
    def std_scaled_reward(self) -> float or None:
        """
        The standard deviation of scaled reward of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._std_scaled_reward

    @property
    def avg_episode_length(self) -> float or None:
        """
        The average episode length of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._avg_episode_length

    @property
    def avg_action_duration(self) -> float or None:
        """
        The average action duration in millisecond (msec) of all episodes of the volley.
        It is None if volley is not setup or if volley has not finished execution.
        """
        return self._avg_action_duration

    @property
    def rewards(self) -> []:
        """
        The list of rewards per step grouped by episode of all episodes already executed.
        """
        reward_list: [] = []
        # Note: each reward block consists in a set of arrays for each parallel step in an episode
        for reward_block in self._rewards:
            for i in range(len(reward_block[0])):
                episode_rewards: [] = []
                for parallel_step in reward_block:
                    if not numpy.isnan(parallel_step[i]):
                        episode_rewards.append(parallel_step[i])
                reward_list.append(episode_rewards)
        return reward_list

    @property
    def total_rewards(self) -> []:
        """
        The list of total rewards of all episodes already executed.
        """
        return self._total_rewards

    @property
    def scaled_rewards(self) -> []:
        """
        The list of scaled rewards of all episodes already executed.
        """
        return self._scaled_rewards

    @property
    def episode_lengths(self) -> []:
        """
        The list of episode lengths of all episodes already executed.
        """
        return self._episode_lengths

    @property
    def action_durations(self) -> []:
        """
        The list of average action durations of all episodes already executed.
        """
        return self._actions_durations
