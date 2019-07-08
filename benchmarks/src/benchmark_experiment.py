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
import numpy
import math

# Import required src

from usienarl import Experiment, Environment, Agent, Interface
from .vpg_agent import VPGAgent


class BenchmarkExperiment(Experiment):
    """
    TODO: summary
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None,
                 episode_length_max: int = math.inf):
        # Define benchmark agent attributes
        self._validation_threshold: float = validation_threshold
        self._episode_length_max: int = episode_length_max
        # Generate the base experiment
        super(BenchmarkExperiment, self).__init__(name, environment, agent, interface)

    def _pre_train(self,
                   logger: logging.Logger,
                   episodes: int, session,
                   render: bool = False):
        for episode in range(episodes):
            # Initialize episode completion flag
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self._environment.reset(logger, session)
            # Initialize the step of the episode
            step: int = 0
            # Execute actions until the episode is completed or the maximum length is exceeded
            while not episode_done and not step > self._episode_length_max:
                # Increase the step counter
                step += 1
                # Execute a pre-train step
                state_next, reward, episode_done = self._pre_train_step(logger, session, state_current, render)
                # Update the current state with the previously next state
                state_current = state_next
            # Print intermediate pre-train completion (every 1/10 of the total pre-training episodes)
            if episode % (episodes // 10) == 0 and episode > 0:
                logger.info("Pre-trained for " + str(episode) + " episodes")

    def _train(self,
               logger: logging.Logger,
               episodes: int, session,
               render: bool = False):
        # Choose how to train the agent depending on its type
        if not isinstance(self._agent, VPGAgent):
            scores: numpy.ndarray = self._train_temporal_difference_agents(logger, episodes, session, render)
        else:
            scores: numpy.ndarray = self._train_policy_optimization_agents(logger, episodes, session, render)
        # Return average score over given episodes
        return numpy.average(scores)

    def _train_policy_optimization_agents(self,
                                          logger: logging.Logger,
                                          episodes: int, session,
                                          render: bool = False) -> numpy.ndarray:
        """
        Train logic for policy optimization agents.
        For the parameters check the original _train method.
        """
        # Define list of scores
        scores: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        for episode in range(episodes):
            # Initialize score and episode completion flag
            episode_score: float = 0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self._environment.reset(logger, session)
            # Initialize the step of the episode
            step: int = 0
            # Execute actions until the episode is completed or the maximum length is exceeded
            while not episode_done and not step > self._episode_length_max:
                # Increment the step counter
                step += 1
                # Execute a train step
                state_next, reward, episode_done = self._train_step(logger, session, state_current, render)
                # Update score for this episode with the given reward
                episode_score += reward
                # Update the current state with the previously next state
                state_current = state_next
            # Update the agent internal model according to its own logic and only if it's a PO agent
            self._agent.update(logger, session, episode, episodes, step)
            # Add the episode reward to the list of rewards
            scores[episode] = episode_score
        # Return back the scores array to the train method
        return scores

    def _train_temporal_difference_agents(self,
                                          logger: logging.Logger,
                                          episodes: int, session,
                                          render: bool = False) -> numpy.ndarray:
        """
        Train logic for temporal difference agents.
        For the parameters check the original _train method.
        """
        # Define list of scores
        scores: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        for episode in range(episodes):
            # Initialize score and episode completion flag
            episode_score: float = 0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self._environment.reset(logger, session)
            # Initialize the step of the episode
            step: int = 0
            # Execute actions until the episode is completed or the maximum length is exceeded
            while not episode_done and not step > self._episode_length_max:
                # Increment the step counter
                step += 1
                # Execute a train step
                state_next, reward, episode_done = self._train_step(logger, session, state_current, render)
                # If episode completed set the next state to None
                if episode_done:
                    state_next = None
                # Update the agent internal model according to its own logic
                self._agent.update(logger, session, episode, episodes, step)
                # Update score for this episode with the given reward
                episode_score += reward
                # Update the current state with the previously next state
                state_current = state_next
            # Add the episode reward to the list of rewards
            scores[episode] = episode_score
        # Return back the scores array to the train method
        return scores

    def _inference(self,
                   logger: logging.Logger,
                   episodes: int, session,
                   render: bool = False):
        # Define list of scores
        scores: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        for episode in range(episodes):
            # Initialize score and episode completion flag
            episode_score: float = 0.0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self._environment.reset(logger, session)
            # Initialize the step of the episode
            step: int = 0
            # Execute actions until the episode is completed or the maximum length is exceeded
            while not episode_done and not step > self._episode_length_max:
                # Increase the step counter
                step += 1
                # Execute an inference step
                state_next, reward, episode_done = self._inference_step(logger, session, state_current, render)
                # Update score for this episode with the given reward
                episode_score += reward
                # Update the current state with the previously next state
                state_current = state_next
            # Add the episode reward to the list of rewards
            scores[episode] = episode_score
        # Return average score over given episodes
        return numpy.average(scores)

    def _is_validated(self,
                      total_training_episodes: int, total_training_steps: int,
                      average_validation_score: float, average_training_score: float) -> bool:
        # Check if average validation score is over validation threshold
        if average_validation_score >= self._validation_threshold:
            return True
        return False

    def _is_successful(self,
                       total_training_episodes: int, total_training_steps: int,
                       average_test_score: float, best_test_score: float) -> bool:
        # Benchmark are always solved when validated over one-hundred episodes
        return True
