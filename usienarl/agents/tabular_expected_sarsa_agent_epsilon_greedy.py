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
import random
import tensorflow

# Import usienarl

from usienarl import Agent, Interface, SpaceType
from usienarl.models import TabularExpectedSARSA


class TabularExpectedSARSAAgentEpsilonGreedy(Agent):
    """
    Tabular Expected SARSA agent with epsilon greedy exploration policy.
    It is supplied with a Tabular Expected SARSA model.

    The batch size define how many steps from the prioritized experience replay built-in buffer should be fed into
    the model when updating. The agent uses a 1-step temporal difference algorithm, i.e. it is updated at every step.

    During training, agent will chose a random action with probability epsilon. Such probability is called exploration
    rate and decays at every completed episode of its decay value, while remaining bounded by its defined max and min
    values.
    """

    def __init__(self,
                 name: str,
                 model: TabularExpectedSARSA,
                 summary_save_every_steps: int = 500,
                 batch_size: int = 32,
                 exploration_rate_max: float = 1.0, exploration_rate_min: float = 0.001,
                 exploration_rate_decay: float = 0.001):
        # Define internal attributes
        self._model: TabularExpectedSARSA = model
        self._exploration_rate_max: float = exploration_rate_max
        self._exploration_rate_min: float = exploration_rate_min
        self._exploration_rate_decay: float = exploration_rate_decay
        self._summary_save_every_steps: int = summary_save_every_steps
        self._batch_size: int = batch_size
        # Define empty attributes
        self._last_training_step: int = 0
        self._last_summary_save_step: int = 0
        self._exploration_rate: float or None = None
        # Generate base agent
        super(TabularExpectedSARSAAgentEpsilonGreedy, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the model and return a flag stating if generation was successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    self._parallel,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal attributes
        self._last_training_step = 0
        self._last_summary_save_step = 0
        # Initialize the model
        self._model.initialize(logger, session)
        # Reset exploration rate to its starting value (the max)
        self._exploration_rate = self._exploration_rate_max

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current: numpy.ndarray,
                   warmup_step: int, warmup_episode: int):
        # Act randomly
        return interface.sample_agent_action(logger, session)

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int):
        # Choose an action according to the epsilon greedy approach: best action predicted by the model or random action
        action: numpy.ndarray = self._model.get_action_with_highest_q_value(session, agent_observation_current, interface.possible_agent_actions(logger, session))
        if self._exploration_rate > 0:
            random_action: numpy.ndarray = interface.sample_agent_action(logger, session)
            for i in range(self._parallel):
                if random.uniform(0, 1.0) < self._exploration_rate:
                    action[i] = random_action[i]
        # Return the exploration action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current: numpy.ndarray,
                      inference_step: int, inference_episode: int):
        # Act with the best policy according to the model
        return self._model.get_action_with_highest_q_value(session, agent_observation_current, interface.possible_agent_actions(logger, session))

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current: numpy.ndarray,
                             agent_action: numpy.ndarray,
                             reward: numpy.ndarray,
                             episode_done: numpy.ndarray,
                             agent_observation_next: numpy.ndarray,
                             warmup_step: int, warmup_episode: int):
        # Adjust the next observation if the episode is done
        agent_observation_next[episode_done] = 0
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), agent_observation_next.copy(), episode_done.copy())

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current: numpy.ndarray,
                            agent_action: numpy.ndarray,
                            reward: numpy.ndarray,
                            episode_done: numpy.ndarray,
                            agent_observation_next: numpy.ndarray,
                            train_step: int, train_episode: int):
        # Adjust the next observation if the episode is done
        agent_observation_next[episode_done] = 0
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(),
                                 agent_observation_next.copy(), episode_done.copy())
        # Get the number of steps actually done in the environment according to parallelization
        for step in range(self._last_training_step, train_step):
            # Update the model and get current loss and absolute errors
            loss, absolute_errors = self._model.update(session, self._model.buffer.get(self._batch_size))
            # Update the buffer with absolute error
            self._model.buffer.update(absolute_errors)
            # Save the summary at the current step if it is the appropriate time and if required
            if self._summary_writer is not None:
                if step - self._last_summary_save_step >= self._summary_save_every_steps:
                    # Generate the tensorflow summary on the loss and add it to the writer
                    summary = tensorflow.Summary()
                    summary.value.add(tag="loss", simple_value=loss)
                    self._summary_writer.add_summary(summary, step)
                    # Update last summary save step
                    self._last_summary_save_step = step
        # Update last training step
        self._last_training_step = train_step

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current: numpy.ndarray,
                                agent_action: numpy.ndarray,
                                reward: numpy.ndarray,
                                episode_done: numpy.ndarray,
                                agent_observation_next: numpy.ndarray,
                                inference_step: int, inference_episode: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: numpy.ndarray,
                                episode_total_reward: numpy.ndarray,
                                warmup_step: int, warmup_episode: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory()

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory()
        # Decrease the exploration rate by its decay value
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate - self._exploration_rate_decay)

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: numpy.ndarray,
                                   episode_total_reward: numpy.ndarray,
                                   inference_step: int, inference_episode: int):
        pass

    @property
    def saved_variables(self):
        # Return the trainable variables of the agent model in experiment/agent _scope
        return self._model.trainable_variables

    @property
    def warmup_steps(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_steps
