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

# Import usienarl

from usienarl import Agent, Interface, SpaceType
from usienarl.td_models import DoubleDeepQLearning


class DoubleDeepQLearningAgentEpsilonGreedy(Agent):
    """
    Double Deep Q-Learning agent with epsilon greedy exploration policy.
    It is supplied with a Double Deep Q-Learning model (DDQN).

    The weight copy step interval defines after how many steps per interval the target network weights should be
    updated with the main network weights.

    The batch size define how many steps from the prioritized experience replay built-in buffer should be fed into
    the model when updating. The agent uses a 1-step temporal difference algorithm, i.e. it is updated at every step.

    During training, agent will chose a random action with probability epsilon. Such probability is called exploration
    rate and decays at every completed episode of its decay value, while remaining bounded by its defined max and min
    values.

    Note: default model is suited for simple environment without complex dynamics (for example, without an action mask).
    It is advised to implement a custom model for research task using default ones as templates.
    """

    def __init__(self,
                 name: str,
                 model: DoubleDeepQLearning,
                 weight_copy_step_interval: int,
                 batch_size: int = 1,
                 exploration_rate_max: float = 1.0, exploration_rate_min: float = 0.001,
                 exploration_rate_decay: float = 0.001):
        # Define agent attributes
        self._model: DoubleDeepQLearning = model
        self._exploration_rate_max: float = exploration_rate_max
        self._exploration_rate_min: float = exploration_rate_min
        self._exploration_rate_decay: float = exploration_rate_decay
        # Define internal agent attributes
        self._weight_copy_step_interval: int = weight_copy_step_interval
        self._batch_size: int = batch_size
        self._current_absolute_errors = None
        self._current_loss = None
        self._exploration_rate: float = None
        # Generate base agent
        super(DoubleDeepQLearningAgentEpsilonGreedy, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the model and return a flag stating if generation was successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_absolute_errors = None
        self._current_loss = None
        # Initialize the model
        self._model.initialize(logger, session)
        # Reset exploration rate to its starting value (the max)
        self._exploration_rate = self._exploration_rate_max
        # Run the weight copy operation to uniform main and target networks
        self._model.copy_weight(session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current):
        # Act randomly
        action = interface.get_random_agent_action(logger, session)
        # Return the random action
        return action

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current):
        # Choose an action according to the epsilon greedy approach: best action predicted by the model or random action
        if self._exploration_rate > 0 and random.uniform(0, 1.0) < self._exploration_rate:
            action = interface.get_random_agent_action(logger, session)
        else:
            action = self._model.get_best_action(session, agent_observation_current)
        # Return the exploration action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current):
        # Act with the best policy according to the model
        action = self._model.get_best_action(session, agent_observation_current)
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current,
                             agent_action, reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_steps_volley: int):
        # Adjust the next observation if None (final step)
        last_step: bool = False
        if agent_observation_next is None:
            last_step = True
            if self._observation_space_type == SpaceType.discrete:
                agent_observation_next = 0
            else:
                agent_observation_next = numpy.zeros(self._observation_space_shape, dtype=float)
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next, last_step)

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        # Adjust the next observation if None (final step)
        last_step: bool = False
        if agent_observation_next is None:
            last_step = True
            if self._observation_space_type == SpaceType.discrete:
                agent_observation_next = 0
            else:
                agent_observation_next = numpy.zeros(self._observation_space_shape, dtype=float)
        # After each weight step interval update the target network weights with the main network weights
        if train_step_absolute % self._weight_copy_step_interval == 0:
            logger.info("Copying weights from main network to target network...")
            self._model.copy_weight(session)
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next, last_step)
        # Update the model and save current loss and absolute errors
        summary, self._current_loss, self._current_absolute_errors = self._model.update(session, self._model.buffer.get(self._batch_size))
        # Update the buffer with the computed absolute error
        self._model.buffer.update(self._current_absolute_errors)
        # Update the summary at the absolute current step
        self._summary_writer.add_summary(summary, train_step_absolute)

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current,
                                agent_action,
                                reward: float,
                                agent_observation_next,
                                inference_step_current: int,
                                inference_episode_current: int,
                                inference_episode_volley: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_steps_volley: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        # Decrease the exploration rate by its decay value
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate - self._exploration_rate_decay)

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: float,
                                   episode_total_reward: float,
                                   inference_episode_current: int,
                                   inference_episode_volley: int):
        pass

    @property
    def trainable_variables(self):
        # Return the trainable variables of the agent model in experiment/agent _scope
        return self._model.trainable_variables

    @property
    def warmup_steps(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_steps
