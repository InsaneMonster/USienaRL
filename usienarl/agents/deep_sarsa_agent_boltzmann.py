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
import tensorflow

# Import usienarl

from usienarl import Agent, Interface, SpaceType
from usienarl.models import DeepSARSA
from usienarl.utils import softmax


class DeepSARSAAgentBoltzmann(Agent):
    """
    Deep SARSA agent with Boltzmann sampling exploration policy.
    It is supplied with a Deep SARSA model.

    The weight copy step interval defines after how many steps per interval the target network weights should be
    updated with the main network weights.

    The batch size define how many steps from the prioritized experience replay built-in buffer should be fed into
    the model when updating. The agent uses a 1-step temporal difference algorithm, i.e. it is updated at every step.

    During training, agent samples according to the value of the temperature. Such temperatures decays at every
    completed episode of its decay value, while remaining bounded by its defined max and min values.

    Note: default model is suited for simple environment without complex dynamics (for example, without an action mask).
    It is advised to implement a custom model for research task using default ones as templates.
    """

    def __init__(self,
                 name: str,
                 model: DeepSARSA,
                 summary_save_every_steps: int = 500,
                 weight_copy_every_steps: int = 100,
                 batch_size: int = 64,
                 temperature_max: float = 1.0, temperature_min: float = 0.001,
                 temperature_decay: float = 0.001):
        # Define internal attributes
        self._model: DeepSARSA = model
        self._temperature_max: float = temperature_max
        self._temperature_min: float = temperature_min
        self._temperature_decay: float = temperature_decay
        self._summary_save_every_steps: int = summary_save_every_steps
        self._weight_copy_every_steps: int = weight_copy_every_steps
        self._batch_size: int = batch_size
        # Define empty attributes
        self._last_training_step: int = 0
        self._last_summary_save_step: int = 0
        self._last_weight_copy_step: int = 0
        self._next_warmup_action = None
        self._next_train_action = None
        self._temperature: float or None = None
        # Generate base agent
        super(DeepSARSAAgentBoltzmann, self).__init__(name)

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
        self._last_weight_copy_step = 0
        self._next_warmup_action = None
        self._next_train_action = None
        # Initialize the model
        self._model.initialize(logger, session)
        # Reset temperature to its starting value (the max)
        self._temperature = self._temperature_max
        # Run the weight copy operation to uniform main and target networks
        self._model.copy_weights(session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current: numpy.ndarray,
                   warmup_step: int, warmup_episode: int):
        # Act randomly or with the already chosen random action, if any
        action = self._next_warmup_action.copy() if self._next_warmup_action is not None else None
        if action is None:
            action = interface.sample_agent_action(logger, session)
        # Return the random action
        return action

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int):
        # Get the best action predicted by the model or use the already predicted action, if any
        action = self._next_train_action.copy() if self._next_train_action is not None else None
        if action is None:
            # Act according to boltzmann approach: get the softmax over all the q-values predicted by the model
            output = softmax(self._model.get_q_values(session, agent_observation_current, interface.possible_agent_actions(logger, session)) / self._temperature)
            # Get an action for each row of the output
            action: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
            for i in range(self._parallel):
                # Make sure probability rows sums up to 1.0
                probability_row: numpy.ndarray = output[i] / output[i].sum()
                action_value = numpy.random.choice(probability_row, p=probability_row)
                # Set the chosen action as the index of such chosen action value
                action[i] = numpy.argmax(probability_row == action_value)
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
        if self._observation_space_type == SpaceType.discrete:
            agent_observation_next[episode_done] = 0
        else:
            agent_observation_next[episode_done] = numpy.zeros(self._observation_space_shape, dtype=float)
        # Choose randomly the next action
        self._next_warmup_action = interface.sample_agent_action(logger, session)
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), agent_observation_next.copy(), self._next_warmup_action.copy(), episode_done.copy())

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
        if self._observation_space_type == SpaceType.discrete:
            agent_observation_next[episode_done] = 0
        else:
            agent_observation_next[episode_done] = numpy.zeros(self._observation_space_shape, dtype=float)
        # Act according to boltzmann approach: get the softmax over all the q-values predicted by the model
        output = softmax(self._model.get_q_values(session, agent_observation_current, interface.possible_agent_actions(logger, session)) / self._temperature)
        # Get an action for each row of the output
        action: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
        for i in range(self._parallel):
            # Make sure probability rows sums up to 1.0
            probability_row: numpy.ndarray = output[i] / output[i].sum()
            action_value = numpy.random.choice(probability_row, p=probability_row)
            # Set the chosen action as the index of such chosen action value
            action[i] = numpy.argmax(probability_row == action_value)
        # Set the next train action
        self._next_train_action = action.copy()
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(),
                                 agent_observation_next.copy(), self._next_train_action.copy(), episode_done.copy())
        # Get the number of steps actually done in the environment according to parallelization
        for step in range(self._last_training_step, train_step):
            # After each weight step interval update the target network weights with the main network weights
            if step - self._last_weight_copy_step >= self._weight_copy_every_steps:
                self._model.copy_weights(session)
                # Update last weight copy step
                self._last_weight_copy_step = step
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
        # Reset predicted next warmup action
        self._next_warmup_action = None

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory()
        # Reset predicted next train action
        self._next_train_action = None
        # Decrease temperature rate by its decay value
        self._temperature = max(self._temperature_min, self._temperature - self._temperature_decay)

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
        # Return the trainable variables of the agent model in experiment/agent scope
        return self._model.trainable_variables

    @property
    def warmup_steps(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_steps
