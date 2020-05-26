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
from usienarl.models import TabularSARSA
from usienarl.utils import softmax


class TabularSARSAAgentDirichlet(Agent):
    """
    Tabular SARSA agent with Dirichlet sampling exploration policy.
    It is supplied with a Tabular SARSA model and an exploration policy.

    The batch size define how many steps from the prioritized experience replay built-in buffer should be fed into
    the model when updating. The agent uses a 1-step temporal difference algorithm, i.e. it is updated at every step.

    During training, agent samples from the probability distribution of its computed values and the dirichlet probability
    distribution with given alpha, according to a trade-off parameter which is updated at the end of each episode.
    """

    def __init__(self,
                 name: str,
                 model: TabularSARSA,
                 summary_save_every_steps: int = 500,
                 batch_size: int = 32,
                 alpha: float = 1.0,
                 dirichlet_trade_off_min: float = 0.5, dirichlet_trade_off_max: float = 1.0,
                 dirichlet_trade_off_update: float = 0.001):
        # Define internal attributes
        self._model: TabularSARSA = model
        self._alpha: float = alpha
        self._dirichlet_trade_off_min: float = dirichlet_trade_off_min
        self._dirichlet_trade_off_max: float = dirichlet_trade_off_max
        self._dirichlet_trade_off_update: float = dirichlet_trade_off_update
        self._summary_save_every_steps: int = summary_save_every_steps
        self._batch_size: int = batch_size
        # Define empty attributes
        self._last_training_step: int = 0
        self._last_summary_save_step: int = 0
        self._next_warmup_action = None
        self._next_train_action = None
        self._dirichlet_trade_off: float or None = None
        # Generate base agent
        super(TabularSARSAAgentDirichlet, self).__init__(name)

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
        self._next_warmup_action = None
        self._next_train_action = None
        # Initialize the model
        self._model.initialize(logger, session)
        # Reset trade-off to its starting value (the min)
        self._dirichlet_trade_off = self._dirichlet_trade_off_min

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
            # Get the possible actions at the current step of the environment by its interface
            possible_actions: [] = interface.possible_agent_actions(logger, session)
            # Act according to dirichlet approach: first get the softmax over all the q-values predicted by the model
            prior_probabilities = softmax(self._model.get_q_values(session, agent_observation_current, possible_actions))
            # Then generate a dirichlet distribution (d) with parameter alpha
            dirichlet_probabilities: numpy.ndarray = numpy.empty(prior_probabilities.shape)
            for i in range(prior_probabilities.shape[0]):
                dirichlet_probabilities[i] = numpy.random.dirichlet(self._alpha * numpy.ones(prior_probabilities[i].size), 1)
            # Get a random action value (random output) using x * p + (1 - x) * d as probability distribution where x is the trade-off
            # Note: use a multiplicative to adjust dirichlet probabilities accordingly
            multiplicative_mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
            for i in range(self._parallel):
                multiplicative_mask[i, possible_actions[i]] = 1.0
            output = self._dirichlet_trade_off * prior_probabilities + (1 - self._dirichlet_trade_off) * (dirichlet_probabilities * multiplicative_mask)
            # Get an action for each row of the output
            action: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
            for i in range(self._parallel):
                # Make sure probability rows sums up to 1.0
                probability_row: numpy.ndarray = output[i] / output[i].sum()
                q_value = numpy.random.choice(probability_row, p=probability_row)
                # Set the chosen action as the index of such chosen action value
                action[i] = numpy.argmax(probability_row == q_value)
        # Return the action
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
        agent_observation_next[episode_done] = 0
        # Get the possible actions at the current step of the environment by its interface
        possible_actions: [] = interface.possible_agent_actions(logger, session)
        # Act according to dirichlet approach: first get the softmax over all the q-values predicted by the model
        prior_probabilities = softmax(self._model.get_q_values(session, agent_observation_current, possible_actions))
        # Then generate a dirichlet distribution (d) with parameter alpha
        dirichlet_probabilities: numpy.ndarray = numpy.empty(prior_probabilities.shape)
        for i in range(prior_probabilities.shape[0]):
            dirichlet_probabilities[i] = numpy.random.dirichlet(self._alpha * numpy.ones(prior_probabilities[i].size), 1)
        # Get a random action value (random output) using x * p + (1 - x) * d as probability distribution where x is the trade-off
        # Note: use a multiplicative to adjust dirichlet probabilities accordingly
        multiplicative_mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
        for i in range(self._parallel):
            multiplicative_mask[i, possible_actions[i]] = 1.0
        output = self._dirichlet_trade_off * prior_probabilities + (1 - self._dirichlet_trade_off) * (dirichlet_probabilities * multiplicative_mask)
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
        # Increase trade-off by its update value
        self._dirichlet_trade_off = min(self._dirichlet_trade_off_max, self._dirichlet_trade_off + self._dirichlet_trade_off_update)

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
